"""Spatial cross-validation training for rhododendron tile classification.

Run from the Scripts folder:
    python train.py
or in a notebook:
    from train import CONFIG, run
    CONFIG["tile_size_px"] = 500
    run(CONFIG)

For each seed and each fold, it trains MobileNetV2 on the other folds and
tests on the held-out fold. Everything (settings, loss per epoch, metrics,
confusion matrices) goes into one text file: ../Results/<run_name>.txt
"""
import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

# ---------------------------------------------------------------------------
# 1. Settings: everything you might change lives here
# ---------------------------------------------------------------------------
CONFIG = {
    "run_name": None,                  # None = built from the settings below
    "tile_size_px": 1000,              # 250, 500 or 1000
    "max_order": None,                 # e.g. 250 = train only on tiles with order <= 250
    "seeds": [0, 1, 2],                # one full CV run per seed
    "folds": None,                     # None = all folds; e.g. [0] for a quick test

    # Data
    "grid_dir": "../Grids",
    "image_dir": "../ImageFiles/InitialComparison",
    "results_dir": "../Results",
    "forest": "KD02",
    "statuses": ["done"],
    "greyscale": False,                # True for the greyscale CNN baseline

    # Model
    "resize_to_px": None,              # None = native resolution (250/500/1000 px in); e.g. 224 to resize
    "weights": "imagenet",             # None = random start (testing only)
    "train_base": True,                # True = fine-tune all of MobileNetV2; False = train the head only
    "dropout": 0.2,

    # Training
    "learning_rate": 1e-4,
    "batch_size": 32,
    "epochs": 20,
    "steps_per_epoch": 50,             # fixed, so every tile size gets the same number of updates
    "class_weights": True,             # weight positives by how rare they are
    "augment": True,                   # random flips and 90-degree rotations (training only)
    "contrast_range": [0.8, 1.2],      # random contrast factor; None = off
    "brightness_delta": 0.1,           # random brightness shift, as a fraction of the 0-255 range; None = off
    "threshold": 0.5,                  # probability cut-off for F1 and the confusion matrix

    # Saving (off for now; switch on for the final analysis)
    "save_models": False,
    "save_predictions": False,
}


# ---------------------------------------------------------------------------
# 2. Data: which images, which labels, which fold
# ---------------------------------------------------------------------------
def load_tiles(cfg):
    """One row per image: file path, label, fold, order."""
    size = cfg["tile_size_px"]
    tiles = gpd.read_file(Path(cfg["grid_dir"]) / f"tiles_{size}px.gpkg")
    tiles = tiles[(tiles["forest"] == cfg["forest"]) & (tiles["role"] == "train")
                  & tiles["status"].isin(cfg["statuses"]) & tiles["fold"].notna()].copy()
    tiles["fold"] = tiles["fold"].astype(int)
    tiles["path"] = [str(Path(cfg["image_dir"]) / str(size) / f"{t}_{size}_{r}_{c}.png")
                     for t, r, c in zip(tiles["tile_id"], tiles["sub_row"], tiles["sub_col"])]
    missing = [p for p in tiles["path"] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing, e.g. {missing[0]}")
    return tiles[["tile_id", "sub_row", "sub_col", "block", "order", "fold", "label", "cover_fraction", "path"]]


def make_dataset(paths, labels, cfg, training):
    """A tf.data pipeline: read PNG -> resize -> (augment) -> scale -> batch."""
    size = cfg["resize_to_px"]

    def load(path, label):
        img = tf.cast(tf.io.decode_png(tf.io.read_file(path), channels=3), tf.float32)   # 0-255
        if size is not None:
            img = tf.image.resize(img, (size, size), antialias=True)
        if cfg["greyscale"]:
            img = tf.repeat(tf.reduce_mean(img, axis=-1, keepdims=True), 3, axis=-1)
        return img, label

    def augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
        if cfg["brightness_delta"]:                                     # whole image lighter or darker
            img = tf.image.random_brightness(img, cfg["brightness_delta"] * 255)
        if cfg["contrast_range"]:                                       # spread around the mean squeezed or stretched
            img = tf.image.random_contrast(img, *cfg["contrast_range"])
        return tf.clip_by_value(img, 0, 255), label

    def scale(img, label):
        return keras.applications.mobilenet_v2.preprocess_input(img), label   # 0-255 -> -1..1

    ds = tf.data.Dataset.from_tensor_slices((list(paths), np.asarray(labels, dtype="float32")))
    if training:
        ds = ds.shuffle(len(paths), reshuffle_each_iteration=True).repeat()
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if training and cfg["augment"]:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(scale, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(cfg["batch_size"]).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# 3. Model: pretrained MobileNetV2 with a new one-output head
# ---------------------------------------------------------------------------
def build_model(cfg):
    size = cfg["resize_to_px"] or cfg["tile_size_px"]
    base = keras.applications.MobileNetV2(input_shape=(size, size, 3), include_top=False,
                                          weights=cfg["weights"], pooling="avg")
    base.trainable = cfg["train_base"]
    inputs = keras.Input((size, size, 3))
    x = base(inputs, training=False)                                   # batch-norm layers stay in inference mode
    x = keras.layers.Dropout(cfg["dropout"])(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)           # probability of rhododendron
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(cfg["learning_rate"]),
                  loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="roc_auc"),
                           keras.metrics.AUC(curve="PR", name="avg_precision")])
    return model


# ---------------------------------------------------------------------------
# 4. Metrics on held-out predictions
# ---------------------------------------------------------------------------
def score(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    both_classes = len(np.unique(labels)) == 2
    return {"n": len(labels), "positives": int(labels.sum()),
            "roc_auc": roc_auc_score(labels, probs) if both_classes else np.nan,
            "avg_precision": average_precision_score(labels, probs) if both_classes else np.nan,
            "f1": f1_score(labels, preds, zero_division=0),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def describe(name, m):
    return (f"{name}: n={m['n']} positives={m['positives']}  "
            f"ROC-AUC={m['roc_auc']:.3f}  AP={m['avg_precision']:.3f}  F1={m['f1']:.3f}  "
            f"precision={m['precision']:.3f}  recall={m['recall']:.3f}\n"
            f"    confusion matrix     predicted 0   predicted 1\n"
            f"    actual 0 (absent)    {m['tn']:>11}   {m['fp']:>11}\n"
            f"    actual 1 (present)   {m['fn']:>11}   {m['tp']:>11}\n")


# ---------------------------------------------------------------------------
# 5. The cross-validation loop
# ---------------------------------------------------------------------------
def run(cfg):
    cfg = dict(cfg)
    if cfg["run_name"] is None:
        order = "all" if cfg["max_order"] is None else cfg["max_order"]
        grey = "_grey" if cfg["greyscale"] else ""
        cfg["run_name"] = f"size{cfg['tile_size_px']}_order{order}{grey}"
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / f"{cfg['run_name']}.txt"           # one file per run, all in one folder
    if report_path.exists():
        raise FileExistsError(f"{report_path} already exists. Change run_name or delete it.")
    extras_dir = results_dir / cfg["run_name"]                     # only used if saving models/predictions

    tiles = load_tiles(cfg)
    folds = cfg["folds"] if cfg["folds"] is not None else sorted(int(f) for f in tiles["fold"].unique())
    report = open(report_path, "w")

    def log(text):
        print(text)
        report.write(text + "\n")
        report.flush()

    log(f"Run {cfg['run_name']}  started {datetime.now():%Y-%m-%d %H:%M}")
    log("Settings: " + json.dumps(cfg))
    log(f"{len(tiles)} images, {int(tiles['label'].sum())} positive, folds {folds}\n")

    fold_rows, predictions = [], []
    for seed in cfg["seeds"]:
        for fold in folds:
            keras.utils.set_random_seed(seed)          # same start for every setting
            train = tiles[tiles["fold"] != fold]
            if cfg["max_order"] is not None:
                train = train[train["order"] <= cfg["max_order"]]
            test = tiles[tiles["fold"] == fold]

            weights = None
            if cfg["class_weights"]:
                p = train["label"].mean()
                weights = {0: round(float(0.5 / (1 - p)), 3), 1: round(float(0.5 / p), 3)}   # each class gives half the loss
            log(f"--- seed {seed}, fold {fold}: train {len(train)} ({int(train['label'].sum())} pos), "
                f"test {len(test)} ({int(test['label'].sum())} pos), class weights {weights}")

            model = build_model(cfg)
            history = model.fit(make_dataset(train["path"], train["label"], cfg, training=True),
                                epochs=cfg["epochs"], steps_per_epoch=cfg["steps_per_epoch"],
                                class_weight=weights, verbose=2)
            h = history.history
            log("    epoch    loss   train ROC-AUC   train AP")
            for e in range(len(h["loss"])):
                log(f"    {e + 1:>5}  {h['loss'][e]:.4f}  {h['roc_auc'][e]:>13.3f}  {h['avg_precision'][e]:>9.3f}")

            probs = model.predict(make_dataset(test["path"], test["label"], cfg, training=False),
                                  verbose=0).ravel()
            m = score(test["label"].to_numpy(), probs, cfg["threshold"])
            fold_rows.append({"seed": seed, "fold": fold, **m})
            predictions.append(test.drop(columns="path").assign(seed=seed, prob=probs))
            log(describe(f"Held-out fold {fold}", m))

            if cfg["save_models"]:
                extras_dir.mkdir(exist_ok=True)
                model.save(extras_dir / f"model_seed{seed}_fold{fold}.keras")

    # Summary: mean and SD across folds, plus pooled out-of-fold predictions per seed
    per_fold = pd.DataFrame(fold_rows)
    preds = pd.concat(predictions)
    if cfg["save_predictions"]:
        extras_dir.mkdir(exist_ok=True)
        preds.to_csv(extras_dir / "predictions.csv", index=False)

    log("=" * 70 + "\nSUMMARY")
    log("Per fold:")
    log(per_fold[["seed", "fold", "n", "positives", "roc_auc", "avg_precision", "f1", "precision", "recall"]]
        .round(3).to_string(index=False))
    log("\nAcross folds (mean +/- SD over all seeds and folds):")
    for c in ["roc_auc", "avg_precision", "f1", "precision", "recall"]:
        log(f"    {c:<14} {per_fold[c].mean():.3f} +/- {per_fold[c].std():.3f}")
    pooled = []
    for seed, p in preds.groupby("seed"):
        m = score(p["label"].to_numpy(), p["prob"].to_numpy(), cfg["threshold"])
        pooled.append({"seed": seed, **m})
        log("\n" + describe(f"Pooled out-of-fold, seed {seed}", m))
    log(f"Finished {datetime.now():%Y-%m-%d %H:%M}")
    report.close()
    return per_fold, pd.DataFrame(pooled)


if __name__ == "__main__":
    run(CONFIG)