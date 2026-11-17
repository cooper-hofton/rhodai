"""Non-CNN baselines on the same tiles and folds as train.py.

  1. GLCM texture features  + random forest
  2. Colour statistics      + logistic regression

Run from the Scripts folder on your own machine (CPU only, no TensorFlow):
    python baselines.py
or in a notebook:
    from baselines import CONFIG, run
    run(CONFIG)

Features are computed once and cached to ../Results/baseline_features_500px.csv,
so re-runs only refit the models. Each model writes a results file in the same
format as train.py, plus predictions.csv for precision-recall curves.
"""
import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# 1. Settings (match the CNN runs)
# ---------------------------------------------------------------------------
CONFIG = {
    "tile_size_px": 500,
    "max_order": 1000,                 # same training subset as cnn_rgb_500 / cnn_grey_500
    "forest": "KD02",
    "statuses": ["done"],
    "grid_dir": "../Grids",
    "image_dir": "../ImageFiles/InitialComparison",
    "results_dir": "../Results",
    "threshold": 0.5,

    # GLCM (texture)
    "glcm_levels": 32,                 # grey levels after quantising 0-255
    "glcm_distances": [1, 3],          # pixels (1 px = ~2.1 cm)
    "glcm_angles_deg": [0, 45, 90, 135],
    "glcm_props": ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"],

    # Random forest
    "rf_trees": 500,
    "rf_seed": 0,

    # Logistic regression: C chosen by inner CV on the training blocks only
    "lr_C_grid": [0.01, 0.1, 1, 10],
    "lr_inner_folds": 4,
}


# ---------------------------------------------------------------------------
# 2. Tiles (same filtering as train.py)
# ---------------------------------------------------------------------------
def load_tiles(cfg):
    size = cfg["tile_size_px"]
    t = gpd.read_file(Path(cfg["grid_dir"]) / f"tiles_{size}px.gpkg")
    t = t[(t["forest"] == cfg["forest"]) & (t["role"] == "train")
          & t["status"].isin(cfg["statuses"]) & t["fold"].notna()].copy()
    if cfg["max_order"] is not None:
        t = t[t["order"] <= cfg["max_order"]]
    t["fold"] = t["fold"].astype(int)
    t["path"] = [str(Path(cfg["image_dir"]) / str(size) / f"{i}_{size}_{r}_{c}.png")
                 for i, r, c in zip(t["tile_id"], t["sub_row"], t["sub_col"])]
    missing = [p for p in t["path"] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing, e.g. {missing[0]}")
    return t[["tile_id", "sub_row", "sub_col", "block", "order", "fold", "label",
              "cover_fraction", "path"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Features
# ---------------------------------------------------------------------------
STATS = {"mean": np.mean, "std": np.std,
         "p10": lambda a: np.percentile(a, 10), "p50": np.median,
         "p90": lambda a: np.percentile(a, 90)}


def colour_features(rgb):
    """Statistics that ignore where pixels are in the tile (colour only)."""
    r, g, b = [rgb[..., i].ravel() for i in range(3)]
    total = r + g + b + 1e-6
    chrom = {"r_chrom": r / total, "g_chrom": g / total, "b_chrom": b / total}
    f = {}
    for name, band in [("R", r), ("G", g), ("B", b), *chrom.items()]:
        for s, fn in STATS.items():
            f[f"col_{name}_{s}"] = fn(band)
    exg = 2 * chrom["g_chrom"] - chrom["r_chrom"] - chrom["b_chrom"]
    f["col_ExG_mean"], f["col_ExG_std"] = exg.mean(), exg.std()
    return f


def glcm_features(rgb, cfg):
    """Texture from grey-level co-occurrence (depends on pixel arrangement)."""
    grey = rgb.mean(axis=2)                                    # same greyscale as the CNN baseline
    q = np.clip(grey / 256 * cfg["glcm_levels"], 0, cfg["glcm_levels"] - 1).astype(np.uint8)
    angles = np.deg2rad(cfg["glcm_angles_deg"])
    m = graycomatrix(q, distances=cfg["glcm_distances"], angles=angles,
                     levels=cfg["glcm_levels"], symmetric=True, normed=True)
    f = {}
    for prop in cfg["glcm_props"]:
        v = graycoprops(m, prop)                               # shape (distances, angles)
        for i, d in enumerate(cfg["glcm_distances"]):
            f[f"glcm_{prop}_d{d}_mean"] = v[i].mean()          # average over directions
            f[f"glcm_{prop}_d{d}_range"] = np.ptp(v[i])        # how direction-dependent
    return f


def compute_features(tiles, cfg):
    cache = Path(cfg["results_dir"]) / f"baseline_features_{cfg['tile_size_px']}px.csv"
    if cache.exists():
        feats = pd.read_csv(cache)
        if set(tiles["path"]) <= set(feats["path"]):
            print(f"Using cached features: {cache}")
            return tiles.merge(feats, on="path", how="left")
    rows = []
    for path in tqdm(tiles["path"], desc="features"):
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        rows.append({"path": path, **colour_features(rgb), **glcm_features(rgb, cfg)})
    feats = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(cache, index=False)
    return tiles.merge(feats, on="path", how="left")


# ---------------------------------------------------------------------------
# 4. Metrics (same as train.py)
# ---------------------------------------------------------------------------
def score(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    both = len(np.unique(labels)) == 2
    return {"n": len(labels), "positives": int(labels.sum()),
            "roc_auc": roc_auc_score(labels, probs) if both else np.nan,
            "avg_precision": average_precision_score(labels, probs) if both else np.nan,
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
# 5. Models
# ---------------------------------------------------------------------------
def fit_rf(X, y, groups, cfg):
    model = RandomForestClassifier(n_estimators=cfg["rf_trees"], class_weight="balanced",
                                   random_state=cfg["rf_seed"], n_jobs=-1)
    model.fit(X, y)
    importance = pd.Series(model.feature_importances_, index=X.columns)
    return model, importance, "500 trees, balanced class weights"


def fit_lr(X, y, groups, cfg):
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=5000))
    n_inner = min(cfg["lr_inner_folds"], groups.nunique())
    search = GridSearchCV(pipe, {"logisticregression__C": cfg["lr_C_grid"]},
                          scoring="average_precision", cv=GroupKFold(n_inner))
    search.fit(X, y, groups=groups)                            # inner CV by block, training folds only
    coefs = pd.Series(search.best_estimator_[-1].coef_[0], index=X.columns)
    return search.best_estimator_, coefs, f"C={search.best_params_['logisticregression__C']}"


MODELS = {"glcm_rf_500": ("glcm_", fit_rf, "importance"),
          "colour_lr_500": ("col_", fit_lr, "standardised coefficient")}


# ---------------------------------------------------------------------------
# 6. Cross-validation loop
# ---------------------------------------------------------------------------
def run(cfg):
    tiles = compute_features(load_tiles(cfg), cfg)
    results_dir = Path(cfg["results_dir"])
    folds = sorted(tiles["fold"].unique())
    summary = {}

    for name, (prefix, fit, weight_name) in MODELS.items():
        report_path = results_dir / f"{name}.txt"
        if report_path.exists():
            print(f"{report_path} exists, skipping. Delete it to rerun.")
            continue
        cols = [c for c in tiles.columns if c.startswith(prefix)]
        out = open(report_path, "w")

        def log(text):
            print(text); out.write(text + "\n"); out.flush()

        log(f"Run {name}  started {datetime.now():%Y-%m-%d %H:%M}  |  CPU (scikit-learn)")
        log("Settings: " + json.dumps(cfg))
        log(f"{len(tiles)} images, {int(tiles['label'].sum())} positive, {len(cols)} features, folds {folds}\n")

        fold_rows, preds, weights = [], [], []
        for fold in folds:
            train, test = tiles[tiles["fold"] != fold], tiles[tiles["fold"] == fold]
            model, w, note = fit(train[cols], train["label"], train["block"], cfg)
            probs = model.predict_proba(test[cols])[:, 1]
            m = score(test["label"].to_numpy(), probs, cfg["threshold"])
            fold_rows.append({"seed": 0, "fold": fold, **m})
            preds.append(test.drop(columns=["path", *[c for c in tiles.columns if c.startswith(("glcm_", "col_"))]])
                         .assign(seed=0, prob=probs))
            weights.append(w)
            log(f"--- fold {fold}: train {len(train)} ({int(train['label'].sum())} pos), "
                f"test {len(test)} ({int(test['label'].sum())} pos), {note}")
            log(describe(f"Held-out fold {fold}", m))

        per_fold = pd.DataFrame(fold_rows)
        p = pd.concat(preds)
        (results_dir / name).mkdir(parents=True, exist_ok=True)
        p.to_csv(results_dir / name / "predictions.csv", index=False)

        log("=" * 70 + "\nSUMMARY\nPer fold:")
        log(per_fold[["fold", "n", "positives", "roc_auc", "avg_precision", "f1", "precision", "recall"]]
            .round(3).to_string(index=False))
        log("\nAcross folds (mean +/- SD):")
        for c in ["roc_auc", "avg_precision", "f1", "precision", "recall"]:
            log(f"    {c:<14} {per_fold[c].mean():.3f} +/- {per_fold[c].std():.3f}")
        pooled = score(p["label"].to_numpy(), p["prob"].to_numpy(), cfg["threshold"])
        log("\n" + describe("Pooled out-of-fold", pooled))
        top = pd.concat(weights, axis=1).mean(axis=1)
        top = top.reindex(top.abs().sort_values(ascending=False).index).head(10)
        log(f"Top 10 features by mean {weight_name} across folds:")
        for f, v in top.items():
            log(f"    {f:<34} {v:+.4f}")
        log(f"\nFinished {datetime.now():%Y-%m-%d %H:%M}")
        out.close()
        summary[name] = per_fold
    return summary


if __name__ == "__main__":
    run(CONFIG)
