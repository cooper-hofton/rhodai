from osgeo import gdal
gdal.UseExceptions()

def make_half_image(path, out_path):
    options = gdal.TranslateOptions(
        widthPct=50,
        heightPct=50,
        resampleAlg="average",
        creationOptions = ['TILED=YES', 'COMPRESS=DEFLATE', 'PREDICTOR=2', 'BIGTIFF=YES'],
        callback=gdal.TermProgress_nocb,
    )
    gdal.Translate(out_path, path, options=options)

make_half_image("../LargeImages/KD06-1-20241203-7_IXM150_mos.tif", "../LargeImages/HALFRES-KD06-1-20241203-7_IXM150_mos.tif")
