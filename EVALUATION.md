# Scientific evaluation

NTRO-SRM separates two questions that are often conflated in satellite-image
super-resolution:

1. **Source consistency:** Does averaging the 2.5 m reconstruction back to the
   Sentinel-2 grid reproduce the observed reflectance?
2. **Reference accuracy:** Does the reconstruction agree with independently
   observed high-resolution imagery?

Source consistency can be calculated for every inference job. Reference
accuracy requires a georeferenced, spectrally compatible high-resolution image
covering the same location and acquisition period.

## Command line

```bash
python -m ntro_srm.evaluation path/to/sr_2.5m.tif \
  --source path/to/sentinel2_10m.tif \
  --reference path/to/reference_2.5m.tif \
  --output outputs/evaluation.json
```

At least one of `--source` and `--reference` is required. Inputs may use
different grids or coordinate systems as long as they are georeferenced. The
evaluator aligns common bands by GeoTIFF descriptions or `BAND_n` tags. Standard
untagged 3-, 4-, 10-, 12-, and 13-band Sentinel-2 layouts are recognized.

Radiometric scaling is explicit. Predictions and references are assumed to be
normalized reflectance by default, while the Sentinel-2 source is divided by
10,000. Use `--prediction-scale`, `--source-scale`, or `--reference-scale` to
provide a different divisor, such as `--reference-scale 255` for an 8-bit
reference. This avoids guessing a raster's meaning from its maximum value.

## Metrics

| Metric | Purpose | Better value |
|---|---|---|
| MAE / RMSE | Absolute reflectance error | Lower |
| Bias | Systematic brightening or darkening | Closer to zero |
| PSNR | Pixel fidelity on aligned imagery | Higher |
| SSIM | Local luminance, contrast, and structure agreement | Closer to one |
| SAM | Spectral-vector angle in degrees | Lower |
| ERGAS | Scale-normalized multispectral error | Lower |
| UIQI | Universal image-quality agreement | Closer to one |
| SCC | Correlation of high-frequency spatial detail | Higher |
| NDVI / NDWI drift | Preservation of analytical spectral indices | Lower |

Metrics are calculated over pixels valid in every compared band. Results are
reported per band as well as in aggregate.

## Operational analysis

The web service runs inexpensive analysis for every completed job:

- consistency after area-downsampling the 2.5 m product to the 10 m grid;
- a novelty-based confidence and hallucination-risk surface;
- NDVI, NDRE, SAVI, EVI, NDWI, MNDWI, NDBI, BSI, NBR, and NDMI products;
- a JSON and Markdown QA report with an explicit reconstruction caveat.

Wald synthesis validation and test-time-augmentation uncertainty are optional.
Wald evaluates a controlled 40 m to 10 m reconstruction against the real 10 m
observation and a bicubic baseline. It does not directly establish accuracy at
2.5 m. Test-time augmentation estimates output sensitivity across equivalent
flips and rotations, but costs one full inference per member.

## Interpretation limits

- PSNR and SSIM are sensitive to small spatial misregistration. Inspect the
  alignment before comparing models.
- A low source-consistency error only confirms that the SR product preserves the
  low-resolution observation after downsampling.
- High-resolution reference data should be close in acquisition time. Seasonal,
  atmospheric, or land-cover changes otherwise become part of the reported error.
- RGB-only references cannot validate Sentinel-2 red-edge and SWIR bands.
