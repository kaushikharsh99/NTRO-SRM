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

## Metrics

| Metric | Purpose | Better value |
|---|---|---|
| MAE / RMSE | Absolute reflectance error | Lower |
| Bias | Systematic brightening or darkening | Closer to zero |
| PSNR | Pixel fidelity on aligned imagery | Higher |
| SSIM | Local luminance, contrast, and structure agreement | Closer to one |
| SAM | Spectral-vector angle in degrees | Lower |
| ERGAS | Scale-normalized multispectral error | Lower |
| NDVI / NDWI drift | Preservation of analytical spectral indices | Lower |

Metrics are calculated over pixels valid in every compared band. Results are
reported per band as well as in aggregate.

## Interpretation limits

- PSNR and SSIM are sensitive to small spatial misregistration. Inspect the
  alignment before comparing models.
- A low source-consistency error only confirms that the SR product preserves the
  low-resolution observation after downsampling.
- High-resolution reference data should be close in acquisition time. Seasonal,
  atmospheric, or land-cover changes otherwise become part of the reported error.
- RGB-only references cannot validate Sentinel-2 red-edge and SWIR bands.
