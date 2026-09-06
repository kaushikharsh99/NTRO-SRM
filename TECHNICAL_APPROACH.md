# Technical approach for SIH26142

## Objective

NTRO-SRM converts a georeferenced Sentinel-2 Level-2A observation from its 10 m
working grid to a 2.5 m multispectral product. The system is designed to improve
spatial detail while constraining spectral drift and preserving the source CRS,
bounds, band identities, and reflectance scale.

## Current implementation

The operational baseline currently provides:

1. Sentinel-2 ingestion for common 3-, 4-, 10-, 12-, and 13-band stacks.
2. Extraction of the ten required bands and alignment of lower-resolution bands.
3. Reflectance normalization and nodata sanitization.
4. Tiled inference using pretrained ESAOpenSR SEN2SR backbones.
5. A ten-band 2.5 m GeoTIFF with preserved geographic extent.
6. Reference-based and source-consistency evaluation with a JSON audit report.
7. A differentiable spectral/spatial objective for the paired-data fine-tuning phase.
8. Wald synthesis validation against a bicubic baseline.
9. Novelty and test-time-augmentation uncertainty estimation.
10. Ten thematic indices for crop, water, urban, moisture, and burn analysis.
11. A 93-pair geographically split dataset manifest covering twelve sites.

The pretrained weights provide the baseline. The team-owned technical work is
the multispectral ingestion, geospatial delivery, evaluation framework, and the
spectrally constrained fine-tuning objective. A project-trained checkpoint is
still required before claiming model-level improvement over the baseline.

## Processing pipeline

```text
Sentinel-2 L2A
  -> band identification and quality masking
  -> 10 m working-grid alignment
  -> reflectance normalization
  -> tiled SR inference with overlap blending
  -> physical-range constraint
  -> 2.5 m georeferenced GeoTIFF
  -> source-consistency evaluation
  -> paired-reference evaluation when HR data is available
```

## Paired-training objective

For an SR prediction `P`, high-resolution target `T`, and low-resolution
observation `S`, the implemented objective is:

```text
L = wpixel * Lcharbonnier(P, T)
  + wspectral * Lcosine-spectrum(P, T)
  + wgradient * Lgradient(P, T)
  + wsource * L1(area-downsample(P), S)
  + wrange * Lphysical-range(P)
```

- The pixel term reconstructs observed HR reflectance and is robust to outliers.
- The spectral term preserves the direction of each multispectral pixel vector.
- The gradient term rewards agreement at spatial boundaries.
- The paired-reference terms use each sample's supervision mask, so a real RGB/NIR
  reference cannot train red-edge or SWIR bands it did not measure.
- The source term prevents high-frequency reconstruction from changing the
  observed 10 m signal when the output is aggregated back to the sensor grid.
- The range term penalizes reflectance outside the configured physical interval.

All components are separately available for experiment logging and ablation.

## Evaluation protocol

Evaluation has two explicitly labelled levels:

### Level 1: source consistency

The 2.5 m result is area-averaged to the source grid and compared with the input.
This detects radiometric drift, band inconsistency, geographic displacement, and
violations of the observation. It cannot confirm that reconstructed fine detail
exists on the ground.

### Level 2: paired-reference accuracy

A high-resolution reference is reprojected to the output grid and common bands
are evaluated using MAE, RMSE, bias, PSNR, SSIM, SAM, ERGAS, and spectral-index
drift. Results must be reported by scene, land-cover class, and band. Geographic
train, validation, and test separation is required to avoid leakage.

## Next model milestone

1. Connect `PairedS2Dataset` and `SpectralSpatialLoss` in a trainer; the loss already
   consumes the dataset's per-band supervision mask.
2. Smoke-train SEN2SR-Lite and verify checkpoint save/resume and decreasing validation loss.
3. Fine-tune the baseline on Wald pairs, then the available real RGB/NIR pairs.
4. Compare bicubic, the untouched pretrained checkpoint, and the fine-tuned checkpoint.
5. Run an ablation that removes each loss term in turn.
6. Validate one downstream task, such as field-boundary extraction or flood mapping.
7. Expand the real paired set and quantify temporal and alignment error.

## Claims and limitations

- Super-resolved pixels are model reconstructions, not direct 2.5 m sensor measurements.
- Source-consistency scores are quality-control measurements, not HR accuracy scores.
- Reference comparisons are meaningful only when imagery is well aligned and
  sufficiently close in acquisition time.
- RGB references cannot validate the Sentinel-2 red-edge and SWIR outputs.
