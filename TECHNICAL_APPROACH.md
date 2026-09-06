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

1. Build a versioned manifest of temporally matched Sentinel-2/HR scene pairs.
2. Record cloud cover, acquisition-time difference, CRS, alignment error, and split.
3. Fine-tune one baseline backbone using `SpectralSpatialLoss`.
4. Compare bicubic, the untouched pretrained checkpoint, and the fine-tuned checkpoint.
5. Run an ablation that removes each loss term in turn.
6. Validate one downstream task, such as field-boundary extraction or flood mapping.
7. Add checkpoint-ensemble or Monte Carlo uncertainty after accuracy is established.

## Claims and limitations

- Super-resolved pixels are model reconstructions, not direct 2.5 m sensor measurements.
- Source-consistency scores are quality-control measurements, not HR accuracy scores.
- Reference comparisons are meaningful only when imagery is well aligned and
  sufficiently close in acquisition time.
- RGB references cannot validate the Sentinel-2 red-edge and SWIR outputs.
