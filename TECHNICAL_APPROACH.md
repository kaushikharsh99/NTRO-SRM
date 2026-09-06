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

1. ~~Connect `PairedS2Dataset` and `SpectralSpatialLoss` in a trainer~~ DONE:
   `src/ntro_srm/training/trainer.py` + `scripts/finetune_lite.py`. Full-scene
   real RGBN pairs are auto-tiled to 32px (`_tile_grid`), with in-memory scene
   caching, train-only flip/rot90 augmentation, cosine schedule, configurable
   loss weights, and real-only val tracking.
   Fixed en route: NAIP radiometry (DNB 0-255 was ÷10000 → 14x too dark; now
   per-band mean/std matched to the LR tile), pair-type filter, resume-LR.
2. ~~Smoke-train SEN2SR-Lite and verify checkpoint save/resume~~ DONE on
   Apple MPS: smoke `val 0.0259 -> 0.0240`, resume continues `-> 0.0236`.
   Shipped `outputs/finetune/lite_ft_best.pt` (Wald 5ep @1e-4, then mixed
   8ep @5e-5 cosine, grad 0.4, augment; 161 train / 84 val tiles).
   Held-out Wald TEST (27 tiles, unseen geography), identical metric:
   pixel 0.01423->0.01321, spectral 0.00858->0.00677 (-21%),
   gradient 0.02196->0.02130, source 0.00655->0.00394 (-40%).
   Sample scene consistency RMSE 0.01234->0.01000. See `tests/test_trainer.py`
   and `scripts/compare_ft.py`. A hotter 12ep @2e-4 continuation degraded
   val (2 real training scenes memorize) and was discarded.
3. Fine-tune the baseline on Wald pairs, then the available real RGB/NIR pairs.
4. Compare bicubic, the untouched pretrained checkpoint, and the fine-tuned checkpoint.
5. Run an ablation that removes each loss term in turn.
6. Validate one downstream task, such as field-boundary extraction or flood mapping.
7. Expand the real paired set and quantify temporal and alignment error.

## Distillation milestone (Swin2SR -> Lite, DONE 2026-09-07)

Wald/NAIP fine-tuning moved Lite by mean |FT-base| ~0.003 reflectance
(visually negligible), while the Swin2SR teacher disagrees with Lite by mean
|Swin-Lite| ~0.08 (p99 ~0.3) at edges. Distillation transfers that edge
reconstruction into a Lite-speed student (~2s inference, same deployment).

- Teacher cache: `scripts/cache_teacher.py` rendered 80 native 128px LR tiles
  (62 train / 18 val across 9 sites; test sites IND_ASSAM_FLOOD,
  IND_DELHI_URBAN, IND_MUMBAI_COAST never distilled) to 512px pseudo-targets
  (`datasets/distill_cache/`, git-ignored, reproducible, ~55 min on MPS).
- Student: `src/ntro_srm/training/distill.py` + `scripts/distill_lite.py`,
  init from `lite_ft_best.pt`, edge-weighted loss
  (pixel 1.0 / spectral 0.10 / gradient 0.60 / source 0.30 / range 0.05),
  Adam 3e-5 cosine, augment. Continuation v2 @2e-5 x10ep.
  Shipped `outputs/finetune/lite_distill_best.pt` (= v2 best).
  Val vs teacher: 0.01295 -> 0.01161 (run 1) -> 0.01117 (v2), monotonic.
- Held-out generalization (IND_DELHI_URBAN test chip): distilled Lite is
  closest to the Swin teacher in pixel AND gradient fidelity on both test
  tiles (beats base and FT; FT is farther from the teacher than base -
  Wald training smooths toward the proxy). Source-consistency RMSE beats base
  on both scenes (Mountain Lake 0.01138 vs 0.01234; Delhi 0.01874 vs 0.02058).
- Honest limits: teacher targets are pseudo-targets, not ground truth, so
  teacher bias can transfer; full-scene RGB differences remain subtle
  (Delhi distill-vs-base mean 0.0031, p99 0.0168, concentrated at edges and
  red-edge/SWIR fusion bands); raw gradient energy ranks base > distill > FT,
  i.e. the gain is better-placed edges + radiometry, not global "sharpness".
  Visuals: `outputs/comparisons/distill_delhi_{rgb_3way,zoom_3way}.png`
  ([base | FT | distill], zoom centred on peak-diff edge).
- Serving: `find_ft_checkpoint` prefers `lite_distill_best.pt`; `lite-ft`
  (Web/CLI/REST) and `scripts/compare_ft.py` pick it up with no flag changes
  (`--checkpoint` overrides for ablations). Tests: `tests/test_distill.py`
  (3 tests, CPU, synthetic cache).

## Claims and limitations

- Super-resolved pixels are model reconstructions, not direct 2.5 m sensor measurements.
- Source-consistency scores are quality-control measurements, not HR accuracy scores.
- Reference comparisons are meaningful only when imagery is well aligned and
  sufficiently close in acquisition time.
- RGB references cannot validate the Sentinel-2 red-edge and SWIR outputs.
