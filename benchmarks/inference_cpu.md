# Inference benchmark (legacy vs optimized)

Input: sample 256x256 | device=cpu | repeats=2 | tile=128 overlap=32

| path | mean ms | speedup | max|legacy-opt| | mean|.| |
|---|---|---|---|---|
| legacy (1/tile, hard-crop) | 4519.9 | 1.00x | - | - |
| optimized batch=1 linear-blend | 2692.5 | 1.68x | 3.67e-01 | 3.59e-04 |
| optimized batch=4 linear-blend | 2029.4 | 2.23x | 3.67e-01 | 3.59e-04 |
| optimized batch=4 average-blend | 2041.5 | 2.21x | - | - |

Notes: legacy = upstream single-tile hard-crop path (`use_legacy_tiling=True`); optimized batches tiles and uses partition-of-unity blending. Small numerical differences vs legacy are expected at overlap seams (that is the seam fix).
