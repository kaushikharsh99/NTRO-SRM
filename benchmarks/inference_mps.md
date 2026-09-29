# Inference benchmark (legacy vs optimized)

Input: sample 256x256 | device=mps | repeats=1 | tile=128 overlap=32

| path | mean ms | speedup | max|legacy-opt| | mean|.| |
|---|---|---|---|---|
| legacy (1/tile, hard-crop) | 808.6 | 1.00x | - | - |
| optimized batch=1 linear-blend | 452.5 | 1.79x | 3.67e-01 | 3.59e-04 |
| optimized batch=4 linear-blend | 351.5 | 2.30x | 3.67e-01 | 3.59e-04 |
| optimized batch=4 average-blend | 349.4 | 2.31x | - | - |

Notes: legacy = upstream single-tile hard-crop path (`use_legacy_tiling=True`); optimized batches tiles and uses partition-of-unity blending. Small numerical differences vs legacy are expected at overlap seams (that is the seam fix).
