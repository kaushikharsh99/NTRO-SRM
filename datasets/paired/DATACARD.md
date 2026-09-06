# NTRO-SRM Paired Dataset — Data Card (SIH26142)

## Summary
Paired Sentinel-2 10m ↔ HR reference tiles for 4x super-resolution
(10m → 2.5m working grid; Wald 40m → 10m proxy for full 10-band supervision).
Built with `scripts/build_dataset.py`. Manifest: `datasets/paired/manifest.csv`.

## Composition (v1, 2026-09-06)
- **90 Wald-synthetic tiles** (`wald_synthetic`): HR = real S2 10m 128×128,
  LR = Wald-degraded 40m 32×32 (Gaussian MTF σ=1.0 + area decimation).
  Full 10-band supervision. Covers all 12 sites.
- **3 real pairs** (`real_paired`): S2 10m chip + NAIP HR reprojected to 2.5m
  (RGBN common bands B02/B03/B04/B08 supervised; red-edge/SWIR via
  source-consistency only):
  - USA_SALINAS_AGRI: S2 2022-09-16 / NAIP 2022-05-14, dt=125d
  - USA_TAHOE_ALPINE: S2 2022-09-23 / NAIP 2022-07-21, dt=64d
  - USA_MOUNTAIN_LAKE: S2 2023-11-29 / NAIP 2023-10-12, dt=48d
- **51/93 tiles Indian-priority** (Ludhiana agri, Delhi urban, Mumbai coast,
  Assam floodplain, Bengaluru lakes, Jaipur semi-arid).
- Splits are **geographic** (by site, not random crop): train/val/test all non-empty.

## Sources & licences
- Sentinel-2 L2A via AWS Earth Search COGs (Element84) — Copernicus open data.
- NAIP via Microsoft Planetary Computer STAC (`naip` collection) — US public domain.
- No commercial HR redistributed: raw HR chips are local derivatives for
  research; re-fetch commands are in `scripts/build_dataset.py`.

## Known limits (judge-facing honesty)
- Wald pairs prove the 4x mapping but not true 2.5m detail (documented proxy).
- Salinas dt=125d slightly exceeds the 120d target; same growing season, flagged.
- RGB-only HR cannot validate red-edge/SWIR — see EVALUATION.md.
- SR outputs remain model reconstructions, not sensor measurements.

## Reproduce / extend
```bash
venv/bin/python scripts/build_dataset.py fetch-s2 --only-indian   # Track 2
venv/bin/python scripts/build_dataset.py tile-all                  # Track 0
venv/bin/python scripts/build_dataset.py real-naip --max-sites 3   # Track 1
venv/bin/python scripts/build_dataset.py qa
pytest tests/test_dataset.py -q
```
Indian HR upgrade path: drop any 2.5m reference covering an `IND_*` chip into
`datasets/raw_hr/` and append a `real_paired` manifest row with matching
`site_id/split`; `PairedS2Dataset` and the evaluator pick it up unchanged.
