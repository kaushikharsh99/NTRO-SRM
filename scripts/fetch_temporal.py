#!/usr/bin/env python3
"""Fetch K co-registered multi-date Sentinel-2 chips per site (temporal SR).

For each site, the reference chip in datasets/raw_s2/<SITE>.tif defines the
common grid. Up to K-1 additional low-cloud dates are fetched via Earth Search,
reprojected onto the reference grid, integer-shift refined (±3px on B04 red),
and stored in datasets/raw_temporal/<SITE>_<YYYYMMDD>.tif with a manifest at
datasets/raw_temporal/temporal_manifest.csv.

Usage:
  venv/bin/python scripts/fetch_temporal.py --sites IND_PUNJAB_AGRI,USA_SALINAS_AGRI --dates 3
  venv/bin/python scripts/fetch_temporal.py --indian-first --max-sites 4 --dates 3
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import rasterio  # noqa: E402
from rasterio.warp import Resampling, reproject  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch co-registered temporal S2 stacks.")
    p.add_argument("--sites", default=None, help="Comma-separated site_ids (default: all).")
    p.add_argument("--indian-first", action="store_true")
    p.add_argument("--only-indian", action="store_true")
    p.add_argument("--max-sites", type=int, default=None)
    p.add_argument("--dates", type=int, default=3, help="Total dates per site incl. reference.")
    p.add_argument("--date-from", default="2023-01-01")
    p.add_argument("--date-to", default="2025-12-31")
    p.add_argument("--max-cloud", type=float, default=15.0)
    p.add_argument("--max-dt-days", type=float, default=45.0,
                   help="Drop candidate dates farther than this from the reference chip date.")
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def ref_grid(site_path: Path):
    with rasterio.open(site_path) as src:
        return src.crs, src.transform, src.width, src.height, src.count


def reproject_to_ref(src_path: Path, dest_path: Path, crs, transform, w, h, count) -> None:
    with rasterio.open(src_path) as src:
        meta = src.meta.copy()
        meta.update(crs=crs, transform=transform, width=w, height=h, count=count)
        with rasterio.open(dest_path, "w", **meta) as dst:
            for i in range(1, count + 1):
                reproject(
                    rasterio.band(src, i), rasterio.band(dst, i),
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=transform, dst_crs=crs,
                    resampling=Resampling.bilinear,
                )
            dst.update_tags(**{k: str(v) for k, v in (src.tags() or {}).items()})


def integer_shift_fix(path: Path, max_shift: int = 3) -> tuple[int, int]:
    """Phase-correlation integer shift of all bands vs B04 self grid.

    Compares B04 against a 1px-blurred copy to estimate sub-grid misregistration
    left by reprojection, then rolls all bands. Returns (dy, dx) applied.
    """
    with rasterio.open(path, "r+") as ds:
        red = ds.read(4).astype(np.float32)
        # 3x3 box blur via shifts (avoids a scipy dependency).
        padded = np.pad(red, 1, mode="edge")
        smooth = sum(padded[i : i + red.shape[0], j : j + red.shape[1]]
                     for i in range(3) for j in range(3)) / 9.0
        f1 = np.fft.fft2(red - red.mean())
        f2 = np.fft.fft2(smooth - smooth.mean())
        xcorr = np.abs(np.fft.ifft2(f1 * np.conj(f2)))
        peak = np.unravel_index(np.argmax(xcorr), xcorr.shape)
        dy = int(peak[0] if peak[0] <= max_shift else peak[0] - xcorr.shape[0])
        dx = int(peak[1] if peak[1] <= max_shift else peak[1] - xcorr.shape[1])
        dy = max(-max_shift, min(max_shift, dy))
        dx = max(-max_shift, min(max_shift, dx))
        if dy or dx:
            for i in range(1, ds.count + 1):
                arr = ds.read(i)
                ds.write(np.roll(arr, shift=(dy, dx), axis=(0, 1)), i)
        return dy, dx


def main() -> int:
    args = parse_args()
    from ntro_srm.training.aois import ALL_AOIS
    from ntro_srm.web.schemas import AOI, SentinelSearchRequest
    from ntro_srm.web.services.sentinel_service import EarthSearchProvider

    aois = sorted(ALL_AOIS, key=lambda a: (0 if a.indian_priority else 1, a.site_id))
    if args.only_indian:
        aois = [a for a in aois if a.indian_priority]
    if args.sites:
        wanted = {s.strip() for s in args.sites.split(",") if s.strip()}
        aois = [a for a in aois if a.site_id in wanted]
    if args.max_sites:
        aois = aois[: args.max_sites]
    if not aois:
        print("No sites selected.", file=sys.stderr)
        return 1

    tmp_dir = REPO_ROOT / "datasets" / "raw_temporal" / "_fetch_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_dir = REPO_ROOT / "datasets" / "raw_temporal"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "temporal_manifest.csv"
    have = set()
    if manifest.is_file():
        with manifest.open(newline="", encoding="utf-8") as fh:
            have = {(r["site_id"], r["scene_id"]) for r in csv.DictReader(fh) if r.get("site_id")}

    provider = EarthSearchProvider(timeout=args.timeout)
    header = ["site_id", "scene_id", "date", "cloud_pct", "dt_days", "is_reference",
              "path", "shift_dy", "shift_dx"]
    rows_written = 0
    for a in aois:
        ref_path = REPO_ROOT / "datasets" / "raw_s2" / f"{a.site_id}.tif"
        if not ref_path.is_file():
            print(f"[temporal] {a.site_id}: no reference chip, skipping", file=sys.stderr)
            continue
        aoi = AOI(min_lon=a.bbox[0], min_lat=a.bbox[1], max_lon=a.bbox[2], max_lat=a.bbox[3])
        try:
            resp = provider.search(SentinelSearchRequest(
                aoi=aoi, date_from=args.date_from, date_to=args.date_to,
                max_cloud_cover=args.max_cloud, limit=20))
        except Exception as e:
            print(f"[temporal] {a.site_id} search FAIL: {e}", file=sys.stderr)
            continue
        crs, transform, w, h, count = ref_grid(ref_path)
        ref_date = None
        with rasterio.open(ref_path) as _r:
            ref_date = (_r.tags() or {}).get("SCENE_DATE", "")

        def dt_days(scene) -> float:
            try:
                d0 = datetime.fromisoformat(ref_date.replace("Z", "+00:00"))
                d1 = datetime.fromisoformat((scene.datetime or "").replace("Z", "+00:00"))
                return abs((d1 - d0).days)
            except ValueError:
                return 1e9

        # Rank by cloud cover, then temporal distance; drop far dates so the
        # stack shows the same ground state (fusion needs small change).
        cands = sorted(resp.scenes or [], key=lambda s: (s.cloud_cover, dt_days(s)))
        scenes = [s for s in cands if dt_days(s) <= args.max_dt_days][: max(args.dates, 1)]
        if not scenes:
            print(f"[temporal] {a.site_id}: no scenes within {args.max_dt_days}d", file=sys.stderr)
            continue
        for rank, scene in enumerate(scenes):
            if (a.site_id, scene.id) in have and not args.overwrite:
                print(f"[temporal] cached {a.site_id} {scene.id[:28]}")
                continue
            date = (scene.datetime or "")[:10]
            dest = out_dir / f"{a.site_id}_{date.replace('-', '')}.tif"
            try:
                raw = provider.fetch_aoi_bands(scene.id, aoi, tmp_dir)
                reproject_to_ref(Path(raw), dest, crs, transform, w, h, count)
                dy, dx = integer_shift_fix(dest)
                dt = ""
                try:
                    if ref_date and scene.datetime:
                        d0 = datetime.fromisoformat(ref_date.replace("Z", "+00:00"))
                        d1 = datetime.fromisoformat(scene.datetime.replace("Z", "+00:00"))
                        dt = str(abs((d1 - d0).days))
                except ValueError:
                    pass
                is_ref = "1" if rank == 0 else "0"
                new_file = not manifest.is_file()
                with manifest.open("a", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=header, lineterminator="\n")
                    if new_file:
                        writer.writeheader()
                    writer.writerow({"site_id": a.site_id, "scene_id": scene.id,
                                     "date": scene.datetime, "cloud_pct": scene.cloud_cover,
                                     "dt_days": dt, "is_reference": is_ref,
                                     "path": str(dest.relative_to(REPO_ROOT)),
                                     "shift_dy": dy, "shift_dx": dx})
                rows_written += 1
                print(f"[temporal] {a.site_id} {date} cloud={scene.cloud_cover}% shift=({dy},{dx})")
            except Exception as e:
                print(f"[temporal] {a.site_id} {scene.id[:28]} fetch FAIL: {e}", file=sys.stderr)
    print(f"[temporal] done +{rows_written} rows -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
