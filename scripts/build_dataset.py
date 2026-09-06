#!/usr/bin/env python3
"""SIH26142 dataset builder — 3 tracks, Indian priority.

Usage:
  venv/bin/python scripts/build_dataset.py wald-from-local --input datasets/sample_s2/sample_s2_l2a.tif --site USA_MOUNTAIN_LAKE
  venv/bin/python scripts/build_dataset.py fetch-s2 --indian-first --max-sites 6
  venv/bin/python scripts/build_dataset.py real-naip --max-sites 3
  venv/bin/python scripts/build_dataset.py tile-all
  venv/bin/python scripts/build_dataset.py qa
  venv/bin/python scripts/build_dataset.py all  (fetch-s2 -> tile-all -> real-naip -> qa)

Outputs:
  datasets/raw_s2/<SITE>.tif      10-band S2 chips (uint16)
  datasets/raw_hr/<SITE>_hr.tif   HR reference resampled to 2.5m (where available)
  datasets/paired/manifest.csv    tile-level manifest (see training/dataset.py)
  datasets/paired/stats.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

MANIFEST_HEADER = ["pair_id", "site_id", "indian_priority", "landcover", "task", "split",
                   "pair_type", "s2_path", "hr_path", "s2_date", "hr_date", "dt_days",
                   "cloud_pct", "crs", "align_rmse_px", "tile_y", "tile_x",
                   "lr_h", "lr_w", "common_bands", "notes"]


def repo() -> Path:
    return REPO_ROOT


def manifest_path() -> Path:
    return REPO_ROOT / "datasets" / "paired" / "manifest.csv"


def load_rows() -> list[dict]:
    mp = manifest_path()
    if not mp.is_file():
        return []
    with mp.open(newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r.get("pair_id")]


def save_rows(rows: list[dict]) -> Path:
    mp = manifest_path()
    mp.parent.mkdir(parents=True, exist_ok=True)
    # de-duplicate by pair_id, keep last
    seen: dict[str, dict] = {}
    for r in rows:
        seen[r["pair_id"]] = r
    ordered = sorted(seen.values(), key=lambda r: (r.get("site_id", ""), r.get("pair_id", "")))
    with mp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_HEADER, lineterminator="\n")
        w.writeheader()
        for r in ordered:
            w.writerow({k: r.get(k, "") for k in MANIFEST_HEADER})
    return mp


def cmd_wald_from_local(args) -> int:
    from ntro_srm.data.sentinel2 import Sentinel2Reader
    from ntro_srm.training.aois import ALL_AOIS
    from ntro_srm.training.wald import tile_pair_indices, valid_tile_mask

    site = next((a for a in ALL_AOIS if a.site_id == args.site), None)
    if site is None:
        print(f"Unknown site {args.site}", file=sys.stderr)
        return 2
    raw_dir = REPO_ROOT / "datasets" / "raw_s2"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"{site.site_id}.tif"

    reader = Sentinel2Reader(args.input)
    data = reader.read()
    arr = data.tensor.numpy()  # (10,H,W) native dtype
    # write 10-band chip
    meta = {"driver": "GTiff", "dtype": arr.dtype, "count": 10,
            "height": data.height, "width": data.width,
            "crs": data.crs, "transform": data.transform,
            "compress": "deflate", "tiled": True}
    with rasterio.open(dest, "w", **meta) as dst:
        dst.write(arr)
        for i, b in enumerate(data.band_names, start=1):
            dst.set_band_description(i, b)
        dst.update_tags(SITE=site.site_id, SOURCE=str(Path(args.input).name),
                        BANDS=",".join(data.band_names))
    print(f"[wald-from-local] chip {dest} {data.width}x{data.height} {data.crs}")

    # tile into manifest rows (Wald: HR window 128, LR 32)
    tile, stride = args.tile_hr, args.stride_hr
    corners = tile_pair_indices(data.height, data.width, tile=tile, stride=stride)
    rows = load_rows()
    kept = 0
    for k, (y, x) in enumerate(corners):
        window = arr[:, y:y + tile, x:x + tile]
        if window.shape[-2:] != (tile, tile):
            continue
        if not valid_tile_mask(window, nodata_fraction_max=0.05):
            continue
        rows.append({
            "pair_id": f"{site.site_id}_wald_{y:04d}_{x:04d}", "site_id": site.site_id,
            "indian_priority": int(site.indian_priority), "landcover": site.landcover,
            "task": site.task, "split": site.split, "pair_type": "wald_synthetic",
            "s2_path": str(dest.relative_to(REPO_ROOT)), "hr_path": "",
            "s2_date": args.s2_date, "hr_date": args.s2_date, "dt_days": 0,
            "cloud_pct": args.cloud_pct, "crs": str(data.crs), "align_rmse_px": 0.0,
            "tile_y": y, "tile_x": x, "lr_h": tile // 4, "lr_w": tile // 4,
            "common_bands": "B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12",
            "notes": f"wald scale=4 from {Path(args.input).name}",
        })
        kept += 1
    mp = save_rows(rows)
    print(f"[wald-from-local] +{kept} tiles -> {mp} (total {len(load_rows())})")
    return 0


def cmd_fetch_s2(args) -> int:
    from ntro_srm.training.aois import ALL_AOIS
    from ntro_srm.web.schemas import AOI, SentinelSearchRequest
    from ntro_srm.web.services.sentinel_service import EarthSearchProvider

    aois = sorted(ALL_AOIS, key=lambda a: (0 if a.indian_priority else 1, a.site_id))
    if args.indian_first:
        pass  # already sorted
    if args.only_indian:
        aois = [a for a in aois if a.indian_priority]
    if args.max_sites:
        aois = aois[:args.max_sites]
    if args.site:
        aois = [a for a in aois if a.site_id == args.site]

    provider = EarthSearchProvider(timeout=args.timeout)
    out_dir = REPO_ROOT / "datasets" / "raw_s2"
    out_dir.mkdir(parents=True, exist_ok=True)
    ok, fail = 0, []
    for a in aois:
        dest = out_dir / f"{a.site_id}.tif"
        if dest.is_file() and not args.overwrite:
            print(f"[fetch-s2] cached {a.site_id}")
            ok += 1
            continue
        bbox = a.bbox
        aoi = AOI(min_lon=bbox[0], min_lat=bbox[1], max_lon=bbox[2], max_lat=bbox[3])
        req = SentinelSearchRequest(aoi=aoi, date_from=args.date_from,
                                    date_to=args.date_to,
                                    max_cloud_cover=args.max_cloud,
                                    limit=5)
        try:
            resp = provider.search(req)
        except Exception as e:
            print(f"[fetch-s2] {a.site_id} search FAIL: {e}", file=sys.stderr)
            fail.append(a.site_id)
            continue
        if not resp.scenes:
            print(f"[fetch-s2] {a.site_id} no scenes (cloud<{args.max_cloud})", file=sys.stderr)
            fail.append(a.site_id)
            continue
        # prefer lowest cloud
        scene = sorted(resp.scenes, key=lambda s: s.cloud_cover)[0]
        try:
            path = provider.fetch_aoi_bands(scene.id, aoi, out_dir)
            # rename to site file for stable manifest paths
            import shutil
            if Path(path).resolve() != dest.resolve():
                shutil.copyfile(path, dest)
            with rasterio.open(dest, "r+") as dst:
                dst.update_tags(SITE=a.site_id, SCENE=scene.id,
                                SCENE_DATE=scene.datetime, CLOUD=scene.cloud_cover)
            print(f"[fetch-s2] {a.site_id} <- {scene.id} cloud={scene.cloud_cover}% {scene.datetime}")
            ok += 1
        except Exception as e:
            print(f"[fetch-s2] {a.site_id} fetch FAIL: {e}", file=sys.stderr)
            fail.append(a.site_id)
    print(f"[fetch-s2] done ok={ok} fail={fail}")
    return 0 if ok else 1


def _reproject_to_s2_grid(hr_src_path: str, s2_chip: Path, dest: Path,
                           target_gsd_m: float = 2.5) -> tuple[int, str]:
    """Reproject an HR file onto the S2 chip CRS/bounds at 4x resolution.

    Returns (common_band_count, hr_date_hint).
    """
    import rasterio.windows as _w  # noqa
    with rasterio.open(s2_chip) as s2:
        crs = s2.crs
        bounds = s2.bounds
        h_hr, w_hr = s2.height * 4, s2.width * 4
        transform_hr = rasterio.transform.from_bounds(
            bounds.left, bounds.bottom, bounds.right, bounds.top, w_hr, h_hr)
    with rasterio.open(hr_src_path) as src:
        count = min(src.count, 4)
        out = np.empty((count, h_hr, w_hr), dtype=np.float32)
        for i in range(1, count + 1):
            reproject(rasterio.band(src, i), out[i - 1],
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform_hr, dst_crs=crs,
                      resampling=Resampling.bilinear,
                      dst_nodata=np.nan)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dest, "w", driver="GTiff", dtype="float32",
                       count=count, height=h_hr, width=w_hr,
                       crs=crs, transform=transform_hr,
                       compress="deflate", tiled=True) as dst:
        dst.write(out)
        for i in range(1, count + 1):
            dst.set_band_description(i, f"HR_RGBN_{i}")
    return count, ""


def cmd_real_naip(args) -> int:
    """Fetch NAIP HR via Planetary Computer STAC for US sites and pair with S2 chips."""
    import requests
    from ntro_srm.training.aois import ALL_AOIS

    pc = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    raw_s2 = REPO_ROOT / "datasets" / "raw_s2"
    raw_hr = REPO_ROOT / "datasets" / "raw_hr"
    raw_hr.mkdir(parents=True, exist_ok=True)
    sites = [a for a in ALL_AOIS if a.site_id in
             ("USA_SALINAS_AGRI", "USA_TAHOE_ALPINE", "USA_MOUNTAIN_LAKE")]
    if args.site:
        sites = [a for a in sites if a.site_id == args.site]
    rows = load_rows()
    added = 0
    import rasterio.windows as _windows
    from rasterio.warp import transform_bounds as _tb
    for a in sites[:args.max_sites]:
        chip = raw_s2 / f"{a.site_id}.tif"
        if not chip.is_file():
            print(f"[real-naip] {a.site_id} missing S2 chip, skipping (run fetch-s2 first)")
            continue
        dest = raw_hr / f"{a.site_id}_hr_2p5m.tif"
        if dest.is_file() and not args.overwrite:
            print(f"[real-naip] cached {a.site_id}")
            continue
        bbox = a.bbox
        try:
            r = requests.post(pc, json={"collections": ["naip"],
                                        "bbox": bbox, "limit": 5},
                              timeout=25)
            r.raise_for_status()
            feats = r.json().get("features", [])
        except Exception as e:
            print(f"[real-naip] {a.site_id} PC search FAIL: {e}")
            continue
        if not feats:
            print(f"[real-naip] {a.site_id} no NAIP items")
            continue
        feat = feats[0]
        # Prefer the NAIP acquisition closest in time to the S2 chip.
        s2_date_hint = ""
        try:
            with rasterio.open(chip) as _s2:
                s2_date_hint = str(_s2.tags().get("SCENE_DATE", ""))[:10]
        except Exception:
            pass
        if s2_date_hint:
            try:
                s2_dt = datetime.fromisoformat(s2_date_hint)
                def _absd(f):
                    try:
                        return abs((datetime.fromisoformat(
                            str(f.get("properties", {}).get("datetime", ""))[:10]) - s2_dt).days)
                    except Exception:
                        return 10 ** 9
                feat = sorted(feats, key=_absd)[0]
            except Exception:
                feat = sorted(feats, key=lambda f: f.get("properties", {}).get("datetime", ""))[-1]
        else:
            feat = sorted(feats, key=lambda f: f.get("properties", {}).get("datetime", ""))[-1]
        props = feat.get("properties", {})
        assets = feat.get("assets", {})
        href = (assets.get("image", {}) or {}).get("href")
        if not href:
            print(f"[real-naip] {a.site_id} no image asset")
            continue
        try:
            tok = requests.get(
                "https://planetarycomputer.microsoft.com/api/sas/v1/sign",
                params={"href": href}, timeout=15)
            if tok.ok and tok.json().get("token"):
                href = href + ("&" if "?" in href else "?") + tok.json()["token"]
        except Exception:
            pass
        try:
            # Windowed COG read: only the AOI, not the whole tile.
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                              GDAL_HTTP_TIMEOUT="30",
                              GDAL_HTTP_MAX_RETRY="3",
                              GDAL_HTTP_RETRY_DELAY="2"):
                with rasterio.open(href) as src:
                    minx, miny, maxx, maxy = _tb("EPSG:4326", src.crs, *bbox)
                    win = _windows.from_bounds(minx, miny, maxx, maxy,
                                               src.transform).round_offsets().round_lengths()
                    win = win.intersection(
                        _windows.Window(0, 0, src.width, src.height))
                    if win.width < 8 or win.height < 8:
                        print(f"[real-naip] {a.site_id} window too small, skipping")
                        continue
                    # cap: NAIP 1m over 2.56km = ~2560px; downsample on read if larger
                    out_h = min(int(win.height), 1400)
                    out_w = min(int(win.width), 1400)
                    data = src.read(list(range(1, min(src.count, 4) + 1)),
                                    window=win,
                                    out_shape=(min(src.count, 4), out_h, out_w),
                                    resampling=Resampling.bilinear)
                    win_transform = _windows.transform(win, src.transform)
                    # scale transform for the out_shape decimation
                    sx, sy = win.width / out_w, win.height / out_h
                    win_transform = rasterio.Affine(win_transform.a * sx, 0, win_transform.c,
                                                    0, win_transform.e * sy, win_transform.f)
                    src_crs = src.crs
            tmp = raw_hr / f"{a.site_id}_src.tif"
            with rasterio.open(tmp, "w", driver="GTiff", dtype=data.dtype,
                               count=data.shape[0], height=out_h, width=out_w,
                               crs=src_crs, transform=win_transform,
                               compress="deflate", tiled=True) as dst:
                dst.write(data)
            n_bands, _ = _reproject_to_s2_grid(str(tmp), chip, dest)
            tmp.unlink(missing_ok=True)
            hr_date = str(props.get("datetime", ""))[:10]
            s2_date = ""
            with rasterio.open(chip) as s2:
                s2_date = str(s2.tags().get("SCENE_DATE", ""))[:10]
            try:
                d1 = datetime.fromisoformat(s2_date) if s2_date else None
                d2 = datetime.fromisoformat(hr_date) if hr_date else None
                dt = abs((d2 - d1).days) if (d1 and d2) else -1
            except Exception:
                dt = -1
            rows.append({
                "pair_id": f"{a.site_id}_real_0000_0000", "site_id": a.site_id,
                "indian_priority": 0, "landcover": a.landcover, "task": a.task,
                "split": a.split, "pair_type": "real_paired",
                "s2_path": str(chip.relative_to(REPO_ROOT)),
                "hr_path": str(dest.relative_to(REPO_ROOT)),
                "s2_date": s2_date, "hr_date": hr_date, "dt_days": dt,
                "cloud_pct": "", "crs": "", "align_rmse_px": "",
                "tile_y": 0, "tile_x": 0, "lr_h": 256, "lr_w": 256,
                "common_bands": "B02,B03,B04,B08",
                "notes": f"NAIP {feat.get('id','')} bands={n_bands}",
            })
            added += 1
            print(f"[real-naip] {a.site_id} OK dt={dt}d hr={hr_date}")
        except Exception as e:
            print(f"[real-naip] {a.site_id} download/reproject FAIL: {e}")
        finally:
            _tmp = locals().get("tmp")
            if _tmp is not None and _tmp.is_file() and _tmp.name.endswith("_src.tif"):
                try:
                    if _tmp.stat().st_size > 450 * 1024 * 1024:
                        _tmp.unlink(missing_ok=True)
                except OSError:
                    pass
    if added:
        save_rows(rows)
    print(f"[real-naip] +{added} real pairs (total {len(load_rows())})")
    return 0


def cmd_tile_all(args) -> int:
    from ntro_srm.training.aois import ALL_AOIS
    from ntro_srm.training.wald import tile_pair_indices, valid_tile_mask

    raw_s2 = REPO_ROOT / "datasets" / "raw_s2"
    chips = sorted(raw_s2.glob("*.tif"))
    if not chips:
        print("[tile-all] no chips in datasets/raw_s2 (run fetch-s2 or wald-from-local first)",
              file=sys.stderr)
        return 1
    by_site = {a.site_id: a for a in ALL_AOIS}
    rows = [r for r in load_rows() if r.get("pair_type") != "wald_synthetic"]
    n_add = 0
    for chip in chips:
        site_id = chip.stem
        # allow sample-derived chips named after site
        site = by_site.get(site_id)
        if site is None:
            print(f"[tile-all] unknown site for {chip.name}, skipping")
            continue
        with rasterio.open(chip) as src:
            h, w = src.height, src.width
            sample = src.read(list(range(1, min(src.count, 10) + 1)),
                              window=rasterio.windows.Window(0, 0, min(w, 128), min(h, 128)))
        # cloud/date tags if present
        s2_date, cloud = "", ""
        with rasterio.open(chip) as src:
            tags = src.tags()
            s2_date = str(tags.get("SCENE_DATE", ""))[:10]
            cloud = str(tags.get("CLOUD", ""))
            crs = str(src.crs)
        tile, stride = args.tile_hr, args.stride_hr
        with rasterio.open(chip) as src:
            full = src.read()
        for (y, x) in tile_pair_indices(h, w, tile=tile, stride=stride):
            win = full[:, y:y + tile, x:x + tile]
            if win.shape[-2:] != (tile, tile):
                continue
            if not valid_tile_mask(win.astype(np.float32)):
                continue
            # skip tiles already present
            pid = f"{site_id}_wald_{y:04d}_{x:04d}"
            rows.append({
                "pair_id": pid, "site_id": site_id,
                "indian_priority": int(site.indian_priority),
                "landcover": site.landcover, "task": site.task, "split": site.split,
                "pair_type": "wald_synthetic",
                "s2_path": str(chip.relative_to(REPO_ROOT)), "hr_path": "",
                "s2_date": s2_date, "hr_date": s2_date, "dt_days": 0,
                "cloud_pct": cloud, "crs": crs, "align_rmse_px": 0.0,
                "tile_y": y, "tile_x": x, "lr_h": tile // 4, "lr_w": tile // 4,
                "common_bands": "B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12",
                "notes": "wald scale=4",
            })
            n_add += 1
    save_rows(rows)
    print(f"[tile-all] +{n_add} wald tiles (total {len(load_rows())})")
    return 0


def cmd_qa(args) -> int:
    rows = load_rows()
    stats = {"generated_at": datetime.now(timezone.utc).isoformat(),
             "total": len(rows),
             "by_type": {}, "by_split": {}, "by_site": {},
             "indian_priority_tiles": 0}
    for r in rows:
        stats["by_type"][r.get("pair_type", "?")] = stats["by_type"].get(r.get("pair_type", "?"), 0) + 1
        stats["by_split"][r.get("split", "?")] = stats["by_split"].get(r.get("split", "?"), 0) + 1
        stats["by_site"][r.get("site_id", "?")] = stats["by_site"].get(r.get("site_id", "?"), 0) + 1
        if str(r.get("indian_priority")) == "1":
            stats["indian_priority_tiles"] += 1
    # file existence check (sample N=50 for speed)
    missing = 0
    for r in rows[:50]:
        for k in ("s2_path", "hr_path"):
            p = (r.get(k) or "").strip()
            if not p:
                continue
            if not (REPO_ROOT / p).is_file():
                missing += 1
    stats["missing_files_in_first50"] = missing
    out = REPO_ROOT / "datasets" / "paired" / "stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    # hard QA gates for SIH
    ok = True
    if stats["total"] == 0:
        print("QA FAIL: empty manifest", file=sys.stderr)
        ok = False
    if stats["indian_priority_tiles"] == 0:
        print("QA FAIL: zero indian-priority tiles", file=sys.stderr)
        ok = False
    for s in ("train", "val", "test"):
        if stats["by_split"].get(s, 0) == 0:
            print(f"QA WARN: empty split {s}", file=sys.stderr)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="SIH26142 3-track dataset builder")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("wald-from-local")
    p.add_argument("--input", required=True)
    p.add_argument("--site", required=True)
    p.add_argument("--s2-date", default="2018-08-25")
    p.add_argument("--cloud-pct", default=0.0, type=float)
    p.add_argument("--tile-hr", default=128, type=int)
    p.add_argument("--stride-hr", default=96, type=int)
    p.set_defaults(func=cmd_wald_from_local)

    p = sub.add_parser("fetch-s2")
    p.add_argument("--indian-first", action="store_true", default=True)
    p.add_argument("--only-indian", action="store_true", default=False)
    p.add_argument("--max-sites", type=int, default=12)
    p.add_argument("--site", default=None)
    p.add_argument("--date-from", default="2023-01-01")
    p.add_argument("--date-to", default="2025-12-31")
    p.add_argument("--max-cloud", default=10.0, type=float)
    p.add_argument("--timeout", default=25, type=int)
    p.add_argument("--overwrite", action="store_true", default=False)
    p.set_defaults(func=cmd_fetch_s2)

    p = sub.add_parser("real-naip")
    p.add_argument("--max-sites", type=int, default=3)
    p.add_argument("--site", default=None)
    p.add_argument("--overwrite", action="store_true", default=False)
    p.set_defaults(func=cmd_real_naip)

    p = sub.add_parser("tile-all")
    p.add_argument("--tile-hr", default=128, type=int)
    p.add_argument("--stride-hr", default=96, type=int)
    p.set_defaults(func=cmd_tile_all)

    p = sub.add_parser("qa")
    p.set_defaults(func=cmd_qa)

    p = sub.add_parser("all")
    p.add_argument("--max-sites", type=int, default=12)
    p.add_argument("--skip-naip", action="store_true", default=False)
    p.set_defaults(func=None)

    args = ap.parse_args()
    if args.cmd == "all":
        a2 = argparse.Namespace(indian_first=True, only_indian=False,
                                max_sites=args.max_sites, site=None,
                                date_from="2023-01-01", date_to="2025-12-31",
                                max_cloud=10.0, timeout=25, overwrite=False)
        rc = cmd_fetch_s2(a2)
        rc2 = cmd_tile_all(argparse.Namespace(tile_hr=128, stride_hr=96))
        rc3 = 0 if args.skip_naip else cmd_real_naip(
            argparse.Namespace(max_sites=3, site=None, overwrite=False))
        rc4 = cmd_qa(argparse.Namespace())
        return rc or rc2 or rc3 or rc4
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
