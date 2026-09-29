#!/usr/bin/env python3
"""Cache Swin2SR teacher targets for Lite distillation (one-time, slow).

Swin takes ~40s per 128px tile on MPS, so targets are precomputed once to
datasets/distill_cache/*.pt and reused across training runs. Skips tiles that
are already cached (safe to resume / re-run).

Usage:
  venv/bin/python scripts/cache_teacher.py --smoke            # 2 tiles, wiring test
  venv/bin/python scripts/cache_teacher.py                    # full train+val cache (~1h on MPS)
  venv/bin/python scripts/cache_teacher.py --split train      # train sites only
  venv/bin/python scripts/cache_teacher.py --device cpu --max-tiles 4
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ntro_srm.models.sen2sr import SEN2SRModel  # noqa: E402
from ntro_srm.training.distill import (  # noqa: E402
    DISTILL_TRAIN_SITES,
    DISTILL_VAL_SITES,
    TILE_LR,
    DistillTile,
    cache_file,
    extract_lr,
    plan_distill_tiles,
    read_normalized_s2,
)
from ntro_srm.training.trainer import forward_native  # noqa: E402
from ntro_srm.utils.device import empty_device_cache, select_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache Swin2SR teacher targets.")
    p.add_argument("--cache-dir", default="datasets/distill_cache")
    p.add_argument("--raw-s2-dir", default="datasets/raw_s2")
    p.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"],
                   help="Default: auto (CUDA > MPS > CPU).")
    p.add_argument("--tile", type=int, default=TILE_LR)
    p.add_argument("--train-stride", type=int, default=64)
    p.add_argument("--val-stride", type=int, default=128)
    p.add_argument("--split", default="all", choices=["all", "train", "val"])
    p.add_argument("--max-tiles", type=int, default=None)
    p.add_argument("--smoke", action="store_true",
                   help="Cache 2 train tiles only (wiring test).")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


@torch.no_grad()
def render_teacher(teacher: SEN2SRModel, lr: torch.Tensor) -> torch.Tensor:
    """Distill-scale forward: 128px LR -> 512px pseudo-HR, no grad."""
    batch = lr.unsqueeze(0).to(teacher.device)
    sr = forward_native(teacher, batch)
    return sr.squeeze(0).detach().cpu().float()


def main() -> int:
    args = parse_args()
    device = select_device(args.device)
    cache_dir = (REPO_ROOT / args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    tiles: list[DistillTile] = []
    if args.split in ("all", "train"):
        tiles += plan_distill_tiles(REPO_ROOT / args.raw_s2_dir, DISTILL_TRAIN_SITES,
                                    args.tile, args.train_stride)
    if args.split in ("all", "val"):
        tiles += plan_distill_tiles(REPO_ROOT / args.raw_s2_dir, DISTILL_VAL_SITES,
                                    args.tile, args.val_stride)
    if args.smoke:
        train = [t for t in tiles if t.site_id in DISTILL_TRAIN_SITES][:2]
        val = [t for t in tiles if t.site_id in DISTILL_VAL_SITES][:1]
        tiles = train + val
    if args.max_tiles is not None:
        tiles = tiles[: args.max_tiles]
    print(f"device={device} tiles={len(tiles)} cache={cache_dir}")

    if not args.overwrite:
        pending = [t for t in tiles if not cache_file(cache_dir, t).is_file()]
        print(f"cached={len(tiles) - len(pending)} pending={len(pending)}")
        tiles = pending
    if not tiles:
        print("Nothing to do.")
        return 0

    print("[cache] loading Swin2SR teacher (slow first import)...", flush=True)
    t0 = time.perf_counter()
    teacher = SEN2SRModel(model_variant="swin2sr", device=str(device), auto_download=False)
    teacher.set_trainable(False)
    teacher.eval()
    print(f"[cache] teacher ready in {time.perf_counter() - t0:.1f}s", flush=True)

    scenes: dict[str, torch.Tensor] = {}
    done, failed = 0, 0
    t_all = time.perf_counter()
    for i, tile in enumerate(tiles):
        t_iter = time.perf_counter()
        try:
            if tile.s2_path not in scenes:
                scenes[tile.s2_path] = read_normalized_s2(REPO_ROOT / tile.s2_path)
                # Keep at most 2 scenes resident (each ~10x260x260 float32).
                while len(scenes) > 2:
                    scenes.pop(next(iter(scenes)))
            lr = extract_lr(scenes[tile.s2_path], tile)
            try:
                hr = render_teacher(teacher, lr)
            except Exception as exc:  # MPS OOM fallback: retry the tile on CPU
                print(f"  [warn] {tile.site_id}@{tile.y},{tile.x} on {device} failed "
                      f"({exc}); retrying on CPU", flush=True)
                empty_device_cache(device)
                gc.collect()
                cpu_teacher = SEN2SRModel(model_variant="swin2sr", device="cpu",
                                          auto_download=False)
                cpu_teacher.set_trainable(False)
                cpu_teacher.eval()
                hr = render_teacher(cpu_teacher, lr)
                del cpu_teacher
            if not torch.isfinite(hr).all():
                raise ValueError("non-finite teacher output")
            if float(hr.mean()) <= 0 or float(hr.mean()) > 1.5:
                raise ValueError(f"implausible teacher mean {float(hr.mean())}")
            torch.save({"teacher_hr": hr,
                        "site_id": tile.site_id, "y": tile.y, "x": tile.x, "t": tile.t,
                        "teacher": "swin2sr"}, cache_file(cache_dir, tile))
            done += 1
        except Exception as exc:  # noqa: BLE001 - log and continue; rerun fills gaps
            failed += 1
            print(f"  [error] {tile.site_id}@{tile.y},{tile.x}: {exc}", flush=True)
        empty_device_cache(device)
        if device.type == "mps":
            gc.collect()
        dt = time.perf_counter() - t_iter
        remain = (len(tiles) - i - 1) * dt
        print(f"[{i + 1}/{len(tiles)}] {tile.site_id}@{tile.y},{tile.x} "
              f"{dt:.1f}s ETA~{remain / 60:.0f}min done={done} failed={failed}", flush=True)

    print(f"[cache] finished in {(time.perf_counter() - t_all) / 60:.1f}min "
          f"done={done} failed={failed} cache={cache_dir}")
    return 1 if failed and not done else 0


if __name__ == "__main__":
    raise SystemExit(main())
