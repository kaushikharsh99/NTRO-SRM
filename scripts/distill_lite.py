#!/usr/bin/env python3
"""Distill Swin2SR (teacher) into SEN2SR-Lite (student) for visible edge gains.

Student keeps Lite's ~2s inference; training chases the teacher's 2.5m
pseudo-targets (cached by scripts/cache_teacher.py) with an edge-weighted
SpectralSpatialLoss: pixel 1.0 + spectral 0.10 + gradient 0.60 + source 0.30.

Usage:
  venv/bin/python scripts/distill_lite.py --smoke     # wiring test (2 epochs, tiny batches)
  venv/bin/python scripts/distill_lite.py             # full run (10 epochs, cosine)
  venv/bin/python scripts/distill_lite.py --epochs 15 --lr 5e-5 --grad-weight 0.8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ntro_srm.training.distill import DistillConfig, run_distill  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distill Swin2SR into SEN2SR-Lite.")
    p.add_argument("--cache-dir", default="datasets/distill_cache")
    p.add_argument("--raw-s2-dir", default="datasets/raw_s2")
    p.add_argument("--output", default="outputs/finetune/lite_distill.pt")
    p.add_argument("--init-checkpoint", default=None,
                   help="Student init weights. Default: best FT weights, else pretrained Lite.")
    p.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-weight", type=float, default=0.6,
                   help="Spatial-gradient term weight (higher = sharper edges).")
    p.add_argument("--source-weight", type=float, default=0.3)
    p.add_argument("--scheduler", default="cosine", choices=["none", "cosine"])
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--smoke", action="store_true",
                   help="Quick wiring test: 2 epochs, 4 train / 2 val batches.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config = DistillConfig(
        cache_dir=args.cache_dir,
        raw_s2_dir=args.raw_s2_dir,
        output=args.output,
        init_checkpoint=args.init_checkpoint,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=2 if args.smoke else args.epochs,
        batch_size=args.batch_size,
        max_train_batches=4 if args.smoke else args.max_train_batches,
        max_val_batches=2 if args.smoke else args.max_val_batches,
        w_gradient=args.grad_weight,
        w_source=args.source_weight,
        scheduler="none" if args.smoke else args.scheduler,
        augment_train=not args.no_augment and not args.smoke,
        train_tile_limit=2 if args.smoke else None,
        val_tile_limit=1 if args.smoke else None,
    )
    print(f"cache={config.cache_dir} epochs={config.epochs} batch={config.batch_size} "
          f"device={config.device or 'auto'} init={config.init_checkpoint or 'auto(FT)'}")
    result = run_distill(config)
    print(f"device={result['device']} val_before={result['val_before']:.6f} "
          f"val_best={result['val_best']:.6f} output={result['output']}")
    for row in result["history"]:
        lr_txt = f" lr={row['lr']:.2e}" if "lr" in row else ""
        print(f"epoch={int(row['epoch'])} train={row['train']:.6f} val={row['val']:.6f}{lr_txt}")
    history_path = str(Path(config.output).with_suffix(".json"))
    Path(history_path).write_text(json.dumps(result["history"], indent=2))
    print(f"history={history_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
