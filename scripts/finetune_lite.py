#!/usr/bin/env python3
"""Fine-tune SEN2SR-Lite on paired Sentinel-2 data (Wald + real RGBN).

Backbone natively forwards 128x128 -> 512x512; 32x32 manifest tiles are
center-padded to 128 and center-cropped after forward (see training/trainer.py).

Usage:
  venv/bin/python scripts/finetune_lite.py --smoke                     # wiring test
  venv/bin/python scripts/finetune_lite.py --epochs 5                  # real fine-tune
  venv/bin/python scripts/finetune_lite.py --epochs 5 --resume outputs/finetune/lite_ft.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ntro_srm.training.trainer import TrainConfig, run_finetune  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune SEN2SR-Lite with SpectralSpatialLoss.")
    p.add_argument("--manifest", default="datasets/paired/manifest.csv")
    p.add_argument("--output", default="outputs/finetune/lite_ft.pt")
    p.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"],
                   help="Default: auto (CUDA > MPS > CPU).")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-weight", type=float, default=0.1,
                   help="Spatial-gradient term weight (higher = sharper edges).")
    p.add_argument("--scheduler", default="none", choices=["none", "cosine"])
    p.add_argument("--pair-types", default="wald",
                   help="Comma list: 'wald' (proxy only) or 'wald,real_paired' (mixed).")
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=False,
                   help="Random flip/rot90 tile augmentation for training.")
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--resume", default=None, help="Checkpoint .pt to resume from.")
    p.add_argument("--smoke", action="store_true",
                   help="Quick wiring test: 2 epochs, 8 train / 4 val batches.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    smoke_train, smoke_val = 8, 4
    alias = {"wald": "wald_synthetic"}
    pair_types = tuple(alias.get(b.strip(), b.strip())
                       for b in args.pair_types.split(",") if b.strip())
    config = TrainConfig(
        manifest=args.manifest,
        output=args.output,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=2 if args.smoke else args.epochs,
        batch_size=args.batch_size,
        max_train_batches=smoke_train if args.smoke else args.max_train_batches,
        max_val_batches=smoke_val if args.smoke else args.max_val_batches,
        resume=args.resume,
        pair_types=pair_types,
        w_gradient=args.grad_weight,
        scheduler="none" if args.smoke else args.scheduler,
        augment_train=args.augment and not args.smoke,
    )
    print(f"manifest={config.manifest} epochs={config.epochs} batch={config.batch_size} "
          f"device={config.device or 'auto'} resume={config.resume}")
    result = run_finetune(config)
    print(f"device={result['device']} val_before={result['val_before']:.6f} "
          f"val_best={result['val_best']:.6f} output={result['output']}")
    for row in result["history"]:
        extra = f" val_real={row['val_real']:.6f}" if "val_real" in row else ""
        lr_txt = f" lr={row['lr']:.2e}" if "lr" in row else ""
        print(f"epoch={int(row['epoch'])} train={row['train']:.6f} val={row['val']:.6f}{extra}{lr_txt}")
    history_path = str(Path(config.output).with_suffix(".json"))
    Path(history_path).write_text(json.dumps(result["history"], indent=2))
    print(f"history={history_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
