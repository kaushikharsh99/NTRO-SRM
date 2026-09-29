"""Swin2SR -> SEN2SR-Lite knowledge distillation.

Why this exists: fine-tuning Lite on Wald/NAIP pairs only moves the weights a
little (mean |FT-base| ~0.003 reflectance, visually negligible). The Swin2SR
teacher disagrees with Lite far more (mean |Swin-Lite| ~0.08, p99 ~0.3 on real
tiles), mostly at edges — exactly the visible detail we want, transferred into
a Lite-speed student (~2s vs ~12min inference).

Protocol (honest labels):
- Teacher outputs are pseudo-targets, NOT ground truth. Distillation transfers
  the teacher's edge reconstruction at Lite speed; it cannot create verified
  2.5m detail. Source-consistency (area-downsample vs observed 10m) stays in
  the loss to anchor the student to the real observation.
- Geographic split is inherited from the paired manifest site splits: test
  sites (IND_ASSAM_FLOOD, IND_DELHI_URBAN, IND_MUMBAI_COAST) are NEVER distilled.
- Teacher targets are cached once by scripts/cache_teacher.py (Swin is ~40s per
  128px tile on MPS) so training itself stays fast and repeatable.

Tile geometry: LR tiles are 128x128 at 10m (native Swin/Lite forward, no pad
waste) -> 512x512 pseudo-HR at 2.5m. One teacher forward per tile.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ntro_srm.models.sen2sr import SEN2SRModel
from ntro_srm.preprocessing.sentinel2 import normalize_sentinel2_l2a
from ntro_srm.training.dataset import PairedSample
from ntro_srm.training.losses import LossWeights, SpectralSpatialLoss
from ntro_srm.training.trainer import (
    build_model,
    evaluate,
    forward_native,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)
from ntro_srm.utils.device import select_device

# Geographic split mirrors datasets/paired/manifest.csv site splits.
# Test sites are deliberately absent: never distill what you evaluate on.
DISTILL_TRAIN_SITES = (
    "DE_FRANKFURT_URBAN",
    "IND_JAIPUR_SEMIARID",
    "IND_PUNJAB_AGRI",
    "UAE_DUBAI_COAST",
    "USA_MOUNTAIN_LAKE",
    "USA_SALINAS_AGRI",
)
DISTILL_VAL_SITES = (
    "IND_BENGALURU_LAKE",
    "NL_ROTTERDAM_PORT",
    "USA_TAHOE_ALPINE",
)

TILE_LR = 128
UPSCALE = 4


@dataclass(frozen=True)
class DistillTile:
    site_id: str
    s2_path: str
    y: int
    x: int
    t: int = TILE_LR


def scene_size(s2_path: Path) -> tuple[int, int]:
    """Return (height, width) of a 10-band S2 chip without reading pixels."""
    with rasterio.open(s2_path) as src:
        return int(src.height), int(src.width)


def tile_origins(length: int, tile: int, stride: int) -> list[int]:
    """Non-wasteful 1D origins covering [0, length) with a tile window."""
    if length <= tile:
        return [0]
    origins = list(range(0, length - tile + 1, stride))
    if origins[-1] + tile < length:
        origins.append(length - tile)
    return origins


def plan_distill_tiles(
    raw_s2_dir: str | Path,
    sites: tuple[str, ...] | list[str],
    tile: int = TILE_LR,
    stride: int = 64,
) -> list[DistillTile]:
    """Tile each site's S2 chip into LR windows (geometry only, no pixels)."""
    raw_s2_dir = Path(raw_s2_dir)
    tiles: list[DistillTile] = []
    for site in sites:
        s2_path = raw_s2_dir / f"{site}.tif"
        if not s2_path.is_file():
            raise FileNotFoundError(f"S2 chip missing for {site}: {s2_path}")
        h, w = scene_size(s2_path)
        for y in tile_origins(h, tile, stride):
            for x in tile_origins(w, tile, stride):
                tiles.append(DistillTile(site, str(s2_path), y, x, tile))
    return tiles


def cache_file(cache_dir: str | Path, tile: DistillTile) -> Path:
    return Path(cache_dir) / f"{tile.site_id}_y{tile.y:04d}_x{tile.x:04d}_t{tile.t}.pt"


def read_normalized_s2(s2_path: str | Path) -> torch.Tensor:
    """Read a raw S2 chip and normalize to [0,1] reflectance (10,H,W)."""
    with rasterio.open(s2_path) as src:
        arr = src.read().astype(np.float32)
    s2 = torch.from_numpy(arr)
    return normalize_sentinel2_l2a(s2, mode="auto", nodata_value=None).float()


def extract_lr(scene: torch.Tensor, tile: DistillTile) -> torch.Tensor:
    return scene[:, tile.y : tile.y + tile.t, tile.x : tile.x + tile.t].contiguous()


class DistillS2Dataset(Dataset):
    """LR tiles + cached Swin teacher targets.

    Returns PairedSample-compatible items (band_mask = ones(10): the teacher
    supervises all ten bands, unlike NAIP real pairs which cover RGBN only),
    so the existing collate/train/evaluate helpers work unchanged.
    """

    def __init__(
        self,
        tiles: list[DistillTile],
        cache_dir: str | Path,
        augment: bool = False,
        scenes: dict[str, torch.Tensor] | None = None,
    ) -> None:
        if not tiles:
            raise ValueError("DistillS2Dataset needs at least one tile")
        self.tiles = tiles
        self.cache_dir = Path(cache_dir)
        self.augment = augment
        self._scenes: dict[str, torch.Tensor] = dict(scenes or {})
        missing = [
            str(cache_file(self.cache_dir, t))
            for t in tiles
            if not cache_file(self.cache_dir, t).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} teacher targets missing in {self.cache_dir} "
                f"(e.g. {missing[0]}). Run scripts/cache_teacher.py first."
            )

    def __len__(self) -> int:
        return len(self.tiles)

    def _scene(self, tile: DistillTile) -> torch.Tensor:
        if tile.s2_path not in self._scenes:
            self._scenes[tile.s2_path] = read_normalized_s2(tile.s2_path)
        return self._scenes[tile.s2_path]

    def __getitem__(self, idx: int) -> PairedSample:
        tile = self.tiles[idx]
        lr = extract_lr(self._scene(tile), tile).float()
        payload = torch.load(cache_file(self.cache_dir, tile), map_location="cpu")
        hr = payload["teacher_hr"].float()
        expect = (10, tile.t * UPSCALE, tile.t * UPSCALE)
        if tuple(hr.shape) != expect:
            raise ValueError(f"Bad cached target {tile}: {tuple(hr.shape)} != {expect}")
        if self.augment:
            lr, hr = _augment_pair(lr, hr)
        return PairedSample(
            lr=lr.float(),
            hr=hr.float(),
            band_mask=torch.ones(10),
            meta={"pair_id": f"distill::{tile.site_id}@{tile.y},{tile.x}",
                  "pair_type": "swin_distill", "site_id": tile.site_id, "split": ""},
        )


def _augment_pair(lr: torch.Tensor, hr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.rand(()) < 0.5:
        lr, hr = torch.flip(lr, [-1]), torch.flip(hr, [-1])
    if torch.rand(()) < 0.5:
        lr, hr = torch.flip(lr, [-2]), torch.flip(hr, [-2])
    k = int(torch.randint(0, 4, (1,)).item())
    if k:
        lr, hr = torch.rot90(lr, k, [-2, -1]), torch.rot90(hr, k, [-2, -1])
    return lr, hr


def find_distill_checkpoint(workspace_root: str | Path | None = None) -> Path | None:
    """Locate distilled Lite weights, preferring the best-epoch file."""
    root = Path(__file__).resolve().parents[3] if workspace_root is None else Path(workspace_root)
    for name in ("lite_distill_best.pt", "lite_distill.pt"):
        candidate = root / "outputs" / "finetune" / name
        if candidate.is_file():
            return candidate
    return None


@dataclass
class DistillConfig:
    cache_dir: str = "datasets/distill_cache"
    raw_s2_dir: str = "datasets/raw_s2"
    output: str = "outputs/finetune/lite_distill.pt"
    init_checkpoint: str | None = None  # default: best FT weights, else pretrained
    device: str | None = None
    lr: float = 3e-5
    weight_decay: float = 0.0
    epochs: int = 10
    batch_size: int = 2
    tile: int = TILE_LR
    train_stride: int = 64
    val_stride: int = 128
    grad_clip: float = 1.0
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    w_pixel: float = 1.0
    w_spectral: float = 0.10
    w_gradient: float = 0.60
    w_source: float = 0.30
    w_range: float = 0.05
    scheduler: str = "cosine"  # "none" | "cosine"
    augment_train: bool = True
    train_tile_limit: int | None = None  # smoke/debug: use first N train tiles
    val_tile_limit: int | None = None


def build_distill_loaders(
    config: DistillConfig,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, list[DistillTile], list[DistillTile]]:
    from ntro_srm.training.trainer import collate_samples

    train_tiles = plan_distill_tiles(config.raw_s2_dir, DISTILL_TRAIN_SITES,
                                     config.tile, config.train_stride)
    val_tiles = plan_distill_tiles(config.raw_s2_dir, DISTILL_VAL_SITES,
                                   config.tile, config.val_stride)
    if config.train_tile_limit is not None:
        train_tiles = train_tiles[: config.train_tile_limit]
    if config.val_tile_limit is not None:
        val_tiles = val_tiles[: config.val_tile_limit]
    train_loader = DataLoader(
        DistillS2Dataset(train_tiles, config.cache_dir, augment=config.augment_train),
        batch_size=config.batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_samples,
    )
    val_loader = DataLoader(
        DistillS2Dataset(val_tiles, config.cache_dir, augment=False),
        batch_size=config.batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_samples,
    )
    return train_loader, val_loader, train_tiles, val_tiles


def resolve_init_checkpoint(config: DistillConfig) -> str | None:
    if config.init_checkpoint:
        return config.init_checkpoint
    from ntro_srm.training.trainer import find_ft_checkpoint

    ft = find_ft_checkpoint()
    return str(ft) if ft is not None else None


def run_distill(config: DistillConfig) -> dict[str, Any]:
    device = select_device(config.device)
    train_loader, val_loader, train_tiles, val_tiles = build_distill_loaders(config)
    print(f"[distill] tiles: train={len(train_tiles)} val={len(val_tiles)} "
          f"tile={config.tile} batch={config.batch_size}")

    model = build_model(device)
    init_ckpt = resolve_init_checkpoint(config)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.lr, weight_decay=config.weight_decay,
    )
    start_epoch = 0
    if init_ckpt is not None:
        ckpt = load_checkpoint(init_ckpt, model, optimizer)
        # Adopt this run's LR; keep weights + momentum.
        for group in optimizer.param_groups:
            group["lr"] = config.lr
        model.to(device)
        print(f"[distill] initialized student from {init_ckpt} "
              f"(epoch={ckpt.get('epoch')})")
    else:
        print("[distill] no FT checkpoint found; starting from pretrained Lite")

    loss_fn = SpectralSpatialLoss(LossWeights(
        pixel=config.w_pixel, spectral=config.w_spectral,
        gradient=config.w_gradient, source_consistency=config.w_source,
        reflectance_range=config.w_range))
    scheduler = None
    if config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(config.epochs, 1))

    # DistillConfig is a superset of TrainConfig fields used by save_checkpoint;
    # persist as a plain dict via the same file format (variant "lite").
    from ntro_srm.training.trainer import TrainConfig

    train_cfg = TrainConfig(
        manifest=f"distill:{config.cache_dir}", output=config.output,
        device=config.device, lr=config.lr, weight_decay=config.weight_decay,
        epochs=config.epochs, batch_size=config.batch_size,
        max_train_batches=config.max_train_batches,
        max_val_batches=config.max_val_batches, grad_clip=config.grad_clip,
        resume=init_ckpt, pair_types=("swin_distill",),
        w_pixel=config.w_pixel, w_spectral=config.w_spectral,
        w_gradient=config.w_gradient, w_source=config.w_source,
        w_range=config.w_range, scheduler=config.scheduler,
        augment_train=config.augment_train,
    )

    history: list[dict[str, float]] = []
    val0 = evaluate(model, val_loader, loss_fn, device, config.max_val_batches)
    print(f"[distill] val_before={val0:.6f} (vs teacher, lower = closer to Swin)")
    best_val = float("inf")
    for epoch in range(start_epoch, start_epoch + config.epochs):
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            config.max_train_batches, config.grad_clip,
        )
        val_loss = evaluate(model, val_loader, loss_fn, device, config.max_val_batches)
        row: dict[str, float] = {"epoch": float(epoch), "train": train_loss, "val": val_loss}
        if scheduler is not None:
            scheduler.step()
            row["lr"] = float(scheduler.get_last_lr()[0])
        history.append(row)
        print(f"[distill] epoch={epoch} train={train_loss:.6f} val={val_loss:.6f}"
              + (f" lr={row['lr']:.2e}" if "lr" in row else ""))
        ckpt_payload = {"scheduler_state": scheduler.state_dict() if scheduler else None}
        save_checkpoint(config.output, model, optimizer, epoch, min(best_val, val_loss),
                        train_cfg, extra=ckpt_payload)
        if val_loss < best_val:
            best_val = val_loss
            best_path = str(Path(config.output).with_name(Path(config.output).stem + "_best.pt"))
            save_checkpoint(best_path, model, optimizer, epoch, best_val, train_cfg,
                            extra=ckpt_payload)
    return {"val_before": val0, "val_best": best_val, "history": history,
            "output": config.output, "device": str(device),
            "n_train": len(train_tiles), "n_val": len(val_tiles)}
