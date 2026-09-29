"""Fine-tuning trainer wiring PairedS2Dataset + SpectralSpatialLoss + SEN2SR-Lite.

The upstream SEN2SR backbone only supports a native 128x128 forward
(128 -> 512 at 4x). Wald/real manifest tiles are 32x32 LR -> 128x128 HR,
so the trainer center-pads LR to 128, forwards, then center-crops the
prediction back to H*4 x W*4 before scoring. Pad/crop are differentiable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ntro_srm.models.sen2sr import SEN2SRModel
from ntro_srm.training.dataset import PairedSample, PairedS2Dataset
from ntro_srm.training.losses import LossWeights, SpectralSpatialLoss
from ntro_srm.utils.device import select_device

NATIVE_SIZE = 128
UPSCALE = 4


def pad_to_native(lr: torch.Tensor, native: int = NATIVE_SIZE) -> tuple[torch.Tensor, int, int]:
    """Center-pad (B,10,H,W) to (B,10,native,native) with replicate mode."""
    if lr.ndim != 4:
        raise ValueError(f"lr must be (B,10,H,W), got {tuple(lr.shape)}")
    _, _, h, w = lr.shape
    if h > native or w > native:
        raise ValueError(f"tile {h}x{w} exceeds native {native}x{native}; retile dataset")
    pad_h = native - h
    pad_w = native - w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    padded = F.pad(lr, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
    return padded, pad_top, pad_left


def forward_native(model: SEN2SRModel, lr: torch.Tensor) -> torch.Tensor:
    """4x forward for arbitrary small tiles via pad -> native forward -> crop."""
    _, _, h, w = lr.shape
    padded, pad_top, pad_left = pad_to_native(lr)
    sr_full = model(padded)
    y0 = pad_top * UPSCALE
    x0 = pad_left * UPSCALE
    return sr_full[:, :, y0 : y0 + h * UPSCALE, x0 : x0 + w * UPSCALE]


def collate_samples(batch: list[PairedSample]) -> dict[str, Any]:
    return {
        "lr": torch.stack([s.lr for s in batch]),
        "hr": torch.stack([s.hr for s in batch]),
        "band_mask": torch.stack([s.band_mask for s in batch]),
        "meta": [s.meta for s in batch],
    }


def find_ft_checkpoint(workspace_root: str | Path | None = None) -> Path | None:
    """Locate fine-tuned Lite weights, preferring the best-epoch file.

    Distilled weights (Swin2SR -> Lite, scripts/distill_lite.py) take
    precedence: they start from the FT checkpoint and move it toward the
    teacher's edge reconstruction, so they are strictly the newer best.
    """
    if workspace_root is None:
        root = Path(__file__).resolve().parents[3]
    else:
        root = Path(workspace_root)
    for name in ("lite_distill_best.pt", "lite_distill.pt",
                 "lite_ft_best.pt", "lite_ft.pt"):
        candidate = root / "outputs" / "finetune" / name
        if candidate.is_file():
            return candidate
    return None


def build_loaders(
    manifest: str | Path,
    batch_size: int = 4,
    num_workers: int = 0,
    pair_types: list[str] | tuple[str, ...] | None = ("wald_synthetic",),
    augment_train: bool = False,
) -> tuple[DataLoader, DataLoader]:
    # Stage 1 fine-tune is Wald-only by default; pass pair_types=None to mix
    # in tiled real RGBN pairs (true 2.5m detail, supervision mask covers
    # common bands only).
    train_ds = PairedS2Dataset(manifest, split="train", pair_types=pair_types,
                               augment=augment_train)
    val_ds = PairedS2Dataset(manifest, split="val", pair_types=pair_types)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_samples,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_samples,
    )
    return train_loader, val_loader


@dataclass
class TrainConfig:
    manifest: str = "datasets/paired/manifest.csv"
    output: str = "outputs/finetune/lite_ft.pt"
    device: str | None = None
    lr: float = 1e-4
    weight_decay: float = 0.0
    epochs: int = 5
    batch_size: int = 4
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    grad_clip: float = 1.0
    resume: str | None = None
    pair_types: tuple[str, ...] = ("wald_synthetic",)
    w_pixel: float = 1.0
    w_spectral: float = 0.25
    w_gradient: float = 0.1
    w_source: float = 0.5
    w_range: float = 0.05
    scheduler: str = "none"  # "none" | "cosine"
    augment_train: bool = False


def build_model(device: torch.device, trainable: bool = True) -> SEN2SRModel:
    model = SEN2SRModel(model_variant="lite", device=str(device), auto_download=True)
    model.set_trainable(trainable)
    model.to(device)
    return model


def train_one_epoch(
    model: SEN2SRModel,
    loader: DataLoader,
    loss_fn: SpectralSpatialLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    total, count = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        mask = batch["band_mask"].to(device)
        optimizer.zero_grad()
        pred = forward_native(model, lr)
        loss = loss_fn(pred, hr, source=lr, band_mask=mask)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item()
        count += 1
    return total / max(count, 1)


@torch.no_grad()
def evaluate(
    model: SEN2SRModel,
    loader: DataLoader,
    loss_fn: SpectralSpatialLoss,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    was_training = model.model.training
    model.model.eval()
    total, count = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        mask = batch["band_mask"].to(device)
        pred = forward_native(model, lr)
        loss = loss_fn(pred, hr, source=lr, band_mask=mask)
        total += loss.item()
        count += 1
    if was_training:
        model.model.train(True)
    return total / max(count, 1)


def save_checkpoint(
    path: str | Path,
    model: SEN2SRModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float,
    config: TrainConfig,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val": best_val,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(config),
            "variant": "lite",
            **(extra or {}),
        },
        path,
    )
    return path


def load_checkpoint(
    path: str | Path,
    model: SEN2SRModel,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


def run_finetune(config: TrainConfig) -> dict[str, Any]:
    device = select_device(config.device)
    train_loader, val_loader = build_loaders(
        config.manifest, config.batch_size, pair_types=config.pair_types,
        augment_train=config.augment_train,
    )
    try:
        _, real_val_loader = build_loaders(
            config.manifest, config.batch_size, pair_types=("real_paired",))
    except ValueError:
        real_val_loader = None
    model = build_model(device)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.lr, weight_decay=config.weight_decay,
    )
    loss_fn = SpectralSpatialLoss(LossWeights(
        pixel=config.w_pixel, spectral=config.w_spectral,
        gradient=config.w_gradient, source_consistency=config.w_source,
        reflectance_range=config.w_range))
    scheduler = None
    if config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(config.epochs, 1))
    start_epoch = 0
    best_val = float("inf")
    if config.resume:
        ckpt = load_checkpoint(config.resume, model, optimizer)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", best_val))
        # Resume keeps weights + momentum but adopts the new run's LR/schedule.
        # The fresh cosine scheduler (built from config.lr above) is kept as-is.
        for group in optimizer.param_groups:
            group["lr"] = config.lr
        model.to(device)

    history: list[dict[str, float]] = []
    val0 = evaluate(model, val_loader, loss_fn, device, config.max_val_batches)
    for epoch in range(start_epoch, start_epoch + config.epochs):
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            config.max_train_batches, config.grad_clip,
        )
        val_loss = evaluate(model, val_loader, loss_fn, device, config.max_val_batches)
        row: dict[str, float] = {"epoch": float(epoch), "train": train_loss, "val": val_loss}
        if real_val_loader is not None:
            row["val_real"] = evaluate(model, real_val_loader, loss_fn, device,
                                       config.max_val_batches)
        if scheduler is not None:
            scheduler.step()
            row["lr"] = float(scheduler.get_last_lr()[0])
        history.append(row)
        ckpt_payload = {"scheduler_state": scheduler.state_dict() if scheduler else None}
        save_checkpoint(config.output, model, optimizer, epoch, min(best_val, val_loss),
                        config, extra=ckpt_payload)
        if val_loss < best_val:
            best_val = val_loss
            best_path = str(Path(config.output).with_name(Path(config.output).stem + "_best.pt"))
            save_checkpoint(best_path, model, optimizer, epoch, best_val, config,
                            extra=ckpt_payload)
    return {"val_before": val0, "val_best": best_val, "history": history,
            "output": config.output, "device": str(device)}
