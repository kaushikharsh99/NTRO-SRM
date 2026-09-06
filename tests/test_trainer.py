"""Tests for the Lite fine-tuning trainer (pad-forward-crop + checkpointing)."""

import csv
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.crs import CRS
from rasterio.transform import Affine

from ntro_srm.models.sen2sr import SEN2SRModel
from ntro_srm.preprocessing.transforms import S2_10BAND_NAMES
from ntro_srm.training.losses import SpectralSpatialLoss
from ntro_srm.training.trainer import (
    evaluate,
    forward_native,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)
from ntro_srm.training.trainer import TrainConfig


def _write_raster(path: Path, data: np.ndarray) -> None:
    with rasterio.open(
        path, "w", driver="GTiff", width=data.shape[-1], height=data.shape[-2],
        count=data.shape[0], dtype="float32",
        crs=CRS.from_epsg(32643), transform=Affine(10, 0, 500000, 0, -10, 3000000),
    ) as ds:
        ds.write(data.astype(np.float32))
        for i, name in enumerate(S2_10BAND_NAMES, start=1):
            ds.set_band_description(i, name)


def _write_manifest(path: Path, rows: list[dict]) -> None:
    columns = ["pair_id", "site_id", "indian_priority", "landcover", "task", "split",
               "pair_type", "s2_path", "hr_path", "s2_date", "hr_date", "dt_days",
               "cloud_pct", "crs", "align_rmse_px", "tile_y", "tile_x",
               "lr_h", "lr_w", "common_bands", "notes"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


@pytest.fixture(scope="module")
def lite_cpu() -> SEN2SRModel:
    ckpt = Path(__file__).resolve().parents[1] / "checkpoints" / "SEN2SRLite"
    model = SEN2SRModel(model_variant="lite", device="cpu",
                        checkpoint_dir=ckpt, auto_download=False)
    model.set_trainable(True)
    return model


def _tiny_manifest(tmp_path: Path) -> Path:
    rng = np.random.default_rng(11)
    src = tmp_path / "s2.tif"
    _write_raster(src, rng.uniform(0.05, 0.8, (10, 128, 128)).astype(np.float32))
    manifest = tmp_path / "manifest.csv"
    rows = []
    for split, n in (("train", 4), ("val", 2)):
        for i in range(n):
            rows.append({"pair_id": f"{split}-{i}", "site_id": "t", "split": split,
                         "pair_type": "wald_synthetic", "s2_path": str(src),
                         "tile_y": 0, "tile_x": 0, "lr_h": 32, "lr_w": 32})
    _write_manifest(manifest, rows)
    return manifest


def test_forward_native_shapes_and_gradients(lite_cpu: SEN2SRModel) -> None:
    lr = torch.rand(2, 10, 32, 32)
    pred = forward_native(lite_cpu, lr)
    assert pred.shape == (2, 10, 128, 128)
    assert torch.isfinite(pred).all()
    pred.mean().backward()
    grads = [p.grad for p in lite_cpu.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)
    lite_cpu.zero_grad(set_to_none=True)


def test_train_step_save_resume_and_val(tmp_path: Path, lite_cpu: SEN2SRModel) -> None:
    from torch.utils.data import DataLoader

    from ntro_srm.training.dataset import PairedS2Dataset
    from ntro_srm.training.trainer import collate_samples

    torch.manual_seed(0)
    manifest = _tiny_manifest(tmp_path)
    train_loader = DataLoader(PairedS2Dataset(manifest, split="train"),
                              batch_size=2, collate_fn=collate_samples)
    val_loader = DataLoader(PairedS2Dataset(manifest, split="val"),
                            batch_size=2, collate_fn=collate_samples)
    loss_fn = SpectralSpatialLoss()
    device = torch.device("cpu")
    opt = torch.optim.Adam((p for p in lite_cpu.parameters() if p.requires_grad), lr=1e-4)

    val_before = evaluate(lite_cpu, val_loader, loss_fn, device)
    assert np.isfinite(val_before)

    losses = [train_one_epoch(lite_cpu, train_loader, loss_fn, opt, device) for _ in range(3)]
    assert all(np.isfinite(v) for v in losses)
    assert losses[-1] <= losses[0] + 1e-6  # training must not diverge on tiny set

    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt_path, lite_cpu, opt, epoch=2, best_val=min(losses),
                    config=TrainConfig())
    assert ckpt_path.is_file()

    before = {k: v.detach().clone() for k, v in lite_cpu.state_dict().items()}
    fresh = SEN2SRModel(model_variant="lite", device="cpu",
                        checkpoint_dir=Path("checkpoints/SEN2SRLite"), auto_download=False)
    fresh.set_trainable(True)
    opt2 = torch.optim.Adam((p for p in fresh.parameters() if p.requires_grad), lr=1e-4)
    meta = load_checkpoint(ckpt_path, fresh, opt2)
    assert meta["epoch"] == 2
    for key, value in before.items():
        assert torch.equal(value, fresh.state_dict()[key]), f"mismatch after resume: {key}"

    val_after = evaluate(fresh, val_loader, loss_fn, device)
    assert np.isfinite(val_after)
