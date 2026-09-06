"""Tests for multi-temporal dataset and fusion head."""

import csv
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.crs import CRS
from rasterio.transform import Affine

from ntro_srm.preprocessing.transforms import S2_10BAND_NAMES
from ntro_srm.training.fusion import TemporalFusionHead
from ntro_srm.training.temporal import TemporalS2Dataset, temporal_mean_lr


def _write_raster(path: Path, data: np.ndarray) -> None:
    with rasterio.open(
        path, "w", driver="GTiff", width=data.shape[-1], height=data.shape[-2],
        count=data.shape[0], dtype="float32",
        crs=CRS.from_epsg(32643), transform=Affine(10, 0, 500000, 0, -10, 3000000),
    ) as ds:
        ds.write(data.astype(np.float32))
        for i, name in enumerate(S2_10BAND_NAMES, start=1):
            ds.set_band_description(i, name)


def _manifest(path: Path, rows: list[dict], header: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in header})


PAIRED_HEADER = ["pair_id", "site_id", "indian_priority", "landcover", "task", "split",
                 "pair_type", "s2_path", "hr_path", "s2_date", "hr_date", "dt_days",
                 "cloud_pct", "crs", "align_rmse_px", "tile_y", "tile_x",
                 "lr_h", "lr_w", "common_bands", "notes"]
TEMP_HEADER = ["site_id", "scene_id", "date", "cloud_pct", "dt_days",
               "is_reference", "path", "shift_dy", "shift_dx"]


def _two_date_setup(tmp_path: Path):
    rng = np.random.default_rng(3)
    ref = tmp_path / "ref.tif"
    alt = tmp_path / "alt.tif"
    _write_raster(ref, rng.uniform(0.05, 0.8, (10, 128, 128)).astype(np.float32))
    _write_raster(alt, rng.uniform(0.05, 0.8, (10, 128, 128)).astype(np.float32))
    paired = tmp_path / "manifest.csv"
    _manifest(paired, [{"pair_id": "w1", "site_id": "s", "split": "train",
                        "pair_type": "wald_synthetic", "s2_path": str(ref),
                        "tile_y": 0, "tile_x": 0, "lr_h": 32, "lr_w": 32}], PAIRED_HEADER)
    temporal = tmp_path / "temporal.csv"
    _manifest(temporal, [
        {"site_id": "s", "scene_id": "a", "date": "2025-01-01", "cloud_pct": 0,
         "path": str(ref)},
        {"site_id": "s", "scene_id": "b", "date": "2025-01-06", "cloud_pct": 1,
         "path": str(alt)},
    ], TEMP_HEADER)
    return paired, temporal


def test_temporal_dataset_stacks_frames(tmp_path: Path) -> None:
    paired, temporal = _two_date_setup(tmp_path)
    ds = TemporalS2Dataset(paired, temporal, split="train", frames=2)
    assert len(ds) == 1
    s = ds[0]
    assert s["lr"].shape == (2, 10, 32, 32)
    assert s["hr"].shape == (10, 128, 128)
    assert s["band_mask"].sum().item() == 10.0
    assert torch.isfinite(s["lr"]).all() and torch.isfinite(s["hr"]).all()


def test_temporal_dataset_needs_full_stack(tmp_path: Path) -> None:
    paired, temporal = _two_date_setup(tmp_path)
    with pytest.raises(ValueError, match="No temporal tiles"):
        TemporalS2Dataset(paired, temporal, split="train", frames=3)


def test_mean_fusion_shape() -> None:
    lr = torch.rand(2, 3, 10, 32, 32)
    assert temporal_mean_lr(lr).shape == (2, 10, 32, 32)


def test_fusion_head_weights_and_gradients() -> None:
    head = TemporalFusionHead(frames=3)
    sr = torch.rand(2, 3, 10, 32, 32)
    lr = torch.rand(2, 3, 10, 8, 8)
    out = head(sr, lr)
    assert out.shape == (2, 10, 32, 32)
    assert torch.isfinite(out).all()
    w = head.weight_map(sr, lr)
    assert w.shape == (2, 3, 32, 32)
    assert torch.allclose(w.sum(dim=1), torch.ones(2, 32, 32), atol=1e-5)
    out.mean().backward()
    assert any(p.grad is not None for p in head.parameters())
