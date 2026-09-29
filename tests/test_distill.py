"""Tests for Swin->Lite distillation (synthetic teacher cache, CPU-only)."""

from pathlib import Path

import torch

from ntro_srm.training.distill import (
    DistillS2Dataset,
    DistillTile,
    cache_file,
    plan_distill_tiles,
    tile_origins,
)
from ntro_srm.training.trainer import TrainConfig, collate_samples, save_checkpoint


def test_tile_origins_cover_and_clip() -> None:
    assert tile_origins(100, 128, 64) == [0]
    origins = tile_origins(257, 128, 64)
    assert origins[0] == 0
    assert origins[-1] + 128 == 257
    assert all(b - a <= 64 for a, b in zip(origins, origins[1:]))


def test_plan_and_dataset_with_fake_teacher(tmp_path: Path) -> None:
    import numpy as np
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    from ntro_srm.preprocessing.transforms import S2_10BAND_NAMES

    raw = tmp_path / "raw_s2"
    raw.mkdir()
    rng = np.random.default_rng(7)
    for site in ("DE_FRANKFURT_URBAN", "IND_BENGALURU_LAKE"):
        with rasterio.open(
            raw / f"{site}.tif", "w", driver="GTiff", width=130, height=132,
            count=10, dtype="uint16",
            crs=CRS.from_epsg(32632), transform=Affine(10, 0, 500000, 0, -10, 3000000),
        ) as ds:
            ds.write((rng.uniform(500, 5000, (10, 132, 130))).astype("uint16"))
            for i, name in enumerate(S2_10BAND_NAMES, start=1):
                ds.set_band_description(i, name)

    tiles = plan_distill_tiles(raw, ("DE_FRANKFURT_URBAN",), tile=128, stride=64)
    assert len(tiles) == 4  # 132px dim -> origins [0, 4]; 130px dim -> [0, 2]
    assert all(isinstance(t, DistillTile) for t in tiles)

    cache = tmp_path / "cache"
    cache.mkdir()
    for t in tiles:
        torch.save({"teacher_hr": torch.rand(10, 512, 512) * 0.6 + 0.05},
                   cache_file(cache, t))
    ds = DistillS2Dataset(tiles, cache, augment=True)
    sample = ds[0]
    assert sample.lr.shape == (10, 128, 128)
    assert sample.hr.shape == (10, 512, 512)
    assert sample.band_mask.equal(torch.ones(10))
    assert torch.isfinite(sample.lr).all() and torch.isfinite(sample.hr).all()

    batch = collate_samples([ds[0], ds[1]])
    assert batch["lr"].shape == (2, 10, 128, 128)
    assert batch["hr"].shape == (2, 10, 512, 512)


def test_distill_train_step_cpu(tmp_path: Path) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from torch.utils.data import DataLoader

    from ntro_srm.models.sen2sr import SEN2SRModel
    from ntro_srm.training.distill import DistillS2Dataset
    from ntro_srm.training.losses import LossWeights, SpectralSpatialLoss
    from ntro_srm.training.trainer import evaluate, train_one_epoch

    torch.manual_seed(0)
    tiles = [DistillTile("S", "nowhere.tif", 0, 0, 32)]
    cache = tmp_path / "c"
    cache.mkdir()
    torch.save({"teacher_hr": torch.rand(10, 128, 128) * 0.5 + 0.1},
               cache_file(cache, tiles[0]))

    class _FakeScenes(DistillS2Dataset):
        def _scene(self, tile):  # type: ignore[override]
            return torch.rand(10, 32, 32) * 0.5 + 0.05

    loader = DataLoader(_FakeScenes(tiles, cache), batch_size=1,
                        collate_fn=collate_samples)
    model = SEN2SRModel(model_variant="lite", device="cpu",
                        checkpoint_dir=Path("checkpoints/SEN2SRLite"),
                        auto_download=False)
    model.set_trainable(True)
    loss_fn = SpectralSpatialLoss(LossWeights(pixel=1.0, spectral=0.1, gradient=0.6,
                                              source_consistency=0.3,
                                              reflectance_range=0.05))
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    val_before = evaluate(model, loader, loss_fn, torch.device("cpu"))
    train_one_epoch(model, loader, loss_fn, opt, torch.device("cpu"))
    val_after = evaluate(model, loader, loss_fn, torch.device("cpu"))
    assert val_before == val_before  # finite check below covers NaN
    assert all(v == v for v in (val_before, val_after))
    ckpt = tmp_path / "d.pt"
    save_checkpoint(ckpt, model, opt, epoch=0, best_val=val_after, config=TrainConfig())
    assert ckpt.is_file()
