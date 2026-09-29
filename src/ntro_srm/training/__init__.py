"""Training objectives for spectrally consistent Sentinel-2 SR."""

from ntro_srm.training.aois import ALL_AOIS, GLOBAL_AOIS, INDIAN_AOIS, CatalogAOI
from ntro_srm.training.dataset import PairedS2Dataset, PairedSample
from ntro_srm.training.losses import LossWeights, SpectralSpatialLoss
from ntro_srm.training.trainer import (
    TrainConfig,
    build_loaders,
    evaluate,
    find_ft_checkpoint,
    forward_native,
    load_checkpoint,
    run_finetune,
    save_checkpoint,
    train_one_epoch,
)
from ntro_srm.training.wald import tile_pair_indices, valid_tile_mask, wald_degrade

__all__ = [
    "ALL_AOIS",
    "CatalogAOI",
    "GLOBAL_AOIS",
    "INDIAN_AOIS",
    "LossWeights",
    "PairedS2Dataset",
    "PairedSample",
    "SpectralSpatialLoss",
    "TrainConfig",
    "build_loaders",
    "evaluate",
    "find_ft_checkpoint",
    "forward_native",
    "load_checkpoint",
    "run_finetune",
    "save_checkpoint",
    "train_one_epoch",
    "tile_pair_indices",
    "valid_tile_mask",
    "wald_degrade",
]
