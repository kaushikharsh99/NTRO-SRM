"""Wald-protocol degradation: simulate 10m->2.5m training from real 10m S2.

We learn the 4x mapping at one scale lower (40m -> 10m) so every Sentinel-2
scene — including Indian AOIs with no commercial HR — yields valid 10-band
paired data. Gaussian blur approximates the sensor MTF before decimation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def gaussian_kernel(kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    ax = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
    k1d = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    k1d /= k1d.sum()
    return torch.outer(k1d, k1d)


def wald_degrade(
    hr_10m: torch.Tensor,
    scale: int = 4,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Blur + decimate a (C,H,W) or (B,C,H,W) 10m stack to synthetic 40m LR.

    The returned LR upsampled back to (H,W) with area-averaging matches the
    shape convention used by SpectralSpatialLoss(source=...).
    """
    squeeze = hr_10m.ndim == 3
    x = hr_10m.unsqueeze(0).float() if squeeze else hr_10m.float()
    b, c, h, w = x.shape
    kernel = gaussian_kernel(kernel_size, sigma).to(x.device, x.dtype)
    kernel = kernel.expand(c, 1, kernel_size, kernel_size)
    pad = kernel_size // 2
    blurred = F.conv2d(x, kernel, padding=pad, groups=c)
    lr_small = F.avg_pool2d(blurred, kernel_size=scale)  # (B,C,H/4,W/4)
    out = lr_small.squeeze(0) if squeeze else lr_small
    return out


def tile_pair_indices(
    height: int, width: int, tile: int = 128, stride: int = 96
) -> list[tuple[int, int]]:
    """Sliding-window top-left corners over the HR (10m working) grid."""
    ys = list(range(0, max(1, height - tile + 1), stride))
    xs = list(range(0, max(1, width - tile + 1), stride))
    if ys[-1] + tile < height:
        ys.append(height - tile)
    if xs[-1] + tile < width:
        xs.append(width - tile)
    return [(y, x) for y in ys for x in xs]


def valid_tile_mask(tile: np.ndarray, nodata_fraction_max: float = 0.05) -> bool:
    flat = tile.reshape(tile.shape[0], -1)
    invalid = ~np.isfinite(flat).all(axis=0)
    return bool(invalid.mean() <= nodata_fraction_max)
