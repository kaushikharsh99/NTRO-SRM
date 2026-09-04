"""Sentinel-2 Level-2A surface reflectance normalization pipeline.

Transforms raw integer or scaled satellite reflectances into clean, physically-consistent
floating point values in [0.0, 1.0] suitable for super-resolution model ingestion.
"""

from __future__ import annotations

import warnings
from typing import Literal, Optional, Union

import torch

NormalizationMode = Literal["s2_10000", "already_normalized", "auto"]


def normalize_sentinel2_l2a(
    tensor: torch.Tensor,
    mode: NormalizationMode = "auto",
    scale_factor: float = 10000.0,
    nodata_value: Optional[float] = 65535.0,
    fill_value: float = 0.0,
    clamp_min: float = 0.0,
    clamp_max: Optional[float] = 1.5,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Normalize Sentinel-2 Level-2A reflectance data for super-resolution models.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor of shape (10, H, W) or (B, 10, H, W).
    mode : {"s2_10000", "already_normalized", "auto"}, default="auto"
        Normalization strategy:
        - "s2_10000": Explicitly divides integer/scaled reflectance by 10,000.
        - "already_normalized": Assumes values are already in [0.0, 1.0]; does not divide.
        - "auto": Checks value range. If maximum exceeds 1.5, applies 1/10000 scaling;
          otherwise assumes already normalized.
    scale_factor : float, default=10000.0
        Sentinel-2 Level-2A reflectance quantification value (ESA specification).
    nodata_value : float, optional, default=65535.0
        Sensor nodata sentinel value (standard for S2 L2A uint16) to replace with fill_value.
    fill_value : float, default=0.0
        Value used to replace nodata, NaNs, and Infs.
    clamp_min : float, default=0.0
        Lower physical bound for surface reflectance.
    clamp_max : float, optional, default=1.5
        Upper physical bound (allows slight specular highlights and cloud margins).
    dtype : torch.dtype, default=torch.float32
        Output tensor data type.

    Returns
    -------
    torch.Tensor
        Clean float32 tensor approximately in [0.0, 1.0] with no NaNs, Infs, or negative values.
    """
    if not tensor.is_floating_point():
        tensor = tensor.to(dtype)
    else:
        tensor = tensor.clone()

    # 1. Handle explicit nodata sentinels if specified
    if nodata_value is not None:
        tensor = torch.where(tensor == nodata_value, torch.tensor(fill_value, dtype=dtype), tensor)

    # 2. Sanitize NaNs and Infs
    tensor = torch.nan_to_num(tensor, nan=fill_value, posinf=fill_value, neginf=fill_value)

    # 3. Determine normalization scale
    if mode == "s2_10000":
        tensor = tensor / scale_factor
    elif mode == "already_normalized":
        if tensor.max() > 10.0:
            warnings.warn(
                f"[NTRO-SRM Preprocessing] mode='already_normalized' was specified, but input "
                f"maximum value is {tensor.max().item():.2f} (> 10.0). "
                f"This data appears to be raw unscaled Sentinel-2 reflectance.",
                UserWarning,
                stacklevel=2,
            )
    elif mode == "auto":
        max_val = tensor.max().item()
        if max_val > 1.5:
            # Typical raw S2 integer range [0, 10000]
            tensor = tensor / scale_factor
        else:
            # Already in [0.0, 1.0]
            pass
    else:
        raise ValueError(
            f"Unknown normalization mode: '{mode}'. "
            f"Supported modes: ['s2_10000', 'already_normalized', 'auto']"
        )

    # 4. Enforce non-negative physical constraint
    if clamp_max is not None:
        tensor = torch.clamp(tensor, min=clamp_min, max=clamp_max)
    else:
        tensor = torch.clamp(tensor, min=clamp_min)

    return tensor
