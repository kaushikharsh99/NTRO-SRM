"""Preprocessing transformations for Sentinel-2 satellite imagery.

Includes reflectance normalization, NaN/Inf handling, non-negative clamping,
and band-order conversions between 10-band Sentinel-2 stacks and 4-band RGBN stacks.
"""

from typing import Optional, Sequence, Union
import torch

# Standard Sentinel-2 10-band surface reflectance ordering (10m and 20m bands)
S2_10BAND_NAMES: list[str] = [
    "B02",  # Blue (10m)       - Index 0
    "B03",  # Green (10m)      - Index 1
    "B04",  # Red (10m)        - Index 2
    "B05",  # Red Edge 1 (20m) - Index 3
    "B06",  # Red Edge 2 (20m) - Index 4
    "B07",  # Red Edge 3 (20m) - Index 5
    "B08",  # NIR (10m)        - Index 6
    "B8A",  # Narrow NIR (20m) - Index 7
    "B11",  # SWIR 1 (20m)     - Index 8
    "B12",  # SWIR 2 (20m)     - Index 9
]

# Standard 4-band RGBN ordering used by optical SR models
RGBN_BAND_NAMES: list[str] = [
    "B04",  # Red   - Index 0 in RGBN
    "B03",  # Green - Index 1 in RGBN
    "B02",  # Blue  - Index 2 in RGBN
    "B08",  # NIR   - Index 3 in RGBN
]

# Indices in 10-band stack corresponding to RGBN
S2_TO_RGBN_INDICES: list[int] = [2, 1, 0, 6]

# Indices in 10-band stack corresponding to 20m RSWIR bands
S2_TO_RSWIR_INDICES: list[int] = [3, 4, 5, 7, 8, 9]


def normalize_reflectance(
    tensor: torch.Tensor,
    scale: float = 10000.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert raw integer Sentinel-2 Level-2A reflectance to [0.0, 1.0] range.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor with values typically in [0, 10000].
    scale : float, default=10000.0
        Reflectance scaling factor from ESA Sentinel-2 L2A specification.
    dtype : torch.dtype, default=torch.float32
        Target floating point data type.

    Returns
    -------
    torch.Tensor
        Normalized reflectance tensor.
    """
    if not tensor.is_floating_point():
        tensor = tensor.to(dtype)
    return tensor / scale


def denormalize_reflectance(
    tensor: torch.Tensor,
    scale: float = 10000.0,
    clamp_max: Optional[float] = 10000.0,
    dtype: torch.dtype = torch.int16,
) -> torch.Tensor:
    """Convert normalized [0.0, 1.0] reflectance back to Sentinel-2 integer scale.

    Parameters
    ----------
    tensor : torch.Tensor
        Normalized reflectance tensor.
    scale : float, default=10000.0
        Reflectance scaling factor.
    clamp_max : float, optional
        Maximum clamp threshold before integer conversion.
    dtype : torch.dtype, default=torch.int16
        Target integer data type.

    Returns
    -------
    torch.Tensor
        Un-normalized integer reflectance tensor.
    """
    scaled = tensor * scale
    if clamp_max is not None:
        scaled = torch.clamp(scaled, min=0.0, max=clamp_max)
    return scaled.to(dtype)


def handle_nans(
    tensor: torch.Tensor,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Replace NaN, positive infinity, and negative infinity with a fill value.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor possibly containing NaN/Inf values.
    fill_value : float, default=0.0
        Replacement value.

    Returns
    -------
    torch.Tensor
        Sanitized tensor with no NaN or Inf elements.
    """
    return torch.nan_to_num(
        tensor,
        nan=fill_value,
        posinf=fill_value,
        neginf=fill_value,
    )


def clamp_non_negative(
    tensor: torch.Tensor,
    min_value: float = 0.0,
    max_value: Optional[float] = None,
) -> torch.Tensor:
    """Clamp tensor values to enforce physical reflectance constraints (>= 0.0).

    Parameters
    ----------
    tensor : torch.Tensor
        Input reflectance tensor.
    min_value : float, default=0.0
        Lower bound. Surface reflectance cannot physically be negative.
    max_value : float, optional
        Upper bound (e.g. 1.0 or 1.5 for hyper-reflective clouds).

    Returns
    -------
    torch.Tensor
        Clamped tensor.
    """
    if max_value is not None:
        return torch.clamp(tensor, min=min_value, max=max_value)
    return torch.clamp(tensor, min=min_value)


def s2_10band_to_rgbn(tensor: torch.Tensor) -> torch.Tensor:
    """Extract RGBN channels [B04, B03, B02, B08] from a 10-band Sentinel-2 stack.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor of shape (B, 10, H, W) or (10, H, W) ordered as
        [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12].

    Returns
    -------
    torch.Tensor
        Tensor of shape (B, 4, H, W) or (4, H, W) ordered as [B04, B03, B02, B08].
    """
    channel_dim = -3
    if tensor.shape[channel_dim] != 10:
        raise ValueError(
            f"Expected 10 channels at dimension {channel_dim}, but got {tensor.shape[channel_dim]}. "
            f"Tensor shape: {tensor.shape}"
        )

    if tensor.ndim == 4:
        return tensor[:, S2_TO_RGBN_INDICES]
    elif tensor.ndim == 3:
        return tensor[S2_TO_RGBN_INDICES]
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {tensor.shape}")


def rgbn_to_s2_10band(
    rgbn: torch.Tensor,
    rswir: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct a full 10-band Sentinel-2 stack from RGBN and 6 RSWIR bands.

    Parameters
    ----------
    rgbn : torch.Tensor
        RGBN tensor of shape (B, 4, H, W) or (4, H, W) ordered [B04, B03, B02, B08].
    rswir : torch.Tensor
        RSWIR tensor of shape (B, 6, H, W) or (6, H, W) ordered
        [B05, B06, B07, B8A, B11, B12].

    Returns
    -------
    torch.Tensor
        Full 10-band stack ordered [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12].
    """
    is_4d = rgbn.ndim == 4
    if not is_4d:
        rgbn = rgbn.unsqueeze(0)
        rswir = rswir.unsqueeze(0)

    if rgbn.shape[1] != 4:
        raise ValueError(f"Expected 4 channels for rgbn, got {rgbn.shape[1]}")
    if rswir.shape[1] != 6:
        raise ValueError(f"Expected 6 channels for rswir, got {rswir.shape[1]}")

    # rgbn order: [B04 (Red)=0, B03 (Green)=1, B02 (Blue)=2, B08 (NIR)=3]
    # rswir order: [B05=0, B06=1, B07=2, B8A=3, B11=4, B12=5]
    b02 = rgbn[:, 2:3]
    b03 = rgbn[:, 1:2]
    b04 = rgbn[:, 0:1]
    b05 = rswir[:, 0:1]
    b06 = rswir[:, 1:2]
    b07 = rswir[:, 2:3]
    b08 = rgbn[:, 3:4]
    b8a = rswir[:, 3:4]
    b11 = rswir[:, 4:5]
    b12 = rswir[:, 5:6]

    stacked = torch.cat([b02, b03, b04, b05, b06, b07, b08, b8a, b11, b12], dim=1)

    if not is_4d:
        stacked = stacked.squeeze(0)

    return stacked
