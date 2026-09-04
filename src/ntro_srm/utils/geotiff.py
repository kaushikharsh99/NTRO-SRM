"""Geospatial GeoTIFF writing utilities with exact georeferencing and metadata tagging."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
import torch

from ntro_srm.preprocessing.transforms import S2_10BAND_NAMES


def compute_sr_transform(
    input_transform: Affine,
    scale_factor: float = 4.0,
) -> Affine:
    """Calculate the output geotransform for 4x super-resolution.

    Preserves the upper-left origin coordinate while dividing pixel dimensions
    by the super-resolution scaling factor (e.g. 10m -> 2.5m).

    Parameters
    ----------
    input_transform : Affine
        Transform of the low-resolution 10m raster.
    scale_factor : float, default=4.0
        Upscaling factor.

    Returns
    -------
    Affine
        Scaled geotransform for the super-resolved output.
    """
    # Affine(a, b, c, d, e, f)
    # a = pixel width, b = row rotation, c = x-origin (upper-left)
    # d = col rotation, e = pixel height (typically negative), f = y-origin (upper-left)
    return Affine(
        input_transform.a / scale_factor,
        input_transform.b / scale_factor,
        input_transform.c,
        input_transform.d / scale_factor,
        input_transform.e / scale_factor,
        input_transform.f,
    )


def write_sr_geotiff(
    output_path: Union[str, Path],
    tensor: Union[torch.Tensor, np.ndarray],
    transform: Affine,
    crs: Union[CRS, str],
    model_name: str = "SEN2SR-Lite",
    input_gsd: str = "10m",
    output_gsd: str = "2.5m",
    upscale_factor: int = 4,
    nodata: Optional[float] = None,
    extra_tags: Optional[dict] = None,
) -> Path:
    """Write a 10-band super-resolved tensor to a georeferenced GeoTIFF with full metadata.

    Parameters
    ----------
    output_path : str or Path
        Destination file path. Parent directories are created if missing.
    tensor : torch.Tensor or np.ndarray
        Array of shape (10, H_sr, W_sr) containing super-resolved reflectance.
    transform : Affine
        Target geotransform (nominally ~2.5m pixel size).
    crs : CRS or str
        Coordinate Reference System.
    model_name : str, default="SEN2SR-Lite"
        Name of the super-resolution model used.
    input_gsd : str, default="10m"
        Nominal low-resolution GSD description.
    output_gsd : str, default="2.5m"
        Nominal super-resolution GSD description.
    upscale_factor : int, default=4
        Spatial magnification factor.
    nodata : float, optional
        Nodata value if applicable.
    extra_tags : dict, optional
        Additional metadata tags to write to the GeoTIFF header.

    Returns
    -------
    Path
        Resolved path to the written GeoTIFF file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert PyTorch tensor to numpy
    if isinstance(tensor, torch.Tensor):
        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise ValueError(f"Batch dimension > 1 not supported for single GeoTIFF write: {tensor.shape}")
            tensor = tensor.squeeze(0)
        data = tensor.detach().cpu().numpy()
    else:
        data = np.asarray(tensor)

    if data.ndim != 3:
        raise ValueError(f"Expected 3D array (channels, height, width), got shape {data.shape}")

    num_bands, height, width = data.shape
    if num_bands != 10:
        raise ValueError(f"Expected exactly 10 bands for Sentinel-2 SR output, but got {num_bands}")

    # Validate finite numerical properties
    if not np.isfinite(data).all():
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure float32 dtype
    if data.dtype != np.float32:
        data = data.astype(np.float32)

    # Enforce non-negativity
    data = np.maximum(data, 0.0)

    # Prepare standard GeoTIFF metadata tags
    tags: dict[str, str] = {
        "MODEL": model_name,
        "INPUT_GSD": input_gsd,
        "OUTPUT_GSD": output_gsd,
        "UPSCALE_FACTOR": str(upscale_factor),
    }

    # Band-specific tags
    for idx, band_name in enumerate(S2_10BAND_NAMES, start=1):
        tags[f"BAND_{idx}"] = band_name

    if extra_tags:
        tags.update({str(k): str(v) for k, v in extra_tags.items()})

    # Rasterio driver configuration
    meta = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": nodata,
        "width": width,
        "height": height,
        "count": 10,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(data)
        dst.update_tags(**tags)

        # Set individual band descriptions
        for idx, band_name in enumerate(S2_10BAND_NAMES, start=1):
            dst.set_band_description(idx, f"Sentinel-2 {band_name} (2.5m SR)")

    return output_path
