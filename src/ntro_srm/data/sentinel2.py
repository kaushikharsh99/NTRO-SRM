"""Sentinel-2 data ingestion and multi-resolution band alignment.

Supports standard multi-band GeoTIFFs (10-band and 12-band stacks), individual band rasters,
and explicitly resamples 20m bands onto the 10m spatial grid using GDAL/rasterio bilinear interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject
import torch

from ntro_srm.preprocessing.transforms import S2_10BAND_NAMES

# Native spatial resolutions for Sentinel-2 MSI surface reflectance bands
S2_10M_BANDS: list[str] = ["B02", "B03", "B04", "B08"]
S2_20M_BANDS: list[str] = ["B05", "B06", "B07", "B8A", "B11", "B12"]
S2_60M_BANDS: list[str] = ["B01", "B09", "B10"]

# Standard 12-band order commonly exported by Google Earth Engine (COPERNICUS/S2_SR_HARMONIZED)
# and the SEN2NEON benchmark dataset
S2_12BAND_ORDER: list[str] = [
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B11", "B12"
]


@dataclass
class Sentinel2RasterData:
    """Container holding extracted Sentinel-2 raster data and geospatial metadata.

    Attributes
    ----------
    tensor : torch.Tensor
        Multi-spectral tensor of shape (10, H, W) containing float32 or uint16 reflectance.
    band_names : list[str]
        Strictly ordered band names: [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12].
    transform : Affine
        Affine geotransform of the 10m grid.
    crs : CRS
        Coordinate Reference System.
    bounds : tuple[float, float, float, float]
        Bounding box in CRS coordinates (left, bottom, right, top).
    resolution : tuple[float, float]
        Pixel resolution (res_x, res_y) in CRS units (nominally ~10.0m).
    width : int
        Width of the 10m raster grid in pixels.
    height : int
        Height of the 10m raster grid in pixels.
    metadata : dict
        Original raster metadata and tags.
    nodata : float or None
        Nodata value if specified in the raster.
    """

    tensor: torch.Tensor
    band_names: list[str]
    transform: Affine
    crs: CRS
    bounds: tuple[float, float, float, float]
    resolution: tuple[float, float]
    width: int
    height: int
    metadata: dict
    nodata: Optional[float] = None


def resample_band_to_grid(
    src_data: np.ndarray,
    src_transform: Affine,
    src_crs: CRS,
    dst_transform: Affine,
    dst_crs: CRS,
    dst_shape: tuple[int, int],
    resampling_method: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """Explicitly resample a 2D raster band onto a target spatial grid.

    Used to align 20m Sentinel-2 bands onto the 10m reference grid before
    super-resolution processing.

    Parameters
    ----------
    src_data : np.ndarray
        2D source array of shape (H_src, W_src).
    src_transform : Affine
        Affine transform of the source band (e.g. ~20m pixel spacing).
    src_crs : CRS
        Coordinate reference system of source band.
    dst_transform : Affine
        Target affine transform (e.g. 10m pixel spacing).
    dst_crs : CRS
        Coordinate reference system of target grid.
    dst_shape : tuple[int, int]
        Target spatial dimensions (H_dst, W_dst).
    resampling_method : Resampling, default=Resampling.bilinear
        GDAL/rasterio resampling algorithm (bilinear recommended for continuous reflectance).

    Returns
    -------
    np.ndarray
        Resampled 2D array of shape (H_dst, W_dst) matching dst_transform.
    """
    dst_data = np.zeros(dst_shape, dtype=src_data.dtype)
    reproject(
        source=src_data,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling_method,
    )
    return dst_data


class Sentinel2Reader:
    """Ingests Sentinel-2 imagery from GeoTIFFs, extracting and ordering the 10 required bands.

    Handles:
    - 10-band stacks (direct 1:1 mapping).
    - 12-band stacks (extracts 10m RGBN + 20m RSWIR, discarding 60m atmospheric bands).
    - Rasters with explicit band descriptions or metadata tags.
    - Automatic 20m -> 10m deterministic resampling if bands differ in spatial resolution.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        band_indices: Optional[Dict[str, int]] = None,
        resampling_method: Resampling = Resampling.bilinear,
    ) -> None:
        """Initialize reader for a Sentinel-2 raster file.

        Parameters
        ----------
        file_path : str or Path
            Path to Sentinel-2 GeoTIFF raster file.
        band_indices : dict[str, int], optional
            Explicit 1-based band index mapping for each of the 10 bands.
            Example: {'B02': 1, 'B03': 2, ...}
            If None, auto-detected from metadata tags, band descriptions, or band count.
        resampling_method : Resampling, default=Resampling.bilinear
            Resampling method used for resolution alignment.
        """
        self.file_path = Path(file_path)
        if not self.file_path.is_file():
            raise FileNotFoundError(f"Sentinel-2 file not found: {self.file_path}")

        self.resampling_method = resampling_method
        self.explicit_band_indices = band_indices

    def _resolve_band_indices(self, src: rasterio.DatasetReader) -> dict[str, int]:
        """Determine 1-based band indices for all 10 required bands."""
        if self.explicit_band_indices is not None:
            # Verify all 10 bands are mapped
            missing = [b for b in S2_10BAND_NAMES if b not in self.explicit_band_indices]
            if missing:
                raise ValueError(f"Explicit band_indices missing required bands: {missing}")
            return self.explicit_band_indices

        # Check band descriptions if present
        descriptions = [d.upper() if d else "" for d in (src.descriptions or [])]
        found_mapping: dict[str, int] = {}
        for target_band in S2_10BAND_NAMES:
            for idx, desc in enumerate(descriptions, start=1):
                if target_band in desc or target_band.replace("0", "") in desc:
                    found_mapping[target_band] = idx
                    break

        if len(found_mapping) == 10:
            return found_mapping

        # Check metadata tags
        tags = src.tags()
        for target_band in S2_10BAND_NAMES:
            for k, v in tags.items():
                if target_band in v.upper():
                    # Check if key implies band number
                    parts = k.upper().split("_")
                    for p in parts:
                        if p.isdigit():
                            found_mapping[target_band] = int(p)

        if len(found_mapping) == 10:
            return found_mapping

        # Standard heuristics based on band count
        if src.count == 10:
            # Assumed direct 10-band stack [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
            return {band: idx for idx, band in enumerate(S2_10BAND_NAMES, start=1)}

        elif src.count == 12:
            # Standard GEE / SEN2NEON 12-band stack:
            # 1:B01, 2:B02, 3:B03, 4:B04, 5:B05, 6:B06, 7:B07, 8:B08, 9:B8A, 10:B09, 11:B11, 12:B12
            mapping_12 = {
                "B02": 2,
                "B03": 3,
                "B04": 4,
                "B05": 5,
                "B06": 6,
                "B07": 7,
                "B08": 8,
                "B8A": 9,
                "B11": 11,
                "B12": 12,
            }
            return mapping_12

        elif src.count == 13:
            # Level-1C / Level-2A full 13-band archive (including B10 cirrus)
            mapping_13 = {
                "B02": 2,
                "B03": 3,
                "B04": 4,
                "B05": 5,
                "B06": 6,
                "B07": 7,
                "B08": 8,
                "B8A": 9,
                "B11": 12,
                "B12": 13,
            }
            return mapping_13

        elif src.count == 4:
            # 4-band Sentinel-2 RGBN stack (B02, B03, B04, B08)
            return {
                "B02": 1,
                "B03": 2,
                "B04": 3,
                "B05": 3,
                "B06": 4,
                "B07": 4,
                "B08": 4,
                "B8A": 4,
                "B11": 3,
                "B12": 3,
            }

        elif src.count == 3:
            # Standard 3-band RGB stack (B04:Red=1, B03:Green=2, B02:Blue=3)
            return {
                "B02": 3,
                "B03": 2,
                "B04": 1,
                "B05": 1,
                "B06": 2,
                "B07": 2,
                "B08": 2,
                "B8A": 2,
                "B11": 1,
                "B12": 1,
            }

        elif src.count == 1:
            # Single band greyscale
            return {band: 1 for band in S2_10BAND_NAMES}

        raise ValueError(
            f"Unable to auto-detect 10 Sentinel-2 bands in file with {src.count} bands. "
            f"Descriptions: {descriptions}. "
            f"Please provide an explicit band_indices dictionary mapping: {S2_10BAND_NAMES}"
        )

    def read(self) -> Sentinel2RasterData:
        """Read raster data, extract the 10 bands, and return Sentinel2RasterData.

        Returns
        -------
        Sentinel2RasterData
            Extracted 10-band tensor, transform, and geospatial properties.
        """
        with rasterio.open(self.file_path) as src:
            band_indices = self._resolve_band_indices(src)

            dst_crs = src.crs
            dst_transform = src.transform
            dst_width = src.width
            dst_height = src.height
            dst_shape = (dst_height, dst_width)
            dst_res = src.res

            band_arrays: list[np.ndarray] = []

            for band_name in S2_10BAND_NAMES:
                band_idx = band_indices[band_name]
                arr = src.read(band_idx)

                # Check if this band needs resolution alignment
                if arr.shape != dst_shape:
                    # Explicit 20m -> 10m resampling
                    resampled = resample_band_to_grid(
                        src_data=arr,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        dst_shape=dst_shape,
                        resampling_method=self.resampling_method,
                    )
                    band_arrays.append(resampled)
                else:
                    band_arrays.append(arr)

            # Stack into (10, H, W) array
            stacked = np.stack(band_arrays, axis=0)

            # Convert to PyTorch tensor (preserves native dtype before normalization)
            tensor = torch.from_numpy(stacked)

            return Sentinel2RasterData(
                tensor=tensor,
                band_names=S2_10BAND_NAMES,
                transform=dst_transform,
                crs=dst_crs,
                bounds=(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top),
                resolution=dst_res,
                width=dst_width,
                height=dst_height,
                metadata=src.meta.copy(),
                nodata=src.nodata,
            )
