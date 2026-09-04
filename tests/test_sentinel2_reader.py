"""Unit tests for Sentinel-2 data reader and resolution alignment."""

from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
import torch

from ntro_srm.data.sentinel2 import (
    S2_10M_BANDS,
    S2_20M_BANDS,
    Sentinel2Reader,
    resample_band_to_grid,
)
from ntro_srm.preprocessing.sentinel2 import normalize_sentinel2_l2a
from ntro_srm.preprocessing.transforms import S2_10BAND_NAMES


@pytest.fixture
def sample_s2_path() -> Path:
    """Path to real downloaded Sentinel-2 sample tile."""
    p = Path(__file__).resolve().parents[1] / "datasets" / "sample_s2" / "sample_s2_l2a.tif"
    if not p.is_file():
        pytest.skip(f"Sample Sentinel-2 file not found at {p}")
    return p


@pytest.fixture
def synthetic_multires_raster(tmp_path: Path) -> Path:
    """Create a temporary 12-band GeoTIFF with known metadata and band values."""
    file_path = tmp_path / "synthetic_12band.tif"
    width, height = 128, 128
    crs = CRS.from_epsg(32617)
    transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0)

    # 12 bands, each band populated with its band index * 1000
    data = np.zeros((12, height, width), dtype=np.uint16)
    for b in range(12):
        data[b, :, :] = (b + 1) * 1000

    # Add a nodata pixel
    data[:, 0, 0] = 65535

    with rasterio.open(
        file_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=12,
        dtype=np.uint16,
        crs=crs,
        transform=transform,
        nodata=65535,
    ) as dst:
        dst.write(data)

    return file_path


class TestSentinel2Reader:
    """Test suite for Sentinel2Reader."""

    def test_read_real_s2_sample(self, sample_s2_path: Path):
        reader = Sentinel2Reader(sample_s2_path)
        data = reader.read()

        assert data.tensor.ndim == 3
        assert data.tensor.shape[0] == 10
        assert data.tensor.shape[1] == 256
        assert data.tensor.shape[2] == 256
        assert data.band_names == S2_10BAND_NAMES
        assert data.crs == CRS.from_epsg(32617)
        assert data.resolution == (10.0, 10.0)
        assert data.width == 256
        assert data.height == 256

    def test_band_extraction_and_ordering(self, synthetic_multires_raster: Path):
        reader = Sentinel2Reader(synthetic_multires_raster)
        data = reader.read()

        assert data.tensor.shape == (10, 128, 128)
        assert data.band_names == S2_10BAND_NAMES

        # Expected mappings from 12-band stack:
        # B02: band 2 -> value 2000
        # B03: band 3 -> value 3000
        # B04: band 4 -> value 4000
        # B05: band 5 -> value 5000
        # B06: band 6 -> value 6000
        # B07: band 7 -> value 7000
        # B08: band 8 -> value 8000
        # B8A: band 9 -> value 9000
        # B11: band 11 -> value 11000
        # B12: band 12 -> value 12000
        expected_vals = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 11000, 12000]
        for i, val in enumerate(expected_vals):
            # Check a non-nodata pixel
            assert data.tensor[i, 10, 10].item() == val

    def test_explicit_resampling_20m_to_10m(self):
        # 20m source: 64x64 with 20m pixel spacing
        src_data = np.ones((64, 64), dtype=np.float32) * 500.0
        src_transform = Affine(20.0, 0.0, 500000.0, 0.0, -20.0, 4000000.0)
        src_crs = CRS.from_epsg(32617)

        # 10m target: 128x128 with 10m pixel spacing
        dst_transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0)
        dst_crs = CRS.from_epsg(32617)
        dst_shape = (128, 128)

        resampled = resample_band_to_grid(
            src_data=src_data,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_shape=dst_shape,
            resampling_method=Resampling.bilinear,
        )

        assert resampled.shape == (128, 128)
        assert np.isclose(resampled.mean(), 500.0, atol=1e-2)

    def test_normalization_and_nodata_handling(self, synthetic_multires_raster: Path):
        reader = Sentinel2Reader(synthetic_multires_raster)
        data = reader.read()

        norm = normalize_sentinel2_l2a(
            data.tensor,
            mode="s2_10000",
            nodata_value=data.nodata,
        )

        assert norm.shape == (10, 128, 128)
        assert norm.dtype == torch.float32
        # Nodata at (0, 0) should be replaced with 0.0
        assert norm[:, 0, 0].sum().item() == 0.0
        # Check normalized range
        assert norm[:, 10, 10].min().item() >= 0.0

    def test_invalid_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            Sentinel2Reader(tmp_path / "non_existent.tif")
