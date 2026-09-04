"""Integration tests for the end-to-end Sentinel-2 super-resolution pipeline."""

from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from ntro_srm.inference.sentinel2_pipeline import Sentinel2SRPipeline, Sentinel2SRResult
from ntro_srm.preprocessing.transforms import S2_10BAND_NAMES


@pytest.fixture(scope="module")
def pipeline() -> Sentinel2SRPipeline:
    """Fixture providing initialized pipeline on CPU (for portable testing)."""
    return Sentinel2SRPipeline(model_variant="lite", device="cpu")


@pytest.fixture
def input_raster_path(tmp_path: Path) -> Path:
    """Provide real sample raster if available, or create a valid 128x128 GeoTIFF."""
    real_sample = Path(__file__).resolve().parents[1] / "datasets" / "sample_s2" / "sample_s2_l2a.tif"
    if real_sample.is_file():
        return real_sample

    # Fallback to creating a synthetic 10-band 128x128 GeoTIFF
    synth_path = tmp_path / "test_s2_input.tif"
    width, height = 128, 128
    crs = CRS.from_epsg(32617)
    transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0)
    data = (np.random.rand(10, height, width) * 3000 + 500).astype(np.uint16)

    with rasterio.open(
        synth_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=10,
        dtype=np.uint16,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)

    return synth_path


class TestSentinel2PipelineIntegration:
    """End-to-end integration test suite."""

    def test_end_to_end_pipeline_and_geotiff(
        self,
        pipeline: Sentinel2SRPipeline,
        input_raster_path: Path,
        tmp_path: Path,
    ):
        output_path = tmp_path / "test_sr_output.tif"

        # 1. Run pipeline prediction
        result: Sentinel2SRResult = pipeline.predict(
            input_path=input_raster_path,
            output_path=output_path,
            normalization_mode="auto",
        )

        # 2. Verify result object
        assert result.output_path == output_path
        assert output_path.is_file(), "Output GeoTIFF file must be created"
        assert result.band_names == S2_10BAND_NAMES
        assert result.inference_time_ms > 0

        # Verify 4x upscaling in memory tensor
        _, in_h, in_w = result.input_shape
        _, out_h, out_w = result.output_shape
        assert out_h == in_h * 4
        assert out_w == in_w * 4

        # 3. Inspect saved GeoTIFF on disk using rasterio
        with rasterio.open(output_path) as dst:
            # Verify band count and dimensions
            assert dst.count == 10
            assert dst.width == in_w * 4
            assert dst.height == in_h * 4
            assert dst.dtypes[0] == "float32"

            # Verify CRS is preserved
            assert dst.crs == result.crs

            # Verify resolution is approximately 2.5m (1/4 of input 10m)
            res_x, res_y = dst.res
            assert np.isclose(res_x, 2.5, atol=1e-2)
            assert np.isclose(res_y, 2.5, atol=1e-2)

            # Verify geographic bounds are preserved
            with rasterio.open(input_raster_path) as src:
                assert np.isclose(dst.bounds.left, src.bounds.left, atol=1e-3)
                assert np.isclose(dst.bounds.right, src.bounds.right, atol=1e-3)
                assert np.isclose(dst.bounds.top, src.bounds.top, atol=1e-3)
                assert np.isclose(dst.bounds.bottom, src.bounds.bottom, atol=1e-3)

            # Verify metadata tags
            tags = dst.tags()
            assert tags["MODEL"] == "SEN2SR-Lite"
            assert tags["INPUT_GSD"] == "10m"
            assert tags["OUTPUT_GSD"] == "2.5m"
            assert tags["UPSCALE_FACTOR"] == "4"
            assert tags["BAND_1"] == "B02"
            assert tags["BAND_10"] == "B12"

            # Read back all data and check numerical validity
            data = dst.read()
            assert bool(np.isfinite(data).all()), "Output GeoTIFF must not contain NaNs or Infs"
            assert bool((data >= 0.0).all()), "Output reflectance must be non-negative"
            assert float(data.mean()) > 0.0, "Output must contain valid signal"
