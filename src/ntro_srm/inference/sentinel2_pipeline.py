"""End-to-end Sentinel-2 super-resolution inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional, Union

from rasterio.crs import CRS
from rasterio.transform import Affine
import torch

from ntro_srm.data.sentinel2 import Sentinel2RasterData, Sentinel2Reader
from ntro_srm.models.sen2sr import SEN2SRModel
from ntro_srm.preprocessing.sentinel2 import NormalizationMode, normalize_sentinel2_l2a
from ntro_srm.preprocessing.transforms import S2_10BAND_NAMES
from ntro_srm.utils.geotiff import compute_sr_transform, write_sr_geotiff


@dataclass
class Sentinel2SRResult:
    """Structured container returned by the Sentinel2SRPipeline.

    Attributes
    ----------
    sr_tensor : torch.Tensor
        Super-resolved 10-band tensor of shape (10, 4*H, 4*W).
    lr_tensor : torch.Tensor
        Input 10-band normalized reflectance tensor of shape (10, H, W).
    input_path : Path
        Source file path of the Sentinel-2 image.
    output_path : Path or None
        Destination file path if saved to GeoTIFF.
    input_shape : tuple[int, int, int]
        Spatial dimensions (C, H, W) of the input 10m grid.
    output_shape : tuple[int, int, int]
        Spatial dimensions (C, 4*H, 4*W) of the output 2.5m grid.
    input_transform : Affine
        Low-resolution geotransform.
    output_transform : Affine
        Super-resolved 2.5m geotransform.
    crs : CRS
        Coordinate Reference System.
    band_names : list[str]
        Band names: [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12].
    inference_time_ms : float
        Elapsed model execution time in milliseconds.
    peak_gpu_memory_mb : float or None
        Peak GPU VRAM allocated during inference if running on CUDA.
    """

    sr_tensor: torch.Tensor
    lr_tensor: torch.Tensor
    input_path: Path
    output_path: Optional[Path]
    input_shape: tuple[int, int, int]
    output_shape: tuple[int, int, int]
    input_transform: Affine
    output_transform: Affine
    crs: CRS
    band_names: list[str]
    inference_time_ms: float
    peak_gpu_memory_mb: Optional[float] = None


class Sentinel2SRPipeline:
    """End-to-end pipeline orchestrating Sentinel-2 ingestion, preprocessing, SR, and GeoTIFF export."""

    def __init__(
        self,
        model_variant: str = "lite",
        device: Optional[Union[str, torch.device]] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize the pipeline with the specified model variant.

        Parameters
        ----------
        model_variant : str, default="lite"
            Super-resolution model architecture ("lite" is default).
        device : str or torch.device, optional
            Compute device ("cuda" or "cpu").
        checkpoint_dir : str or Path, optional
            Path to pretrained model checkpoint directory.
        """
        self.model = SEN2SRModel(
            model_variant=model_variant,
            device=device,
            checkpoint_dir=checkpoint_dir,
            auto_download=True,
        )
        self.device = self.model.device

    def predict(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        normalization_mode: NormalizationMode = "auto",
        overlap: int = 32,
    ) -> Sentinel2SRResult:
        """Execute super-resolution on a real Sentinel-2 raster.

        Parameters
        ----------
        input_path : str or Path
            Path to the input Sentinel-2 GeoTIFF.
        output_path : str or Path, optional
            If provided, writes the 10-band 2.5m super-resolved GeoTIFF to this path.
        normalization_mode : {"s2_10000", "already_normalized", "auto"}, default="auto"
            Normalization mode applied to raw reflectances.
        overlap : int, default=32
            Tiling overlap in pixels when processing images larger than 128x128.

        Returns
        -------
        Sentinel2SRResult
            Structured result containing tensors, transforms, metadata, and performance metrics.
        """
        input_path = Path(input_path)

        # 1. Read input raster and extract 10 Sentinel-2 bands
        reader = Sentinel2Reader(input_path)
        raster_data: Sentinel2RasterData = reader.read()

        # 2. Preprocessing & reflectance normalization
        normalized_tensor = normalize_sentinel2_l2a(
            raster_data.tensor,
            mode=normalization_mode,
            nodata_value=raster_data.nodata,
        )

        # 3. Model execution
        is_cuda = self.device.type == "cuda"
        if is_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        sr_tensor = self.model.predict(
            normalized_tensor,
            auto_normalize=False,  # Already normalized via explicit preprocessing step
            clamp_output=True,
            overlap=overlap,
        )

        if is_cuda:
            torch.cuda.synchronize()
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        else:
            peak_memory_mb = None

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 4. Compute output 2.5m geotransform
        output_transform = compute_sr_transform(raster_data.transform, scale_factor=4.0)

        # 5. Optionally write to GeoTIFF
        saved_path: Optional[Path] = None
        if output_path is not None:
            model_label = "SEN2SR-Swin2SR" if self.model.model_variant == "swin2sr" else "SEN2SR-Lite"
            saved_path = write_sr_geotiff(
                output_path=output_path,
                tensor=sr_tensor,
                transform=output_transform,
                crs=raster_data.crs,
                model_name=model_label,
                input_gsd="10m",
                output_gsd="2.5m",
                upscale_factor=4,
            )

        return Sentinel2SRResult(
            sr_tensor=sr_tensor,
            lr_tensor=normalized_tensor,
            input_path=input_path,
            output_path=saved_path,
            input_shape=tuple(normalized_tensor.shape),
            output_shape=tuple(sr_tensor.shape),
            input_transform=raster_data.transform,
            output_transform=output_transform,
            crs=raster_data.crs,
            band_names=S2_10BAND_NAMES,
            inference_time_ms=elapsed_ms,
            peak_gpu_memory_mb=peak_memory_mb,
        )
