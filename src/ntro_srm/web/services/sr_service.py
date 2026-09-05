"""Super-resolution job processing and preview management service."""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import threading
import time
from typing import Any, Dict, Optional, Tuple
import uuid

import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
import torch
import torch.nn.functional as F

import dotenv

from ntro_srm.data.sentinel2 import Sentinel2Reader
from ntro_srm.inference.sentinel2_pipeline import Sentinel2SRPipeline, Sentinel2SRResult
from ntro_srm.preprocessing.transforms import S2_10BAND_NAMES
from ntro_srm.utils.geotiff import write_sr_geotiff
from ntro_srm.web.services.analysis_service import (
    COMPOSITES,
    AnalysisService,
    available_layer_names,
    layer_catalog,
)
from ntro_srm.web.schemas import (
    AOI,
    InferenceRequest,
    JobProgress,
    SentinelSearchRequest,
    SystemInfoResponse,
)
from ntro_srm.web.services.sentinel_service import (
    CopernicusCDSEProvider,
    EarthSearchProvider,
    LocalDemoProvider,
    SentinelDataProvider,
)


def render_true_color_rgb(
    rgb_arr: np.ndarray,
    ref_rgb: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Render physical surface reflectance to natural, photorealistic True Color RGB.

    Uses shared joint radiometric scaling across R, G, B channels with physical
    reflectance boundaries. Preserves true spectral ratios and avoids neon saturation.
    """
    if ref_rgb is None:
        ref_rgb = rgb_arr

    valid = np.isfinite(ref_rgb)
    if not np.any(valid):
        return np.zeros_like(rgb_arr, dtype=np.uint8)

    # Joint dark point / haze subtraction across visible bands (clamped to prevent shadow clipping)
    floor = float(np.percentile(ref_rgb[valid], 1.0))
    floor = min(max(0.0, floor), 0.03)

    # Physical reflectance ceiling across visible bands (minimum 0.28 to prevent dark scene blowup)
    p99 = float(np.percentile(ref_rgb[valid], 99.0))
    ceiling = max(p99, 0.28)

    denom = ceiling - floor if ceiling > floor else 1.0
    norm = np.clip((rgb_arr - floor) / denom, 0.0, 1.0)

    # Standard sRGB perceptual tone curve (gamma = 1.9)
    gamma_corrected = np.power(norm, 1.0 / 1.9)
    return (gamma_corrected * 255.0).astype(np.uint8)


def render_false_color_cir(
    cir_arr: np.ndarray,
    ref_cir: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Render physical surface reflectance to classic False Color Infrared (CIR).

    Applies calibrated physical reflectance scaling for NIR (B08) and visible (B04, B03)
    to produce rich ruby/crimson vegetation without neon distortion.
    """
    if ref_cir is None:
        ref_cir = cir_arr

    nir_ref = ref_cir[..., 0]
    vis_ref = ref_cir[..., 1:3]

    valid_nir = np.isfinite(nir_ref)
    valid_vis = np.isfinite(vis_ref)

    # NIR scaling (healthy vegetation peaks around 0.60 - 0.85)
    nir_p99 = float(np.percentile(nir_ref[valid_nir], 99.0)) if np.any(valid_nir) else 0.70
    nir_ceil = max(nir_p99, 0.70)
    nir_floor = min(max(0.0, float(np.percentile(nir_ref[valid_nir], 1.0))), 0.05) if np.any(valid_nir) else 0.0

    # Visible (Red & Green) scaling
    vis_p99 = float(np.percentile(vis_ref[valid_vis], 99.0)) if np.any(valid_vis) else 0.28
    vis_ceil = max(vis_p99, 0.28)
    vis_floor = min(max(0.0, float(np.percentile(vis_ref[valid_vis], 1.0))), 0.03) if np.any(valid_vis) else 0.0

    out = np.zeros_like(cir_arr, dtype=np.float32)
    nir_denom = nir_ceil - nir_floor if nir_ceil > nir_floor else 1.0
    vis_denom = vis_ceil - vis_floor if vis_ceil > vis_floor else 1.0

    out[..., 0] = np.clip((cir_arr[..., 0] - nir_floor) / nir_denom, 0.0, 1.0)
    out[..., 1] = np.clip((cir_arr[..., 1] - vis_floor) / vis_denom, 0.0, 1.0)
    out[..., 2] = np.clip((cir_arr[..., 2] - vis_floor) / vis_denom, 0.0, 1.0)

    # Gamma tone curve
    out = np.power(out, 1.0 / 1.9)
    return (out * 255.0).astype(np.uint8)


# Nominal upper reflectance bounds per Sentinel-2 band, used to keep the radiometric
# stretch physically anchored instead of purely percentile-driven. Index positions follow
# the canonical 10-band order [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12].
BAND_CEILING: tuple[float, ...] = (0.28, 0.28, 0.28, 0.35, 0.50, 0.60, 0.70, 0.70, 0.55, 0.45)
BAND_FLOOR_CAP: tuple[float, ...] = (0.03, 0.03, 0.03, 0.04, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05)
DISPLAY_GAMMA: float = 1.9


def render_composite(
    stack: np.ndarray,
    reference: Optional[np.ndarray],
    band_indices: Tuple[int, int, int],
) -> np.ndarray:
    """Render a three-band composite from a 10-band reflectance stack.

    True colour and colour-infrared reuse the calibrated joint-scaling renderers so their
    appearance is unchanged; every other composite (SWIR, agriculture, geology) is stretched
    per channel between a haze-subtracted floor and a physically anchored ceiling, which is
    the correct treatment when the three channels span very different reflectance regimes.

    Parameters
    ----------
    stack : numpy.ndarray
        Reflectance stack of shape ``(10, H, W)`` to be rendered.
    reference : numpy.ndarray, optional
        Stack whose statistics drive the stretch, so that the native, bicubic and
        super-resolved renders of the same scene stay radiometrically comparable.
        Defaults to ``stack``.
    band_indices : tuple[int, int, int]
        Band positions mapped to the red, green and blue display channels.

    Returns
    -------
    numpy.ndarray
        ``(H, W, 3)`` uint8 image.
    """
    ref = stack if reference is None else reference
    r, g, b = band_indices

    if band_indices == (2, 1, 0):
        return render_true_color_rgb(
            np.transpose(stack[[r, g, b]], (1, 2, 0)),
            ref_rgb=np.transpose(ref[[r, g, b]], (1, 2, 0)),
        )
    if band_indices == (6, 2, 1):
        return render_false_color_cir(
            np.transpose(stack[[r, g, b]], (1, 2, 0)),
            ref_cir=np.transpose(ref[[r, g, b]], (1, 2, 0)),
        )

    out = np.zeros((stack.shape[1], stack.shape[2], 3), dtype=np.float32)
    for channel, band in enumerate(band_indices):
        ref_band = ref[band]
        valid = np.isfinite(ref_band)
        if np.any(valid):
            floor = min(max(0.0, float(np.percentile(ref_band[valid], 1.0))), BAND_FLOOR_CAP[band])
            ceiling = max(float(np.percentile(ref_band[valid], 99.0)), BAND_CEILING[band])
        else:
            floor, ceiling = 0.0, BAND_CEILING[band]
        denom = ceiling - floor if ceiling > floor else 1.0
        out[..., channel] = np.clip((stack[band] - floor) / denom, 0.0, 1.0)

    out = np.power(np.nan_to_num(out, nan=0.0), 1.0 / DISPLAY_GAMMA)
    return (out * 255.0).astype(np.uint8)


class SRService:
    """Manages super-resolution inference pipeline and background job lifecycle."""

    def __init__(
        self,
        workspace_root: Path,
        device: Optional[str] = None,
        model_variant: str = "lite",
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.output_base = self.workspace_root / "outputs" / "web_jobs"
        self.output_base.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.workspace_root / "datasets" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Compute device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_variant = model_variant
        self._lock = threading.Lock()
        self._pipelines: dict[str, Sentinel2SRPipeline] = {}
        # Pre-warm default Lite pipeline
        self._init_pipeline()

        # Load environment variables from .env if present
        dotenv.load_dotenv(self.workspace_root / ".env")

        # Data providers
        self.cdse_provider = CopernicusCDSEProvider()
        self.earth_search_provider = EarthSearchProvider()
        sample_path = self.workspace_root / "datasets" / "sample_s2" / "sample_s2_l2a.tif"
        self.demo_provider = LocalDemoProvider(sample_path) if sample_path.is_file() else None

        # Active provider: CDSE if configured, else Earth Search
        self.active_provider: SentinelDataProvider = (
            self.cdse_provider if self.cdse_provider.is_configured() else self.earth_search_provider
        )

        # Scientific quality-assessment and thematic-product service
        self.analysis = AnalysisService(self.workspace_root)

        # Background worker pool (single worker initially for GPU safety)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.jobs: dict[str, JobProgress] = {}

    def _init_pipeline(self) -> None:
        """Preload the default SEN2SR-Lite pipeline."""
        try:
            self.get_pipeline("lite")
        except Exception as e:
            print(f"[SRService Warning] Default pipeline initialization delayed: {e}")

    def get_pipeline(self, variant: str = "lite") -> Sentinel2SRPipeline:
        """Fetch or instantiate the requested model pipeline variant."""
        normalized = "swin2sr" if variant.lower() in ("swin", "swin2sr", "sen2sr") else "lite"
        with self._lock:
            if normalized in self._pipelines:
                return self._pipelines[normalized]

            # VRAM safety check before instantiating Swin2SR
            if normalized == "swin2sr" and self.device == "cuda":
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    free_bytes, _ = torch.cuda.mem_get_info()
                    free_gb = free_bytes / (1024**3)
                    if free_gb < 2.2:
                        raise RuntimeError(
                            f"Insufficient GPU VRAM to run SEN2SR-Swin2SR ({free_gb:.2f} GB free, "
                            f">=2.2 GB required). Please close other GPU processes or use SEN2SR-Lite."
                        )

            checkpoint_folder = "SEN2SR" if normalized == "swin2sr" else "SEN2SRLite"
            checkpoint_dir = self.workspace_root / "checkpoints" / checkpoint_folder

            pipeline = Sentinel2SRPipeline(
                model_variant=normalized,
                device=self.device,
                checkpoint_dir=checkpoint_dir,
            )
            self._pipelines[normalized] = pipeline
            print(f"[SRService] Initialized and cached '{normalized}' pipeline on {self.device}")
            return pipeline

    def get_system_info(self) -> SystemInfoResponse:
        """Query GPU, compute environment status, and model metadata."""
        cuda_avail = torch.cuda.is_available()
        gpu_name = "CPU"
        vram_total = None
        vram_free = None

        if cuda_avail:
            gpu_name = torch.cuda.get_device_name(0)
            mem_info = torch.cuda.mem_get_info()
            vram_free = round(mem_info[0] / (1024**3), 2)
            vram_total = round(mem_info[1] / (1024**3), 2)

        checkpoint_lite_ready = (
            self.workspace_root / "checkpoints" / "SEN2SRLite" / "mlm.json"
        ).is_file()
        checkpoint_swin_ready = (
            self.workspace_root / "checkpoints" / "SEN2SR" / "mlm.json"
        ).is_file()

        models_available = [
            {
                "id": "lite",
                "name": "SEN2SR-Lite",
                "tag": "Default",
                "params": "~0.47M",
                "architecture": "CNN (SPAN)",
                "description": "Fast 10-band baseline model",
                "purpose": "Fast baseline",
                "bands": 10,
                "input_gsd": "10m",
                "output_gsd": "2.5m",
                "upscale_factor": 4,
                "ready": checkpoint_lite_ready,
            },
            {
                "id": "swin2sr",
                "name": "SEN2SR-Swin2SR",
                "tag": "High-Capacity",
                "params": "~12.9M",
                "architecture": "Vision Transformer + MambaSR",
                "description": "Higher-capacity 10-band model",
                "purpose": "Higher-capacity model",
                "bands": 10,
                "input_gsd": "10m",
                "output_gsd": "2.5m",
                "upscale_factor": 4,
                "ready": checkpoint_swin_ready,
            },
        ]

        cdse_ok = self.cdse_provider.is_configured()
        provider_name = "Copernicus Data Space (CDSE)" if cdse_ok else "AWS Earth Search"

        return SystemInfoResponse(
            cuda_available=cuda_avail,
            device_name=gpu_name,
            vram_total_gb=vram_total,
            vram_free_gb=vram_free,
            model_variant="SEN2SR-Lite",
            models_available=models_available,
            upscale_factor=4,
            max_aoi_pixels=512 * 512,
            max_aoi_km2=30.0,
            checkpoint_ready=checkpoint_lite_ready and checkpoint_swin_ready,
            active_provider=provider_name,
            cdse_configured=cdse_ok,
        )

    def create_job(self, request: InferenceRequest) -> str:
        """Enqueue new super-resolution processing job."""
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        now = time.time()

        # Validate AOI limits
        if request.aoi and not request.is_demo:
            est_px = request.aoi.estimated_s2_pixels_10m
            if est_px > 512 * 512:
                raise ValueError(
                    f"Selected AOI has ~{est_px:,} pixels at 10m, exceeding maximum allowed limit (262,144 px / 512x512). "
                    f"Please select a smaller area (~5km x 5km max)."
                )

        initial_progress = JobProgress(
            job_id=job_id,
            status="pending",
            progress_step="Job enqueued in background scheduler",
            progress_percent=5,
            created_at=now,
            updated_at=now,
        )

        with self._lock:
            self.jobs[job_id] = initial_progress

        # Dispatch async task
        self.executor.submit(self._run_job, job_id, request)
        return job_id

    def get_job_progress(self, job_id: str) -> Optional[JobProgress]:
        """Fetch current status of a job."""
        with self._lock:
            return self.jobs.get(job_id)

    def _update_progress(
        self, job_id: str, status: str, step: str, percent: int, error: Optional[str] = None, result: Optional[dict] = None
    ) -> None:
        with self._lock:
            if job_id in self.jobs:
                self.jobs[job_id].status = status
                self.jobs[job_id].progress_step = step
                self.jobs[job_id].progress_percent = percent
                self.jobs[job_id].updated_at = time.time()
                if error:
                    self.jobs[job_id].error_message = error
                if result:
                    self.jobs[job_id].result = result

    def _run_job(self, job_id: str, request: InferenceRequest) -> None:
        """Execute complete pipeline asynchronously."""
        job_dir = self.output_base / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        start_time = time.time()

        try:
            self._update_progress(job_id, "processing", "Validating AOI and parameters...", 10)

            # Resolve requested model variant
            requested_model = getattr(request, "model", "lite") or "lite"
            model_variant = "swin2sr" if str(requested_model).lower() in ("swin", "swin2sr", "sen2sr") else "lite"
            model_display_name = "SEN2SR-Swin2SR" if model_variant == "swin2sr" else "SEN2SR-Lite"
            model_tag = "swin" if model_variant == "swin2sr" else "lite"

            self._update_progress(job_id, "processing", f"Loading {model_display_name} weights...", 15)
            pipeline = self.get_pipeline(model_variant)

            # 1. Acquire Input Data
            input_raster: Path
            effective_scene_id: str = "DEMO_MLBS_20180825_S2L2A"

            if request.custom_upload_path:
                self._update_progress(job_id, "processing", "Loading uploaded GeoTIFF raster...", 25)
                uploaded_file = Path(request.custom_upload_path)
                if not uploaded_file.is_file():
                    raise FileNotFoundError(f"Uploaded GeoTIFF not found: {uploaded_file}")
                input_raster = uploaded_file
                effective_scene_id = f"UPLOAD_{uploaded_file.stem}"

            elif request.is_demo:
                self._update_progress(job_id, "processing", "Loading local demo Sentinel-2 scene...", 25)
                if not self.demo_provider:
                    raise FileNotFoundError("Local demo dataset not found at datasets/sample_s2/sample_s2_l2a.tif")
                input_raster = self.demo_provider.fetch_aoi_bands("demo", request.aoi, job_dir)
                effective_scene_id = "DEMO_MLBS_20180825_S2L2A"

            else:
                if not request.aoi:
                    raise ValueError("No Area of Interest (AOI) or image provided.")

                target_scene_id = request.scene_id

                # If no scene_id is specified or set to "auto", automatically discover the best scene
                if not target_scene_id or target_scene_id == "auto":
                    self._update_progress(job_id, "processing", "Auto-discovering best cloud-free Sentinel-2 scene...", 20)
                    now_dt = datetime.now(timezone.utc)
                    search_req = SentinelSearchRequest(
                        aoi=request.aoi,
                        date_from=(now_dt - timedelta(days=365)).strftime("%Y-%m-%d"),
                        date_to=now_dt.strftime("%Y-%m-%d"),
                        max_cloud_cover=35.0,
                        limit=10,
                    )
                    search_res = None
                    try:
                        search_res = self.active_provider.search(search_req)
                    except Exception as err:
                        print(f"[SRService] Primary search error: {err}")

                    if (not search_res or not search_res.scenes) and (self.active_provider != self.earth_search_provider):
                        try:
                            search_res = self.earth_search_provider.search(search_req)
                        except Exception as err2:
                            print(f"[SRService] Fallback search error: {err2}")

                    if search_res and search_res.scenes:
                        best_scene = min(search_res.scenes, key=lambda s: s.cloud_cover)
                        target_scene_id = best_scene.id
                        print(f"[SRService] Auto-selected scene {target_scene_id} ({best_scene.cloud_cover}% cloud)")
                    else:
                        # Fallback: if AOI is near demo coordinates (-80.59 to -80.55, 37.41 to 37.44) or demo available
                        if self.demo_provider:
                            print("[SRService] Using local reference demo raster as fallback")
                            target_scene_id = "DEMO_MLBS_20180825_S2L2A"
                            input_raster = self.demo_provider.fetch_aoi_bands("demo", request.aoi, job_dir)
                        else:
                            raise ValueError(
                                "No cloud-free Sentinel-2 scenes found in this area over the past year. "
                                "Please select another area or load a preset location."
                            )

                effective_scene_id = target_scene_id

                # If input_raster not yet loaded by demo fallback, stream from online provider
                if "input_raster" not in locals() or input_raster is None:
                    use_cdse = self.cdse_provider.is_configured() and ("AWS" not in target_scene_id)
                    provider_label = "Copernicus CDSE" if use_cdse else "AWS Earth Search"
                    self._update_progress(job_id, "processing", f"Streaming 10 Sentinel-2 bands from {provider_label}...", 30)

                    def stac_progress(msg: str, pct: int):
                        mapped_pct = int(30 + (pct / 100.0) * 20)
                        self._update_progress(job_id, "processing", msg, mapped_pct)

                    try:
                        if use_cdse:
                            input_raster = self.cdse_provider.fetch_aoi_bands(
                                scene_id=target_scene_id,
                                aoi=request.aoi,
                                output_dir=self.cache_dir,
                                progress_callback=stac_progress,
                            )
                        else:
                            input_raster = self.earth_search_provider.fetch_aoi_bands(
                                scene_id=target_scene_id,
                                aoi=request.aoi,
                                output_dir=self.cache_dir,
                                progress_callback=stac_progress,
                            )
                    except Exception as fetch_err:
                        print(f"[SRService] Primary fetch failed: {fetch_err}, trying fallback...")
                        if use_cdse:
                            try:
                                input_raster = self.earth_search_provider.fetch_aoi_bands(
                                    scene_id=target_scene_id,
                                    aoi=request.aoi,
                                    output_dir=self.cache_dir,
                                    progress_callback=stac_progress,
                                )
                            except Exception:
                                raise fetch_err
                        elif self.demo_provider:
                            input_raster = self.demo_provider.fetch_aoi_bands("demo", request.aoi, job_dir)
                        else:
                            raise fetch_err

            # 2. Run Super-Resolution Pipeline
            self._update_progress(job_id, "processing", f"Running {model_display_name} neural super-resolution...", 55)
            output_sr_geotiff = job_dir / f"{job_id}_sen2sr_{model_tag}_2.5m.tif"

            sr_result = pipeline.predict(
                input_path=input_raster,
                output_path=output_sr_geotiff,
                overlap=request.overlap,
            )

            # Ensure canonical alias {job_id}_sr_2.5m.tif exists for download convenience
            compat_sr_geotiff = job_dir / f"{job_id}_sr_2.5m.tif"
            if output_sr_geotiff != compat_sr_geotiff:
                try:
                    import os, shutil
                    if compat_sr_geotiff.exists():
                        compat_sr_geotiff.unlink()
                    try:
                        os.link(output_sr_geotiff, compat_sr_geotiff)
                    except OSError:
                        shutil.copyfile(output_sr_geotiff, compat_sr_geotiff)
                except Exception as alias_err:
                    print(f"[SRService] Alias creation note: {alias_err}")

            # 3. Persist the preprocessed 10 m observation as a canonical 10-band GeoTIFF.
            # Everything downstream (previews, lazy layers, the pixel probe, the QA report)
            # reads this file, so band order and radiometry are identical everywhere.
            lr_geotiff = job_dir / f"{job_id}_native_10m.tif"
            try:
                write_sr_geotiff(
                    output_path=lr_geotiff,
                    tensor=sr_result.lr_tensor,
                    transform=sr_result.input_transform,
                    crs=sr_result.crs,
                    model_name="Observed (no super-resolution)",
                    input_gsd="10m",
                    output_gsd="10m",
                    upscale_factor=1,
                )
            except Exception as lr_err:
                print(f"[SRService] Native 10 m export note: {lr_err}")
                lr_geotiff = Path(input_raster)

            # 4. Generate Web Map Preview Overlays (RGB, CIR, Bicubic)
            self._update_progress(job_id, "processing", "Generating interactive map visual overlays...", 78)
            preview_meta = self._generate_previews(
                lr_tensor=sr_result.lr_tensor,
                sr_path=output_sr_geotiff,
                dest_dir=job_dir,
            )

            # 5. Scientific quality assessment, uncertainty and thematic products.
            analysis_payload: dict[str, Any] = {}
            confidence_geotiff = job_dir / f"{job_id}_confidence_2.5m.tif"
            if request.run_analysis:
                members = self._resolve_ensemble_size(request, model_variant)
                try:
                    def _predict(tile: torch.Tensor) -> torch.Tensor:
                        return pipeline.model.predict(
                            tile,
                            auto_normalize=False,
                            clamp_output=True,
                            overlap=request.overlap,
                        )

                    artifacts = self.analysis.run(
                        job_id=job_id,
                        job_dir=job_dir,
                        lr_tensor=sr_result.lr_tensor,
                        sr_tensor=sr_result.sr_tensor,
                        predict_fn=_predict,
                        scene_meta={
                            "scene_id": effective_scene_id,
                            "model": model_display_name,
                            "model_variant": model_variant,
                            "device": self.device,
                            "crs": str(sr_result.crs),
                            "native_gsd": "10.0m",
                            "output_gsd": "2.50m",
                            "input_shape": list(sr_result.input_shape),
                            "output_shape": list(sr_result.output_shape),
                        },
                        run_wald=request.run_wald_validation,
                        uncertainty_members=members,
                        crs=sr_result.crs,
                        sr_transform=sr_result.output_transform,
                        progress_callback=lambda msg, pct: self._update_progress(
                            job_id, "processing", msg, pct
                        ),
                    )
                    analysis_payload = artifacts.payload
                except Exception as an_err:
                    print(f"[SRService] Quality assessment failed: {an_err}")
                    analysis_payload = {"warnings": [f"Quality assessment failed: {an_err}"]}

            total_time = round(time.time() - start_time, 2)
            self._update_progress(job_id, "processing", "Finalizing product metadata...", 96)

            result_payload = {
                "job_id": job_id,
                "is_demo": request.is_demo,
                "scene_id": effective_scene_id,
                "sr_geotiff_path": str(output_sr_geotiff),
                "sr_geotiff_filename": output_sr_geotiff.name,
                "lr_geotiff_path": str(lr_geotiff),
                "confidence_geotiff_path": (
                    str(confidence_geotiff) if confidence_geotiff.is_file() else None
                ),
                "input_geotiff_path": str(input_raster),
                "processing_time_sec": total_time,
                "device_used": self.device,
                "model": model_display_name,
                "model_variant": model_variant,
                "peak_vram_mb": round(sr_result.peak_gpu_memory_mb, 1) if sr_result.peak_gpu_memory_mb else None,
                "inference_time_ms": round(sr_result.inference_time_ms, 1),
                "native_gsd": "10.0m",
                "output_gsd": "2.50m",
                "upscale_factor": 4,
                "bands_count": 10,
                "input_shape": list(sr_result.input_shape),
                "output_shape": list(sr_result.output_shape),
                "crs": str(sr_result.crs),
                "bounds_wgs84": preview_meta["bounds_wgs84"],
                "leaflet_bounds": preview_meta["leaflet_bounds"],
                "previews": preview_meta["previews"],
                "analysis": analysis_payload,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }

            self._update_progress(
                job_id,
                status="completed",
                step="Super-resolution complete",
                percent=100,
                result=result_payload,
            )
            print(f"[SRService] Job {job_id} completed successfully in {total_time}s")

        except Exception as e:
            err_msg = str(e)
            print(f"[SRService Error] Job {job_id} failed: {err_msg}")
            self._update_progress(
                job_id,
                status="failed",
                step="Processing failed",
                percent=100,
                error=err_msg,
            )

    def _resolve_ensemble_size(self, request: InferenceRequest, model_variant: str) -> int:
        """Decide how many test-time-augmentation members the uncertainty pass may use.

        The ensemble costs one extra full-resolution forward pass per member. That is a
        couple of seconds on SEN2SR-Lite but many minutes on the Vision Transformer, so the
        default is opt-in for the heavy model and enabled for the fast one.

        Parameters
        ----------
        request : InferenceRequest
            Job request; ``uncertainty_members`` of ``None`` selects the automatic policy.
        model_variant : str
            Resolved model identifier (``"lite"`` or ``"swin2sr"``).

        Returns
        -------
        int
            Ensemble size; ``0`` selects the free novelty-only estimator.
        """
        requested = request.uncertainty_members
        if requested is None:
            return 0 if model_variant == "swin2sr" else 4
        return max(0, min(8, int(requested)))

    def _generate_previews(self, lr_tensor: torch.Tensor, sr_path: Path, dest_dir: Path) -> dict:
        """Generate georeferenced RGB, CIR, and Bicubic PNG previews for Leaflet."""
        # 1. Preprocessed 10-band observation in [0, 1]
        lr_np = lr_tensor.detach().cpu().float().numpy()  # (10, H, W)

        # 2. Read SR
        with rasterio.open(sr_path) as src:
            sr_np = src.read().astype(np.float32)  # (10, 4H, 4W)
            src_crs = src.crs
            src_bounds = src.bounds

        # 3. Compute Bicubic
        t_lr = torch.from_numpy(lr_np).unsqueeze(0)
        t_bicubic = F.interpolate(
            t_lr,
            size=(sr_np.shape[1], sr_np.shape[2]),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        bicubic_np = torch.clamp(t_bicubic.squeeze(0), 0.0, 1.0).numpy()

        # Reproject bounding box to WGS84 for Leaflet
        try:
            if src_crs is None:
                min_lon, min_lat, max_lon, max_lat = float(src_bounds.left), float(src_bounds.bottom), float(src_bounds.right), float(src_bounds.top)
            else:
                min_lon, min_lat, max_lon, max_lat = transform_bounds(
                    src_crs, "EPSG:4326", *src_bounds
                )
        except Exception:
            min_lon, min_lat, max_lon, max_lat = -80.5868, 37.4138, -80.5577, 37.4370
        leaflet_bounds = [[min_lat, min_lon], [max_lat, max_lon]]
        bounds_wgs84 = {
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat,
        }

        # 4. Render the composites that the map opens with. Every remaining composite,
        # spectral index and analysis layer is materialised lazily on first request.
        stacks = {"lr": lr_np, "bicubic": bicubic_np, "sr": sr_np}
        for key in ("rgb", "cir"):
            band_idx = COMPOSITES[key]["bands"]
            for source, stack in stacks.items():
                image = render_composite(stack, lr_np, band_idx)
                Image.fromarray(image).save(
                    dest_dir / f"{source}_{key}.png", format="PNG", optimize=True
                )

        job_id = dest_dir.name
        previews = {
            name: f"/api/sr/jobs/{job_id}/preview/{name}" for name in available_layer_names()
        }

        return {
            "bounds_wgs84": bounds_wgs84,
            "leaflet_bounds": leaflet_bounds,
            "previews": previews,
        }

    # ------------------------------------------------------------------
    # Job introspection helpers used by the REST layer
    # ------------------------------------------------------------------
    def list_jobs(self, limit: int = 25) -> list[dict[str, Any]]:
        """Summarise recent jobs, newest first, for the session history panel.

        Parameters
        ----------
        limit : int, default=25
            Maximum number of entries to return.

        Returns
        -------
        list[dict]
            One compact record per job, safe to render without fetching the full result.
        """
        with self._lock:
            jobs = sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)[:limit]
            records = []
            for job in jobs:
                result = job.result or {}
                analysis = result.get("analysis") or {}
                summary = analysis.get("summary") or {}
                records.append(
                    {
                        "job_id": job.job_id,
                        "status": job.status,
                        "created_at": job.created_at,
                        "updated_at": job.updated_at,
                        "progress_percent": job.progress_percent,
                        "scene_id": result.get("scene_id"),
                        "model": result.get("model"),
                        "output_shape": result.get("output_shape"),
                        "processing_time_sec": result.get("processing_time_sec"),
                        "leaflet_bounds": result.get("leaflet_bounds"),
                        "verdict": summary.get("verdict"),
                        "error_message": job.error_message,
                    }
                )
            return records

    def resolve_preview(self, job_id: str, layer_name: str) -> Path:
        """Return the on-disk PNG for a preview layer, rendering it if necessary.

        Parameters
        ----------
        job_id : str
            Job identifier.
        layer_name : str
            Layer identifier, e.g. ``sr_ndvi`` or ``bicubic_swir``.

        Returns
        -------
        Path
            Path to the cached PNG.

        Raises
        ------
        ValueError
            If the layer name is unknown.
        FileNotFoundError
            If the job has not produced the rasters the layer needs.
        """
        job = self.get_job_progress(job_id)
        if not job or job.status != "completed" or not job.result:
            raise FileNotFoundError(f"Job '{job_id}' has no completed products yet.")

        result = job.result
        job_dir = self.output_base / job_id
        sr_path = Path(result["sr_geotiff_path"])
        lr_path = Path(result.get("lr_geotiff_path") or result["input_geotiff_path"])
        confidence = result.get("confidence_geotiff_path")
        return self.analysis.render_layer(
            job_dir=job_dir,
            layer_name=layer_name,
            lr_path=lr_path,
            sr_path=sr_path,
            confidence_path=Path(confidence) if confidence else None,
        )

    def probe_pixel(self, job_id: str, lat: float, lon: float) -> dict[str, Any]:
        """Sample the observed and super-resolved spectra at a geographic coordinate.

        Parameters
        ----------
        job_id : str
            Completed job to probe.
        lat, lon : float
            WGS84 coordinate.

        Returns
        -------
        dict
            Payload for the web UI's spectral inspector.

        Raises
        ------
        FileNotFoundError
            If the job has not completed.
        ValueError
            If the coordinate lies outside the patch.
        """
        job = self.get_job_progress(job_id)
        if not job or job.status != "completed" or not job.result:
            raise FileNotFoundError(f"Job '{job_id}' has no completed products yet.")

        result = job.result
        confidence = result.get("confidence_geotiff_path")
        uncertainty_meta = ((result.get("analysis") or {}).get("uncertainty")) or {}
        return self.analysis.probe_pixel(
            lat=lat,
            lon=lon,
            lr_path=Path(result.get("lr_geotiff_path") or result["input_geotiff_path"]),
            sr_path=Path(result["sr_geotiff_path"]),
            confidence_path=Path(confidence) if confidence else None,
            uncertainty_meta=uncertainty_meta,
        )

    @staticmethod
    def layer_catalog() -> dict[str, Any]:
        """Describe every renderable layer, index and metric for the UI layer picker."""
        return layer_catalog()
