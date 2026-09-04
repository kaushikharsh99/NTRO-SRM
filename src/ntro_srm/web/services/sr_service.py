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

            # 3. Generate Web Map Preview Overlays (RGB, CIR, Bicubic)
            self._update_progress(job_id, "processing", "Generating interactive map visual overlays...", 80)
            preview_meta = self._generate_previews(
                lr_path=input_raster,
                sr_path=output_sr_geotiff,
                dest_dir=job_dir,
            )

            total_time = round(time.time() - start_time, 2)
            self._update_progress(job_id, "processing", "Finalizing product metadata...", 95)

            result_payload = {
                "job_id": job_id,
                "is_demo": request.is_demo,
                "scene_id": effective_scene_id,
                "sr_geotiff_path": str(output_sr_geotiff),
                "sr_geotiff_filename": output_sr_geotiff.name,
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

    def _generate_previews(self, lr_path: Path, sr_path: Path, dest_dir: Path) -> dict:
        """Generate georeferenced RGB, CIR, and Bicubic PNG previews for Leaflet."""
        # 1. Read LR
        reader = Sentinel2Reader(lr_path)
        lr_data = reader.read()
        lr_tensor = lr_data.tensor.float()
        if lr_tensor.max() > 2.0:
            lr_tensor = lr_tensor / 10000.0
        lr_np = lr_tensor.numpy()  # (10, H, W)

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

        # 4. Color Channels:
        # True Color RGB: B04 (index 2), B03 (index 1), B02 (index 0)
        # False Color CIR: B08 (index 6), B04 (index 2), B03 (index 1)
        lr_rgb_raw = np.transpose(lr_np[[2, 1, 0]], (1, 2, 0))
        bic_rgb_raw = np.transpose(bicubic_np[[2, 1, 0]], (1, 2, 0))
        sr_rgb_raw = np.transpose(sr_np[[2, 1, 0]], (1, 2, 0))

        lr_cir_raw = np.transpose(lr_np[[6, 2, 1]], (1, 2, 0))
        bic_cir_raw = np.transpose(bicubic_np[[6, 2, 1]], (1, 2, 0))
        sr_cir_raw = np.transpose(sr_np[[6, 2, 1]], (1, 2, 0))

        # Photorealistic True Color RGB with shared radiometric calibration
        lr_rgb_u8 = render_true_color_rgb(lr_rgb_raw, ref_rgb=lr_rgb_raw)
        bic_rgb_u8 = render_true_color_rgb(bic_rgb_raw, ref_rgb=lr_rgb_raw)
        sr_rgb_u8 = render_true_color_rgb(sr_rgb_raw, ref_rgb=lr_rgb_raw)

        # Standard False Color Infrared (CIR)
        lr_cir_u8 = render_false_color_cir(lr_cir_raw, ref_cir=lr_cir_raw)
        bic_cir_u8 = render_false_color_cir(bic_cir_raw, ref_cir=lr_cir_raw)
        sr_cir_u8 = render_false_color_cir(sr_cir_raw, ref_cir=lr_cir_raw)

        # Save preview images
        Image.fromarray(lr_rgb_u8).save(dest_dir / "lr_rgb.png", format="PNG", optimize=True)
        Image.fromarray(bic_rgb_u8).save(dest_dir / "bicubic_rgb.png", format="PNG", optimize=True)
        Image.fromarray(sr_rgb_u8).save(dest_dir / "sr_rgb.png", format="PNG", optimize=True)

        Image.fromarray(lr_cir_u8).save(dest_dir / "lr_cir.png", format="PNG", optimize=True)
        Image.fromarray(bic_cir_u8).save(dest_dir / "bicubic_cir.png", format="PNG", optimize=True)
        Image.fromarray(sr_cir_u8).save(dest_dir / "sr_cir.png", format="PNG", optimize=True)

        return {
            "bounds_wgs84": bounds_wgs84,
            "leaflet_bounds": leaflet_bounds,
            "previews": {
                "lr_rgb": f"/api/sr/jobs/{dest_dir.name}/preview/lr_rgb",
                "sr_rgb": f"/api/sr/jobs/{dest_dir.name}/preview/sr_rgb",
                "bicubic_rgb": f"/api/sr/jobs/{dest_dir.name}/preview/bicubic_rgb",
                "lr_cir": f"/api/sr/jobs/{dest_dir.name}/preview/lr_cir",
                "sr_cir": f"/api/sr/jobs/{dest_dir.name}/preview/sr_cir",
                "bicubic_cir": f"/api/sr/jobs/{dest_dir.name}/preview/bicubic_cir",
            },
        }
