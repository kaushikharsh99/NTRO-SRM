"""REST API endpoints for NTRO-SRM Super-Resolution Web Application."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Optional
import uuid

from fastapi import APIRouter, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from ntro_srm.web.schemas import (
    AOI,
    HealthResponse,
    InferenceRequest,
    JobListResponse,
    JobProgress,
    PixelProbeResponse,
    SentinelSearchRequest,
    SentinelSearchResponse,
    SystemInfoResponse,
)
from ntro_srm.web.services.sentinel_service import EarthSearchProvider, LocalDemoProvider
from ntro_srm.web.services.sr_service import SRService

router = APIRouter(prefix="/api")


def get_sr_service(request: Request) -> SRService:
    """Dependency to retrieve singleton SRService instance."""
    return request.app.state.sr_service


@router.get("/system-info", response_model=SystemInfoResponse)
def get_system_info(request: Request) -> SystemInfoResponse:
    """Query local GPU device, VRAM, and model checkpoint status."""
    service = get_sr_service(request)
    return service.get_system_info()


@router.get("/health", response_model=HealthResponse)
def get_health(request: Request) -> HealthResponse:
    """Liveness and readiness probe reporting device, checkpoints, and catalog status."""
    from ntro_srm import __version__ as ntro_version

    service = get_sr_service(request)
    info = service.get_system_info()
    active = sum(1 for j in service.jobs.values() if j.status in ("pending", "processing"))
    return HealthResponse(
        status="ok",
        version=ntro_version,
        device=service.device,
        cuda_available=info.cuda_available,
        checkpoints_ready=info.checkpoint_ready,
        catalog_provider=info.active_provider,
        active_jobs=active,
    )


@router.get("/layers")
def get_layer_catalog(request: Request) -> dict:
    """Describe every renderable map layer, spectral index, and quality metric.

    The web UI builds its layer picker, index legends, and metric chips from this
    payload, so new indices become available in the interface without a front-end change.
    """
    service = get_sr_service(request)
    return service.layer_catalog()


@router.post("/sentinel/search", response_model=SentinelSearchResponse)
def search_sentinel_scenes(
    search_req: SentinelSearchRequest,
    request: Request,
) -> SentinelSearchResponse:
    """Search public STAC catalog (AWS Earth Search) for Sentinel-2 L2A acquisitions."""
    # Check AOI area limit
    if search_req.aoi.area_km2 > 100.0:
        raise HTTPException(
            status_code=400,
            detail=f"AOI area is {search_req.aoi.area_km2:.1f} km², which exceeds the maximum search limit (100 km²)."
        )

    service = get_sr_service(request)
    try:
        response = service.active_provider.search(search_req)
        if response.total == 0 and service.active_provider != service.earth_search_provider:
            # Fallback to Earth Search if CDSE returns 0
            response = service.earth_search_provider.search(search_req)
        return response
    except Exception as e:
        # Fallback to Earth Search if primary provider errors
        try:
            return service.earth_search_provider.search(search_req)
        except Exception:
            raise HTTPException(status_code=502, detail=f"Sentinel-2 catalog search failed: {e}")


@router.get("/demo/info")
def get_demo_info(request: Request) -> dict:
    """Return local pre-downloaded Sentinel-2 demo scene metadata."""
    service = get_sr_service(request)
    if not service.demo_provider:
        raise HTTPException(status_code=404, detail="Demo scene not found in local workspace.")
    return service.demo_provider.get_demo_info()


@router.post("/sr/upload")
async def upload_geotiff_patch(
    request: Request,
    file: UploadFile = File(...),
    model: str = Query(default="lite", description="Model variant: 'lite' or 'swin2sr'"),
    run_analysis: bool = Query(default=True, description="Run quality assessment and thematic products"),
    run_wald_validation: bool = Query(default=True, description="Run Wald's synthesis validation"),
    uncertainty_members: Optional[int] = Query(
        default=None, ge=0, le=8, description="Test-time-augmentation ensemble size (None = automatic)"
    ),
) -> dict:
    """Upload a custom Sentinel-2 GeoTIFF patch and enqueue super-resolution inference."""
    service = get_sr_service(request)

    if not file.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(status_code=400, detail="Only GeoTIFF files (.tif, .tiff) are supported.")

    uploads_dir = service.output_base / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    file_id = uuid.uuid4().hex[:8]
    safe_filename = Path(file.filename).name.replace(" ", "_")
    target_path = uploads_dir / f"{file_id}_{safe_filename}"

    with open(target_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Inspect GeoTIFF metadata with rasterio
    try:
        import rasterio
        with rasterio.open(target_path) as src:
            w = src.width
            h = src.height
            count = src.count
            crs_str = str(src.crs)
            if w * h > 600 * 600:
                target_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=400,
                    detail=f"Uploaded image has {w}×{h} ({w*h:,} px), exceeding the 512×512 tile limit. Please crop a smaller patch."
                )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted GeoTIFF file: {e}")

    # Launch inference job
    req = InferenceRequest(
        custom_upload_path=str(target_path),
        is_demo=False,
        overlap=32,
        model=model,
        run_analysis=run_analysis,
        run_wald_validation=run_wald_validation,
        uncertainty_members=uncertainty_members,
    )
    job_id = service.create_job(req)
    return {
        "job_id": job_id,
        "filename": file.filename,
        "dimensions": [w, h],
        "bands": count,
        "crs": crs_str,
        "status": "pending",
        "message": f"Uploaded {file.filename} ({w}×{h}, {count} bands). Super-resolution job started.",
    }


@router.post("/sr/process")
def launch_sr_inference(
    req: InferenceRequest,
    request: Request,
) -> dict:
    """Launch asynchronous 4x super-resolution job."""
    service = get_sr_service(request)

    if not req.is_demo and req.aoi is None and not req.custom_upload_path:
        raise HTTPException(status_code=400, detail="AOI is required for real Sentinel-2 processing.")
    
    if not req.is_demo and not req.scene_id and not req.custom_upload_path:
        req.scene_id = "auto"

    try:
        job_id = service.create_job(req)
        return {"job_id": job_id, "status": "pending", "message": "Super-resolution job started"}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to enqueue SR job: {e}")


@router.get("/sr/jobs/{job_id}", response_model=JobProgress)
def get_job_status(job_id: str, request: Request) -> JobProgress:
    """Poll progress status and metadata of an active or completed SR job."""
    service = get_sr_service(request)
    job = service.get_job_progress(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


@router.get("/sr/jobs", response_model=JobListResponse)
def list_jobs(
    request: Request,
    limit: int = Query(default=25, ge=1, le=100, description="Maximum jobs to return"),
) -> JobListResponse:
    """List recent super-resolution jobs, newest first, for the session history panel."""
    service = get_sr_service(request)
    records = service.list_jobs(limit=limit)
    return JobListResponse(jobs=records, total=len(records))


@router.api_route("/sr/jobs/{job_id}/preview/{layer_name}", methods=["GET", "HEAD"])
def get_preview_layer(
    job_id: str,
    layer_name: str,
    request: Request,
) -> FileResponse:
    """Serve a georeferenced PNG overlay for interactive Leaflet display.

    Composites and spectral-index layers beyond the two rendered eagerly at job completion
    are materialised on first request and cached in the job directory, so the map can offer
    every band combination and thematic product without inflating job runtime.
    """
    service = get_sr_service(request)

    # Fast path: an already-rendered layer needs no job-state lookup.
    cached = service.output_base / job_id / f"{layer_name}.png"
    if cached.is_file():
        return FileResponse(cached, media_type="image/png")

    try:
        img_path = service.resolve_preview(job_id, layer_name)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to render layer '{layer_name}': {e}")

    return FileResponse(img_path, media_type="image/png")


@router.get("/sr/jobs/{job_id}/analysis")
def get_job_analysis(job_id: str, request: Request) -> dict:
    """Return the quality-assessment block for a completed job.

    Contains the Wald synthesis scores against the bicubic baseline, the radiometric
    consistency verdict, the uncertainty summary, per-index statistics, and links to the
    downloadable QA report.
    """
    service = get_sr_service(request)
    job = service.get_job_progress(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job.status != "completed" or not job.result:
        raise HTTPException(status_code=400, detail=f"Job '{job_id}' is not completed yet.")
    return job.result.get("analysis") or {}


@router.get("/sr/jobs/{job_id}/pixel", response_model=PixelProbeResponse)
def probe_pixel(
    job_id: str,
    request: Request,
    lat: float = Query(..., ge=-90.0, le=90.0, description="WGS84 latitude"),
    lon: float = Query(..., ge=-180.0, le=180.0, description="WGS84 longitude"),
) -> PixelProbeResponse:
    """Sample the 10-band spectrum, spectral indices, and confidence at one coordinate.

    This is what makes the super-resolved product inspectable rather than merely viewable:
    the caller gets the observed 10 m spectrum and the reconstructed 2.5 m spectrum side by
    side, so spectral fidelity can be judged pixel by pixel.
    """
    service = get_sr_service(request)
    try:
        payload = service.probe_pixel(job_id, lat, lon)
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pixel probe failed: {e}")
    return PixelProbeResponse(**payload)


@router.get("/sr/jobs/{job_id}/download/{file_type}")
def download_result_file(
    job_id: str,
    file_type: str,
    request: Request,
) -> FileResponse:
    """Download the scientific 10-band GeoTIFF or PNG preview."""
    service = get_sr_service(request)
    job = service.get_job_progress(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job.status != "completed" or not job.result:
        raise HTTPException(status_code=400, detail=f"Job '{job_id}' is not completed yet.")

    job_dir = service.output_base / job_id

    if file_type in ["geotiff", "sr_geotiff", "tif"]:
        tif_path = Path(job.result["sr_geotiff_path"])
        if not tif_path.is_file():
            raise HTTPException(status_code=404, detail="GeoTIFF file missing on disk.")
        filename = f"NTRO_SRM_2.5m_{job.result['scene_id']}_{job_id}.tif"
        return FileResponse(
            tif_path,
            media_type="image/tiff",
            filename=filename,
        )

    elif file_type == "rgb":
        rgb_path = job_dir / "sr_rgb.png"
        if not rgb_path.is_file():
            raise HTTPException(status_code=404, detail="RGB preview file missing.")
        return FileResponse(
            rgb_path,
            media_type="image/png",
            filename=f"NTRO_SRM_RGB_{job_id}.png",
        )

    elif file_type == "cir":
        cir_path = job_dir / "sr_cir.png"
        if not cir_path.is_file():
            raise HTTPException(status_code=404, detail="CIR preview file missing.")
        return FileResponse(
            cir_path,
            media_type="image/png",
            filename=f"NTRO_SRM_CIR_{job_id}.png",
        )

    elif file_type in ("native", "lr", "native_geotiff"):
        lr_path = Path(job.result.get("lr_geotiff_path") or job.result["input_geotiff_path"])
        if not lr_path.is_file():
            raise HTTPException(status_code=404, detail="Native 10 m GeoTIFF missing on disk.")
        return FileResponse(
            lr_path,
            media_type="image/tiff",
            filename=f"NTRO_SRM_native_10m_{job.result['scene_id']}_{job_id}.tif",
        )

    elif file_type == "confidence":
        conf = job.result.get("confidence_geotiff_path")
        if not conf or not Path(conf).is_file():
            raise HTTPException(
                status_code=404,
                detail="No confidence raster was produced for this job (quality assessment disabled).",
            )
        return FileResponse(
            Path(conf),
            media_type="image/tiff",
            filename=f"NTRO_SRM_confidence_2.5m_{job_id}.tif",
        )

    elif file_type in ("report", "report-json"):
        report_path = job_dir / "quality_report.json"
        if not report_path.is_file():
            raise HTTPException(status_code=404, detail="Quality report not available for this job.")
        return FileResponse(
            report_path,
            media_type="application/json",
            filename=f"NTRO_SRM_quality_report_{job_id}.json",
        )

    elif file_type in ("report-md", "report_markdown"):
        report_path = job_dir / "quality_report.md"
        if not report_path.is_file():
            raise HTTPException(status_code=404, detail="Quality report not available for this job.")
        return FileResponse(
            report_path,
            media_type="text/markdown",
            filename=f"NTRO_SRM_quality_report_{job_id}.md",
        )

    else:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid file_type '{file_type}'. Supported: 'geotiff', 'native', 'confidence', "
                f"'rgb', 'cir', 'report', 'report-md'."
            ),
        )
