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
    InferenceRequest,
    JobProgress,
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


@router.api_route("/sr/jobs/{job_id}/preview/{layer_name}", methods=["GET", "HEAD"])
def get_preview_layer(
    job_id: str,
    layer_name: str,
    request: Request,
) -> FileResponse:
    """Serve georeferenced PNG overlay for interactive Leaflet display."""
    service = get_sr_service(request)
    valid_layers = [
        "lr_rgb", "sr_rgb", "bicubic_rgb",
        "lr_cir", "sr_cir", "bicubic_cir"
    ]
    if layer_name not in valid_layers:
        raise HTTPException(status_code=400, detail=f"Invalid layer '{layer_name}'. Expected one of {valid_layers}")

    img_path = service.output_base / job_id / f"{layer_name}.png"
    if not img_path.is_file():
        raise HTTPException(status_code=404, detail=f"Preview layer '{layer_name}' not ready or job incomplete.")

    return FileResponse(img_path, media_type="image/png")


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

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file_type '{file_type}'. Supported: 'geotiff', 'rgb', 'cir'."
        )
