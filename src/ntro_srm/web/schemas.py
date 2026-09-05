"""Data validation schemas for NTRO-SRM Web API."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class AOI(BaseModel):
    """Geographic Area of Interest in WGS84 coordinates."""

    min_lon: float = Field(..., ge=-180.0, le=180.0, description="Minimum longitude (West)")
    min_lat: float = Field(..., ge=-90.0, le=90.0, description="Minimum latitude (South)")
    max_lon: float = Field(..., ge=-180.0, le=180.0, description="Maximum longitude (East)")
    max_lat: float = Field(..., ge=-90.0, le=90.0, description="Maximum latitude (North)")

    @field_validator("max_lon")
    @classmethod
    def validate_longitude_order(cls, v: float, info: Any) -> float:
        min_lon = info.data.get("min_lon")
        if min_lon is not None and v <= min_lon:
            raise ValueError(f"max_lon ({v}) must be greater than min_lon ({min_lon})")
        return v

    @field_validator("max_lat")
    @classmethod
    def validate_latitude_order(cls, v: float, info: Any) -> float:
        min_lat = info.data.get("min_lat")
        if min_lat is not None and v <= min_lat:
            raise ValueError(f"max_lat ({v}) must be greater than min_lat ({min_lat})")
        return v

    @property
    def center_lat(self) -> float:
        return (self.min_lat + self.max_lat) / 2.0

    @property
    def center_lon(self) -> float:
        return (self.min_lon + self.max_lon) / 2.0

    @property
    def width_km(self) -> float:
        """Approximate width in kilometers."""
        lat_rad = math.radians(self.center_lat)
        delta_lon = self.max_lon - self.min_lon
        return delta_lon * 111.32 * math.cos(lat_rad)

    @property
    def height_km(self) -> float:
        """Approximate height in kilometers."""
        delta_lat = self.max_lat - self.min_lat
        return delta_lat * 110.574

    @property
    def area_km2(self) -> float:
        return max(0.0, self.width_km * self.height_km)

    @property
    def estimated_s2_pixels_10m(self) -> int:
        """Estimated number of 10m Sentinel-2 pixels."""
        w_px = max(1, int(round((self.width_km * 1000.0) / 10.0)))
        h_px = max(1, int(round((self.height_km * 1000.0) / 10.0)))
        return w_px * h_px

    @property
    def estimated_s2_shape(self) -> tuple[int, int]:
        """Estimated (height_px, width_px) at 10m GSD."""
        w_px = max(1, int(round((self.width_km * 1000.0) / 10.0)))
        h_px = max(1, int(round((self.height_km * 1000.0) / 10.0)))
        return (h_px, w_px)

    def to_bbox_list(self) -> list[float]:
        return [self.min_lon, self.min_lat, self.max_lon, self.max_lat]

    def to_leaflet_bounds(self) -> list[list[float]]:
        """Return [[south, west], [north, east]] for Leaflet."""
        return [[self.min_lat, self.min_lon], [self.max_lat, self.max_lon]]


class SentinelSearchRequest(BaseModel):
    """Request payload for searching Sentinel-2 L2A STAC catalogs."""

    aoi: AOI
    date_from: str = Field(..., description="Start date (YYYY-MM-DD)")
    date_to: str = Field(..., description="End date (YYYY-MM-DD)")
    max_cloud_cover: float = Field(default=20.0, ge=0.0, le=100.0, description="Max cloud coverage %")
    limit: int = Field(default=10, ge=1, le=50, description="Max items to return")


class SentinelSceneItem(BaseModel):
    """Metadata summary of a discovered Sentinel-2 acquisition."""

    id: str
    datetime: str
    cloud_cover: float
    platform: str = "Sentinel-2"
    thumbnail_url: Optional[str] = None
    provider: str = "AWS Earth Search"
    tile_id: Optional[str] = None
    bbox: Optional[list[float]] = None
    assets: Optional[dict[str, str]] = None


class SentinelSearchResponse(BaseModel):
    """List of discovered Sentinel-2 scenes."""

    scenes: list[SentinelSceneItem]
    total: int
    query_aoi: AOI


class InferenceRequest(BaseModel):
    """Request to launch 4x super-resolution processing."""

    aoi: Optional[AOI] = None
    scene_id: Optional[str] = None
    custom_upload_path: Optional[str] = None
    is_demo: bool = Field(default=False, description="Whether to run on local demo scene")
    overlap: int = Field(default=32, ge=8, le=64, description="Sliding window overlap in pixels")
    clamp_output: bool = Field(default=True, description="Clamp reflectance to [0.0, 1.0]")
    model: str = Field(default="lite", description="Model variant: 'lite' or 'swin2sr'")
    run_analysis: bool = Field(
        default=True,
        description="Run quality assessment, uncertainty estimation and thematic products",
    )
    run_wald_validation: bool = Field(
        default=True,
        description=(
            "Run Wald's synthesis protocol (10m -> 40m -> 10m) for quantitative accuracy "
            "assessment. Costs one additional forward pass at 1/16 of the area."
        ),
    )
    uncertainty_members: Optional[int] = Field(
        default=None,
        ge=0,
        le=8,
        description=(
            "Test-time-augmentation ensemble size for uncertainty estimation. Each member is a "
            "full-resolution forward pass. None selects the automatic policy (4 for SEN2SR-Lite, "
            "0 for the Vision Transformer); 0 uses the free novelty-only estimator."
        ),
    )


class JobProgress(BaseModel):
    """Status update for an asynchronous processing job."""

    job_id: str
    status: str = Field(..., description="'pending', 'processing', 'completed', 'failed'")
    progress_step: str
    progress_percent: int = Field(ge=0, le=100)
    error_message: Optional[str] = None
    created_at: float
    updated_at: float
    result: Optional[dict[str, Any]] = None


class SystemInfoResponse(BaseModel):
    """Host computing environment and model availability."""

    cuda_available: bool
    device_name: str
    vram_total_gb: Optional[float] = None
    vram_free_gb: Optional[float] = None
    model_variant: str = "SEN2SR-Lite"
    models_available: list[dict[str, Any]] = Field(default_factory=list)
    upscale_factor: int = 4
    max_aoi_pixels: int = 512 * 512
    max_aoi_km2: float = 30.0
    checkpoint_ready: bool = True
    active_provider: str = "Copernicus Data Space (CDSE)"
    cdse_configured: bool = False


class JobSummary(BaseModel):
    """Compact record of a processing job for the session history panel."""

    job_id: str
    status: str
    created_at: float
    updated_at: float
    progress_percent: int = 0
    scene_id: Optional[str] = None
    model: Optional[str] = None
    output_shape: Optional[list[int]] = None
    processing_time_sec: Optional[float] = None
    leaflet_bounds: Optional[list[list[float]]] = None
    verdict: Optional[str] = None
    error_message: Optional[str] = None


class JobListResponse(BaseModel):
    """Recent processing jobs, newest first."""

    jobs: list[JobSummary] = Field(default_factory=list)
    total: int = 0


class BandSample(BaseModel):
    """Reflectance spectrum sampled at one pixel of a raster."""

    reflectance: list[Optional[float]] = Field(default_factory=list)
    row: int = 0
    col: int = 0


class IndexSample(BaseModel):
    """Spectral index evaluated at a probed pixel, before and after super-resolution."""

    key: str
    name: str
    lr: Optional[float] = None
    sr: Optional[float] = None
    unit: str = ""
    class_label: Optional[str] = None
    class_color: Optional[str] = None


class PixelUncertainty(BaseModel):
    """Per-pixel reconstruction reliability read-out."""

    confidence: Optional[float] = None
    std: Optional[float] = None
    novelty: Optional[float] = None
    risk: Optional[str] = None


class PixelProbeResponse(BaseModel):
    """Full spectral, thematic and reliability read-out at a clicked map coordinate."""

    lat: float
    lon: float
    easting: float
    northing: float
    crs: str
    row: int
    col: int
    band_names: list[str] = Field(default_factory=list)
    wavelengths_nm: list[float] = Field(default_factory=list)
    lr: BandSample
    sr: BandSample
    bicubic: Optional[BandSample] = None
    indices: list[IndexSample] = Field(default_factory=list)
    uncertainty: Optional[PixelUncertainty] = None


class HealthResponse(BaseModel):
    """Liveness and readiness probe for the web application."""

    status: str = "ok"
    version: str
    device: str
    cuda_available: bool
    checkpoints_ready: bool
    catalog_provider: str
    active_jobs: int = 0
