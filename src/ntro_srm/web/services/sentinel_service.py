"""Sentinel-2 data provider service.

Implements STAC catalog searching and windowed Cloud Optimized GeoTIFF (COG)
acquisition for sub-scene Areas of Interest (AOI), as well as a local demo provider.
"""

from __future__ import annotations

import abc
import hashlib
import os
from pathlib import Path
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds
import requests

from ntro_srm.data.sentinel2 import resample_band_to_grid
from ntro_srm.preprocessing.transforms import S2_10BAND_NAMES
from ntro_srm.web.schemas import AOI, SentinelSceneItem, SentinelSearchRequest, SentinelSearchResponse

# Asset key aliases used across AWS Earth Search and Planetary Computer
STAC_BAND_ALIASES: dict[str, list[str]] = {
    "B02": ["blue", "B02", "b02"],
    "B03": ["green", "B03", "b03"],
    "B04": ["red", "B04", "b04"],
    "B05": ["rededge1", "B05", "b05"],
    "B06": ["rededge2", "B06", "b06"],
    "B07": ["rededge3", "B07", "b07"],
    "B08": ["nir", "B08", "b08"],
    "B8A": ["nir08", "B8A", "b8a"],
    "B11": ["swir16", "B11", "b11"],
    "B12": ["swir22", "B12", "b12"],
}


class SentinelDataProvider(abc.ABC):
    """Abstract interface for Sentinel-2 data sources."""

    @abc.abstractmethod
    def search(self, request: SentinelSearchRequest) -> SentinelSearchResponse:
        """Search catalog for matching Sentinel-2 scenes."""
        pass

    @abc.abstractmethod
    def fetch_aoi_bands(
        self,
        scene_id: str,
        aoi: AOI,
        output_dir: Path,
        progress_callback: Optional[callable] = None,
    ) -> Path:
        """Acquire 10-band aligned GeoTIFF for specified AOI."""
        pass


class EarthSearchProvider(SentinelDataProvider):
    """Sentinel-2 L2A provider using AWS Earth Search STAC API."""

    STAC_SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"
    COLLECTION_ID = "sentinel-2-l2a"

    def __init__(self, timeout: int = 15) -> None:
        self.timeout = timeout

    def search(self, request: SentinelSearchRequest) -> SentinelSearchResponse:
        """Query Element84 Earth Search STAC API for Sentinel-2 L2A items."""
        aoi = request.aoi
        payload = {
            "collections": [self.COLLECTION_ID],
            "bbox": aoi.to_bbox_list(),
            "datetime": f"{request.date_from}T00:00:00Z/{request.date_to}T23:59:59Z",
            "query": {"eo:cloud_cover": {"lte": float(request.max_cloud_cover)}},
            "limit": int(request.limit),
            "sortby": [{"field": "properties.datetime", "direction": "desc"}],
        }

        try:
            resp = requests.post(
                self.STAC_SEARCH_URL,
                json=payload,
                timeout=self.timeout,
                headers={"User-Agent": "NTRO-SRM-Web/0.1.0"},
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"STAC search request failed: {e}") from e

        features = data.get("features", [])
        scenes: list[SentinelSceneItem] = []

        for feat in features:
            props = feat.get("properties", {})
            assets = feat.get("assets", {})
            cloud_pct = float(props.get("eo:cloud_cover", 0.0))

            # Extract thumbnail URL if available
            thumb = assets.get("thumbnail", {}).get("href")
            if not thumb:
                thumb = assets.get("visual", {}).get("href")

            # Collect available band hrefs
            band_hrefs: dict[str, str] = {}
            for b_target, aliases in STAC_BAND_ALIASES.items():
                for alias in aliases:
                    if alias in assets and "href" in assets[alias]:
                        band_hrefs[b_target] = assets[alias]["href"]
                        break

            # Only include if essential bands are present
            if len(band_hrefs) >= 4:
                item = SentinelSceneItem(
                    id=feat.get("id", "unknown"),
                    datetime=props.get("datetime", "unknown"),
                    cloud_cover=round(cloud_pct, 1),
                    platform=props.get("platform", "Sentinel-2"),
                    thumbnail_url=thumb,
                    provider="AWS Earth Search",
                    tile_id=props.get("grid:code") or props.get("mgrs:utm_zone"),
                    bbox=feat.get("bbox"),
                    assets=band_hrefs,
                )
                scenes.append(item)

        return SentinelSearchResponse(
            scenes=scenes,
            total=len(scenes),
            query_aoi=aoi,
        )

    def fetch_aoi_bands(
        self,
        scene_id: str,
        aoi: AOI,
        output_dir: Path,
        progress_callback: Optional[callable] = None,
    ) -> Path:
        """Stream windowed 10-band stack for the AOI directly from AWS COGs.

        Avoids downloading complete ~500MB tiles by performing HTTP range requests
        via GDAL/rasterio against Cloud Optimized GeoTIFFs.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Lookup item metadata to get asset URLs
        item_url = f"https://earth-search.aws.element84.com/v1/collections/{self.COLLECTION_ID}/items/{scene_id}"
        try:
            resp = requests.get(item_url, timeout=self.timeout)
            resp.raise_for_status()
            item_data = resp.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to fetch metadata for scene {scene_id}: {e}") from e

        assets = item_data.get("assets", {})
        band_urls: dict[str, str] = {}
        for b_target, aliases in STAC_BAND_ALIASES.items():
            for alias in aliases:
                if alias in assets and "href" in assets[alias]:
                    band_urls[b_target] = assets[alias]["href"]
                    break

        missing = [b for b in S2_10BAND_NAMES if b not in band_urls]
        if missing:
            raise ValueError(f"Scene {scene_id} missing required spectral bands: {missing}")

        # Unique cache filename based on scene and AOI
        aoi_hash = hashlib.md5(f"{scene_id}_{aoi.to_bbox_list()}".encode()).hexdigest()[:10]
        out_geotiff = output_dir / f"s2_{scene_id}_{aoi_hash}_10band.tif"
        if out_geotiff.is_file():
            if progress_callback:
                progress_callback("Found cached 10-band AOI raster", 100)
            return out_geotiff

        # 2. Open reference band (B02 10m) to establish target grid and CRS
        b02_url = band_urls["B02"]
        if progress_callback:
            progress_callback(f"Opening reference band B02 from COG stream...", 15)

        with rasterio.open(b02_url) as ref_src:
            ref_crs = ref_src.crs
            ref_transform = ref_src.transform

            # Transform AOI WGS84 bounding box to native raster CRS (typically UTM)
            min_x, min_y, max_x, max_y = transform_bounds(
                "EPSG:4326", ref_crs, aoi.min_lon, aoi.min_lat, aoi.max_lon, aoi.max_lat
            )

            # Compute window on reference 10m grid
            ref_window = from_bounds(min_x, min_y, max_x, max_y, ref_transform)
            # Round window to integer pixels
            ref_window = ref_window.round_offsets().round_lengths()

            # Clamp window to raster bounds
            ref_window = ref_window.intersection(Window(0, 0, ref_src.width, ref_src.height))

            win_w = int(ref_window.width)
            win_h = int(ref_window.height)
            if win_w < 8 or win_h < 8:
                raise ValueError(f"Selected AOI is too small ({win_w}x{win_h} px at 10m). Minimum is 8x8.")

            target_transform = rasterio.windows.transform(ref_window, ref_transform)
            target_shape = (win_h, win_w)

        # 3. Read and align each of the 10 bands
        stacked_bands = np.zeros((10, win_h, win_w), dtype=np.uint16)

        for i, b_name in enumerate(S2_10BAND_NAMES):
            if progress_callback:
                pct = int(20 + (i / 10.0) * 60)
                progress_callback(f"Streaming band {b_name} ({i+1}/10)...", pct)

            b_url = band_urls[b_name]
            with rasterio.open(b_url) as b_src:
                # If band is native 10m (B02, B03, B04, B08)
                if b_name in ["B02", "B03", "B04", "B08"]:
                    b_win = from_bounds(min_x, min_y, max_x, max_y, b_src.transform).round_offsets().round_lengths()
                    b_win = b_win.intersection(Window(0, 0, b_src.width, b_src.height))
                    band_data = b_src.read(1, window=b_win)

                    # Ensure exact shape match
                    if band_data.shape != target_shape:
                        band_data = resample_band_to_grid(
                            src_data=band_data,
                            src_transform=rasterio.windows.transform(b_win, b_src.transform),
                            src_crs=b_src.crs,
                            dst_transform=target_transform,
                            dst_crs=ref_crs,
                            dst_shape=target_shape,
                            resampling_method=Resampling.bilinear,
                        )
                else:
                    # Native 20m band: read 20m window and explicitly resample to 10m target grid
                    b20_win = from_bounds(min_x, min_y, max_x, max_y, b_src.transform).round_offsets().round_lengths()
                    b20_win = b20_win.intersection(Window(0, 0, b_src.width, b_src.height))
                    raw_20m = b_src.read(1, window=b20_win)

                    b20_transform = rasterio.windows.transform(b20_win, b_src.transform)
                    band_data = resample_band_to_grid(
                        src_data=raw_20m,
                        src_transform=b20_transform,
                        src_crs=b_src.crs,
                        dst_transform=target_transform,
                        dst_crs=ref_crs,
                        dst_shape=target_shape,
                        resampling_method=Resampling.bilinear,
                    )

                stacked_bands[i] = band_data.astype(np.uint16)

        # 4. Write output 10-band GeoTIFF
        if progress_callback:
            progress_callback("Writing aligned 10-band GeoTIFF...", 85)

        with rasterio.open(
            out_geotiff,
            "w",
            driver="GTiff",
            height=win_h,
            width=win_w,
            count=10,
            dtype=np.uint16,
            crs=ref_crs,
            transform=target_transform,
            compress="deflate",
        ) as dst:
            dst.write(stacked_bands)
            dst.update_tags(
                SOURCE_SCENE=scene_id,
                SOURCE_PLATFORM="Sentinel-2",
                BANDS=",".join(S2_10BAND_NAMES),
            )
            for idx, name in enumerate(S2_10BAND_NAMES, start=1):
                dst.update_tags(idx, BAND_NAME=name)

        if progress_callback:
            progress_callback("10-band AOI acquisition complete", 100)

        return out_geotiff


class LocalDemoProvider(SentinelDataProvider):
    """Local offline provider using existing sample dataset."""

    def __init__(self, sample_raster_path: Path) -> None:
        self.sample_path = Path(sample_raster_path).resolve()
        if not self.sample_path.is_file():
            raise FileNotFoundError(f"Local demo raster not found: {self.sample_path}")

        # Cache WGS84 bounds
        with rasterio.open(self.sample_path) as src:
            wgs = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
            self.demo_aoi = AOI(
                min_lon=round(wgs[0], 5),
                min_lat=round(wgs[1], 5),
                max_lon=round(wgs[2], 5),
                max_lat=round(wgs[3], 5),
            )
            self.crs_str = str(src.crs)

    def get_demo_info(self) -> dict:
        """Return demo scene metadata."""
        return {
            "scene_id": "DEMO_MLBS_20180825_S2L2A",
            "aoi": self.demo_aoi.model_dump(),
            "datetime": "2018-08-25T16:00:00Z",
            "cloud_cover": 0.0,
            "tile_id": "17SQC (Mountain Lake, VA)",
            "raster_path": str(self.sample_path),
            "description": "Pre-downloaded Sentinel-2 L2A scene (Mountain Lake Biological Station, VA).",
        }

    def search(self, request: SentinelSearchRequest) -> SentinelSearchResponse:
        """Return the local demo scene as a search match."""
        item = SentinelSceneItem(
            id="DEMO_MLBS_20180825_S2L2A",
            datetime="2018-08-25T16:00:00Z",
            cloud_cover=0.0,
            platform="Sentinel-2A",
            thumbnail_url="/api/demo/thumbnail",
            provider="Local Demo Archive",
            tile_id="17SQC",
            bbox=self.demo_aoi.to_bbox_list(),
        )
        return SentinelSearchResponse(
            scenes=[item],
            total=1,
            query_aoi=request.aoi,
        )

    def fetch_aoi_bands(
        self,
        scene_id: str,
        aoi: AOI,
        output_dir: Path,
        progress_callback: Optional[callable] = None,
    ) -> Path:
        """Return local sample file directly."""
        if progress_callback:
            progress_callback("Using local sample Sentinel-2 raster", 100)
        return self.sample_path


class CopernicusCDSEProvider(SentinelDataProvider):
    """Official Copernicus Data Space Ecosystem (CDSE) / Sentinel Hub provider.

    Uses OAuth 2.0 client credentials authentication to query the CDSE Catalog API
    and streams 10-band Sentinel-2 L2A Float32 GeoTIFFs directly via the Process API.
    """

    TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    CATALOG_URL = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
    PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        timeout: int = 25,
    ) -> None:
        self.client_id = (
            client_id
            or os.environ.get("CDSE_CLIENT_ID")
            or os.environ.get("COPERNICUS_CLIENT_ID")
        )
        self.client_secret = (
            client_secret
            or os.environ.get("CDSE_CLIENT_SECRET")
            or os.environ.get("COPERNICUS_CLIENT_SECRET")
        )
        self.timeout = timeout
        self._token: Optional[str] = None
        self._token_expiry: float = 0.0
        self._lock = threading.Lock()

    def is_configured(self) -> bool:
        """Check whether valid CDSE client credentials exist."""
        return bool(self.client_id and self.client_secret)

    def _get_token(self) -> str:
        """Acquire or return cached OAuth bearer token."""
        with self._lock:
            if self._token and time.time() < self._token_expiry:
                return self._token

            if not self.is_configured():
                raise ValueError(
                    "Copernicus CDSE credentials not configured. Please set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET."
                )

            try:
                resp = requests.post(
                    self.TOKEN_URL,
                    data={
                        "grant_type": "client_credentials",
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                self._token = data["access_token"]
                expires_in = data.get("expires_in", 3600)
                self._token_expiry = time.time() + expires_in - 60
                return self._token
            except Exception as e:
                raise RuntimeError(f"Copernicus CDSE OAuth authentication failed: {e}") from e

    def search(self, request: SentinelSearchRequest) -> SentinelSearchResponse:
        """Query CDSE STAC/Catalog API for matching Sentinel-2 L2A acquisitions."""
        token = self._get_token()
        aoi = request.aoi

        payload = {
            "collections": ["sentinel-2-l2a"],
            "datetime": f"{request.date_from}T00:00:00Z/{request.date_to}T23:59:59Z",
            "bbox": aoi.to_bbox_list(),
            "limit": int(request.limit),
        }

        try:
            resp = requests.post(
                self.CATALOG_URL,
                json=payload,
                headers={"Authorization": f"Bearer {token}"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"CDSE catalog search failed: {e}") from e

        features = data.get("features", [])
        scenes: list[SentinelSceneItem] = []

        for feat in features:
            props = feat.get("properties", {})
            cloud_pct = float(props.get("eo:cloud_cover", 0.0))
            if cloud_pct > float(request.max_cloud_cover):
                continue

            assets = feat.get("assets", {})
            thumb = assets.get("thumbnail", {}).get("href")

            # Extract tile ID if available
            feat_id = feat.get("id", "unknown")
            tile_id = None
            if "_" in feat_id:
                parts = feat_id.split("_")
                for p in parts:
                    if p.startswith("T") and len(p) == 6:
                        tile_id = p
                        break

            item = SentinelSceneItem(
                id=feat_id,
                datetime=props.get("datetime", "unknown"),
                cloud_cover=round(cloud_pct, 1),
                platform=props.get("platform", "Sentinel-2"),
                thumbnail_url=thumb,
                provider="Copernicus Data Space (CDSE)",
                tile_id=tile_id,
                bbox=feat.get("bbox"),
            )
            scenes.append(item)

        return SentinelSearchResponse(
            scenes=scenes,
            total=len(scenes),
            query_aoi=aoi,
        )

    def fetch_aoi_bands(
        self,
        scene_id: str,
        aoi: AOI,
        output_dir: Path,
        progress_callback: Optional[callable] = None,
    ) -> Path:
        """Stream a 10-band Float32 GeoTIFF for the AOI directly via the CDSE Process API."""
        token = self._get_token()
        output_dir.mkdir(parents=True, exist_ok=True)

        aoi_hash = hashlib.md5(f"{scene_id}_{aoi.to_bbox_list()}".encode()).hexdigest()[:10]
        out_geotiff = output_dir / f"cdse_{aoi_hash}_10band.tif"
        if out_geotiff.is_file():
            if progress_callback:
                progress_callback("Found cached Copernicus 10-band raster", 100)
            return out_geotiff

        if progress_callback:
            progress_callback("Connecting to Copernicus Data Space Ecosystem...", 20)

        # Calculate pixel dimensions matching ~10m GSD
        w_px = max(8, int(round((aoi.width_km * 1000.0) / 10.0)))
        h_px = max(8, int(round((aoi.height_km * 1000.0) / 10.0)))

        # Extract date from scene ID if available (e.g. S2A_MSIL2A_20230831T155829_...)
        date_str = ""
        parts = scene_id.split("_")
        for p in parts:
            if len(p) >= 8 and p[:8].isdigit():
                date_str = f"{p[:4]}-{p[4:6]}-{p[6:8]}"
                break

        if date_str:
            time_filter = {
                "from": f"{date_str}T00:00:00Z",
                "to": f"{date_str}T23:59:59Z",
            }
        else:
            time_filter = {"from": "2020-01-01T00:00:00Z", "to": "2026-12-31T23:59:59Z"}

        evalscript = """//VERSION=3
function setup() {
  return {
    input: [{
      bands: ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'],
      units: 'REFLECTANCE'
    }],
    output: {
      bands: 10,
      sampleType: 'FLOAT32'
    }
  };
}

function evaluatePixel(sample) {
  return [
    sample.B02, sample.B03, sample.B04, sample.B05,
    sample.B06, sample.B07, sample.B08, sample.B8A,
    sample.B11, sample.B12
  ];
}
"""

        payload = {
            "input": {
                "bounds": {
                    "bbox": aoi.to_bbox_list(),
                    "properties": {
                        "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
                    },
                },
                "data": [{
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": time_filter,
                    },
                }],
            },
            "output": {
                "width": w_px,
                "height": h_px,
                "responses": [{
                    "identifier": "default",
                    "format": {"type": "image/tiff"},
                }],
            },
            "evalscript": evalscript,
        }

        if progress_callback:
            progress_callback("Requesting 10-band Float32 GeoTIFF from CDSE...", 50)

        try:
            resp = requests.post(
                self.PROCESS_URL,
                json=payload,
                headers={"Authorization": f"Bearer {token}"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"CDSE Process API request failed: {e}") from e

        out_geotiff.write_bytes(resp.content)

        # Update tags with standard band names
        try:
            with rasterio.open(out_geotiff, "r+") as dst:
                dst.update_tags(
                    SOURCE_SCENE=scene_id,
                    SOURCE_PROVIDER="Copernicus Data Space Ecosystem (CDSE)",
                    BANDS=",".join(S2_10BAND_NAMES),
                )
                for idx, name in enumerate(S2_10BAND_NAMES, start=1):
                    dst.update_tags(idx, BAND_NAME=name)
        except Exception:
            pass

        if progress_callback:
            progress_callback("Copernicus 10-band acquisition complete", 100)

        return out_geotiff

