"""Unit and integration tests for NTRO-SRM Web API."""

from pathlib import Path
import time
import pytest
from fastapi.testclient import TestClient

from ntro_srm.web.app import create_app
from ntro_srm.web.schemas import AOI, SentinelSearchRequest


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Provide TestClient with CPU device for fast, portable CI/CD testing."""
    project_root = Path(__file__).resolve().parents[1]
    app = create_app(workspace_root=project_root, device="cpu")
    return TestClient(app)


class TestWebAPI:
    """Test suite for the web endpoints and services."""

    def test_system_info_endpoint(self, client: TestClient):
        resp = client.get("/api/system-info")
        assert resp.status_code == 200
        data = resp.json()
        assert "cuda_available" in data
        assert "device_name" in data
        assert data["model_variant"] == "SEN2SR-Lite"
        assert "models_available" in data
        assert len(data["models_available"]) == 2
        model_ids = [m["id"] for m in data["models_available"]]
        assert "lite" in model_ids
        assert "swin2sr" in model_ids
        assert data["upscale_factor"] == 4
        assert data["max_aoi_pixels"] == 512 * 512

    def test_aoi_schema_validation(self):
        # Valid AOI
        aoi = AOI(min_lon=-80.60, min_lat=37.40, max_lon=-80.55, max_lat=37.45)
        assert aoi.width_km > 0.0
        assert aoi.height_km > 0.0
        assert aoi.area_km2 > 0.0
        assert aoi.estimated_s2_pixels_10m > 0

        # Invalid: max_lon < min_lon
        with pytest.raises(ValueError):
            AOI(min_lon=-80.50, min_lat=37.40, max_lon=-80.60, max_lat=37.45)

        # Invalid: max_lat < min_lat
        with pytest.raises(ValueError):
            AOI(min_lon=-80.60, min_lat=37.50, max_lon=-80.55, max_lat=37.40)

    def test_demo_info_endpoint(self, client: TestClient):
        resp = client.get("/api/demo/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["scene_id"] == "DEMO_MLBS_20180825_S2L2A"
        assert "aoi" in data
        assert data["cloud_cover"] == 0.0

    def test_excessive_aoi_rejected(self, client: TestClient):
        # Extremely large AOI (10x10 degrees ~ 1000km x 1000km)
        big_aoi = {
            "min_lon": -80.0,
            "min_lat": 30.0,
            "max_lon": -70.0,
            "max_lat": 40.0,
        }
        resp = client.post(
            "/api/sr/process",
            json={
                "aoi": big_aoi,
                "scene_id": "test_scene",
                "is_demo": False,
            },
        )
        assert resp.status_code == 400
        assert "exceeding maximum allowed limit" in resp.json()["detail"]

    def test_missing_aoi_rejected_for_real_scenes(self, client: TestClient):
        resp = client.post(
            "/api/sr/process",
            json={
                "aoi": None,
                "scene_id": "test_scene",
                "is_demo": False,
            },
        )
        assert resp.status_code == 400
        assert "AOI is required" in resp.json()["detail"]

    def test_demo_job_workflow_and_downloads(self, client: TestClient):
        # 1. Launch demo inference job
        resp = client.post(
            "/api/sr/process",
            json={"is_demo": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        job_id = data["job_id"]

        # 2. Poll job status until completed (timeout 30s)
        completed = False
        for _ in range(60):
            status_resp = client.get(f"/api/sr/jobs/{job_id}")
            assert status_resp.status_code == 200
            st = status_resp.json()
            if st["status"] == "completed":
                completed = True
                break
            elif st["status"] == "failed":
                pytest.fail(f"Demo job failed: {st.get('error_message')}")
            time.sleep(0.5)

        assert completed, "Demo job timed out before completion"

        # 3. Verify job result structure
        res = st["result"]
        assert res["upscale_factor"] == 4
        assert res["bands_count"] == 10
        assert "previews" in res
        assert "leaflet_bounds" in res

        # 4. Test preview endpoints
        p_rgb = client.get(f"/api/sr/jobs/{job_id}/preview/sr_rgb")
        assert p_rgb.status_code == 200
        assert p_rgb.headers["content-type"] == "image/png"

        p_lr = client.get(f"/api/sr/jobs/{job_id}/preview/lr_rgb")
        assert p_lr.status_code == 200

        p_cir = client.get(f"/api/sr/jobs/{job_id}/preview/sr_cir")
        assert p_cir.status_code == 200

        # 5. Test GeoTIFF download
        download_resp = client.get(f"/api/sr/jobs/{job_id}/download/geotiff")
        assert download_resp.status_code == 200
        assert download_resp.headers["content-type"] == "image/tiff"
        assert len(download_resp.content) > 1000

        # 6. Test RGB PNG download
        download_png = client.get(f"/api/sr/jobs/{job_id}/download/rgb")
        assert download_png.status_code == 200
        assert download_png.headers["content-type"] == "image/png"

    def test_homepage_renders(self, client: TestClient):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "NTRO-SRM" in resp.text
        assert "SEN2SR-Lite" in resp.text
        assert "leaflet.js" in resp.text

    def test_cdse_provider_authentication_and_search(self):
        from ntro_srm.web.services.sentinel_service import CopernicusCDSEProvider
        provider = CopernicusCDSEProvider()
        assert provider.is_configured() is True
        token = provider._get_token()
        assert token is not None and len(token) > 50

        aoi = AOI(min_lon=-80.60, min_lat=37.40, max_lon=-80.55, max_lat=37.45)
        search_req = SentinelSearchRequest(
            aoi=aoi,
            date_from="2023-08-01",
            date_to="2023-08-31",
            max_cloud_cover=20.0,
            limit=2,
        )
        res = provider.search(search_req)
        assert res.total >= 1
        assert res.scenes[0].provider == "Copernicus Data Space (CDSE)"

