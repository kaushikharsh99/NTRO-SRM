"""Tests for the quality-assessment, layer-catalogue and pixel-probe web endpoints.

These exercise the analysis surface added on top of the super-resolution API: the layer
registry the UI builds itself from, the lazily rendered spectral-index and analysis
overlays, the per-pixel spectral probe, the session job list, and the downloadable QA
report. Everything runs on the bundled demo scene on CPU.
"""

from pathlib import Path
import time

import pytest
from fastapi.testclient import TestClient

from ntro_srm.web.app import create_app
from ntro_srm.web.services.analysis_service import (
    ANALYSIS_LAYERS,
    COMPOSITES,
    available_layer_names,
    parse_layer_name,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_SCENE = PROJECT_ROOT / "datasets" / "sample_s2" / "sample_s2_l2a.tif"


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Provide a TestClient pinned to CPU for portable CI execution."""
    app = create_app(workspace_root=PROJECT_ROOT, device="cpu")
    return TestClient(app)


@pytest.fixture(scope="module")
def completed_job(client: TestClient) -> dict:
    """Run the bundled demo scene once and share the finished job across tests."""
    if not SAMPLE_SCENE.is_file():
        pytest.skip("Bundled Sentinel-2 demo scene is not present in this checkout.")

    resp = client.post(
        "/api/sr/process",
        json={
            "is_demo": True,
            "run_analysis": True,
            "run_wald_validation": True,
            "uncertainty_members": 0,
        },
    )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]

    payload = None
    for _ in range(240):
        st = client.get(f"/api/sr/jobs/{job_id}").json()
        if st["status"] == "completed":
            payload = st
            break
        if st["status"] == "failed":
            pytest.skip(f"Demo super-resolution unavailable in this environment: {st.get('error_message')}")
        time.sleep(0.5)

    assert payload is not None, "Demo job did not finish within the allotted time"
    return payload


class TestLayerRegistry:
    """The layer registry must stay self-consistent — the UI is generated from it."""

    def test_every_advertised_layer_parses(self):
        for name in available_layer_names():
            source, product, kind = parse_layer_name(name)
            assert source in ("lr", "sr", "bicubic")
            assert kind in ("composite", "index", "analysis")
            assert product

    def test_analysis_layers_are_super_resolved_only(self):
        for key in ANALYSIS_LAYERS:
            assert f"sr_{key}" in available_layer_names()
            assert f"lr_{key}" not in available_layer_names()

    def test_unknown_layer_is_rejected(self):
        with pytest.raises(ValueError):
            parse_layer_name("sr_not_a_layer")
        with pytest.raises(ValueError):
            parse_layer_name("nonsense")

    def test_composites_reference_valid_band_positions(self):
        for spec in COMPOSITES.values():
            assert len(spec["bands"]) == 3
            assert all(0 <= b < 10 for b in spec["bands"])


class TestLayerCatalogEndpoint:
    """The /api/layers payload drives the layer picker, legends and metric chips."""

    def test_catalog_shape(self, client: TestClient):
        resp = client.get("/api/layers")
        assert resp.status_code == 200
        data = resp.json()

        for key in ("composites", "indices", "analysis", "metric_meta", "band_names", "wavelengths_nm"):
            assert key in data, f"missing '{key}' in layer catalogue"

        assert len(data["band_names"]) == 10
        assert len(data["wavelengths_nm"]) == 10
        assert data["band_names"][0] == "B02"
        assert data["band_names"][-1] == "B12"

    def test_index_entries_carry_ui_metadata(self, client: TestClient):
        data = client.get("/api/layers").json()
        keys = {entry["key"] for entry in data["indices"]}
        assert {"ndvi", "ndwi", "ndbi", "nbr"} <= keys

        ndvi = [e for e in data["indices"] if e["key"] == "ndvi"][0]
        for field in ("name", "formula", "cmap", "vmin", "vmax", "application", "legend_hex"):
            assert field in ndvi, f"index entry missing '{field}'"
        assert len(ndvi["legend_hex"]) == 9
        assert all(h.startswith("#") and len(h) == 7 for h in ndvi["legend_hex"])

    def test_metric_meta_is_complete(self, client: TestClient):
        meta = client.get("/api/layers").json()["metric_meta"]
        assert {"psnr_db", "ssim", "sam_deg", "ergas"} <= set(meta)
        for key, entry in meta.items():
            for field in ("label", "unit", "better", "good", "excellent", "description"):
                assert field in entry, f"metric '{key}' is missing '{field}'"
            assert entry["better"] in ("higher", "lower")


class TestHealthEndpoint:
    def test_health_reports_runtime_state(self, client: TestClient):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["device"] == "cpu"
        assert isinstance(data["cuda_available"], bool)
        assert data["version"]


class TestJobListing:
    def test_empty_or_populated_listing_is_well_formed(self, client: TestClient):
        resp = client.get("/api/sr/jobs?limit=5")
        assert resp.status_code == 200
        data = resp.json()
        assert "jobs" in data and "total" in data
        assert data["total"] == len(data["jobs"])
        for job in data["jobs"]:
            assert {"job_id", "status", "created_at", "updated_at"} <= set(job)


class TestAnalysisProducts:
    """End-to-end assertions on a real completed job."""

    def test_analysis_block_present(self, completed_job: dict):
        analysis = completed_job["result"].get("analysis")
        assert analysis, "job result carries no analysis block"
        assert "summary" in analysis
        assert analysis["summary"].get("verdict") in (
            "validated", "validated-with-caveats", "inconclusive"
        )

    def test_reconstruction_caveat_is_always_stated(self, completed_job: dict):
        caveats = completed_job["result"]["analysis"]["summary"].get("caveats", [])
        assert any("not directly observed" in c for c in caveats), (
            "the scientific-reconstruction caveat must always be reported"
        )

    def test_consistency_check_ran(self, completed_job: dict):
        consistency = completed_job["result"]["analysis"].get("consistency")
        assert consistency, "radiometric consistency is free to compute and must always run"
        assert "passed" in consistency
        assert "max_abs_bias" in consistency
        assert len(consistency["per_band_bias"]) == 10

    def test_indices_computed_with_statistics(self, completed_job: dict):
        indices = completed_job["result"]["analysis"].get("indices") or []
        keys = {i["key"] for i in indices}
        assert {"ndvi", "ndwi", "ndbi"} <= keys

        ndvi = [i for i in indices if i["key"] == "ndvi"][0]
        assert -1.0 <= ndvi["mean"] <= 1.0
        assert 0.0 <= ndvi["valid_fraction"] <= 1.0
        assert "delta" in ndvi and "edge_gain" in ndvi["delta"]

    def test_uncertainty_estimated_without_an_ensemble(self, completed_job: dict):
        unc = completed_job["result"]["analysis"].get("uncertainty")
        assert unc, "the free novelty-only uncertainty estimate must always be produced"
        assert 0.0 <= unc["reliability_score"] <= 100.0
        assert unc["hallucination_risk"] in ("low", "moderate", "elevated", "high")
        assert unc["method"] == "novelty-only"

    def test_analysis_endpoint_matches_job_result(self, client: TestClient, completed_job: dict):
        job_id = completed_job["job_id"]
        resp = client.get(f"/api/sr/jobs/{job_id}/analysis")
        assert resp.status_code == 200
        assert resp.json()["summary"] == completed_job["result"]["analysis"]["summary"]


class TestLazyPreviewRendering:
    """Layers beyond the eagerly rendered composites materialise on first request."""

    @pytest.mark.parametrize("layer", ["sr_ndvi", "lr_ndvi", "sr_swir", "bicubic_rgb", "sr_confidence"])
    def test_layer_renders_and_caches(self, client: TestClient, completed_job: dict, layer: str):
        job_id = completed_job["job_id"]
        resp = client.get(f"/api/sr/jobs/{job_id}/preview/{layer}")
        assert resp.status_code == 200, f"{layer}: {resp.text[:200]}"
        assert resp.headers["content-type"] == "image/png"
        assert len(resp.content) > 500

        # Second request must be served from the cached PNG on disk.
        again = client.get(f"/api/sr/jobs/{job_id}/preview/{layer}")
        assert again.status_code == 200
        assert len(again.content) == len(resp.content)

    def test_invalid_layer_returns_400(self, client: TestClient, completed_job: dict):
        job_id = completed_job["job_id"]
        resp = client.get(f"/api/sr/jobs/{job_id}/preview/sr_bogus")
        assert resp.status_code == 400

    def test_unknown_job_returns_404(self, client: TestClient):
        resp = client.get("/api/sr/jobs/job_missing/preview/sr_ndvi")
        assert resp.status_code == 404


class TestPixelProbe:
    def test_probe_returns_paired_spectra(self, client: TestClient, completed_job: dict):
        job_id = completed_job["job_id"]
        bounds = completed_job["result"]["leaflet_bounds"]
        lat = (bounds[0][0] + bounds[1][0]) / 2.0
        lon = (bounds[0][1] + bounds[1][1]) / 2.0

        resp = client.get(f"/api/sr/jobs/{job_id}/pixel", params={"lat": lat, "lon": lon})
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert len(data["band_names"]) == 10
        assert len(data["wavelengths_nm"]) == 10
        assert len(data["lr"]["reflectance"]) == 10
        assert len(data["sr"]["reflectance"]) == 10
        assert data["crs"].startswith("EPSG:")

        finite = [v for v in data["sr"]["reflectance"] if v is not None]
        assert finite, "probe returned no finite super-resolved reflectance"
        assert all(-0.5 <= v <= 3.0 for v in finite), "reflectance outside a plausible range"

        keys = {i["key"] for i in data["indices"]}
        assert {"ndvi", "ndwi"} <= keys

    def test_probe_outside_patch_is_rejected(self, client: TestClient, completed_job: dict):
        job_id = completed_job["job_id"]
        resp = client.get(f"/api/sr/jobs/{job_id}/pixel", params={"lat": -60.0, "lon": 120.0})
        assert resp.status_code == 400

    def test_probe_requires_a_completed_job(self, client: TestClient):
        resp = client.get("/api/sr/jobs/job_missing/pixel", params={"lat": 0.0, "lon": 0.0})
        assert resp.status_code == 404


class TestReportDownloads:
    @pytest.mark.parametrize(
        "file_type,content_type",
        [("report", "application/json"), ("report-md", "text/markdown"), ("native", "image/tiff")],
    )
    def test_new_download_targets(self, client: TestClient, completed_job: dict, file_type, content_type):
        job_id = completed_job["job_id"]
        resp = client.get(f"/api/sr/jobs/{job_id}/download/{file_type}")
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"].startswith(content_type.split(";")[0])
        assert len(resp.content) > 100

    def test_markdown_report_is_readable(self, client: TestClient, completed_job: dict):
        job_id = completed_job["job_id"]
        text = client.get(f"/api/sr/jobs/{job_id}/download/report-md").text
        assert "NTRO-SRM" in text or "Quality" in text
        assert "not directly observed" in text

    def test_unknown_file_type_rejected(self, client: TestClient, completed_job: dict):
        job_id = completed_job["job_id"]
        resp = client.get(f"/api/sr/jobs/{job_id}/download/nope")
        assert resp.status_code == 400
