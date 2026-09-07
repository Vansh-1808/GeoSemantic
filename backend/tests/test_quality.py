"""
Phase 9 Image Quality Assessment & Confounding Factors test suite.
Tests deterministic quality evaluation across:
1. Clear imagery
2. Cloud-heavy imagery
3. Poor-quality / nodata / corrupted imagery
4. REST API endpoints (/api/quality/tile, /api/quality/scene, /quality/...)
"""
from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.services.spectral_analysis import spectral_analysis_service


@pytest.fixture(scope="module")
def client():
    from app.main import app
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def clear_image_path(tmp_path: Path) -> Path:
    """Create a synthetic clear satellite scene (healthy vegetation & urban, moderate brightness, zero cloud)."""
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    # Natural green vegetation and soil: B=40, G=120, R=60
    img[:] = (40, 120, 60)
    p = tmp_path / "clear_test.png"
    cv2.imwrite(str(p), img)
    return p


@pytest.fixture
def cloud_heavy_image_path(tmp_path: Path) -> Path:
    """Create a synthetic cloud-heavy scene (80% bright white pixels: B=240, G=240, R=240)."""
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[:] = (40, 90, 50)
    # Add large thick cloud block across 80% of pixels
    img[:105, :] = (245, 245, 245)
    p = tmp_path / "cloud_heavy_test.png"
    cv2.imwrite(str(p), img)
    return p


@pytest.fixture
def poor_quality_nodata_path(tmp_path: Path) -> Path:
    """Create a synthetic corrupted / nodata tile (70% unmapped black border pixels)."""
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    # Only 30% has data
    img[:38, :38] = (60, 90, 70)
    # Remaining 70%+ is pure black (0, 0, 0)
    p = tmp_path / "poor_nodata_test.png"
    cv2.imwrite(str(p), img)
    return p


@pytest.fixture
def noisy_striped_image_path(tmp_path: Path) -> Path:
    """Create a synthetic sensor-banded / noisy image with severe row jitter."""
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    for r in range(128):
        # Alternating stark high/low rows
        val = 200 if r % 2 == 0 else 40
        img[r, :] = (val, val, val)
    p = tmp_path / "noisy_striped_test.png"
    cv2.imwrite(str(p), img)
    return p


class TestDeterministicQualityAssessment:
    """Verify core image quality analysis logic under varying confounding factors."""

    def test_clear_imagery_evaluates_suitable(self, clear_image_path: Path):
        result = spectral_analysis_service.analyze_quality(clear_image_path)
        assert result["quality_score"] >= 0.75
        assert result["cloud_score"] < 0.10
        assert result["nodata_score"] < 0.05
        assert result["suitability"] == "Suitable"
        assert result["usable_for_change_analysis"] is True
        assert result["satellite_mask_available"] is False
        assert len(result["caveats"]) >= 2

    def test_cloud_heavy_imagery_evaluates_unsuitable(self, cloud_heavy_image_path: Path):
        result = spectral_analysis_service.analyze_quality(cloud_heavy_image_path)
        assert result["cloud_score"] >= 0.60
        assert result["suitability"] in ("Caution", "Poor Quality")
        assert result["usable_for_change_analysis"] is False
        # Quality score must be heavily penalized by cloud occlusion
        assert result["quality_score"] < 0.60

    def test_poor_quality_nodata_imagery_evaluates_poor(self, poor_quality_nodata_path: Path):
        result = spectral_analysis_service.analyze_quality(poor_quality_nodata_path)
        assert result["nodata_score"] >= 0.60
        assert result["quality_score"] < 0.40
        assert result["suitability"] == "Poor Quality"
        assert result["usable_for_change_analysis"] is False

    def test_sensor_noise_detected(self, noisy_striped_image_path: Path):
        result = spectral_analysis_service.analyze_quality(noisy_striped_image_path)
        assert result["noise_score"] > 0.10

    def test_missing_or_corrupt_file_graceful_fallback(self):
        result = spectral_analysis_service.analyze_quality("non_existent_file_xyz_123.png")
        assert result["quality_score"] == 0.0
        assert result["suitability"] == "Poor Quality"
        assert result["usable_for_change_analysis"] is False
        assert result["nodata_score"] == 1.0


class TestQualityAPIEndpoints:
    """Verify REST API endpoints for tile and scene quality."""

    def test_get_scene_quality_with_valid_scene(self, client: TestClient):
        # Using ingested Perambur scene
        sid = "4bdf2d40-b492-49b5-afc9-1f07c721ddbd"
        res = client.get(f"/api/quality/scene/{sid}")
        assert res.status_code == 200
        data = res.json()
        assert "quality_score" in data
        assert "suitability" in data
        assert "tile_count" in data
        assert data["tile_count"] > 0
        assert "tiles_suitable" in data
        assert "usable_for_change_analysis" in data
        assert data["suitability"] in ("Suitable", "Caution", "Poor Quality")

    def test_root_quality_scene_endpoint(self, client: TestClient):
        sid = "ba579301-c79c-457e-a37d-08d5df27551e"
        res = client.get(f"/quality/scene/{sid}")
        assert res.status_code == 200
        assert res.json()["tile_count"] > 0

    def test_get_tile_quality_with_valid_tile(self, client: TestClient):
        tid = "2419358d-40da-4b34-b8fb-b51f8898d73d"
        res = client.get(f"/api/quality/tile/{tid}")
        assert res.status_code == 200
        data = res.json()
        assert data["tile_id"] == tid
        assert "quality_score" in data
        assert "cloud_score" in data
        assert "shadow_score" in data
        assert "nodata_score" in data
        assert "suitability" in data
        assert "usable_for_change_analysis" in data
        assert data["satellite_mask_available"] is False
        assert len(data["caveats"]) > 0

    def test_get_quality_nonexistent_ids_return_404(self, client: TestClient):
        fake_id = uuid.uuid4()
        res_scene = client.get(f"/api/quality/scene/{fake_id}")
        assert res_scene.status_code == 404

        res_tile = client.get(f"/api/quality/tile/{fake_id}")
        assert res_tile.status_code == 404
