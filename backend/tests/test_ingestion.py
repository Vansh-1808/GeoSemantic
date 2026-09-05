"""
Tests for geospatial metadata extraction and tile generation.
These use synthetic GeoTIFF files generated in memory — no real imagery required.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

# Make backend app importable
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def synthetic_geotiff(tmp_path: Path) -> Path:
    """Create a small 512×512 synthetic GeoTIFF for testing."""
    out = tmp_path / "test_scene_20230615.tif"
    width, height = 512, 512
    transform = from_bounds(77.0, 28.0, 77.5, 28.5, width, height)
    data = np.random.randint(50, 200, (3, height, width), dtype=np.uint8)

    with rasterio.open(
        out,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs=CRS.from_epsg(4326),
        transform=transform,
    ) as ds:
        ds.write(data)
        ds.update_tags(
            ACQUISITIONDATETIME="2023-06-15T10:30:00+05:30",
            SENSOR_ID="TEST-MSI",
            SPACECRAFT_ID="TEST-SAT",
        )
    return out


@pytest.fixture
def invalid_file(tmp_path: Path) -> Path:
    """Create a non-raster file."""
    f = tmp_path / "not_a_tiff.tif"
    f.write_bytes(b"this is not a geotiff")
    return f


# ── Validation tests ──────────────────────────────────────────

class TestValidation:
    def test_valid_geotiff_passes(self, synthetic_geotiff: Path):
        from app.services.ingestion import _validate_geotiff
        _validate_geotiff(synthetic_geotiff)  # Should not raise

    def test_invalid_file_raises(self, invalid_file: Path):
        from app.services.ingestion import _validate_geotiff
        with pytest.raises(ValueError):
            _validate_geotiff(invalid_file)

    def test_missing_file_raises(self, tmp_path: Path):
        from app.services.ingestion import _validate_geotiff
        with pytest.raises(FileNotFoundError):
            _validate_geotiff(tmp_path / "nonexistent.tif")


# ── Metadata extraction tests ─────────────────────────────────

class TestMetadataExtraction:
    def test_extracts_basic_metadata(self, synthetic_geotiff: Path):
        from app.services.ingestion import _extract_metadata
        meta = _extract_metadata(synthetic_geotiff)

        assert meta["width"] == 512
        assert meta["height"] == 512
        assert meta["band_count"] == 3
        assert meta["crs_epsg"] == 4326

    def test_extracts_bbox_in_4326(self, synthetic_geotiff: Path):
        from app.services.ingestion import _extract_metadata
        meta = _extract_metadata(synthetic_geotiff)

        w, s, e, n = meta["bbox_4326"]
        assert w < e, "West must be less than East"
        assert s < n, "South must be less than North"
        assert -180 <= w <= 180
        assert -90 <= s <= 90

    def test_parses_acquisition_date(self, synthetic_geotiff: Path):
        from app.services.ingestion import _extract_metadata
        meta = _extract_metadata(synthetic_geotiff)

        assert meta["acquisition_date"] is not None
        assert meta["acquisition_date"].year == 2023
        assert meta["acquisition_date"].month == 6
        assert meta["acquisition_date"].day == 15

    def test_parses_sensor_info(self, synthetic_geotiff: Path):
        from app.services.ingestion import _extract_metadata
        meta = _extract_metadata(synthetic_geotiff)

        assert meta["sensor"] == "TEST-MSI"
        assert meta["platform"] == "TEST-SAT"

    def test_crs_wkt_not_empty(self, synthetic_geotiff: Path):
        from app.services.ingestion import _extract_metadata
        meta = _extract_metadata(synthetic_geotiff)

        assert meta["crs_wkt"] and len(meta["crs_wkt"]) > 10


# ── Quality assessment tests ──────────────────────────────────

class TestQualityAssessment:
    def test_returns_quality_score(self, synthetic_geotiff: Path):
        from app.services.ingestion import _assess_quality, _extract_metadata
        meta = _extract_metadata(synthetic_geotiff)
        quality = _assess_quality(synthetic_geotiff, meta)

        assert "quality_score" in quality
        assert 0.0 <= quality["quality_score"] <= 1.0

    def test_nodata_ratio_in_range(self, synthetic_geotiff: Path):
        from app.services.ingestion import _assess_quality, _extract_metadata
        meta = _extract_metadata(synthetic_geotiff)
        quality = _assess_quality(synthetic_geotiff, meta)

        assert 0.0 <= quality["nodata_ratio"] <= 1.0

    def test_cloud_cover_assessed(self, synthetic_geotiff: Path):
        from app.services.ingestion import _assess_quality, _extract_metadata
        meta = _extract_metadata(synthetic_geotiff)
        quality = _assess_quality(synthetic_geotiff, meta)

        # Should have cloud estimate for 3-band imagery
        assert "cloud_cover_pct" in quality
        assert 0.0 <= quality["cloud_cover_pct"] <= 100.0


# ── Checksum tests ────────────────────────────────────────────

class TestChecksum:
    def test_checksum_is_hex(self, synthetic_geotiff: Path):
        from app.services.ingestion import _compute_checksum
        cs = _compute_checksum(synthetic_geotiff)

        assert len(cs) == 64
        int(cs, 16)  # Should not raise if valid hex

    def test_same_file_same_checksum(self, synthetic_geotiff: Path):
        from app.services.ingestion import _compute_checksum
        cs1 = _compute_checksum(synthetic_geotiff)
        cs2 = _compute_checksum(synthetic_geotiff)
        assert cs1 == cs2
