import pytest
import numpy as np
import cv2
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock

from app.services.change_detection import (
    ChangeDetectionService,
    CONSTRUCTION_KEYWORDS,
    VEGETATION_LOSS_KEYWORDS,
    VEGETATION_GAIN_KEYWORDS,
)
from app.models.tile import Tile
from app.models.change_event import ChangeEvent
from app.schemas.change import ChangeAnalyzeRequest
from app.services.confidence_engine import confidence_engine, ConfidenceEvaluationResult


@pytest.fixture
def mock_db():
    db = AsyncMock()
    return db


def test_service_initialization(mock_db):
    """Test service initialization both with and without explicit db argument."""
    # Without db argument (H2 decoupled style)
    service_no_db = ChangeDetectionService()
    assert service_no_db.vector_store is not None
    assert service_no_db.spectral_service is not None
    assert service_no_db.embedding_service is not None

    # With db argument (backwards compatibility)
    service_with_db = ChangeDetectionService(mock_db)
    assert service_with_db.vector_store is not None


def test_align_images_success():
    service = ChangeDetectionService()

    # Create two synthetic images that are slightly shifted
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img1, (20, 20), (80, 80), (255, 255, 255), -1)

    img2 = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img2, (25, 20), (85, 80), (255, 255, 255), -1)  # Shifted by 5px right

    img2_aligned, reg_quality = service._align_images(img1, img2)

    # Should align cleanly
    assert reg_quality > 0.9

    # Difference should be very small after alignment
    diff = cv2.absdiff(img1, img2_aligned)
    mse = np.mean(diff ** 2)
    assert mse < 100


def test_compute_visual_difference():
    service = ChangeDetectionService()

    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img1, (20, 20), (80, 80), (255, 255, 255), -1)

    # Very different image
    img2 = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img2, (40, 40), (60, 60), (128, 128, 128), -1)

    score = service._compute_visual_difference(img1, img2)
    assert score > 0.1

    # Identical images
    score_identical = service._compute_visual_difference(img1, img1)
    assert score_identical == 0.0


def test_compute_semantic_difference():
    service = ChangeDetectionService()

    # Mock vector store
    mock_vs = MagicMock()
    service.vector_store = mock_vs

    tile1 = Tile(id="uuid1", remoteclip_vector_id="vec1")
    tile2 = Tile(id="uuid2", remoteclip_vector_id="vec2")

    # Mock points
    pt1 = MagicMock()
    pt1.vector = [1.0, 0.0, 0.0]

    pt2 = MagicMock()
    pt2.vector = [0.0, 1.0, 0.0]  # Orthogonal (cosine sim = 0, distance = 1.0)

    def side_effect(collection_name, ids, with_vectors):
        if ids[0] == "vec1":
            return [pt1]
        if ids[0] == "vec2":
            return [pt2]
        return []

    mock_vs.client.retrieve.side_effect = side_effect

    score = service._compute_semantic_difference(tile1, tile2)
    assert score == 1.0


def test_compute_tile_iou():
    service = ChangeDetectionService()

    # Case 1: Identical bounding boxes -> IoU = 1.0
    t1 = Tile(bbox_west=10.0, bbox_south=20.0, bbox_east=11.0, bbox_north=21.0)
    t2 = Tile(bbox_west=10.0, bbox_south=20.0, bbox_east=11.0, bbox_north=21.0)
    assert service._compute_tile_iou(t1, t2) == 1.0

    # Case 2: Partial overlap
    # t1: [0, 2] x [0, 1], Area = 2
    # t2: [1, 3] x [0, 1], Area = 2
    # Intersection: [1, 2] x [0, 1], Area = 1
    # Union: 2 + 2 - 1 = 3 -> IoU = 1/3
    t3 = Tile(bbox_west=0.0, bbox_south=0.0, bbox_east=2.0, bbox_north=1.0)
    t4 = Tile(bbox_west=1.0, bbox_south=0.0, bbox_east=3.0, bbox_north=1.0)
    iou = service._compute_tile_iou(t3, t4)
    assert abs(iou - (1.0 / 3.0)) < 1e-4

    # Case 3: Completely disjoint
    t5 = Tile(bbox_west=0.0, bbox_south=0.0, bbox_east=1.0, bbox_north=1.0)
    t6 = Tile(bbox_west=5.0, bbox_south=5.0, bbox_east=6.0, bbox_north=6.0)
    assert service._compute_tile_iou(t5, t6) == 0.0

    # Case 4: Missing bbox fallback to centroid distance
    t7 = Tile(bbox_west=None, center_lon=75.0, center_lat=25.0)
    t8 = Tile(bbox_west=None, center_lon=75.0001, center_lat=25.0001)
    assert service._compute_tile_iou(t7, t8) == 1.0


def test_scenes_overlap():
    service = ChangeDetectionService()

    scene1_tiles = [
        Tile(bbox_west=10.0, bbox_south=10.0, bbox_east=12.0, bbox_north=12.0)
    ]
    scene2_overlapping = [
        Tile(bbox_west=11.0, bbox_south=11.0, bbox_east=13.0, bbox_north=13.0)
    ]
    scene3_disjoint = [
        Tile(bbox_west=50.0, bbox_south=50.0, bbox_east=52.0, bbox_north=52.0)
    ]

    assert service._scenes_overlap(scene1_tiles, scene2_overlapping) is True
    assert service._scenes_overlap(scene1_tiles, scene3_disjoint) is False


def test_change_keywords():
    """Verify keyword sets contain essential domain descriptors."""
    assert "construction" in CONSTRUCTION_KEYWORDS
    assert "urban" in CONSTRUCTION_KEYWORDS
    assert "deforestation" in VEGETATION_LOSS_KEYWORDS
    assert "reforestation" in VEGETATION_GAIN_KEYWORDS
    assert "afforestation" in VEGETATION_GAIN_KEYWORDS


def test_confidence_engine_evaluation():
    """Verify confidence engine produces calibrated scores and correct decision tiers."""
    result = confidence_engine.evaluate_signals(
        visual_change=0.8,
        semantic_change=0.75,
        image_quality=0.85,
        cloud_score=0.02,
        shadow_score=0.01,
        registration_confidence=0.95,
        sensor_compatibility=1.0,
        seasonal_compatibility=0.9,
        spectral_evidence=0.6,
        spatial_consistency=0.7,
    )
    assert 0.0 <= result.final_confidence <= 1.0
    assert result.confidence_tier in ("High Confidence", "Medium Confidence", "Low Confidence", "Suppressed")
    assert result.is_suppressed is False

    # Cloud suppression test: high cloud should suppress or heavily penalize
    cloudy_result = confidence_engine.evaluate_signals(
        visual_change=0.9,
        semantic_change=0.8,
        image_quality=0.2,
        cloud_score=0.60,  # Exceeds cloud suppression limit (0.25)
        shadow_score=0.0,
        registration_confidence=0.9,
        sensor_compatibility=1.0,
        seasonal_compatibility=0.8,
        spectral_evidence=0.5,
        spatial_consistency=0.5,
    )
    assert cloudy_result.is_suppressed is True or cloudy_result.confidence_tier == "Suppressed" or cloudy_result.final_confidence < 0.40


@pytest.mark.asyncio
async def test_analyze_change_smoke_no_tiles(mock_db):
    """Verify analyze_change handles empty database gracefully."""
    service = ChangeDetectionService()

    # Mock scalars to return an empty result
    mock_result = MagicMock()
    mock_result.all.return_value = []
    mock_db.scalars.return_value = mock_result

    req = ChangeAnalyzeRequest(
        aoi_wkt=None,
        start_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
        limit=10,
    )

    count, events = await service.analyze_change(req, mock_db)
    assert count == 0
    assert events == []
