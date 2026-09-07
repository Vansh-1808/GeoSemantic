import pytest
import numpy as np
import cv2
from datetime import datetime

from app.services.change_detection import ChangeDetectionService
from app.models.tile import Tile
from app.models.change_event import ChangeEvent

from unittest.mock import MagicMock

@pytest.fixture
def mock_db():
    db = MagicMock()
    return db

def test_align_images_success(mock_db):
    service = ChangeDetectionService(mock_db)
    
    # Create two synthetic images that are slightly shifted
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img1, (20, 20), (80, 80), (255, 255, 255), -1)
    
    img2 = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img2, (25, 20), (85, 80), (255, 255, 255), -1) # Shifted by 5px right
    
    img2_aligned, reg_quality = service._align_images(img1, img2)
    
    # Should align perfectly
    assert reg_quality > 0.9
    
    # Difference should be very small now
    diff = cv2.absdiff(img1, img2_aligned)
    mse = np.mean(diff ** 2)
    assert mse < 100 # almost zero

def test_compute_visual_difference(mock_db):
    service = ChangeDetectionService(mock_db)
    
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img1, (20, 20), (80, 80), (255, 255, 255), -1)
    
    # Very different image
    img2 = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img2, (40, 40), (60, 60), (128, 128, 128), -1)
    
    score = service._compute_visual_difference(img1, img2)
    assert score > 0.1 # Should be somewhat high
    
    # Identical images
    score_identical = service._compute_visual_difference(img1, img1)
    assert score_identical == 0.0

def test_compute_semantic_difference(mock_db):
    service = ChangeDetectionService(mock_db)
    
    # Mock vector store
    mock_vs = MagicMock()
    service.vector_store = mock_vs
    
    tile1 = Tile(id="uuid1", remoteclip_vector_id="vec1")
    tile2 = Tile(id="uuid2", remoteclip_vector_id="vec2")
    
    # Mock points
    pt1 = MagicMock()
    pt1.vector = [1.0, 0.0, 0.0]
    
    pt2 = MagicMock()
    pt2.vector = [0.0, 1.0, 0.0] # Orthogonal (cosine sim = 0, distance = 1.0)
    
    def side_effect(collection_name, ids, with_vectors):
        if ids[0] == "vec1": return [pt1]
        if ids[0] == "vec2": return [pt2]
        return []
        
    mock_vs.client.retrieve.side_effect = side_effect
    
    score = service._compute_semantic_difference(tile1, tile2)
    
    assert score == 1.0 # Max distance
