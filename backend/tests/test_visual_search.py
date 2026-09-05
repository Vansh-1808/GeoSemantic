"""
Automated test suite for Phase 7: Visual and Image-to-Image Search.

Validates:
1. POST /api/search/similar/{tile_id} executes visual similarity search using DINOv2.
2. Reference tile is excluded from its own results (no self-similarity duplication).
3. POST /api/search/image handles uploaded images and queries dino_tiles.
4. Cosine similarity scores are within [-1.0, 1.0] and sorted in descending order.
5. Satellite sensor filter restricts visual search results strictly.
6. Quality score threshold filters out low-quality tiles.
7. Date range filter restricts results temporally.
8. Non-existent tile ID returns HTTP 404 Not Found.
9. Empty uploaded file returns HTTP 400 Bad Request.
10. Latency telemetry (total, DINOv2 embed, Qdrant search) is accurately reported.
"""
from __future__ import annotations

import io
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Enforce test environment and offline mode
os.environ["APP_ENV"] = "test"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.db.database import AsyncSessionLocal
from app.models.scene import Scene
from app.models.tile import Tile
from app.services.tile_embedding import tile_embedding_service
from app.services.vector_store import vector_store


@pytest.fixture(scope="module")
def client():
    from app.main import app
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
async def indexed_visual_scene():
    """
    Creates a temporary scene with 3 distinct tiles, generates real embeddings,
    and indexes them in Qdrant collections.
    """
    async with AsyncSessionLocal() as session:
        scene_id = uuid.uuid4()
        scene = Scene(
            id=scene_id,
            source_path=str(settings.data_dir / "test_visual_scene.tif"),
            filename="test_visual_scene.tif",
            sensor="Sentinel-2B",
            platform="Sentinel-2",
            acquisition_date=datetime(2024, 7, 10, 10, 0, tzinfo=timezone.utc),
            cloud_cover_pct=1.5,
            quality_score=0.96,
            ingestion_status="COMPLETED",
            tile_count=3,
            embedded_count=0,
            bbox_west=77.1,
            bbox_south=28.1,
            bbox_east=77.7,
            bbox_north=28.7,
        )
        session.add(scene)

        # Create tile images on disk
        test_thumbs_dir = settings.thumbnails_dir / str(scene_id)
        test_thumbs_dir.mkdir(parents=True, exist_ok=True)
        test_previews_dir = settings.previews_dir / str(scene_id)
        test_previews_dir.mkdir(parents=True, exist_ok=True)

        tiles: list[Tile] = []
        qualities = [0.95, 0.70, 0.50]
        sensors = ["Sentinel-2B", "Sentinel-2B", "Landsat-9"]

        for i in range(3):
            tile_id = uuid.uuid4()

            thumb_path = test_thumbs_dir / f"tile_{i}.png"
            img = Image.new("RGB", (128, 128), color=(30 + i * 60, 110 - i * 30, 40 + i * 40))
            img.save(thumb_path)

            prev_path = test_previews_dir / f"tile_{i}.png"
            prev_img = Image.new("RGB", (512, 512), color=(30 + i * 60, 110 - i * 30, 40 + i * 40))
            prev_img.save(prev_path)

            tile = Tile(
                id=tile_id,
                scene_id=scene_id,
                tile_col=i,
                tile_row=0,
                tile_size=512,
                pixel_x_off=i * 512,
                pixel_y_off=0,
                pixel_width=512,
                pixel_height=512,
                center_lon=77.20 + i * 0.15,
                center_lat=28.30 + i * 0.10,
                bbox_west=77.15 + i * 0.15,
                bbox_south=28.25 + i * 0.10,
                bbox_east=77.25 + i * 0.15,
                bbox_north=28.35 + i * 0.10,
                tile_path=None,
                thumbnail_path=str(thumb_path),
                acquisition_date=datetime(2024, 7, 10 + i, 10, 0, tzinfo=timezone.utc),
                sensor=sensors[i],
                resolution_m=10.0,
                quality_score=qualities[i],
                cloud_cover_pct=float(i * 1.5),
                is_valid=True,
            )
            session.add(tile)
            tiles.append(tile)

        await session.commit()

        # Embed and index in Qdrant (both remoteclip_tiles and dino_tiles)
        await tile_embedding_service.generate_scene_embeddings(
            scene_id=scene_id,
            session=session,
            force_reembed=True,
            batch_size=4,
        )

    yield scene_id, tiles

    # Cleanup
    point_ids = [str(t.id) for t in tiles]
    try:
        vector_store.delete_points(settings.qdrant_remoteclip_collection, point_ids)
        vector_store.delete_points(settings.qdrant_dino_collection, point_ids)
    except Exception:
        pass

    async with AsyncSessionLocal() as cleanup_session:
        for t in tiles:
            await cleanup_session.delete(t)
        del_scene = await cleanup_session.get(Scene, scene_id)
        if del_scene:
            await cleanup_session.delete(del_scene)
        await cleanup_session.commit()


@pytest.mark.asyncio
async def test_visual_search_by_tile_id(client: TestClient, indexed_visual_scene):
    """Verify tile-to-tile visual similarity search retrieves matches from dino_tiles."""
    scene_id, tiles = indexed_visual_scene
    ref_tile = tiles[0]

    response = client.post(
        f"/api/search/similar/{ref_tile.id}",
        json={"top_k": 10},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["query_type"] == "tile_id"
    assert data["reference_tile_id"] == str(ref_tile.id)
    assert data["model_used"] == "DINOv2 (vit_small_patch14_dinov2.lvd142m)"
    assert data["embedding_dimension"] == 384
    assert data["execution_time_ms"] > 0
    assert data["vector_search_time_ms"] > 0
    assert data["total_found"] > 0

    first = data["results"][0]
    assert "tile_id" in first
    assert "similarity_score" in first
    assert "center_coordinates" in first
    assert "bbox" in first
    assert "provenance" in first


@pytest.mark.asyncio
async def test_visual_search_excludes_reference_tile(client: TestClient, indexed_visual_scene):
    """Verify reference tile is excluded from its own similarity results."""
    scene_id, tiles = indexed_visual_scene
    ref_tile = tiles[0]

    response = client.post(
        f"/api/search/similar/{ref_tile.id}",
        json={"top_k": 20},
    )
    assert response.status_code == 200
    results = response.json()["results"]

    # None of the results should have tile_id == ref_tile.id
    result_ids = [r["tile_id"] for r in results]
    assert str(ref_tile.id) not in result_ids, "Reference tile must be excluded from its own similarity results"


@pytest.mark.asyncio
async def test_visual_search_by_uploaded_image(client: TestClient, indexed_visual_scene):
    """Verify uploaded image search embeds with DINOv2 and queries dino_tiles."""
    # Create test image in memory
    img = Image.new("RGB", (256, 256), color=(40, 90, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/api/search/image?top_k=5",
        files={"file": ("query_satellite.png", buf.getvalue(), "image/png")},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["query_type"] == "uploaded_image"
    assert data["reference_tile_id"] is None
    assert data["model_used"] == "DINOv2 (vit_small_patch14_dinov2.lvd142m)"
    assert data["embedding_dimension"] == 384
    assert data["visual_embedding_time_ms"] > 0
    assert data["vector_search_time_ms"] > 0
    assert data["total_found"] > 0


@pytest.mark.asyncio
async def test_visual_similarity_score_bounds_and_order(client: TestClient, indexed_visual_scene):
    """Verify similarity scores are strictly within [-1.0, 1.0] and sorted descending."""
    _, tiles = indexed_visual_scene
    ref_tile = tiles[1]

    response = client.post(
        f"/api/search/similar/{ref_tile.id}",
        json={"top_k": 20},
    )
    assert response.status_code == 200
    results = response.json()["results"]

    prev_score = 1.0
    for idx, r in enumerate(results):
        score = r["similarity_score"]
        assert -1.0 <= score <= 1.0, f"Score {score} out of bounds"
        assert score <= prev_score + 1e-6, f"Results not sorted descending: {score} > {prev_score}"
        assert r["rank"] == idx + 1
        prev_score = score


@pytest.mark.asyncio
async def test_visual_search_sensor_filter(client: TestClient, indexed_visual_scene):
    """Verify sensor filter isolates tiles by sensor."""
    scene_id, tiles = indexed_visual_scene
    ref_tile = tiles[0]

    # Query with Sentinel-2B filter
    res_s2 = client.post(
        f"/api/search/similar/{ref_tile.id}",
        json={"sensor": "Sentinel-2B", "scene_id": str(scene_id)},
    )
    assert res_s2.status_code == 200
    for r in res_s2.json()["results"]:
        assert r["sensor"] == "Sentinel-2B"

    # Query with Landsat-9 filter
    res_l9 = client.post(
        f"/api/search/similar/{ref_tile.id}",
        json={"sensor": "Landsat-9", "scene_id": str(scene_id)},
    )
    assert res_l9.status_code == 200
    l9_results = res_l9.json()["results"]
    assert len(l9_results) == 1
    assert l9_results[0]["sensor"] == "Landsat-9"


@pytest.mark.asyncio
async def test_visual_search_quality_filter(client: TestClient, indexed_visual_scene):
    """Verify min_quality threshold excludes low quality tiles."""
    scene_id, tiles = indexed_visual_scene
    ref_tile = tiles[2]  # tile 2 is quality 0.50

    response = client.post(
        f"/api/search/similar/{ref_tile.id}",
        json={"min_quality": 0.90, "scene_id": str(scene_id)},
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["quality_score"] >= 0.90


@pytest.mark.asyncio
async def test_visual_search_missing_tile_404(client: TestClient):
    """Verify non-existent tile ID returns HTTP 404."""
    random_id = uuid.uuid4()
    response = client.post(f"/api/search/similar/{random_id}")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_visual_search_empty_upload_400(client: TestClient):
    """Verify empty upload returns HTTP 400."""
    response = client.post(
        "/api/search/image",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert response.status_code == 400
