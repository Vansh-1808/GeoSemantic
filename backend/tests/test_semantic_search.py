"""
Automated test suite for Phase 6: Semantic Text-to-Satellite Retrieval.

Validates:
1. Natural language query generates valid ranked tile results from Qdrant.
2. Cosine similarity scores are within [-1.0, 1.0] and sorted in descending order.
3. Satellite sensor filter restricts results strictly to matching sensor.
4. Min quality threshold filters out lower quality tiles.
5. Acquisition date range filter limits results temporally.
6. Area of Interest (AOI) bounding box filter restricts results geographically.
7. Provenance metadata is attached to each retrieved result item.
8. Query latency telemetry is measured and reported accurately.
9. GET /api/search/filters exposes available sensors, date ranges, and index counts.
10. POST /api/search/text functions as a valid alias endpoint.
"""
from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import select

# Enforce test environment and offline mode
os.environ["APP_ENV"] = "test"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.db.database import AsyncSessionLocal
from app.models.provenance import ProvenanceRecord
from app.models.scene import Scene
from app.models.tile import Tile
from app.services.tile_embedding import tile_embedding_service
from app.services.vector_store import vector_store


@pytest.fixture(scope="module")
def client():
    from app.main import app
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
async def indexed_test_scene():
    """
    Creates a temporary scene with 3 distinct tiles, generates real embeddings,
    and indexes them in Qdrant collections.
    """
    async with AsyncSessionLocal() as session:
        scene_id = uuid.uuid4()
        scene = Scene(
            id=scene_id,
            source_path=str(settings.data_dir / "test_search_scene.tif"),
            filename="test_search_scene.tif",
            sensor="Sentinel-2B",
            platform="Sentinel-2",
            acquisition_date=datetime(2024, 6, 15, 11, 0, tzinfo=timezone.utc),
            cloud_cover_pct=3.0,
            quality_score=0.95,
            ingestion_status="COMPLETED",
            tile_count=3,
            embedded_count=0,
            bbox_west=77.0,
            bbox_south=28.0,
            bbox_east=77.6,
            bbox_north=28.6,
        )
        session.add(scene)

        # Create tile images on disk
        test_thumbs_dir = settings.thumbnails_dir / str(scene_id)
        test_thumbs_dir.mkdir(parents=True, exist_ok=True)
        test_previews_dir = settings.previews_dir / str(scene_id)
        test_previews_dir.mkdir(parents=True, exist_ok=True)

        tiles: list[Tile] = []
        qualities = [0.95, 0.75, 0.60]
        sensors = ["Sentinel-2B", "Sentinel-2B", "Landsat-9"]

        for i in range(3):
            tile_id = uuid.uuid4()

            # Thumbnail
            thumb_path = test_thumbs_dir / f"tile_{i}.png"
            img = Image.new("RGB", (128, 128), color=(40 + i * 50, 100 - i * 20, 60 + i * 30))
            img.save(thumb_path)

            # Preview
            prev_path = test_previews_dir / f"tile_{i}.png"
            prev_img = Image.new("RGB", (512, 512), color=(40 + i * 50, 100 - i * 20, 60 + i * 30))
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
                center_lon=77.10 + i * 0.15,
                center_lat=28.20 + i * 0.10,
                bbox_west=77.05 + i * 0.15,
                bbox_south=28.15 + i * 0.10,
                bbox_east=77.15 + i * 0.15,
                bbox_north=28.25 + i * 0.10,
                tile_path=None,
                thumbnail_path=str(thumb_path),
                acquisition_date=datetime(2024, 6, 15 + i, 11, 0, tzinfo=timezone.utc),
                sensor=sensors[i],
                resolution_m=10.0,
                quality_score=qualities[i],
                cloud_cover_pct=float(i * 2.0),
                is_valid=True,
            )
            session.add(tile)
            tiles.append(tile)

        await session.commit()

        # Embed and index all 3 tiles in Qdrant
        await tile_embedding_service.generate_scene_embeddings(
            scene_id=scene_id,
            session=session,
            force_reembed=True,
            batch_size=4,
        )

    yield scene_id, tiles

    # Teardown: delete Qdrant points and DB records
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
async def test_semantic_search_empty_filters(client: TestClient):
    """Verify semantic search handles unmatchable queries gracefully."""
    response = client.post(
        "/api/search/semantic",
        json={
            "query": "airport runway",
            "top_k": 5,
            "sensor": "NonExistentSensor123",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "airport runway"
    assert data["total_found"] == 0
    assert data["results"] == []
    assert data["execution_time_ms"] > 0
    assert data["text_embedding_time_ms"] > 0


@pytest.mark.asyncio
async def test_semantic_search_with_indexed_tiles(client: TestClient, indexed_test_scene):
    """Verify natural language search retrieves indexed tiles from Qdrant with provenance."""
    scene_id, tiles = indexed_test_scene

    response = client.post(
        "/api/search/semantic",
        json={
            "query": "agricultural fields and vegetation canopy",
            "top_k": 10,
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["query"] == "agricultural fields and vegetation canopy"
    assert data["total_found"] >= len(tiles)
    assert data["execution_time_ms"] > 0
    assert data["text_embedding_time_ms"] > 0
    assert data["vector_search_time_ms"] > 0
    assert "RemoteCLIP" in data["model_used"]

    # Verify result item schema
    first = data["results"][0]
    assert "tile_id" in first
    assert "similarity_score" in first
    assert "rank" in first
    assert first["rank"] == 1
    assert "scene_id" in first
    assert "scene_name" in first
    assert "center_coordinates" in first
    assert "lon" in first["center_coordinates"]
    assert "lat" in first["center_coordinates"]
    assert "bbox" in first
    assert "west" in first["bbox"]
    assert "thumbnail_url" in first
    assert "preview_url" in first
    assert "provenance" in first


@pytest.mark.asyncio
async def test_semantic_search_ranking_order(client: TestClient, indexed_test_scene):
    """Verify similarity scores are strictly within [-1.0, 1.0] and sorted descending."""
    response = client.post(
        "/api/search/semantic",
        json={
            "query": "coastal harbor with shipping containers",
            "top_k": 20,
        },
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) > 0

    prev_score = 1.0
    for idx, r in enumerate(results):
        score = r["similarity_score"]
        assert -1.0 <= score <= 1.0, f"Score {score} out of valid cosine range"
        assert score <= prev_score + 1e-6, f"Results not sorted: {score} > {prev_score}"
        assert r["rank"] == idx + 1
        prev_score = score


@pytest.mark.asyncio
async def test_semantic_search_sensor_filter(client: TestClient, indexed_test_scene):
    """Verify satellite sensor filter restricts results strictly to matching sensor."""
    scene_id, tiles = indexed_test_scene

    # Query with Sentinel-2B filter
    res_s2 = client.post(
        "/api/search/semantic",
        json={
            "query": "satellite imagery tile",
            "sensor": "Sentinel-2B",
            "scene_id": str(scene_id),
            "top_k": 10,
        },
    )
    assert res_s2.status_code == 200
    s2_results = res_s2.json()["results"]
    assert len(s2_results) == 2
    for r in s2_results:
        assert r["sensor"] == "Sentinel-2B"

    # Query with Landsat-9 filter
    res_l9 = client.post(
        "/api/search/semantic",
        json={
            "query": "satellite imagery tile",
            "sensor": "Landsat-9",
            "scene_id": str(scene_id),
            "top_k": 10,
        },
    )
    assert res_l9.status_code == 200
    l9_results = res_l9.json()["results"]
    assert len(l9_results) == 1
    assert l9_results[0]["sensor"] == "Landsat-9"


@pytest.mark.asyncio
async def test_semantic_search_quality_filter(client: TestClient, indexed_test_scene):
    """Verify min_quality threshold excludes low-quality tiles."""
    scene_id, tiles = indexed_test_scene

    # With min_quality = 0.90 (only tile 0 has 0.95)
    response = client.post(
        "/api/search/semantic",
        json={
            "query": "satellite imagery tile",
            "min_quality": 0.90,
            "scene_id": str(scene_id),
            "top_k": 10,
        },
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["quality_score"] >= 0.90


@pytest.mark.asyncio
async def test_semantic_search_date_range_filter(client: TestClient, indexed_test_scene):
    """Verify acquisition date range filter restricts results temporally."""
    scene_id, tiles = indexed_test_scene

    # Filter for June 15, 2024 only (should match tile 0)
    response = client.post(
        "/api/search/semantic",
        json={
            "query": "satellite imagery tile",
            "start_date": "2024-06-15T00:00:00Z",
            "end_date": "2024-06-15T23:59:59Z",
            "scene_id": str(scene_id),
            "top_k": 10,
        },
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1

    # Date range in 2020 should match 0 tiles
    res_empty = client.post(
        "/api/search/semantic",
        json={
            "query": "satellite imagery tile",
            "start_date": "2020-01-01T00:00:00Z",
            "end_date": "2020-12-31T23:59:59Z",
            "scene_id": str(scene_id),
            "top_k": 10,
        },
    )
    assert res_empty.status_code == 200
    assert len(res_empty.json()["results"]) == 0


@pytest.mark.asyncio
async def test_semantic_search_aoi_filter(client: TestClient, indexed_test_scene):
    """Verify Area of Interest bounding box filter restricts results geographically."""
    scene_id, tiles = indexed_test_scene

    # AOI covering tile 0 only (center_lon=77.10, center_lat=28.20)
    response = client.post(
        "/api/search/semantic",
        json={
            "query": "satellite imagery tile",
            "aoi": [77.0, 28.15, 77.20, 28.25],
            "scene_id": str(scene_id),
            "top_k": 10,
        },
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["tile_col"] == 0

    # Disjoint AOI (e.g. London coordinates) returns 0 results
    res_disjoint = client.post(
        "/api/search/semantic",
        json={
            "query": "satellite imagery tile",
            "aoi": [-0.2, 51.4, 0.0, 51.6],
            "scene_id": str(scene_id),
            "top_k": 10,
        },
    )
    assert res_disjoint.status_code == 200
    assert len(res_disjoint.json()["results"]) == 0


@pytest.mark.asyncio
async def test_get_search_filters(client: TestClient, indexed_test_scene):
    """Verify GET /api/search/filters returns distinct sensors and date ranges."""
    response = client.get("/api/search/filters")
    assert response.status_code == 200
    data = response.json()

    assert "sensors" in data
    assert isinstance(data["sensors"], list)
    assert "Sentinel-2B" in data["sensors"]
    assert "total_indexed_tiles" in data
    assert data["total_indexed_tiles"] > 0


@pytest.mark.asyncio
async def test_text_search_alias(client: TestClient, indexed_test_scene):
    """Verify POST /api/search/text functions identically to /semantic."""
    scene_id, _ = indexed_test_scene

    resp_sem = client.post(
        "/api/search/semantic",
        json={"query": "test query", "scene_id": str(scene_id)},
    )
    resp_txt = client.post(
        "/api/search/text",
        json={"query": "test query", "scene_id": str(scene_id)},
    )

    assert resp_sem.status_code == 200
    assert resp_txt.status_code == 200
    assert len(resp_sem.json()["results"]) == len(resp_txt.json()["results"])
