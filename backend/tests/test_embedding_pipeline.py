"""
Automated test suite for Phase 5: Real Embedding and Vector Index Pipeline.

Validates:
1. Real tiles generate embeddings.
2. Vectors appear in Qdrant collections (remoteclip_tiles & dino_tiles).
3. Metadata payload is correct (tile_id, scene_id, acquisition_date, sensor, geometry, quality_score, model_version).
4. Re-running does not create accidental duplicates (idempotent upserts).
5. New imagery can be added incrementally without rebuilding existing indexes.
6. Vector statistics measure embedding time, number of vectors, and disk storage usage.
7. PostgreSQL provenance records are created for audit traceability.
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
async def sample_scene_with_tiles():
    """Create a temporary test scene with 2 analysis tiles in the database."""
    async with AsyncSessionLocal() as session:
        scene_id = uuid.uuid4()
        scene = Scene(
            id=scene_id,
            source_path=str(settings.data_dir / "test_scene.tif"),
            filename="test_scene.tif",
            sensor="Sentinel-2B",
            platform="Sentinel-2",
            acquisition_date=datetime(2023, 5, 20, 10, 30, tzinfo=timezone.utc),
            cloud_cover_pct=2.5,
            quality_score=0.92,
            ingestion_status="COMPLETED",
            tile_count=2,
            embedded_count=0,
            bbox_west=77.10,
            bbox_south=28.50,
            bbox_east=77.30,
            bbox_north=28.70,
        )
        session.add(scene)

        # Create temporary tile images on disk
        test_tiles_dir = settings.tiles_dir / str(scene_id)
        test_tiles_dir.mkdir(parents=True, exist_ok=True)
        test_thumbs_dir = settings.thumbnails_dir / str(scene_id)
        test_thumbs_dir.mkdir(parents=True, exist_ok=True)

        tiles: list[Tile] = []
        for i in range(2):
            tile_id = uuid.uuid4()

            # Create synthetic PNG thumbnail
            thumb_path = test_thumbs_dir / f"thumb_{i}.png"
            img = Image.new("RGB", (128, 128), color=(30 + i * 40, 80 + i * 30, 50))
            img.save(thumb_path)

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
                center_lon=77.15 + i * 0.1,
                center_lat=28.55,
                bbox_west=77.10 + i * 0.1,
                bbox_south=28.50,
                bbox_east=77.20 + i * 0.1,
                bbox_north=28.60,
                tile_path=None,
                thumbnail_path=str(thumb_path),
                acquisition_date=scene.acquisition_date,
                sensor=scene.sensor,
                resolution_m=10.0,
                quality_score=0.90 + i * 0.05,
                cloud_cover_pct=1.0,
                is_valid=True,
            )
            session.add(tile)
            tiles.append(tile)

        await session.commit()

        yield scene_id, tiles

        # Cleanup test scene and tiles
        async with AsyncSessionLocal() as cleanup_session:
            sc = await cleanup_session.get(Scene, scene_id)
            if sc:
                await cleanup_session.delete(sc)
                await cleanup_session.commit()

        # Clean Qdrant points
        vector_store.delete_scene_points(str(scene_id))


@pytest.mark.asyncio
class TestEmbeddingPipeline:
    """Core end-to-end tile embedding and vector indexing tests."""

    async def test_real_tiles_generate_embeddings_and_appear_in_qdrant(
        self, sample_scene_with_tiles
    ):
        scene_id, tiles = sample_scene_with_tiles

        async with AsyncSessionLocal() as session:
            result = await tile_embedding_service.generate_scene_embeddings(
                scene_id=scene_id,
                session=session,
                force_reembed=True,
                batch_size=4,
            )

        # 1. Verify embedding generation results
        assert result["status"] == "COMPLETED"
        assert result["total_tiles"] == 2
        assert result["embedded_count"] == 2
        assert result["duration_seconds"] is not None
        assert result["duration_seconds"] > 0

        # 2. Verify vectors appear in Qdrant collections
        clip_coll = settings.qdrant_remoteclip_collection
        dino_coll = settings.qdrant_dino_collection

        tile_id_str = str(tiles[0].id)

        # Check RemoteCLIP collection
        records_clip = vector_store.client.retrieve(
            collection_name=clip_coll,
            ids=[tile_id_str],
            with_vectors=True,
            with_payload=True,
        )
        assert len(records_clip) == 1
        assert len(records_clip[0].vector) == 512

        # Check DINOv2 collection
        records_dino = vector_store.client.retrieve(
            collection_name=dino_coll,
            ids=[tile_id_str],
            with_vectors=True,
            with_payload=True,
        )
        assert len(records_dino) == 1
        assert len(records_dino[0].vector) == 384

    async def test_metadata_payload_correctness(self, sample_scene_with_tiles):
        scene_id, tiles = sample_scene_with_tiles

        async with AsyncSessionLocal() as session:
            await tile_embedding_service.generate_scene_embeddings(
                scene_id=scene_id,
                session=session,
                force_reembed=True,
            )

        tile_id_str = str(tiles[0].id)
        record = vector_store.client.retrieve(
            collection_name=settings.qdrant_remoteclip_collection,
            ids=[tile_id_str],
            with_payload=True,
        )[0]

        payload = record.payload
        assert payload is not None

        # Verify all requirements
        assert payload["tile_id"] == tile_id_str
        assert payload["scene_id"] == str(scene_id)
        assert payload["sensor"] == "Sentinel-2B"
        assert "acquisition_date" in payload
        assert payload["quality_score"] is not None
        assert payload["model_name"] == "RemoteCLIP"
        assert payload["model_version"] == "ViT-B-32"

        # Verify geographic geometry
        assert "geometry" in payload
        geom = payload["geometry"]
        assert geom["type"] == "Polygon"
        assert len(geom["coordinates"][0]) == 5  # closed bounding ring

    async def test_rerun_does_not_create_accidental_duplicates(self, sample_scene_with_tiles):
        scene_id, tiles = sample_scene_with_tiles

        clip_coll = settings.qdrant_remoteclip_collection
        stats_initial = vector_store.get_collection_stats(clip_coll)
        initial_points = stats_initial["points_count"]

        async with AsyncSessionLocal() as session:
            # First run
            res1 = await tile_embedding_service.generate_scene_embeddings(
                scene_id=scene_id,
                session=session,
                force_reembed=True,
            )
            assert res1["embedded_count"] == 2

            stats_after_first = vector_store.get_collection_stats(clip_coll)
            points_after_first = stats_after_first["points_count"]
            assert points_after_first == initial_points + 2

            # Re-run with force_reembed=True (must overwrite in-place, NOT duplicate)
            res2 = await tile_embedding_service.generate_scene_embeddings(
                scene_id=scene_id,
                session=session,
                force_reembed=True,
            )
            assert res2["embedded_count"] == 2

            stats_after_rerun = vector_store.get_collection_stats(clip_coll)
            assert stats_after_rerun["points_count"] == points_after_first  # EXACT SAME COUNT

            # Re-run with force_reembed=False (incremental: skips all)
            res3 = await tile_embedding_service.generate_scene_embeddings(
                scene_id=scene_id,
                session=session,
                force_reembed=False,
            )
            assert res3["skipped_count"] == 2

    async def test_incremental_addition_without_rebuilding(self, sample_scene_with_tiles):
        scene_id_1, tiles_1 = sample_scene_with_tiles

        async with AsyncSessionLocal() as session:
            await tile_embedding_service.generate_scene_embeddings(
                scene_id=scene_id_1,
                session=session,
                force_reembed=True,
            )

        # Create a second scene
        async with AsyncSessionLocal() as session:
            scene_id_2 = uuid.uuid4()
            scene2 = Scene(
                id=scene_id_2,
                source_path=str(settings.data_dir / "test_scene_2.tif"),
                filename="test_scene_2.tif",
                sensor="Landsat-9",
                ingestion_status="COMPLETED",
                tile_count=1,
            )
            session.add(scene2)

            thumb_dir = settings.thumbnails_dir / str(scene_id_2)
            thumb_dir.mkdir(parents=True, exist_ok=True)
            thumb_path = thumb_dir / "thumb_0.png"
            Image.new("RGB", (128, 128), color=(20, 60, 100)).save(thumb_path)

            tile2 = Tile(
                id=uuid.uuid4(),
                scene_id=scene_id_2,
                tile_col=0,
                tile_row=0,
                tile_size=512,
                pixel_x_off=0,
                pixel_y_off=0,
                pixel_width=512,
                pixel_height=512,
                thumbnail_path=str(thumb_path),
                sensor="Landsat-9",
                is_valid=True,
            )
            session.add(tile2)
            await session.commit()

            # Embed scene 2 incrementally
            res = await tile_embedding_service.generate_scene_embeddings(
                scene_id=scene_id_2,
                session=session,
                force_reembed=False,
            )
            assert res["embedded_count"] == 1

            # Verify vectors from scene 1 are still present
            assert vector_store.point_exists(
                settings.qdrant_remoteclip_collection, str(tiles_1[0].id)
            )
            # And scene 2 point exists
            assert vector_store.point_exists(
                settings.qdrant_remoteclip_collection, str(tile2.id)
            )

            # Cleanup scene 2
            sc2 = await session.get(Scene, scene_id_2)
            if sc2:
                await session.delete(sc2)
                await session.commit()
            vector_store.delete_scene_points(str(scene_id_2))

    async def test_provenance_records_created(self, sample_scene_with_tiles):
        scene_id, tiles = sample_scene_with_tiles

        async with AsyncSessionLocal() as session:
            await tile_embedding_service.generate_scene_embeddings(
                scene_id=scene_id,
                session=session,
                force_reembed=True,
            )

            # Check tile provenance
            stmt_tile = select(ProvenanceRecord).where(
                ProvenanceRecord.entity_id == tiles[0].id,
                ProvenanceRecord.operation == "embedding",
            )
            res_tile = await session.execute(stmt_tile)
            tile_record = res_tile.scalars().first()
            assert tile_record is not None
            assert tile_record.details["remoteclip_dim"] == 512
            assert tile_record.details["dino_dim"] == 384

            # Check scene batch provenance
            stmt_scene = select(ProvenanceRecord).where(
                ProvenanceRecord.entity_id == scene_id,
                ProvenanceRecord.operation == "embedding_batch",
            )
            res_scene = await session.execute(stmt_scene)
            scene_record = res_scene.scalars().first()
            assert scene_record is not None
            assert scene_record.details["tiles_embedded"] == 2


class TestEmbeddingAPIEndpoints:
    """REST API endpoints verification."""

    def test_vector_statistics_endpoint(self, client: TestClient):
        res = client.get("/vector/statistics")
        assert res.status_code == 200
        data = res.json()

        assert "total_vectors" in data
        assert "collections" in data
        assert "storage_usage_bytes" in data
        assert "storage_usage_mb" in data
        assert data["storage_usage_mb"] >= 0.0
        assert "remoteclip_tiles" in data["collections"]
        assert "dino_tiles" in data["collections"]

        clip = data["collections"]["remoteclip_tiles"]
        assert clip["vector_size"] == 512
        assert clip["status"] == "green"

        dino = data["collections"]["dino_tiles"]
        assert dino["vector_size"] == 384
        assert dino["status"] == "green"

    def test_embedding_status_endpoint(self, client: TestClient):
        # Using existing scene id from database
        res = client.get("/embedding/status/9374488b-bd32-4311-9507-0f9c79e6485c")
        assert res.status_code == 200
        data = res.json()
        assert data["scene_id"] == "9374488b-bd32-4311-9507-0f9c79e6485c"
        assert "total_tiles" in data
        assert "embedded_tiles" in data
        assert "progress_pct" in data
        assert "status" in data
        assert "models" in data
        assert "collections" in data

    def test_embedding_generate_sync_endpoint(self, client: TestClient):
        res = client.post(
            "/embedding/generate/9374488b-bd32-4311-9507-0f9c79e6485c?sync=true"
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "COMPLETED"
        assert data["scene_id"] == "9374488b-bd32-4311-9507-0f9c79e6485c"
        assert data["total_tiles"] >= 9
        assert data["embedded_count"] >= 9
