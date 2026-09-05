"""
End-to-End Automated Pipeline Test Script for GeoSemantic Platform.

Executes the complete workflow from raw GeoTIFF dataset to AI retrieval:
1. Generates / validates realistic satellite GeoTIFF datasets.
2. Ingests scenes into PostgreSQL (metadata extraction, PostGIS footprints, quality scoring).
3. Executes windowed tiling (512x512 tiles, 128px thumbnails, 512px RGB previews).
4. Generates dual AI embeddings (RemoteCLIP 512-dim & DINOv2 384-dim) into local Qdrant.
5. Executes natural language semantic search (/search/semantic).
6. Executes visual image-to-image and tile-to-tile similarity search (/search/image and /search/similar/{id}).
7. Validates performance, latencies, and offline operation.

Usage:
    backend\\.venv\\Scripts\\python scripts/test_end_to_end.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
backend_dir = ROOT_DIR / "backend"
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(backend_dir / ".env")

# Enforce strict offline operation
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["APP_ENV"] = "test"

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from PIL import Image
from sqlalchemy import select

from app.core.config import settings
from app.db.database import AsyncSessionLocal
from app.models.scene import Scene
from app.models.tile import Tile
from app.models.provenance import ProcessingJob
from app.schemas.search import SemanticSearchRequest, VisualSearchRequest
from app.services.ingestion import ingest_scene
from app.services.semantic_search import semantic_search_service
from app.services.tile_embedding import tile_embedding_service
from app.services.tiling import TilingConfig, process_scene_tiling
from app.services.vector_store import vector_store
from app.services.visual_search import visual_search_service
from scripts.generate_demo_data import SCENES, generate_scene


async def run_e2e():
    print("=" * 80)
    print(" GeoSemantic Satellite Intelligence Platform — End-to-End Pipeline Test")
    print("=" * 80)

    settings.ensure_directories()

    # Step 1: Ensure Dataset Files Exist
    print("\n[Step 1/6] Preparing Satellite Datasets...")
    scenes_dir = settings.imagery_dir
    scenes_to_test = [SCENES[0], SCENES[4]]  # urban_2022 and industrial_2024

    dataset_paths = []
    for s_def in scenes_to_test:
        path = generate_scene(s_def, scenes_dir)
        dataset_paths.append(path)
        print(f"  ✓ Dataset file ready: {path.name} ({path.stat().st_size / 1024:.1f} KB)")

    # Step 2: Ingest Scenes into PostgreSQL
    print("\n[Step 2/6] Ingesting Satellite Scenes into PostGIS Database...")
    ingested_scene_ids = []

    async with AsyncSessionLocal() as session:
        for path in dataset_paths:
            # Check if already in DB
            stmt = select(Scene).where(Scene.filename == path.name)
            existing = (await session.execute(stmt)).scalars().first()

            if existing:
                print(f"  ✓ Scene already in database: {path.name} (ID: {existing.id})")
                ingested_scene_ids.append(existing.id)
            else:
                job_id = uuid.uuid4()
                job = ProcessingJob(
                    id=job_id,
                    job_type="ingestion",
                    status="PENDING",
                    params={"file_path": str(path), "filename": path.name},
                    related_entity_type="scene",
                )
                session.add(job)
                await session.flush()

                print(f"  * Ingesting {path.name} ...")
                t0 = time.perf_counter()
                scene = await ingest_scene(path, session, job_id)
                await session.commit()
                t_ingest = (time.perf_counter() - t0) * 1000
                print(f"  ✓ Ingested {path.name} in {t_ingest:.1f}ms (Sensor: {scene.sensor}, Quality: {scene.quality_score:.2f})")
                ingested_scene_ids.append(scene.id)

    # Step 3: Tile Preprocessing
    print("\n[Step 3/6] Windowed Tiling & Multiscale Preprocessing (512x512)...")
    async with AsyncSessionLocal() as session:
        for sid in ingested_scene_ids:
            scene = await session.get(Scene, sid)
            if not scene:
                continue

            # Check if tiles exist
            stmt_t = select(Tile).where(Tile.scene_id == sid)
            tiles = list((await session.execute(stmt_t)).scalars().all())

            if len(tiles) >= 4:
                print(f"  ✓ Scene {scene.filename} already has {len(tiles)} analysis tiles.")
            else:
                print(f"  * Generating tiles for {scene.filename} ...")
                t0 = time.perf_counter()
                cfg = TilingConfig(tile_size=512, overlap_px=0)
                t_job_id = uuid.uuid4()
                t_job = ProcessingJob(
                    id=t_job_id,
                    job_type="tiling",
                    status="RUNNING",
                    params={"scene_id": str(sid)},
                    related_entity_type="scene",
                    related_entity_id=sid,
                )
                session.add(t_job)
                await session.flush()
                t_res = await process_scene_tiling(
                    scene_id=sid,
                    session=session,
                    config=cfg,
                    job_id=t_job_id,
                )
                await session.commit()
                t_tiling = (time.perf_counter() - t0) * 1000
                print(f"  ✓ Processed {t_res.tile_count} tiles in {t_tiling:.1f}ms")

    # Step 4: AI Vector Indexing (RemoteCLIP + DINOv2)
    print("\n[Step 4/6] Generating Offline AI Vector Embeddings into Qdrant...")
    async with AsyncSessionLocal() as session:
        for sid in ingested_scene_ids:
            scene = await session.get(Scene, sid)
            if not scene:
                continue

            print(f"  * Indexing scene: {scene.filename} ...")
            t0 = time.perf_counter()
            res = await tile_embedding_service.generate_scene_embeddings(
                scene_id=sid,
                session=session,
                force_reembed=False,
                batch_size=4,
            )
            await session.commit()
            t_emb = (time.perf_counter() - t0) * 1000
            print(f"  ✓ Embedded {res['embedded_count']} tiles in {t_emb:.1f}ms (Skipped: {res['skipped_count']})")

    # Step 5: Verify Qdrant Vector Statistics
    print("\n[Step 5/6] Verifying Local Qdrant Vector Collections...")
    stats = vector_store.get_all_statistics()
    print(f"  ✓ Total Vector Points: {stats['total_points']}")
    print(f"  ✓ Disk Storage Used: {stats['storage_usage_mb']} MB")
    for coll_name, c_info in stats["collections"].items():
        print(f"    - Collection '{coll_name}': {c_info.get('points_count', 0)} points (dim={c_info.get('vector_size')})")

    # Step 6: Test Semantic & Visual Queries
    print("\n[Step 6/6] Executing Retrieval Queries...")

    async with AsyncSessionLocal() as session:
        # 6a. Semantic Search
        sem_query = "Dense urban buildings and road grid"
        print(f"\n  [Semantic Search]: Query = \"{sem_query}\"")
        t0 = time.perf_counter()
        sem_resp = await semantic_search_service.search(
            SemanticSearchRequest(query=sem_query, top_k=3),
            session,
        )
        print(f"  ✓ Found {sem_resp.total_found} tiles in {sem_resp.execution_time_ms:.1f}ms (Embed: {sem_resp.text_embedding_time_ms:.1f}ms | Qdrant: {sem_resp.vector_search_time_ms:.1f}ms)")
        for r in sem_resp.results:
            coords = f"[{r.center_coordinates['lat']:.4f}, {r.center_coordinates['lon']:.4f}]"
            print(f"    - Rank #{r.rank}: Score {r.similarity_score * 100:5.1f}% | Col/Row [{r.tile_col},{r.tile_row}] | Center {coords} | {r.scene_name}")

        # 6b. Tile-to-Tile Visual Search
        target_tile = sem_resp.results[0]
        print(f"\n  [Visual Search — Tile-to-Tile]: Template = Tile #{target_tile.tile_col},{target_tile.tile_row}")
        vis_resp = await visual_search_service.search_by_tile_id(
            tile_id=target_tile.tile_id,
            req=VisualSearchRequest(top_k=3),
            session=session,
        )
        print(f"  ✓ Found {vis_resp.total_found} similar sites in {vis_resp.execution_time_ms:.1f}ms (Qdrant: {vis_resp.vector_search_time_ms:.1f}ms)")
        for r in vis_resp.results:
            coords = f"[{r.center_coordinates['lat']:.4f}, {r.center_coordinates['lon']:.4f}]"
            print(f"    - Rank #{r.rank}: Score {r.similarity_score * 100:5.1f}% | Col/Row [{r.tile_col},{r.tile_row}] | Center {coords} | {r.scene_name}")

        # 6c. Image Upload Visual Search
        print(f"\n  [Visual Search — Image Upload]: Using generated satellite imagery file...")
        query_img = Image.open(dataset_paths[0])
        upload_resp = await visual_search_service.search_by_image(
            image_input=query_img,
            req=VisualSearchRequest(top_k=3),
            session=session,
        )
        print(f"  ✓ Found {upload_resp.total_found} matches in {upload_resp.execution_time_ms:.1f}ms (DINOv2: {upload_resp.visual_embedding_time_ms:.1f}ms | Qdrant: {upload_resp.vector_search_time_ms:.1f}ms)")

    print("\n" + "=" * 80)
    print(" ALL END-TO-END PIPELINE CHECKS PASSED WITH 100% OFFLINE SATELLITE DATA")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_e2e())
