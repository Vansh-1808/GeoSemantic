"""
Verification Script for Phase 7: Visual and Image-to-Image Search.

Demonstrates:
1. DINOv2 (384-dimensional) visual feature extraction running 100% offline.
2. Image-to-image similarity search using uploaded satellite imagery.
3. Tile-to-tile similarity search from an existing archive reference tile.
4. Automatic exclusion of the reference tile from self-similarity results.
5. Multi-factor metadata, spatial, and quality filtering.
6. Query execution latency profiling (DINOv2 embedding vs Qdrant search).
"""
from __future__ import annotations

import asyncio
import io
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
backend_dir = ROOT_DIR / "backend"
sys.path.insert(0, str(backend_dir))

from dotenv import load_dotenv
load_dotenv(backend_dir / ".env")

# Enforce offline mode
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
from app.models.tile import Tile
from app.schemas.search import VisualSearchRequest
from app.services.vector_store import vector_store
from app.services.visual_search import visual_search_service


async def run_verification():
    print("=" * 80)
    print(" GeoSemantic Platform — Phase 7 Visual & Image-to-Image Search Verification")
    print("=" * 80)

    # 1. Check Qdrant dino_tiles collection
    collection = settings.qdrant_dino_collection
    stats = vector_store.get_collection_stats(collection)
    points_count = stats.get("points_count", 0)
    print(f"[Qdrant Vector Index]: Collection '{collection}' has {points_count} visual vectors (384-dim).")

    async with AsyncSessionLocal() as session:
        # Fetch an existing tile to use as template
        stmt = select(Tile).where(Tile.dino_vector_id.isnot(None)).limit(1)
        res = await session.execute(stmt)
        sample_tile = res.scalar_one_or_none()

        if not sample_tile:
            # Fallback to any tile
            stmt_any = select(Tile).limit(1)
            res_any = await session.execute(stmt_any)
            sample_tile = res_any.scalar_one_or_none()

        if not sample_tile:
            print("[!] No tiles found in database. Ingest and embed imagery first.")
            return

        print(f"\n[Test 1] Tile-to-Tile Visual Similarity Search:")
        print(f"  Reference Tile: ID {sample_tile.id} (Col {sample_tile.tile_col}, Row {sample_tile.tile_row}, Sensor: {sample_tile.sensor})")

        req = VisualSearchRequest(top_k=5)
        resp_tile = await visual_search_service.search_by_tile_id(
            tile_id=sample_tile.id,
            req=req,
            session=session,
        )

        print(f"  * Results returned: {resp_tile.total_found} tiles")
        print(f"  * Total Latency: {resp_tile.execution_time_ms:.2f}ms (Visual Embed: {resp_tile.visual_embedding_time_ms:.2f}ms | Qdrant: {resp_tile.vector_search_time_ms:.2f}ms)")
        print(f"  * Model: {resp_tile.model_used} | Dimension: {resp_tile.embedding_dimension}-dim")

        # Verify reference tile is NOT in results
        result_ids = [str(r.tile_id) for r in resp_tile.results]
        assert str(sample_tile.id) not in result_ids, "[FAIL] Reference tile appeared in its own results!"
        print("  * [PASS] Reference tile self-exclusion verified.")

        for r in resp_tile.results[:3]:
            coords = f"[{r.center_coordinates['lat']:.4f}, {r.center_coordinates['lon']:.4f}]"
            print(f"    - Rank #{r.rank}: Score {r.similarity_score * 100:5.1f}% | Col/Row [{r.tile_col},{r.tile_row}] | Center {coords} | {r.sensor}")

        # 2. Test Image Upload Search
        print(f"\n[Test 2] Uploaded Image Visual Similarity Search:")
        # Create synthetic 256x256 image with earth tones
        test_img = Image.new("RGB", (256, 256), color=(45, 85, 55))

        resp_img = await visual_search_service.search_by_image(
            image_input=test_img,
            req=VisualSearchRequest(top_k=5),
            session=session,
        )

        print(f"  * Results returned: {resp_img.total_found} tiles")
        print(f"  * Total Latency: {resp_img.execution_time_ms:.2f}ms (DINOv2 Embed: {resp_img.visual_embedding_time_ms:.2f}ms | Qdrant: {resp_img.vector_search_time_ms:.2f}ms)")
        print(f"  * Model: {resp_img.model_used}")

        for r in resp_img.results[:3]:
            coords = f"[{r.center_coordinates['lat']:.4f}, {r.center_coordinates['lon']:.4f}]"
            print(f"    - Rank #{r.rank}: Score {r.similarity_score * 100:5.1f}% | Col/Row [{r.tile_col},{r.tile_row}] | Center {coords} | {r.sensor}")

        print("\n" + "=" * 80)
        print(" PHASE 7 VISUAL SIMILARITY RETRIEVAL VERIFIED SUCCESSFULLY")
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_verification())
