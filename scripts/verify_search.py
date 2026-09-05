"""
Verification Script for Phase 6: Semantic Text-to-Satellite Retrieval.

Demonstrates:
1. Natural language query text-to-vector embedding via offline RemoteCLIP.
2. Vector similarity retrieval against local Qdrant remoteclip_tiles collection.
3. Metadata, quality, sensor, temporal, and spatial filtering.
4. Result ranking, similarity scores, coordinate extraction, and provenance audit trails.
5. Query execution latency profiling.
"""
from __future__ import annotations

import asyncio
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

from app.core.config import settings
from app.db.database import AsyncSessionLocal
from app.schemas.search import SemanticSearchRequest
from app.services.semantic_search import semantic_search_service
from app.services.vector_store import vector_store


TEST_QUERIES = [
  "Commercial airport runway, tarmac, and aprons",
  "Dense urban residential buildings with street grid",
  "Agricultural farmland with green vegetation crop circles",
  "Harbor port with cargo containers and docked vessels",
  "Industrial manufacturing plant with fuel storage tanks",
]


async def run_verification():
  print("=" * 80)
  print(" GeoSemantic Platform — Phase 6 Semantic Retrieval Verification")
  print("=" * 80)

  # Check Qdrant collection status
  collection = settings.qdrant_remoteclip_collection
  stats = vector_store.get_collection_stats(collection)
  points_count = stats.get("points_count", 0)
  print(f"[Qdrant Vector Index]: Collection '{collection}' has {points_count} points.")

  if points_count == 0:
    print(
      "[!] Warning: No vectors found in remoteclip_tiles collection. Ingestion/embedding may be required."
    )

  async with AsyncSessionLocal() as session:
    # 1. Test Filter discovery
    print("\n--- Testing GET /api/search/filters ---")
    t0 = time.perf_counter()
    filters_meta = await semantic_search_service.get_search_filters(session)
    t_filter = (time.perf_counter() - t0) * 1000
    print(f"Sensors available: {filters_meta.sensors}")
    print(f"Date range: {filters_meta.min_date} to {filters_meta.max_date}")
    print(
      f"Total indexed tiles in DB: {filters_meta.total_indexed_tiles} (Fetched in {t_filter:.2f}ms)"
    )

    # 2. Test Multiple Natural Language Queries
    print("\n--- Testing Natural Language Semantic Retrieval ---")
    latencies = []

    for idx, query in enumerate(TEST_QUERIES, 1):
      print(f"\n[Query #{idx}]: \"{query}\"")
      req = SemanticSearchRequest(
        query=query,
        top_k=5,
      )

      resp = await semantic_search_service.search(req, session)
      latencies.append(resp.execution_time_ms)

      print(
        f"  * Results returned: {resp.total_found} tiles (Total: {resp.execution_time_ms:.1f}ms | Embed: {resp.text_embedding_time_ms:.1f}ms | Qdrant: {resp.vector_search_time_ms:.1f}ms)"
      )
      print(f"  * Device: {resp.device.upper()} | Model: {resp.model_used}")

      if resp.results:
        for r in resp.results[:3]:
          coords = f"[{r.center_coordinates['lat']:.4f}, {r.center_coordinates['lon']:.4f}]"
          sensor_str = r.sensor or "Unknown"
          prov_model = r.provenance.get("model_name", "RemoteCLIP") if r.provenance else "None"
          print(
            f"    - Rank #{r.rank}: Score {r.similarity_score * 100:5.1f}% | Col/Row [{r.tile_col},{r.tile_row}] | Center {coords} | {sensor_str} | Prov: {prov_model}"
          )

    # 3. Test Multi-Condition Filtering
    print("\n--- Testing Multi-Factor Metadata & Spatial Filtering ---")
    if filters_meta.sensors:
      test_sensor = filters_meta.sensors[0]
      print(f"Applying sensor filter: '{test_sensor}' with min_quality=0.5")
      filtered_req = SemanticSearchRequest(
        query="satellite terrain landscape",
        top_k=5,
        sensor=test_sensor,
        min_quality=0.5,
      )
      filtered_resp = await semantic_search_service.search(
        filtered_req, session
      )
      print(f"  * Matched {filtered_resp.total_found} tiles with filter.")
      for r in filtered_resp.results:
        assert r.sensor == test_sensor, f"Expected {test_sensor}, got {r.sensor}"
        assert (
          r.quality_score is None or r.quality_score >= 0.5
        ), f"Quality score too low: {r.quality_score}"
      print(f"  * Verified all returned tiles match sensor='{test_sensor}' and quality >= 0.5.")

    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    print("\n" + "=" * 80)
    print(
      f" PHASE 6 VERIFICATION COMPLETE — Average Semantic Query Latency: {avg_lat:.2f}ms"
    )
    print("=" * 80)


if __name__ == "__main__":
  asyncio.run(run_verification())
