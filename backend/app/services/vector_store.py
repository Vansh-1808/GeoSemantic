"""
VectorStoreService: Centralized Qdrant vector database management layer.
Supports both embedded local directory storage and Qdrant server mode.

Collections:
1. 'remoteclip_tiles': 512-dimensional semantic embeddings (Cosine distance).
2. 'dino_tiles': 384-dimensional visual similarity embeddings (Cosine distance).

Strict Singleton:
Avoids OS file lock collisions on Windows when Qdrant operates in local directory mode.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class VectorStoreService:
    """
    Singleton service managing Qdrant vector database collections and operations.
    """

    _instance: Optional["VectorStoreService"] = None

    def __new__(cls) -> "VectorStoreService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self._initialized = True
        self._client: Optional[QdrantClient] = None
        self._initialize_client()
        self.ensure_collections()
        logger.info(
            "vector_store_initialized",
            mode=settings.qdrant_mode,
            path=str(settings.qdrant_local_path) if settings.qdrant_mode == "local" else None,
        )

    def _initialize_client(self) -> None:
        """Initialize Qdrant client based on configuration."""
        if settings.qdrant_mode == "local":
            storage_path = settings.qdrant_local_path
            storage_path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(storage_path))
        else:
            self._client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
            )

    @property
    def client(self) -> QdrantClient:
        """Access the underlying QdrantClient."""
        if self._client is None:
            self._initialize_client()
        return self._client

    def ensure_collections(self) -> None:
        """Ensure both semantic and visual similarity collections exist."""
        client = self.client

        # 1. Semantic embeddings (RemoteCLIP, 512 dimensions)
        remoteclip_coll = settings.qdrant_remoteclip_collection
        if not client.collection_exists(remoteclip_coll):
            client.create_collection(
                collection_name=remoteclip_coll,
                vectors_config=models.VectorParams(
                    size=512,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info("qdrant_collection_created", collection=remoteclip_coll, dim=512)

        # 2. Visual similarity embeddings (DINOv2, 384 dimensions)
        dino_coll = settings.qdrant_dino_collection
        if not client.collection_exists(dino_coll):
            client.create_collection(
                collection_name=dino_coll,
                vectors_config=models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info("qdrant_collection_created", collection=dino_coll, dim=384)

    def upsert_tile_vectors(
        self,
        collection_name: str,
        points: List[models.PointStruct],
    ) -> None:
        """
        Batch upsert vectors into a specific collection.
        Uses tile UUID string as point ID to ensure idempotency and prevent duplicates.
        """
        if not points:
            return

        self.client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )
        logger.debug(
            "vectors_upserted",
            collection=collection_name,
            count=len(points),
        )

    def point_exists(self, collection_name: str, point_id: str) -> bool:
        """Check if a specific tile vector already exists in a collection."""
        try:
            records = self.client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
            )
            return len(records) > 0
        except Exception:
            return False

    def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 20,
        query_filter: Optional[models.Filter] = None,
        with_payload: bool = True,
    ) -> List[Any]:
        """
        Execute similarity search on a collection with optional multi-condition filtering.
        Returns list of ScoredPoint objects sorted by similarity score descending.
        """
        if not self.client.collection_exists(collection_name):
            return []

        res = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=with_payload,
        )
        return res.points

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get telemetry and statistics for a specific collection."""
        try:
            if not self.client.collection_exists(collection_name):
                return {
                    "collection_name": collection_name,
                    "exists": False,
                    "points_count": 0,
                    "status": "not_found",
                }

            info = self.client.get_collection(collection_name)
            points_count = info.points_count or 0
            vectors_count = info.vectors_count if info.vectors_count is not None else points_count

            # Extract vector size from collection config
            vector_size = 512
            distance = "Cosine"
            if hasattr(info.config.params, "vectors") and isinstance(info.config.params.vectors, models.VectorParams):
                vector_size = info.config.params.vectors.size
                distance = info.config.params.vectors.distance.name

            return {
                "collection_name": collection_name,
                "exists": True,
                "points_count": points_count,
                "vectors_count": vectors_count,
                "vector_size": vector_size,
                "distance": distance,
                "status": str(info.status),
            }
        except Exception as exc:
            logger.error("get_collection_stats_failed", collection=collection_name, error=str(exc))
            return {
                "collection_name": collection_name,
                "exists": False,
                "points_count": 0,
                "error": str(exc),
                "status": "error",
            }

    def get_all_statistics(self) -> Dict[str, Any]:
        """
        Consolidated vector store telemetry across all collections,
        measuring vector counts and disk storage usage.
        """
        remoteclip_stats = self.get_collection_stats(settings.qdrant_remoteclip_collection)
        dino_stats = self.get_collection_stats(settings.qdrant_dino_collection)

        total_points = remoteclip_stats.get("points_count", 0) + dino_stats.get("points_count", 0)
        total_vectors = total_points

        # Calculate local disk storage usage
        storage_bytes = 0
        storage_path_str = str(settings.qdrant_local_path)
        if settings.qdrant_mode == "local" and settings.qdrant_local_path.exists():
            for root, _, files in os.walk(settings.qdrant_local_path):
                for f in files:
                    try:
                        storage_bytes += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass

        storage_mb = round(storage_bytes / (1024 * 1024), 2)

        return {
            "total_vectors": total_vectors,
            "total_points": total_points,
            "collections": {
                settings.qdrant_remoteclip_collection: remoteclip_stats,
                settings.qdrant_dino_collection: dino_stats,
            },
            "storage_usage_bytes": storage_bytes,
            "storage_usage_mb": storage_mb,
            "storage_path": storage_path_str,
            "qdrant_mode": settings.qdrant_mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def delete_scene_points(self, scene_id: str) -> None:
        """
        Remove all vectors associated with a scene across all collections.
        """
        filter_selector = models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="scene_id",
                        match=models.MatchValue(value=scene_id),
                    )
                ]
            )
        )
        for coll in [settings.qdrant_remoteclip_collection, settings.qdrant_dino_collection]:
            try:
                if self.client.collection_exists(coll):
                    self.client.delete(
                        collection_name=coll,
                        points_selector=filter_selector,
                        wait=True,
                    )
            except Exception as exc:
                logger.warning("delete_scene_points_failed", collection=coll, error=str(exc))

    def prune_orphaned_vectors(self, active_tile_ids: set[str]) -> dict[str, int]:
        """
        Scan all collections in Qdrant and delete any points whose point IDs
        are not present in the active PostgreSQL database.
        """
        pruned_counts: dict[str, int] = {}
        for coll in [settings.qdrant_remoteclip_collection, settings.qdrant_dino_collection]:
            if not self.client.collection_exists(coll):
                continue

            orphaned_ids: list[str] = []
            offset = None
            while True:
                records, next_offset = self.client.scroll(
                    collection_name=coll,
                    limit=250,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
                for rec in records:
                    if str(rec.id) not in active_tile_ids:
                        orphaned_ids.append(str(rec.id))

                if next_offset is None or not records:
                    break
                offset = next_offset

            if orphaned_ids:
                chunk_size = 250
                for i in range(0, len(orphaned_ids), chunk_size):
                    chunk = orphaned_ids[i : i + chunk_size]
                    self.client.delete(
                        collection_name=coll,
                        points_selector=models.PointIdsList(points=chunk),
                        wait=True,
                    )
                logger.info("pruned_orphaned_vectors", collection=coll, count=len(orphaned_ids))
            pruned_counts[coll] = len(orphaned_ids)

        return pruned_counts

    def close(self) -> None:
        """Clean up Qdrant client connection and release local file lock."""
        if self._client is not None:
            try:
                self._client.close()
                logger.info("vector_store_closed")
            except Exception:
                pass
            self._client = None
            self._initialized = False
            VectorStoreService._instance = None


# Module-level singleton
vector_store = VectorStoreService()
