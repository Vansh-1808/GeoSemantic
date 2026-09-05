"""
TileEmbeddingService: Orchestrates end-to-end embedding generation for satellite tiles.
Pipeline:
  Satellite Tile → Load Image (GeoTIFF/PNG) → Remote Sensing Models (RemoteCLIP + DINOv2) →
  Vector Normalization → Qdrant Collections → PostgreSQL Provenance & Audit Records

Guarantees:
- Idempotency & Zero Duplicate Vectors (Point ID == tile.id UUID string).
- Incremental indexing (skips already-embedded tiles unless force_reembed is True).
- Full geospatial metadata linkage in vector payloads.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from qdrant_client import models
import rasterio
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.models.provenance import ProcessingJob, ProvenanceRecord
from app.models.scene import Scene
from app.models.tile import Tile
from app.services.embedding_service import embedding_service
from app.services.vector_store import vector_store

logger = get_logger(__name__)


class TileEmbeddingService:
    """
    Coordinates tile embedding extraction, vector store persistence,
    and database provenance tracking.
    """

    def __init__(self) -> None:
        self.embedding = embedding_service
        self.vector_store = vector_store

    def load_tile_image(self, tile: Tile) -> Image.Image:
        """
        Load tile imagery into a standard RGB PIL Image.
        Priority:
        1. Original GeoTIFF tile (converts float32/uint16 multispectral/RGB to uint8 RGB).
        2. Generated PNG thumbnail (fallback).
        """
        # 1. Try reading the GeoTIFF tile raster
        if tile.tile_path and Path(tile.tile_path).exists():
            try:
                with rasterio.open(tile.tile_path) as src:
                    arr = src.read()  # (C, H, W)
                    if arr.ndim == 3 and arr.shape[0] > 0:
                        # Extract first 3 bands as RGB (or duplicate if 1 band)
                        if arr.shape[0] >= 3:
                            rgb = arr[:3, :, :]
                        elif arr.shape[0] == 1:
                            rgb = np.repeat(arr, 3, axis=0)
                        else:
                            rgb = arr[:3, :, :]

                        # Normalize to 0-255 uint8
                        if rgb.dtype in (np.float32, np.float64):
                            if rgb.max() <= 1.0:
                                rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
                            else:
                                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                        elif rgb.dtype != np.uint8:
                            p2, p98 = np.percentile(rgb, (2, 98))
                            if p98 > p2:
                                rgb = ((rgb - p2) / (p98 - p2) * 255.0).clip(0, 255).astype(np.uint8)
                            else:
                                rgb = np.clip(rgb, 0, 255).astype(np.uint8)

                        # Reorder to (H, W, C)
                        rgb = np.transpose(rgb, (1, 2, 0))
                        return Image.fromarray(rgb, mode="RGB")
            except Exception as exc:
                logger.warning(
                    "geotiff_tile_read_failed_trying_thumbnail",
                    tile_id=str(tile.id),
                    error=str(exc),
                )

        # 2. Fallback to thumbnail PNG
        if tile.thumbnail_path and Path(tile.thumbnail_path).exists():
            return Image.open(tile.thumbnail_path).convert("RGB")

        raise FileNotFoundError(
            f"No valid image file found for tile {tile.id} (path={tile.tile_path}, thumb={tile.thumbnail_path})"
        )

    async def generate_scene_embeddings(
        self,
        scene_id: uuid.UUID,
        session: AsyncSession,
        force_reembed: bool = False,
        batch_size: int = 8,
        job_id: Optional[uuid.UUID] = None,
        progress_callback: Optional[Callable[[float, str], Any]] = None,
    ) -> Dict[str, Any]:
        """
        Embed all tiles belonging to a scene and index into Qdrant.
        Supports batch processing, incremental insertion, and provenance records.
        """
        start_time = time.perf_counter()

        # Step 1: Validate Scene
        scene = await session.get(Scene, scene_id)
        if not scene:
            raise ValueError(f"Scene {scene_id} not found")

        # Step 2: Query Scene Tiles
        stmt = (
            select(Tile)
            .where(Tile.scene_id == scene_id)
            .order_by(Tile.tile_row, Tile.tile_col)
        )
        result = await session.execute(stmt)
        all_tiles = list(result.scalars().all())

        total_tiles = len(all_tiles)
        if total_tiles == 0:
            return {
                "job_id": str(job_id or uuid.uuid4()),
                "scene_id": str(scene_id),
                "status": "COMPLETED",
                "total_tiles": 0,
                "embedded_count": 0,
                "skipped_count": 0,
                "duration_seconds": 0.0,
                "message": "Scene has no tiles to embed",
            }

        # Filter tiles for incremental embedding
        if force_reembed:
            tiles_to_process = all_tiles
            skipped_count = 0
        else:
            tiles_to_process = [
                t for t in all_tiles
                if t.remoteclip_vector_id is None or t.dino_vector_id is None
            ]
            skipped_count = total_tiles - len(tiles_to_process)

        if not tiles_to_process:
            logger.info("all_tiles_already_embedded", scene_id=str(scene_id), total=total_tiles)
            return {
                "job_id": str(job_id or uuid.uuid4()),
                "scene_id": str(scene_id),
                "status": "COMPLETED",
                "total_tiles": total_tiles,
                "embedded_count": total_tiles,
                "skipped_count": skipped_count,
                "duration_seconds": round(time.perf_counter() - start_time, 3),
                "message": f"All {total_tiles} tiles are already embedded (incremental mode)",
            }

        # Helper to report progress
        async def report(pct: float, msg: str) -> None:
            if progress_callback:
                try:
                    res = progress_callback(pct, msg)
                    if asyncio.iscoroutine(res):
                        await res
                except Exception:
                    pass

        await report(5.0, f"Starting embedding for {len(tiles_to_process)} tiles...")

        # Step 3: Process in batches
        embedded_tiles_count = 0
        now = datetime.now(timezone.utc)

        clip_collection = settings.qdrant_remoteclip_collection
        dino_collection = settings.qdrant_dino_collection

        # Ensure collections exist
        self.vector_store.ensure_collections()

        for batch_idx in range(0, len(tiles_to_process), batch_size):
            chunk = tiles_to_process[batch_idx : batch_idx + batch_size]

            # Load PIL images for chunk
            loaded_items: List[Tuple[Tile, Image.Image]] = []
            for t in chunk:
                try:
                    img = self.load_tile_image(t)
                    loaded_items.append((t, img))
                except Exception as exc:
                    logger.error("tile_image_load_failed", tile_id=str(t.id), error=str(exc))

            if not loaded_items:
                continue

            chunk_tiles = [item[0] for item in loaded_items]
            chunk_imgs = [item[1] for item in loaded_items]

            # 3a. Generate RemoteCLIP embeddings (512-dim)
            clip_embeddings = self.embedding.embed_images(
                chunk_imgs, model_name="RemoteCLIP", normalize=True
            )

            # 3b. Generate DINOv2 embeddings (384-dim)
            dino_embeddings = self.embedding.embed_images(
                chunk_imgs, model_name="DINOv2", normalize=True
            )

            clip_points: List[models.PointStruct] = []
            dino_points: List[models.PointStruct] = []

            for tile_obj, clip_vec, dino_vec in zip(chunk_tiles, clip_embeddings, dino_embeddings):
                point_id = str(tile_obj.id)

                # Base payload linking all required metadata
                base_payload = {
                    "tile_id": str(tile_obj.id),
                    "scene_id": str(tile_obj.scene_id),
                    "tile_col": tile_obj.tile_col,
                    "tile_row": tile_obj.tile_row,
                    "acquisition_date": tile_obj.acquisition_date.isoformat()
                    if tile_obj.acquisition_date
                    else None,
                    "sensor": tile_obj.sensor,
                    "quality_score": tile_obj.quality_score,
                    "cloud_cover_pct": tile_obj.cloud_cover_pct,
                    "resolution_m": tile_obj.resolution_m,
                    "bbox_west": tile_obj.bbox_west,
                    "bbox_south": tile_obj.bbox_south,
                    "bbox_east": tile_obj.bbox_east,
                    "bbox_north": tile_obj.bbox_north,
                    "center_lon": tile_obj.center_lon,
                    "center_lat": tile_obj.center_lat,
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [tile_obj.bbox_west, tile_obj.bbox_south],
                                [tile_obj.bbox_east, tile_obj.bbox_south],
                                [tile_obj.bbox_east, tile_obj.bbox_north],
                                [tile_obj.bbox_west, tile_obj.bbox_north],
                                [tile_obj.bbox_west, tile_obj.bbox_south],
                            ]
                        ],
                    }
                    if tile_obj.bbox_west is not None
                    else None,
                    "embedded_at": now.isoformat(),
                }

                # RemoteCLIP point
                clip_payload = dict(base_payload)
                clip_payload["model_name"] = "RemoteCLIP"
                clip_payload["model_version"] = "ViT-B-32"
                clip_payload["embedding_dimension"] = 512
                clip_points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=clip_vec,
                        payload=clip_payload,
                    )
                )

                # DINOv2 point
                dino_payload = dict(base_payload)
                dino_payload["model_name"] = "DINOv2"
                dino_payload["model_version"] = "vit_small_patch14_dinov2.lvd142m"
                dino_payload["embedding_dimension"] = 384
                dino_points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=dino_vec,
                        payload=dino_payload,
                    )
                )

                # Update Tile ORM record
                tile_obj.remoteclip_vector_id = point_id
                tile_obj.dino_vector_id = point_id
                tile_obj.embedding_model = "RemoteCLIP-ViT-B-32;DINOv2-vit_small_patch14_dinov2"
                tile_obj.embedded_at = now

                # Append processing history
                history = list(tile_obj.processing_history or [])
                history.append(
                    {
                        "operation": "embedding",
                        "remoteclip_dim": 512,
                        "dino_dim": 384,
                        "timestamp": now.isoformat(),
                    }
                )
                tile_obj.processing_history = history

                # Add tile-level provenance record
                provenance = ProvenanceRecord(
                    entity_id=tile_obj.id,
                    entity_type="tile",
                    operation="embedding",
                    details={
                        "scene_id": str(scene_id),
                        "tile_id": str(tile_obj.id),
                        "remoteclip_collection": clip_collection,
                        "remoteclip_dim": 512,
                        "dino_collection": dino_collection,
                        "dino_dim": 384,
                    },
                    operator="system",
                )
                session.add(provenance)

            # 3c. Upsert vectors to Qdrant
            self.vector_store.upsert_tile_vectors(clip_collection, clip_points)
            self.vector_store.upsert_tile_vectors(dino_collection, dino_points)

            embedded_tiles_count += len(chunk_tiles)
            await session.flush()

            # Progress tracking
            progress = 5.0 + (embedded_tiles_count / len(tiles_to_process)) * 85.0
            await report(
                min(90.0, progress),
                f"Embedded {embedded_tiles_count}/{len(tiles_to_process)} tiles into Qdrant...",
            )
            await asyncio.sleep(0)

        # Step 4: Update Scene statistics and create batch Provenance
        total_embedded_for_scene = await session.scalar(
            select(func.count(Tile.id)).where(
                Tile.scene_id == scene_id,
                Tile.remoteclip_vector_id.isnot(None),
            )
        )
        scene.embedded_count = total_embedded_for_scene or 0
        scene.updated_at = datetime.now(timezone.utc)

        duration = round(time.perf_counter() - start_time, 3)

        scene_provenance = ProvenanceRecord(
            entity_id=scene.id,
            entity_type="scene",
            operation="embedding_batch",
            details={
                "tiles_embedded": embedded_tiles_count,
                "tiles_skipped": skipped_count,
                "total_tiles": total_tiles,
                "remoteclip_collection": clip_collection,
                "dino_collection": dino_collection,
                "duration_seconds": duration,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            operator="system",
        )
        session.add(scene_provenance)
        await session.commit()

        await report(100.0, f"Successfully embedded {embedded_tiles_count} tiles in {duration}s")

        logger.info(
            "scene_embedding_completed",
            scene_id=str(scene_id),
            embedded_count=embedded_tiles_count,
            duration=duration,
        )

        return {
            "job_id": str(job_id or uuid.uuid4()),
            "scene_id": str(scene_id),
            "status": "COMPLETED",
            "total_tiles": total_tiles,
            "embedded_count": embedded_tiles_count,
            "skipped_count": skipped_count,
            "duration_seconds": duration,
            "message": f"Successfully indexed {embedded_tiles_count} tiles into Qdrant ({duration}s)",
        }

    async def get_scene_embedding_status(
        self,
        scene_id: uuid.UUID,
        session: AsyncSession,
    ) -> Dict[str, Any]:
        """Query real-time embedding status for a scene."""
        scene = await session.get(Scene, scene_id)
        if not scene:
            raise ValueError(f"Scene {scene_id} not found")

        total_tiles = await session.scalar(
            select(func.count(Tile.id)).where(Tile.scene_id == scene_id)
        ) or 0

        embedded_tiles = await session.scalar(
            select(func.count(Tile.id)).where(
                Tile.scene_id == scene_id,
                Tile.remoteclip_vector_id.isnot(None),
            )
        ) or 0

        pending_tiles = max(0, total_tiles - embedded_tiles)
        progress_pct = (embedded_tiles / total_tiles * 100.0) if total_tiles > 0 else 0.0

        if total_tiles == 0:
            status_str = "NOT_STARTED"
        elif embedded_tiles == 0:
            status_str = "NOT_STARTED"
        elif embedded_tiles == total_tiles:
            status_str = "COMPLETED"
        else:
            status_str = "PARTIAL"

        # Check most recent embedded timestamp
        last_embedded_at = await session.scalar(
            select(func.max(Tile.embedded_at)).where(Tile.scene_id == scene_id)
        )

        # Check active job
        recent_job = await session.scalar(
            select(ProcessingJob)
            .where(
                ProcessingJob.related_entity_id == scene_id,
                ProcessingJob.job_type == "embedding",
            )
            .order_by(ProcessingJob.created_at.desc())
        )

        return {
            "scene_id": scene_id,
            "total_tiles": total_tiles,
            "embedded_tiles": embedded_tiles,
            "pending_tiles": pending_tiles,
            "progress_pct": round(progress_pct, 1),
            "status": status_str,
            "last_embedded_at": last_embedded_at,
            "models": ["RemoteCLIP", "DINOv2"],
            "collections": [
                settings.qdrant_remoteclip_collection,
                settings.qdrant_dino_collection,
            ],
            "active_job_id": recent_job.id if recent_job else None,
        }


# Module-level singleton
tile_embedding_service = TileEmbeddingService()
