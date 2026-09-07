"""
VisualSearchService: Executes visual similarity retrieval using DINOv2 (384-dimensional visual features).
Supports:
1. Image-to-Image Search from uploaded local imagery files (PNG, JPEG, GeoTIFF).
2. Tile-to-Tile Search from an existing satellite tile archive reference.
3. Multi-factor metadata, spatial, and temporal filtering.
4. Automatic exclusion of the reference tile from self-similarity results.
5. PostgreSQL tile/scene metadata and provenance audit enrichment.
"""
from __future__ import annotations

import io
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image
from qdrant_client import models
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.models.provenance import ProvenanceRecord
from app.models.scene import Scene
from app.models.tile import Tile
from app.schemas.search import (
    SemanticSearchResultItem,
    VisualSearchRequest,
    VisualSearchResponse,
)
from app.services.embedding_service import embedding_service
from app.services.spectral_analysis import spectral_analysis_service
from app.services.tile_embedding import tile_embedding_service
from app.services.vector_store import vector_store

logger = get_logger(__name__)


class VisualSearchService:
    """
    Coordinates visual nearest-neighbor vector search against Qdrant dino_tiles,
    query image feature extraction via DINOv2, and PostgreSQL provenance enrichment.
    """

    def __init__(self) -> None:
        self.embedding = embedding_service
        self.tile_embedding = tile_embedding_service
        self.vector_store = vector_store
        self.spectral = spectral_analysis_service

    def _build_qdrant_filter(
        self,
        req: VisualSearchRequest,
        exclude_tile_id: Optional[uuid.UUID] = None,
    ) -> tuple[Optional[models.Filter], Dict[str, Any]]:
        """Construct Qdrant Filter from request parameters."""
        must_conditions: List[models.Condition] = []
        must_not_conditions: List[models.Condition] = []
        filters_applied: Dict[str, Any] = {}

        # 1a. Sensors filter (support multiple or single)
        sensors_list = req.sensors or ([req.sensor] if req.sensor else None)
        if sensors_list:
            if len(sensors_list) == 1:
                must_conditions.append(
                    models.FieldCondition(
                        key="sensor",
                        match=models.MatchValue(value=sensors_list[0]),
                    )
                )
            else:
                must_conditions.append(
                    models.FieldCondition(
                        key="sensor",
                        match=models.MatchAny(any=sensors_list),
                    )
                )
            filters_applied["sensors"] = sensors_list

        # 1b. Quality threshold
        if req.min_quality is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="quality_score",
                    range=models.Range(gte=req.min_quality),
                )
            )
            filters_applied["min_quality"] = req.min_quality

        # 1c. Cloud cover threshold
        if req.max_cloud_cover is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="cloud_cover_pct",
                    range=models.Range(lte=req.max_cloud_cover),
                )
            )
            filters_applied["max_cloud_cover"] = req.max_cloud_cover

        # 1d. Date range
        if req.start_date or req.end_date:
            date_kwargs: Dict[str, Any] = {}
            if req.start_date:
                date_kwargs["gte"] = req.start_date.isoformat()
                filters_applied["start_date"] = req.start_date.isoformat()
            if req.end_date:
                date_kwargs["lte"] = req.end_date.isoformat()
                filters_applied["end_date"] = req.end_date.isoformat()

            must_conditions.append(
                models.FieldCondition(
                    key="acquisition_date",
                    range=models.DatetimeRange(**date_kwargs),
                )
            )

        # 1e. Geospatial filtering (Polygon envelope or Bounding Box)
        has_polygon = bool(req.aoi_polygon and len(req.aoi_polygon) >= 3)
        bbox_coords = req.aoi_bbox or req.aoi

        if has_polygon and req.aoi_polygon:
            lons = [p[0] for p in req.aoi_polygon]
            lats = [p[1] for p in req.aoi_polygon]
            poly_w, poly_s, poly_e, poly_n = min(lons), min(lats), max(lons), max(lats)
            must_conditions.append(
                models.FieldCondition(
                    key="center_lon",
                    range=models.Range(gte=poly_w, lte=poly_e),
                )
            )
            must_conditions.append(
                models.FieldCondition(
                    key="center_lat",
                    range=models.Range(gte=poly_s, lte=poly_n),
                )
            )
            filters_applied["aoi_polygon"] = {
                "vertex_count": len(req.aoi_polygon),
                "envelope": [poly_w, poly_s, poly_e, poly_n],
                "spatial_mode": req.spatial_filter_mode,
            }
        elif bbox_coords and len(bbox_coords) == 4:
            west, south, east, north = bbox_coords
            w, e = min(west, east), max(west, east)
            s, n = min(south, north), max(south, north)
            must_conditions.append(
                models.FieldCondition(
                    key="center_lon",
                    range=models.Range(gte=w, lte=e),
                )
            )
            must_conditions.append(
                models.FieldCondition(
                    key="center_lat",
                    range=models.Range(gte=s, lte=n),
                )
            )
            filters_applied["aoi_bbox"] = [w, s, e, n]

        if req.scene_id:
            must_conditions.append(
                models.FieldCondition(
                    key="scene_id",
                    match=models.MatchValue(value=str(req.scene_id)),
                )
            )
            filters_applied["scene_id"] = str(req.scene_id)

        if exclude_tile_id:
            must_not_conditions.append(
                models.FieldCondition(
                    key="tile_id",
                    match=models.MatchValue(value=str(exclude_tile_id)),
                )
            )
            filters_applied["excluded_tile_id"] = str(exclude_tile_id)

        qdrant_filter = None
        if must_conditions or must_not_conditions:
            qdrant_filter = models.Filter(
                must=must_conditions if must_conditions else None,
                must_not=must_not_conditions if must_not_conditions else None,
            )

        return qdrant_filter, filters_applied

    async def _enrich_results(
        self,
        scored_points: List[Any],
        session: AsyncSession,
    ) -> List[SemanticSearchResultItem]:
        """Enrich Qdrant ScoredPoint results with PostgreSQL Tile and Scene metadata."""
        if not scored_points:
            return []

        tile_uuids = [uuid.UUID(p.id) for p in scored_points]

        stmt = (
            select(Tile, Scene.filename)
            .join(Scene, Tile.scene_id == Scene.id)
            .where(Tile.id.in_(tile_uuids))
        )
        db_res = await session.execute(stmt)
        tile_map = {row[0].id: (row[0], row[1]) for row in db_res.all()}

        prov_stmt = (
            select(ProvenanceRecord)
            .where(
                ProvenanceRecord.entity_id.in_(tile_uuids),
                ProvenanceRecord.operation == "embedding",
            )
            .order_by(ProvenanceRecord.recorded_at.desc())
        )
        prov_res = await session.execute(prov_stmt)
        prov_map: Dict[uuid.UUID, Dict[str, Any]] = {}
        for pr in prov_res.scalars():
            if pr.entity_id not in prov_map:
                prov_map[pr.entity_id] = {
                    "operation": pr.operation,
                    "model_name": pr.details.get("dino_collection", "DINOv2") if pr.details else "DINOv2",
                    "embedding_dimension": pr.details.get("dino_dim", 384) if pr.details else 384,
                    "recorded_at": pr.recorded_at.isoformat(),
                    "operator": pr.operator,
                    "details": pr.details,
                }

        results: List[SemanticSearchResultItem] = []
        for point in scored_points:
            tid = uuid.UUID(point.id)
            payload = point.payload or {}
            tile_info = tile_map.get(tid)

            # Skip any points from Qdrant that no longer exist in PostgreSQL or on disk
            if not tile_info:
                continue

            tile_obj, scene_filename = tile_info
            scene_id = tile_obj.scene_id
            sensor = tile_obj.sensor
            acq_date = tile_obj.acquisition_date
            quality_score = tile_obj.quality_score
            cloud_pct = tile_obj.cloud_cover_pct
            tile_col = tile_obj.tile_col
            tile_row = tile_obj.tile_row
            bbox = {
                "west": tile_obj.bbox_west if tile_obj.bbox_west is not None else payload.get("bbox_west", 0.0),
                "south": tile_obj.bbox_south if tile_obj.bbox_south is not None else payload.get("bbox_south", 0.0),
                "east": tile_obj.bbox_east if tile_obj.bbox_east is not None else payload.get("bbox_east", 0.0),
                "north": tile_obj.bbox_north if tile_obj.bbox_north is not None else payload.get("bbox_north", 0.0),
            }
            center_coords = {
                "lon": tile_obj.center_lon if tile_obj.center_lon is not None else payload.get("center_lon", 0.0),
                "lat": tile_obj.center_lat if tile_obj.center_lat is not None else payload.get("center_lat", 0.0),
            }

            footprint_geojson = {
                "type": "Polygon",
                "coordinates": [
                    [
                        [bbox["west"], bbox["south"]],
                        [bbox["east"], bbox["south"]],
                        [bbox["east"], bbox["north"]],
                        [bbox["west"], bbox["north"]],
                        [bbox["west"], bbox["south"]],
                    ]
                ],
            }

            thumb_path = tile_obj.thumbnail_path or ""
            spectral = self.spectral.analyze_image(thumb_path)

            provenance_data = prov_map.get(tid) or {
                "operation": "embedding",
                "model_name": payload.get("model_name", "DINOv2"),
                "model_version": payload.get("model_version", "vit_small_patch14_dinov2.lvd142m"),
                "embedding_dimension": 384,
                "embedded_at": payload.get("embedded_at"),
            }
            provenance_data["spectral"] = spectral

            # Construct reliable thumbnail and preview URLs
            if tile_obj.thumbnail_path and Path(tile_obj.thumbnail_path).exists():
                thumb_url = f"/thumbnails/{scene_id}/{Path(tile_obj.thumbnail_path).name}"
            else:
                thumb_url = f"/api/tiles/{tid}/thumbnail"

            preview_path = (
                settings.previews_dir / str(scene_id) / f"preview_{tile_col:04d}_{tile_row:04d}.png"
            )
            if preview_path.exists():
                prev_url = f"/previews/{scene_id}/{preview_path.name}"
            else:
                prev_url = f"/api/tiles/{tid}/preview"

            results.append(
                SemanticSearchResultItem(
                    tile_id=tid,
                    similarity_score=round(float(point.score), 4),
                    rank=len(results) + 1,
                    scene_id=scene_id,
                    scene_name=scene_filename,
                    sensor=sensor,
                    acquisition_date=acq_date,
                    quality_score=quality_score,
                    cloud_cover_pct=cloud_pct,
                    tile_col=tile_col,
                    tile_row=tile_row,
                    center_coordinates=center_coords,
                    bbox=bbox,
                    footprint_geojson=footprint_geojson,
                    thumbnail_url=thumb_url,
                    preview_url=prev_url,
                    landcover=spectral,
                    provenance=provenance_data,
                )
            )

        return results

    async def _filter_by_polygon(
        self,
        scored_points: List[Any],
        req: VisualSearchRequest,
        session: AsyncSession,
    ) -> tuple[List[Any], float]:
        """Verify candidate points against exact PostGIS polygon footprint."""
        if not req.aoi_polygon or len(req.aoi_polygon) < 3 or not scored_points:
            return scored_points[:req.top_k], 0.0

        t_spatial_start = time.perf_counter()
        poly_pts = [list(p) for p in req.aoi_polygon]
        if poly_pts[0] != poly_pts[-1]:
            poly_pts.append(poly_pts[0])
        poly_wkt = "POLYGON((" + ", ".join(f"{p[0]} {p[1]}" for p in poly_pts) + "))"

        candidate_uuids = [uuid.UUID(p.id) for p in scored_points]
        spatial_func = "ST_Within" if req.spatial_filter_mode == "within" else "ST_Intersects"
        spatial_sql = text(
            f"SELECT tiles.id FROM tiles WHERE tiles.id = ANY(:candidate_ids) AND {spatial_func}(tiles.footprint, ST_GeomFromText(:wkt, 4326))"
        )
        spatial_res = await session.execute(
            spatial_sql,
            {"candidate_ids": candidate_uuids, "wkt": poly_wkt},
        )
        valid_spatial_ids = set(spatial_res.scalars().all())

        filtered = [p for p in scored_points if uuid.UUID(p.id) in valid_spatial_ids][:req.top_k]
        spatial_ms = round((time.perf_counter() - t_spatial_start) * 1000, 2)
        return filtered, spatial_ms

    async def search_by_image(
        self,
        image_input: Union[Image.Image, bytes, Path, str],
        req: VisualSearchRequest,
        session: AsyncSession,
    ) -> VisualSearchResponse:
        """
        Execute visual similarity search given an uploaded query image.
        Uses offline DINOv2 (384-dimensional features) against Qdrant dino_tiles collection.
        """
        t0 = time.perf_counter()

        # Step 1: Generate 384-dim visual embedding
        t_embed_start = time.perf_counter()
        query_vector = self.embedding.embed_image(
            image=image_input,
            model_name="DINOv2",
            normalize=True,
        )
        visual_embed_ms = round((time.perf_counter() - t_embed_start) * 1000, 2)

        # Step 2: Build filters
        qdrant_filter, filters_applied = self._build_qdrant_filter(req)

        # Step 3: Vector search on dino_tiles
        t_vec_start = time.perf_counter()
        has_polygon = bool(req.aoi_polygon and len(req.aoi_polygon) >= 3)
        fetch_limit = min(100, max(req.top_k * 3, 30)) if has_polygon else req.top_k

        scored_points = self.vector_store.search_vectors(
            collection_name=settings.qdrant_dino_collection,
            query_vector=query_vector,
            limit=fetch_limit,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        vector_search_ms = round((time.perf_counter() - t_vec_start) * 1000, 2)

        # Step 4: Exact PostGIS spatial verification
        filtered_points, spatial_filter_ms = await self._filter_by_polygon(scored_points, req, session)

        # Step 5: Enrich from PostgreSQL
        results = await self._enrich_results(filtered_points, session)
        total_time_ms = round((time.perf_counter() - t0) * 1000, 2)

        logger.info(
            "visual_search_by_image_completed",
            total_found=len(results),
            visual_embed_ms=visual_embed_ms,
            vector_search_ms=vector_search_ms,
            spatial_filter_ms=spatial_filter_ms,
            total_time_ms=total_time_ms,
        )

        return VisualSearchResponse(
            query_type="uploaded_image",
            reference_tile_id=None,
            reference_thumbnail_url=None,
            total_found=len(results),
            execution_time_ms=total_time_ms,
            visual_embedding_time_ms=visual_embed_ms,
            vector_search_time_ms=vector_search_ms,
            spatial_filter_time_ms=spatial_filter_ms,
            device=self.embedding.manager.device,
            model_used="DINOv2 (vit_small_patch14_dinov2.lvd142m)",
            embedding_dimension=384,
            results=results,
            filters_applied=filters_applied,
        )

    async def search_by_tile_id(
        self,
        tile_id: uuid.UUID,
        req: VisualSearchRequest,
        session: AsyncSession,
    ) -> VisualSearchResponse:
        """
        Execute visual similarity search using an existing archive tile as the reference.
        Reuses indexed DINOv2 vector directly and excludes the reference tile from results.
        """
        t0 = time.perf_counter()

        # Step 1: Validate tile exists
        tile = await session.get(Tile, tile_id)
        if not tile:
            raise ValueError(f"Tile {tile_id} not found in archive")

        # Step 2: Retrieve DINOv2 vector (from Qdrant cache or on-the-fly embedding)
        t_embed_start = time.perf_counter()
        query_vector: Optional[List[float]] = None

        try:
            retrieved = self.vector_store.client.retrieve(
                collection_name=settings.qdrant_dino_collection,
                ids=[str(tile_id)],
                with_vectors=True,
            )
            if retrieved and getattr(retrieved[0], "vector", None):
                query_vector = retrieved[0].vector
        except Exception:
            pass

        if query_vector is None:
            # Fallback: compute embedding on the fly from tile image
            img = self.tile_embedding.load_tile_image(tile)
            query_vector = self.embedding.embed_image(img, model_name="DINOv2", normalize=True)

        visual_embed_ms = round((time.perf_counter() - t_embed_start) * 1000, 2)

        # Step 3: Build filters excluding the reference tile itself
        qdrant_filter, filters_applied = self._build_qdrant_filter(req, exclude_tile_id=tile_id)

        # Step 4: Vector search on dino_tiles
        t_vec_start = time.perf_counter()
        has_polygon = bool(req.aoi_polygon and len(req.aoi_polygon) >= 3)
        fetch_limit = min(100, max(req.top_k * 3, 30)) if has_polygon else req.top_k

        scored_points = self.vector_store.search_vectors(
            collection_name=settings.qdrant_dino_collection,
            query_vector=query_vector,
            limit=fetch_limit,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        vector_search_ms = round((time.perf_counter() - t_vec_start) * 1000, 2)

        # Step 5: Exact PostGIS spatial verification
        filtered_points, spatial_filter_ms = await self._filter_by_polygon(scored_points, req, session)

        # Step 6: Enrich results
        results = await self._enrich_results(filtered_points, session)
        total_time_ms = round((time.perf_counter() - t0) * 1000, 2)

        logger.info(
            "visual_search_by_tile_completed",
            reference_tile=str(tile_id),
            total_found=len(results),
            visual_embed_ms=visual_embed_ms,
            vector_search_ms=vector_search_ms,
            spatial_filter_ms=spatial_filter_ms,
            total_time_ms=total_time_ms,
        )

        return VisualSearchResponse(
            query_type="tile_id",
            reference_tile_id=tile_id,
            reference_thumbnail_url=f"/api/tiles/{tile_id}/thumbnail",
            total_found=len(results),
            execution_time_ms=total_time_ms,
            visual_embedding_time_ms=visual_embed_ms,
            vector_search_time_ms=vector_search_ms,
            spatial_filter_time_ms=spatial_filter_ms,
            device=self.embedding.manager.device,
            model_used="DINOv2 (vit_small_patch14_dinov2.lvd142m)",
            embedding_dimension=384,
            results=results,
            filters_applied=filters_applied,
        )


# Module-level singleton
visual_search_service = VisualSearchService()
