"""
SemanticSearchService: Executes natural language semantic retrieval over satellite imagery archives.
Workflow:
  Query String → Text Embedding (RemoteCLIP 512-dim) → Qdrant Cosine Similarity Search →
  Multi-Factor Metadata & Spatial Filters → PostgreSQL Record & Provenance Enrichment →
  Ranked Results with Preview URLs and Telemetry
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client import models
from sqlalchemy import distinct, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.models.provenance import ProvenanceRecord
from app.models.scene import Scene
from app.models.tile import Tile
from app.schemas.search import (
    SearchFiltersResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
    SemanticSearchResultItem,
)
from app.services.embedding_service import embedding_service
from app.services.spectral_analysis import spectral_analysis_service
from app.services.vector_store import vector_store

logger = get_logger(__name__)


class SemanticSearchService:
    """
    Coordinates semantic vector search against Qdrant, metadata/spatial filtering,
    and PostgreSQL provenance enrichment.
    """

    def __init__(self) -> None:
        self.embedding = embedding_service
        self.vector_store = vector_store
        self.spectral = spectral_analysis_service

    async def search(
        self,
        req: SemanticSearchRequest,
        session: AsyncSession,
    ) -> SemanticSearchResponse:
        """
        Execute natural language semantic search across the satellite tile archive.
        Returns ranked tiles with similarity scores, coordinates, preview URLs, and provenance.
        """
        t0 = time.perf_counter()

        # Step 1: Generate normalized 512-dimensional query embedding
        t_embed_start = time.perf_counter()
        query_vector = self.embedding.embed_text(
            text=req.query.strip(),
            model_name="RemoteCLIP",
            normalize=True,
        )
        text_embedding_time_ms = round((time.perf_counter() - t_embed_start) * 1000, 2)

        # Step 2: Build Qdrant filter conditions
        must_conditions: List[models.Condition] = []
        filters_applied: Dict[str, Any] = {}

        # 2a. Sensors filter (support multiple or single)
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

        # 2b. Quality threshold
        if req.min_quality is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="quality_score",
                    range=models.Range(gte=req.min_quality),
                )
            )
            filters_applied["min_quality"] = req.min_quality

        # 2c. Cloud cover threshold
        if req.max_cloud_cover is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="cloud_cover_pct",
                    range=models.Range(lte=req.max_cloud_cover),
                )
            )
            filters_applied["max_cloud_cover"] = req.max_cloud_cover

        # 2d. Date range
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

        # 2e. Geospatial filtering (Polygon envelope or Bounding Box)
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

        qdrant_filter = models.Filter(must=must_conditions) if must_conditions else None

        # Step 3: Query Qdrant vector index
        t_vec_start = time.perf_counter()
        collection_name = settings.qdrant_remoteclip_collection
        fetch_limit = min(200, max(req.top_k * 4, 50))
        scored_points = self.vector_store.search_vectors(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=fetch_limit,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        vector_search_time_ms = round((time.perf_counter() - t_vec_start) * 1000, 2)

        # Step 4: PostGIS Exact Spatial Verification (if polygon AOI provided)
        spatial_filter_time_ms = 0.0
        if has_polygon and scored_points and req.aoi_polygon:
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

            filtered_points = [p for p in scored_points if uuid.UUID(p.id) in valid_spatial_ids]
            spatial_filter_time_ms = round((time.perf_counter() - t_spatial_start) * 1000, 2)
        else:
            filtered_points = scored_points

        # Step 5: Enrich points from PostgreSQL database
        results: List[SemanticSearchResultItem] = []

        if filtered_points:
            tile_uuids = [uuid.UUID(p.id) for p in filtered_points]

            # Fetch matching Tile ORM entities and parent Scene filename
            stmt = (
                select(Tile, Scene.filename)
                .join(Scene, Tile.scene_id == Scene.id)
                .where(Tile.id.in_(tile_uuids), Tile.is_valid == True)
            )
            db_res = await session.execute(stmt)
            tile_map = {row[0].id: (row[0], row[1]) for row in db_res.all()}

            # Fetch provenance records for valid tiles
            prov_stmt = (
                select(ProvenanceRecord)
                .where(
                    ProvenanceRecord.entity_id.in_(list(tile_map.keys())),
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
                        "model_name": pr.details.get("remoteclip_collection", "RemoteCLIP") if pr.details else "RemoteCLIP",
                        "embedding_dimension": pr.details.get("remoteclip_dim", 512) if pr.details else 512,
                        "recorded_at": pr.recorded_at.isoformat(),
                        "operator": pr.operator,
                        "details": pr.details,
                    }

            intent = self.spectral.parse_query_intent(req.query)
            candidate_items: List[Tuple[float, SemanticSearchResultItem]] = []

            for point in filtered_points:
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

                # Extract physical spectral ratios
                thumb_path = tile_obj.thumbnail_path or ""
                spectral = self.spectral.analyze_image(thumb_path)

                # Compute fused score and calibrated display percentage
                raw_score = float(point.score)
                fused_score, calibrated_score = self.spectral.compute_fused_score(
                    raw_similarity=raw_score,
                    spectral=spectral,
                    intent=intent,
                )

                provenance_data = prov_map.get(tid) or {
                    "operation": "embedding",
                    "model_name": payload.get("model_name", "RemoteCLIP"),
                    "model_version": payload.get("model_version", "ViT-B-32"),
                    "embedding_dimension": 512,
                    "embedded_at": payload.get("embedded_at"),
                }
                provenance_data["spectral"] = {
                    "water_pct": spectral["water_pct"],
                    "veg_pct": spectral["veg_pct"],
                    "urban_pct": spectral["urban_pct"],
                    "raw_similarity": round(raw_score, 4),
                    "fused_score": fused_score,
                }

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

                item = SemanticSearchResultItem(
                    tile_id=tid,
                    similarity_score=calibrated_score,
                    rank=0,  # Assigned after reranking
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
                candidate_items.append((fused_score, item))

            # Rerank candidates by fused score (highest relevance first)
            candidate_items.sort(key=lambda x: x[0], reverse=True)

            # Assign 1-based ranks to the top_k results
            for idx, (_, item) in enumerate(candidate_items[:req.top_k], start=1):
                item.rank = idx
                results.append(item)

        total_time_ms = round((time.perf_counter() - t0) * 1000, 2)

        logger.info(
            "semantic_search_completed",
            query=req.query,
            total_found=len(results),
            total_time_ms=total_time_ms,
            text_embed_ms=text_embedding_time_ms,
            vector_search_ms=vector_search_time_ms,
            spatial_filter_ms=spatial_filter_time_ms,
        )

        return SemanticSearchResponse(
            query=req.query,
            total_found=len(results),
            execution_time_ms=total_time_ms,
            text_embedding_time_ms=text_embedding_time_ms,
            vector_search_time_ms=vector_search_time_ms,
            spatial_filter_time_ms=spatial_filter_time_ms,
            device=self.embedding.manager.device,
            model_used="RemoteCLIP (ViT-B-32)",
            results=results,
            filters_applied=filters_applied,
        )

    async def get_search_filters(self, session: AsyncSession) -> SearchFiltersResponse:
        """Fetch available sensors, date ranges, and total index statistics."""
        # Distinct sensors and count per sensor
        sensor_stmt = (
            select(Tile.sensor, func.count(Tile.id))
            .where(Tile.sensor.isnot(None), Tile.remoteclip_vector_id.isnot(None))
            .group_by(Tile.sensor)
        )
        sensor_res = await session.execute(sensor_stmt)
        sensor_counts = {row[0]: row[1] for row in sensor_res.all() if row[0]}
        sensors = sorted(list(sensor_counts.keys()))

        # Min and max dates, quality, and cloud cover
        stats_stmt = (
            select(
                func.min(Tile.acquisition_date),
                func.max(Tile.acquisition_date),
                func.min(Tile.quality_score),
                func.max(Tile.cloud_cover_pct),
            ).where(Tile.remoteclip_vector_id.isnot(None))
        )
        stats_res = await session.execute(stats_stmt)
        min_date_val, max_date_val, min_q, max_cloud = stats_res.first() or (None, None, 0.0, 100.0)

        # Count of embedded tiles in database
        count_res = await session.scalar(
            select(func.count(Tile.id)).where(Tile.remoteclip_vector_id.isnot(None))
        )

        return SearchFiltersResponse(
            sensors=sensors,
            sensor_counts=sensor_counts,
            min_date=min_date_val.isoformat() if min_date_val else None,
            max_date=max_date_val.isoformat() if max_date_val else None,
            total_indexed_tiles=count_res or 0,
            min_quality=float(min_q) if min_q is not None else 0.0,
            max_cloud_cover=float(max_cloud) if max_cloud is not None else 100.0,
        )


# Module-level singleton
semantic_search_service = SemanticSearchService()
