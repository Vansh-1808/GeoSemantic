"""
Semantic and visual similarity search API routes.
Connects natural language queries to the Qdrant vector store via RemoteCLIP.
"""
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_session
from app.schemas.search import (
    SearchFiltersResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
    VisualSearchRequest,
    VisualSearchResponse,
)
from app.services.semantic_search import semantic_search_service
from app.services.visual_search import visual_search_service

router = APIRouter()


@router.post("/semantic", response_model=SemanticSearchResponse)
async def semantic_search_endpoint(
    req: SemanticSearchRequest,
    session: AsyncSession = Depends(get_session),
) -> SemanticSearchResponse:
    """
    Semantic text-to-satellite image retrieval.
    Encodes query text using RemoteCLIP and searches against the Qdrant remoteclip_tiles collection.
    Supports filtering by date range, sensor, area of interest (AOI), and quality score.
    """
    try:
        return await semantic_search_service.search(req=req, session=session)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic search failed: {exc}",
        )


@router.post("/text", response_model=SemanticSearchResponse)
async def text_search_alias(
    req: SemanticSearchRequest,
    session: AsyncSession = Depends(get_session),
) -> SemanticSearchResponse:
    """Alias for /semantic endpoint."""
    return await semantic_search_endpoint(req, session)


@router.get("/filters", response_model=SearchFiltersResponse)
async def get_search_filters_endpoint(
    session: AsyncSession = Depends(get_session),
) -> SearchFiltersResponse:
    """
    Fetch dynamic metadata filter values present in the satellite tile archive
    (sensors, date boundaries, total indexed tiles).
    """
    try:
        return await semantic_search_service.get_search_filters(session=session)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query search filters: {exc}",
        )


# ── Visual & Image-to-Image Search ───────────────────────────────

@router.post("/image", response_model=VisualSearchResponse)
async def visual_image_search_endpoint(
    file: UploadFile = File(..., description="Uploaded satellite imagery file (PNG, JPEG, GeoTIFF)"),
    top_k: int = Query(default=20, ge=1, le=100),
    sensor: Optional[str] = Query(default=None),
    sensors: Optional[list[str]] = Query(default=None),
    min_quality: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    max_cloud_cover: Optional[float] = Query(default=None, ge=0.0, le=100.0),
    start_date: Optional[str] = Query(default=None),
    end_date: Optional[str] = Query(default=None),
    west: Optional[float] = Query(default=None),
    south: Optional[float] = Query(default=None),
    east: Optional[float] = Query(default=None),
    north: Optional[float] = Query(default=None),
    spatial_filter_mode: str = Query(default="intersects"),
    session: AsyncSession = Depends(get_session),
) -> VisualSearchResponse:
    """
    Visual image-to-image similarity search using uploaded satellite imagery.
    Extracts 384-dimensional features via offline DINOv2 and retrieves visually
    similar tiles from the Qdrant dino_tiles collection.
    """
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        # Parse date range
        parsed_start = datetime.fromisoformat(start_date) if start_date else None
        parsed_end = datetime.fromisoformat(end_date) if end_date else None

        # Parse AOI
        aoi_coords = None
        if all(c is not None for c in [west, south, east, north]):
            aoi_coords = [west, south, east, north]  # type: ignore

        # Combine single sensor and multi sensors if provided
        active_sensors = sensors or ([sensor] if sensor else None)

        req = VisualSearchRequest(
            top_k=top_k,
            sensor=sensor,
            sensors=active_sensors,
            min_quality=min_quality,
            max_cloud_cover=max_cloud_cover,
            start_date=parsed_start,
            end_date=parsed_end,
            aoi=aoi_coords,
            aoi_bbox=aoi_coords,
            spatial_filter_mode=spatial_filter_mode if spatial_filter_mode in ("intersects", "within") else "intersects",
        )

        return await visual_search_service.search_by_image(
            image_input=image_bytes,
            req=req,
            session=session,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Visual image search failed: {exc}",
        )


@router.post("/similar/{tile_id}", response_model=VisualSearchResponse)
async def visual_similar_tile_search_endpoint(
    tile_id: uuid.UUID,
    req: Optional[VisualSearchRequest] = None,
    session: AsyncSession = Depends(get_session),
) -> VisualSearchResponse:
    """
    Tile-to-tile visual similarity search.
    Finds visually similar satellite locations given an existing reference tile.
    Reuses indexed DINOv2 features and excludes the query tile from the results.
    """
    try:
        active_req = req or VisualSearchRequest()
        return await visual_search_service.search_by_tile_id(
            tile_id=tile_id,
            req=active_req,
            session=session,
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(val_err),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Visual similarity search failed: {exc}",
        )

