"""
Tile API routes — serve tiles, previews, thumbnails, and tile metadata.
Endpoints:
  GET /tiles                — Paginated list of tiles (filter by scene_id, is_valid, min_quality)
  GET /tiles/{id}           — Full tile metadata
  GET /tiles/{id}/preview   — Serve high-res RGB tile preview PNG
  GET /tiles/{id}/thumbnail — Serve 128x128 tile thumbnail PNG
  GET /tiles/scene/{scene_id} — List tiles for a specific scene
"""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.database import get_session
from app.models.tile import Tile
from app.schemas.scene import TileDetail, TileListResponse, TileSummary

router = APIRouter()


@router.get("", response_model=TileListResponse)
async def list_tiles(
    scene_id: uuid.UUID | None = None,
    is_valid: bool | None = True,
    min_quality: float | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
    session: AsyncSession = Depends(get_session),
):
    """
    List tiles with optional filtering by scene_id, validity, and quality score.
    """
    query = select(Tile).order_by(Tile.tile_row.asc(), Tile.tile_col.asc())

    if scene_id is not None:
        query = query.where(Tile.scene_id == scene_id)
    if is_valid is not None:
        query = query.where(Tile.is_valid == is_valid)
    if min_quality is not None:
        query = query.where(Tile.quality_score >= min_quality)

    count_query = select(func.count()).select_from(query.subquery())
    total = (await session.execute(count_query)).scalar_one()

    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await session.execute(query)
    tiles = result.scalars().all()

    summaries = []
    for t in tiles:
        s = _build_tile_summary(t)
        summaries.append(s)

    return TileListResponse(
        tiles=summaries,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{tile_id}", response_model=TileDetail)
async def get_tile(
    tile_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    """Return full tile metadata including pixel coordinates and URLs."""
    tile = await session.get(Tile, tile_id)
    if not tile:
        raise HTTPException(status_code=404, detail="Tile not found")

    result = TileDetail.model_validate(tile)
    _enrich_tile_urls(tile, result)
    return result


@router.get("/{tile_id}/preview")
async def get_tile_preview(
    tile_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    """Serve the high-resolution RGB tile preview PNG directly."""
    tile = await session.get(Tile, tile_id)
    if not tile:
        raise HTTPException(status_code=404, detail="Tile not found")

    # Check for preview in previews directory
    preview_path = (
        settings.previews_dir / str(tile.scene_id) / f"preview_{tile.tile_col:04d}_{tile.tile_row:04d}.png"
    )
    if preview_path.exists():
        return FileResponse(preview_path, media_type="image/png")

    # Fallback to thumbnail if preview not found
    if tile.thumbnail_path and Path(tile.thumbnail_path).exists():
        return FileResponse(tile.thumbnail_path, media_type="image/png")

    raise HTTPException(status_code=404, detail="Preview not available for this tile")


@router.get("/{tile_id}/thumbnail")
async def get_tile_thumbnail(
    tile_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    """Serve the tile thumbnail PNG directly."""
    tile = await session.get(Tile, tile_id)
    if not tile:
        raise HTTPException(status_code=404, detail="Tile not found")

    if tile.thumbnail_path and Path(tile.thumbnail_path).exists():
        return FileResponse(tile.thumbnail_path, media_type="image/png")

    # Fallback to preview if thumbnail missing
    preview_path = (
        settings.previews_dir / str(tile.scene_id) / f"preview_{tile.tile_col:04d}_{tile.tile_row:04d}.png"
    )
    if preview_path.exists():
        return FileResponse(preview_path, media_type="image/png")

    raise HTTPException(status_code=404, detail="Thumbnail not available for this tile")


@router.get("/scene/{scene_id}", response_model=list[TileSummary])
async def list_tiles_for_scene(
    scene_id: uuid.UUID,
    limit: int = 200,
    session: AsyncSession = Depends(get_session),
):
    """List tiles for a scene — used by the map to show tile coverage."""
    result = await session.execute(
        select(Tile)
        .where(Tile.scene_id == scene_id, Tile.is_valid == True)
        .order_by(Tile.tile_row.asc(), Tile.tile_col.asc())
        .limit(limit)
    )
    tiles = result.scalars().all()
    return [_build_tile_summary(t) for t in tiles]


def _build_tile_summary(t: Tile) -> TileSummary:
    s = TileSummary.model_validate(t)
    _enrich_tile_urls(t, s)
    return s


def _enrich_tile_urls(t: Tile, target: TileSummary | TileDetail) -> None:
    if t.thumbnail_path and Path(t.thumbnail_path).exists():
        target.thumbnail_url = f"/thumbnails/{t.scene_id}/{Path(t.thumbnail_path).name}"
    else:
        target.thumbnail_url = f"/api/tiles/{t.id}/thumbnail"

    preview_path = (
        settings.previews_dir / str(t.scene_id) / f"preview_{t.tile_col:04d}_{t.tile_row:04d}.png"
    )
    if preview_path.exists():
        target.preview_url = f"/previews/{t.scene_id}/{preview_path.name}"
    else:
        target.preview_url = f"/api/tiles/{t.id}/preview"
