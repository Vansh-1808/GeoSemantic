"""
Image Quality Assessment and Confounding Factors API routes.
Endpoints:
  GET  /quality/tile/{tile_id}            — Get quality analysis for a tile
  POST /quality/tile/{tile_id}/compute    — Compute & persist quality for a tile
  GET  /quality/scene/{scene_id}          — Get aggregate quality for a scene
  POST /quality/scene/{scene_id}/compute  — Batch compute & persist quality for all tiles in a scene
"""
from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.db.database import get_session
from app.models.scene import Scene
from app.models.tile import Tile
from app.schemas.quality import (
    SceneComputeQualityResponse,
    SceneQualityResponse,
    TileQualityResponse,
)
from app.services.spectral_analysis import spectral_analysis_service

router = APIRouter()
logger = get_logger(__name__)


def _get_tile_image_path(tile: Tile) -> str | None:
    """Resolve local image file path for a tile."""
    if tile.thumbnail_path and os.path.exists(tile.thumbnail_path):
        return tile.thumbnail_path
    if tile.tile_path and os.path.exists(tile.tile_path):
        return tile.tile_path
    
    # Fallback to standard convention
    candidate = settings.thumbnails_dir / str(tile.scene_id) / f"thumb_{tile.tile_row:04d}_{tile.tile_col:04d}.png"
    if candidate.exists():
        return str(candidate)
    return None


@router.get("/tile/{tile_id}", response_model=TileQualityResponse)
async def get_tile_quality(
    tile_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    """Return complete deterministic quality metrics and confounding factor detection for a tile."""
    tile = await session.get(Tile, tile_id)
    if not tile:
        raise HTTPException(status_code=404, detail=f"Tile {tile_id} not found")

    image_path = _get_tile_image_path(tile)
    if not image_path:
        return TileQualityResponse(
            tile_id=tile.id,
            quality_score=0.0,
            usable_for_change_analysis=False,
            suitability="Poor Quality",
            cloud_score=0.0,
            shadow_score=0.0,
            nodata_score=1.0,
            noise_score=0.0,
            saturation_score=0.0,
            cloud_cover_pct=0.0,
            nodata_ratio=1.0,
            analysis_method="fallback_file_missing",
            satellite_mask_available=False,
            caveats=["Raster file could not be located on disk."],
        )

    res = spectral_analysis_service.analyze_quality(image_path)
    return TileQualityResponse(tile_id=tile.id, **res)


@router.post("/tile/{tile_id}/compute", response_model=TileQualityResponse)
async def compute_tile_quality(
    tile_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    """Compute and persist quality metrics for a single tile directly to the database."""
    tile = await session.get(Tile, tile_id)
    if not tile:
        raise HTTPException(status_code=404, detail=f"Tile {tile_id} not found")

    image_path = _get_tile_image_path(tile)
    if not image_path:
        raise HTTPException(status_code=400, detail=f"Image file for tile {tile_id} not found on disk")

    res = spectral_analysis_service.analyze_quality(image_path)
    
    tile.quality_score = res["quality_score"]
    tile.cloud_cover_pct = res["cloud_cover_pct"]
    tile.nodata_ratio = res["nodata_ratio"]
    tile.is_valid = (res["suitability"] != "Poor Quality")
    
    await session.commit()
    await session.refresh(tile)

    return TileQualityResponse(tile_id=tile.id, **res)


@router.get("/scene/{scene_id}", response_model=SceneQualityResponse)
async def get_scene_quality(
    scene_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    """Aggregate quality across all tiles in a scene with change suitability verdict."""
    scene = await session.get(Scene, scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")

    result = await session.execute(
        select(Tile).where(Tile.scene_id == scene_id).order_by(Tile.tile_row.asc(), Tile.tile_col.asc())
    )
    tiles = result.scalars().all()

    if not tiles:
        return SceneQualityResponse(
            scene_id=scene.id,
            quality_score=scene.quality_score or 0.0,
            suitability="Poor Quality",
            tile_count=0,
            tiles_suitable=0,
            tiles_caution=0,
            tiles_poor=0,
            avg_cloud_cover_pct=0.0,
            avg_nodata_ratio=1.0,
            usable_for_change_analysis=False,
            quality_details={},
        )

    tile_qualities: List[Dict[str, Any]] = []
    for t in tiles:
        img_path = _get_tile_image_path(t)
        if img_path:
            q = spectral_analysis_service.analyze_quality(img_path)
        else:
            q = {
                "quality_score": 0.0,
                "usable_for_change_analysis": False,
                "suitability": "Poor Quality",
                "cloud_score": 0.0,
                "shadow_score": 0.0,
                "nodata_score": 1.0,
                "noise_score": 0.0,
                "saturation_score": 0.0,
                "cloud_cover_pct": 0.0,
                "nodata_ratio": 1.0,
            }
        tile_qualities.append(q)

    total_count = len(tile_qualities)
    suitable_count = sum(1 for q in tile_qualities if q["suitability"] == "Suitable")
    caution_count = sum(1 for q in tile_qualities if q["suitability"] == "Caution")
    poor_count = sum(1 for q in tile_qualities if q["suitability"] == "Poor Quality")

    avg_quality = float(sum(q["quality_score"] for q in tile_qualities) / total_count)
    avg_cloud = float(sum(q["cloud_cover_pct"] for q in tile_qualities) / total_count)
    avg_nodata = float(sum(q["nodata_ratio"] for q in tile_qualities) / total_count)

    suitable_ratio = suitable_count / total_count
    if avg_quality >= 0.75 and suitable_ratio >= 0.70:
        scene_suitability = "Suitable"
        usable_for_change = True
    elif avg_quality >= 0.40 and (suitable_count + caution_count) / total_count >= 0.50:
        scene_suitability = "Caution"
        usable_for_change = False
    else:
        scene_suitability = "Poor Quality"
        usable_for_change = False

    details = {
        "suitable_ratio": round(suitable_ratio, 3),
        "avg_shadow_score": round(float(sum(q.get("shadow_score", 0.0) for q in tile_qualities) / total_count), 4),
        "avg_noise_score": round(float(sum(q.get("noise_score", 0.0) for q in tile_qualities) / total_count), 4),
        "avg_saturation_score": round(float(sum(q.get("saturation_score", 0.0) for q in tile_qualities) / total_count), 4),
    }

    return SceneQualityResponse(
        scene_id=scene.id,
        quality_score=round(avg_quality, 4),
        suitability=scene_suitability,
        tile_count=total_count,
        tiles_suitable=suitable_count,
        tiles_caution=caution_count,
        tiles_poor=poor_count,
        avg_cloud_cover_pct=round(avg_cloud, 2),
        avg_nodata_ratio=round(avg_nodata, 4),
        usable_for_change_analysis=usable_for_change,
        quality_details=details,
    )


@router.post("/scene/{scene_id}/compute", response_model=SceneComputeQualityResponse)
async def compute_scene_quality(
    scene_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    """
    Batch assess all tiles in a scene, update each tile's quality metadata,
    and persist aggregated quality scores to the Scene database record.
    """
    scene = await session.get(Scene, scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")

    result = await session.execute(
        select(Tile).where(Tile.scene_id == scene_id).order_by(Tile.tile_row.asc(), Tile.tile_col.asc())
    )
    tiles = result.scalars().all()

    if not tiles:
        raise HTTPException(status_code=400, detail=f"Scene {scene_id} has no tiles to compute quality for")

    tile_qualities: List[Dict[str, Any]] = []
    for t in tiles:
        img_path = _get_tile_image_path(t)
        if img_path:
            q = spectral_analysis_service.analyze_quality(img_path)
        else:
            q = {
                "quality_score": 0.0,
                "usable_for_change_analysis": False,
                "suitability": "Poor Quality",
                "cloud_score": 0.0,
                "shadow_score": 0.0,
                "nodata_score": 1.0,
                "noise_score": 0.0,
                "saturation_score": 0.0,
                "cloud_cover_pct": 0.0,
                "nodata_ratio": 1.0,
            }
        
        # Persist to tile record in DB
        t.quality_score = q["quality_score"]
        t.cloud_cover_pct = q["cloud_cover_pct"]
        t.nodata_ratio = q["nodata_ratio"]
        t.is_valid = (q["suitability"] != "Poor Quality")
        tile_qualities.append(q)

    total_count = len(tile_qualities)
    suitable_count = sum(1 for q in tile_qualities if q["suitability"] == "Suitable")
    caution_count = sum(1 for q in tile_qualities if q["suitability"] == "Caution")
    poor_count = sum(1 for q in tile_qualities if q["suitability"] == "Poor Quality")

    avg_quality = float(sum(q["quality_score"] for q in tile_qualities) / total_count)
    avg_cloud = float(sum(q["cloud_cover_pct"] for q in tile_qualities) / total_count)
    avg_nodata = float(sum(q["nodata_ratio"] for q in tile_qualities) / total_count)

    suitable_ratio = suitable_count / total_count
    if avg_quality >= 0.75 and suitable_ratio >= 0.70:
        scene_suitability = "Suitable"
        usable_for_change = True
    elif avg_quality >= 0.40 and (suitable_count + caution_count) / total_count >= 0.50:
        scene_suitability = "Caution"
        usable_for_change = False
    else:
        scene_suitability = "Poor Quality"
        usable_for_change = False

    details = {
        "suitable_ratio": round(suitable_ratio, 3),
        "avg_shadow_score": round(float(sum(q.get("shadow_score", 0.0) for q in tile_qualities) / total_count), 4),
        "avg_noise_score": round(float(sum(q.get("noise_score", 0.0) for q in tile_qualities) / total_count), 4),
        "avg_saturation_score": round(float(sum(q.get("saturation_score", 0.0) for q in tile_qualities) / total_count), 4),
    }

    # Persist to Scene record in DB
    scene.quality_score = round(avg_quality, 4)
    scene.cloud_cover_pct = round(avg_cloud, 2)
    scene.quality_details = details

    await session.commit()
    await session.refresh(scene)

    summary = SceneQualityResponse(
        scene_id=scene.id,
        quality_score=round(avg_quality, 4),
        suitability=scene_suitability,
        tile_count=total_count,
        tiles_suitable=suitable_count,
        tiles_caution=caution_count,
        tiles_poor=poor_count,
        avg_cloud_cover_pct=round(avg_cloud, 2),
        avg_nodata_ratio=round(avg_nodata, 4),
        usable_for_change_analysis=usable_for_change,
        quality_details=details,
    )

    return SceneComputeQualityResponse(
        scene_id=scene.id,
        tiles_computed=total_count,
        quality_score=round(avg_quality, 4),
        suitability=scene_suitability,
        summary=summary,
    )
