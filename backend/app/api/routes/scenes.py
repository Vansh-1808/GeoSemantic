"""
Scene processing and metadata API routes.
Endpoints:
  POST /scenes/{id}/process   — Trigger tiling & preprocessing for a scene
  GET  /scenes/{id}           — Get full scene details
  GET  /scenes                — List all ingested scenes
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.database import get_session
from app.models.provenance import ProcessingJob
from app.models.scene import Scene
from app.schemas.scene import (
    ProcessSceneRequest,
    ProcessSceneResponse,
    SceneInDB,
    SceneListResponse,
    SceneSummary,
)
from app.services.tiling import TilingConfig, process_scene_tiling

router = APIRouter()
logger = get_logger(__name__)


@router.post("/{scene_id}/process", response_model=ProcessSceneResponse)
async def process_scene(
    scene_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    request: ProcessSceneRequest = ProcessSceneRequest(),
    sync: bool = Query(default=False, description="Run synchronously and wait for completion"),
    session: AsyncSession = Depends(get_session),
):
    """
    Process an ingested satellite scene into tiles with configurable tile size,
    overlap, and radiometric normalization.
    """
    scene = await session.get(Scene, scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found")

    job_id = uuid.uuid4()
    job = ProcessingJob(
        id=job_id,
        job_type="tiling",
        status="PENDING",
        params={
            "scene_id": str(scene_id),
            "tile_size": request.tile_size,
            "overlap_px": request.overlap_px,
            "normalize": request.normalize,
            "min_valid_pixel_ratio": request.min_valid_pixel_ratio,
        },
        related_entity_type="scene",
        related_entity_id=scene_id,
    )
    session.add(job)
    await session.commit()

    config = TilingConfig(
        tile_size=request.tile_size,
        overlap_px=request.overlap_px,
        normalize=request.normalize,
        min_valid_pixel_ratio=request.min_valid_pixel_ratio,
        generate_preview=request.generate_preview,
    )

    if sync:
        try:
            job.status = "RUNNING"
            job.started_at = datetime.now(timezone.utc)
            await session.commit()

            result = await process_scene_tiling(
                scene_id=scene_id,
                session=session,
                config=config,
                job_id=job_id,
            )
            return ProcessSceneResponse(
                scene_id=scene_id,
                job_id=job_id,
                status="COMPLETED",
                tile_count=result.tile_count,
                processing_time_seconds=result.processing_time_seconds,
                message=f"Tiling completed successfully: {result.tile_count} tiles generated in {result.processing_time_seconds}s",
            )
        except Exception as exc:
            logger.exception("sync_tiling_failed", scene_id=str(scene_id), error=str(exc))
            job.status = "FAILED"
            job.error_message = str(exc)
            job.completed_at = datetime.now(timezone.utc)
            await session.commit()
            raise HTTPException(status_code=500, detail=f"Tiling failed: {exc}")

    # Async background task
    background_tasks.add_task(_run_tiling_background, scene_id, job_id, config)

    return ProcessSceneResponse(
        scene_id=scene_id,
        job_id=job_id,
        status="PROCESSING",
        message=f"Scene tiling initiated with {request.tile_size}x{request.tile_size}px tiles and {request.overlap_px}px overlap",
    )


@router.get("/{scene_id}", response_model=SceneInDB)
async def get_scene_by_id(
    scene_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    """Retrieve full metadata for a single scene."""
    scene = await session.get(Scene, scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found")
    return SceneInDB.model_validate(scene)


@router.get("", response_model=SceneListResponse)
async def list_scenes_endpoint(
    page: int = 1,
    page_size: int = 20,
    status: str | None = None,
    session: AsyncSession = Depends(get_session),
):
    """List scenes with pagination."""
    query = select(Scene).order_by(Scene.created_at.desc())
    if status:
        query = query.where(Scene.ingestion_status == status.upper())

    count_query = select(func.count()).select_from(query.subquery())
    total = (await session.execute(count_query)).scalar_one()

    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await session.execute(query)
    scenes = result.scalars().all()

    return SceneListResponse(
        scenes=[SceneSummary.model_validate(s) for s in scenes],
        total=total,
        page=page,
        page_size=page_size,
    )


# ─────────────────────────────────────────────────────────────
# Background Worker
# ─────────────────────────────────────────────────────────────

async def _run_tiling_background(
    scene_id: uuid.UUID, job_id: uuid.UUID, config: TilingConfig
) -> None:
    from app.db.database import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        try:
            job = await session.get(ProcessingJob, job_id)
            if job:
                job.status = "RUNNING"
                job.started_at = datetime.now(timezone.utc)
                await session.commit()

            await process_scene_tiling(
                scene_id=scene_id,
                session=session,
                config=config,
                job_id=job_id,
            )
        except Exception as exc:
            logger.exception("background_tiling_failed", scene_id=str(scene_id), error=str(exc))
            job = await session.get(ProcessingJob, job_id)
            if job:
                job.status = "FAILED"
                job.error_message = str(exc)
                job.completed_at = datetime.now(timezone.utc)
                await session.commit()
