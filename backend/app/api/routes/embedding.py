"""
API endpoints for satellite tile vector embedding, status tracking, and vector store statistics.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.database import AsyncSessionLocal, get_session
from app.models.provenance import ProcessingJob
from app.models.scene import Scene
from app.schemas.embedding import (
    EmbeddingGenerateRequest,
    EmbeddingJobResponse,
    EmbeddingStatusResponse,
    VectorStatisticsResponse,
)
from app.services.tile_embedding import tile_embedding_service
from app.services.vector_store import vector_store

logger = get_logger(__name__)

router = APIRouter()


async def _run_embedding_background(
    scene_id: uuid.UUID,
    job_id: uuid.UUID,
    force_reembed: bool,
    batch_size: int,
) -> None:
    """Execute embedding generation asynchronously in background task."""
    async with AsyncSessionLocal() as session:
        job = await session.get(ProcessingJob, job_id)
        if not job:
            return

        job.status = "RUNNING"
        job.started_at = datetime.now(timezone.utc)
        await session.commit()

        async def update_progress(pct: float, message: str) -> None:
            async with AsyncSessionLocal() as prog_session:
                pj = await prog_session.get(ProcessingJob, job_id)
                if pj:
                    pj.progress_pct = pct
                    pj.progress_message = message
                    await prog_session.commit()

        try:
            result = await tile_embedding_service.generate_scene_embeddings(
                scene_id=scene_id,
                session=session,
                force_reembed=force_reembed,
                batch_size=batch_size,
                job_id=job_id,
                progress_callback=update_progress,
            )

            job.status = "COMPLETED"
            job.progress_pct = 100.0
            job.completed_at = datetime.now(timezone.utc)
            job.progress_message = result.get("message", "Completed")
            job.result = result
            await session.commit()

        except Exception as exc:
            logger.error("background_embedding_failed", scene_id=str(scene_id), error=str(exc))
            job.status = "FAILED"
            job.completed_at = datetime.now(timezone.utc)
            job.error_message = str(exc)
            await session.commit()


@router.post("/generate/{scene_id}", response_model=EmbeddingJobResponse)
async def generate_embeddings_endpoint(
    scene_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    force_reembed: bool = Query(False, description="If True, re-embed all tiles regardless of existing vector IDs"),
    sync: bool = Query(False, description="If True, run synchronously and wait for completion"),
    batch_size: int = Query(8, ge=1, le=64, description="Tiles per batch"),
    req: Optional[EmbeddingGenerateRequest] = None,
    session: AsyncSession = Depends(get_session),
) -> EmbeddingJobResponse:
    """
    Generate RemoteCLIP and DINOv2 vector embeddings for tiles in a scene.
    Supports both synchronous execution and background task execution.
    """
    # Prefer body params if provided
    if req:
        force_reembed = req.force_reembed
        batch_size = req.batch_size

    scene = await session.get(Scene, scene_id)
    if not scene:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scene {scene_id} not found",
        )

    job_id = uuid.uuid4()
    job = ProcessingJob(
        id=job_id,
        job_type="embedding",
        status="RUNNING" if sync else "PENDING",
        related_entity_id=scene_id,
        related_entity_type="scene",
        progress_pct=0.0,
        progress_message="Starting embedding pipeline...",
        params={"force_reembed": force_reembed, "batch_size": batch_size},
        started_at=datetime.now(timezone.utc) if sync else None,
    )
    session.add(job)
    await session.commit()

    if sync:
        try:
            result = await tile_embedding_service.generate_scene_embeddings(
                scene_id=scene_id,
                session=session,
                force_reembed=force_reembed,
                batch_size=batch_size,
                job_id=job_id,
            )
            job.status = "COMPLETED"
            job.progress_pct = 100.0
            job.completed_at = datetime.now(timezone.utc)
            job.result = result
            await session.commit()

            return EmbeddingJobResponse(
                job_id=job_id,
                scene_id=scene_id,
                status="COMPLETED",
                total_tiles=result["total_tiles"],
                embedded_count=result["embedded_count"],
                skipped_count=result["skipped_count"],
                duration_seconds=result["duration_seconds"],
                message=result["message"],
            )
        except Exception as exc:
            logger.error("sync_embedding_failed", scene_id=str(scene_id), error=str(exc))
            job.status = "FAILED"
            job.completed_at = datetime.now(timezone.utc)
            job.error_message = str(exc)
            await session.commit()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Embedding generation failed: {exc}",
            )
    else:
        background_tasks.add_task(
            _run_embedding_background,
            scene_id=scene_id,
            job_id=job_id,
            force_reembed=force_reembed,
            batch_size=batch_size,
        )
        return EmbeddingJobResponse(
            job_id=job_id,
            scene_id=scene_id,
            status="PENDING",
            total_tiles=scene.tile_count,
            embedded_count=scene.embedded_count,
            skipped_count=0,
            duration_seconds=None,
            message="Tile embedding job queued in background",
        )


@router.get("/status/{scene_id}", response_model=EmbeddingStatusResponse)
async def get_embedding_status_endpoint(
    scene_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> EmbeddingStatusResponse:
    """Get real-time vector embedding progress and status for a scene."""
    try:
        status_info = await tile_embedding_service.get_scene_embedding_status(
            scene_id=scene_id,
            session=session,
        )
        return EmbeddingStatusResponse(**status_info)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )


@router.get("/statistics", response_model=VectorStatisticsResponse)
async def get_vector_statistics_endpoint() -> VectorStatisticsResponse:
    """Get global vector store statistics, vector counts, and storage metrics."""
    try:
        stats = vector_store.get_all_statistics()
        return VectorStatisticsResponse(**stats)
    except Exception as exc:
        logger.error("vector_statistics_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query vector statistics: {exc}",
        )


@router.post("/prune-orphans")
async def prune_orphaned_vectors_endpoint(
    session: AsyncSession = Depends(get_session),
) -> dict:
    """
    Remove any points from Qdrant whose tile IDs no longer exist in PostgreSQL.
    Cleans up old test runs and deleted scenes.
    """
    try:
        from app.models.tile import Tile
        stmt = select(Tile.id)
        res = await session.execute(stmt)
        active_ids = {str(row[0]) for row in res.all()}
        pruned = vector_store.prune_orphaned_vectors(active_ids)
        stats = vector_store.get_all_statistics()
        return {
            "status": "success",
            "active_tiles_in_db": len(active_ids),
            "pruned_by_collection": pruned,
            "current_statistics": stats,
        }
    except Exception as exc:
        logger.error("prune_orphans_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to prune orphaned vectors: {exc}",
        )
