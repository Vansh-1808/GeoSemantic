"""
Ingestion API routes.
POST /api/ingest/scene     — upload and ingest a GeoTIFF
GET  /api/ingest/status/{job_id} — poll job status
GET  /api/ingest/scenes    — list all scenes
GET  /api/ingest/scenes/{scene_id} — scene detail
DELETE /api/ingest/scenes/{scene_id} — remove scene
"""
from __future__ import annotations

import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.db.database import get_session
from app.models.provenance import ProcessingJob
from app.models.scene import Scene
from app.schemas.scene import (
    IngestionRequest,
    IngestionStatus,
    SceneInDB,
    SceneListResponse,
    SceneSummary,
)
from app.services import ingestion as ingestion_svc

router = APIRouter()
logger = get_logger(__name__)

ALLOWED_EXTENSIONS = {".tif", ".tiff", ".geotiff"}
MAX_FILE_SIZE_BYTES = 5 * 1024 ** 3  # 5 GB


@router.post("/scene", response_model=IngestionRequest, status_code=202)
async def ingest_scene(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    """
    Upload a GeoTIFF and start asynchronous ingestion.
    Returns immediately with a job_id for status polling.
    """
    # Validate file extension
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Accepted: {ALLOWED_EXTENSIONS}",
        )

    # Save uploaded file to imagery directory
    dest_path = settings.imagery_dir / file.filename
    if dest_path.exists():
        # Avoid clobbering existing files — append uuid
        stem = Path(file.filename).stem
        dest_path = settings.imagery_dir / f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"

    try:
        with open(dest_path, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                out.write(chunk)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")

    # Create a ProcessingJob record
    job_id = uuid.uuid4()
    job = ProcessingJob(
        id=job_id,
        job_type="ingestion",
        status="PENDING",
        params={"file_path": str(dest_path), "filename": file.filename},
        related_entity_type="scene",
    )
    session.add(job)
    await session.commit()

    # Queue the ingestion pipeline as a background task
    background_tasks.add_task(
        _run_ingestion_background, dest_path, job_id
    )

    logger.info("ingestion_queued", filename=file.filename, job_id=str(job_id))
    return IngestionRequest(
        scene_id=uuid.UUID(int=0),  # Not yet assigned; poll status for actual scene_id
        job_id=job_id,
        message=f"Ingestion started for '{file.filename}'",
        status="PROCESSING",
    )


@router.get("/status/{job_id}", response_model=IngestionStatus)
async def get_ingestion_status(
    job_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    """Poll the status of an ingestion job."""
    job = await session.get(ProcessingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Try to find the scene linked to this job
    scene_id = job.related_entity_id or uuid.UUID(int=0)
    tile_count = 0
    embedded_count = 0

    if job.related_entity_id:
        scene = await session.get(Scene, job.related_entity_id)
        if scene:
            scene_id = scene.id
            tile_count = scene.tile_count
            embedded_count = scene.embedded_count

    return IngestionStatus(
        scene_id=scene_id,
        job_id=job_id,
        status=job.status,
        progress_pct=job.progress_pct,
        message=job.progress_message,
        error=job.error_message,
        tile_count=tile_count,
        embedded_count=embedded_count,
    )


@router.get("/scenes", response_model=SceneListResponse)
async def list_scenes(
    page: int = 1,
    page_size: int = 20,
    status: str | None = None,
    session: AsyncSession = Depends(get_session),
):
    """List all ingested scenes, paginated."""
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


@router.get("/scenes/{scene_id}", response_model=SceneInDB)
async def get_scene(
    scene_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    """Retrieve full scene details."""
    scene = await session.get(Scene, scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found")
    return SceneInDB.model_validate(scene)


@router.delete("/scenes/{scene_id}", status_code=204)
async def delete_scene(
    scene_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    """
    Remove a scene and all its tiles from the database.
    Also removes tile files and thumbnails from disk.
    """
    scene = await session.get(Scene, scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found")

    # Remove tile directory from disk
    for subdir in [settings.tiles_dir, settings.thumbnails_dir, settings.previews_dir]:
        scene_dir = subdir / str(scene_id)
        if scene_dir.exists():
            shutil.rmtree(scene_dir, ignore_errors=True)

    # Remove vectors from Qdrant
    try:
        from app.services.vector_store import vector_store
        vector_store.delete_scene_points(str(scene_id))
    except Exception as exc:
        logger.warning("failed_to_delete_qdrant_points_on_scene_delete", scene_id=str(scene_id), error=str(exc))

    await session.delete(scene)
    await session.commit()
    logger.info("scene_deleted", scene_id=str(scene_id))


# ─────────────────────────────────────────────────────────────
# Background task runner
# ─────────────────────────────────────────────────────────────

async def _run_ingestion_background(file_path: Path, job_id: uuid.UUID) -> None:
    """
    Runs the ingestion service in a fresh database session.
    Background tasks cannot reuse the request's session.
    """
    from app.db.database import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        try:
            job = await session.get(ProcessingJob, job_id)
            if job:
                job.status = "RUNNING"
                job.started_at = datetime.now(timezone.utc)
                await session.commit()

            scene = await ingestion_svc.ingest_scene(file_path, session, job_id)

            # Link scene to job
            job = await session.get(ProcessingJob, job_id)
            if job:
                job.related_entity_id = scene.id
                await session.commit()

            # Auto-generate tile embeddings so the new custom dataset is immediately searchable
            try:
                from app.services.tile_embedding import tile_embedding_service
                logger.info("auto_generating_embeddings_for_ingested_scene", scene_id=str(scene.id))
                await tile_embedding_service.generate_scene_embeddings(
                    scene_id=scene.id,
                    session=session,
                    force_reembed=False,
                    batch_size=8,
                )
            except Exception as emb_exc:
                logger.warning("auto_embedding_generation_failed", scene_id=str(scene.id), error=str(emb_exc))

        except Exception as exc:
            logger.exception("background_ingestion_failed", error=str(exc))
