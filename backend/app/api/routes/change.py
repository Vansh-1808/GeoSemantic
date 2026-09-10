import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.db.database import get_session
from app.models.change_event import ChangeEvent
from app.schemas.change import (
    ChangeAnalyzeRequest,
    ChangeAnalyzeResponse,
    ChangeEventResponse,
)
from app.services.change_detection import ChangeDetectionService

router = APIRouter(tags=["change"])

@router.post("/analyze", response_model=ChangeAnalyzeResponse)
async def analyze_change(
    request: ChangeAnalyzeRequest,
    db: AsyncSession = Depends(get_session)
):
    """Run the multi-temporal change detection pipeline for a given AOI and date range."""
    service = ChangeDetectionService(db)
    
    candidates_found, events = await service.analyze_change(request)
    
    return ChangeAnalyzeResponse(
        message=f"Change detection analysis complete. Found {candidates_found} candidates.",
        candidates_found=candidates_found,
        events=events
    )

@router.get("/{change_id}", response_model=ChangeEventResponse)
async def get_change_event(change_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    """Get a specific change event by ID."""
    event = await db.get(ChangeEvent, change_id)
    if not event:
        raise HTTPException(status_code=404, detail="Change event not found")
    return event

@router.get("", response_model=List[ChangeEventResponse])
async def list_change_events(
    limit: int = 50,
    skip: int = 0,
    min_confidence: float = 0.0,
    db: AsyncSession = Depends(get_session)
):
    """List change events – only returns events whose before/after tiles still exist in the DB."""
    from app.models.tile import Tile
    from sqlalchemy import exists

    # Only show events where BOTH tiles still exist (eliminates stale/orphaned records)
    before_exists = exists().where(Tile.id == ChangeEvent.before_tile_id)
    after_exists = exists().where(Tile.id == ChangeEvent.after_tile_id)

    query = (
        select(ChangeEvent)
        .where(
            and_(
                ChangeEvent.final_confidence >= min_confidence,
                ChangeEvent.before_tile_id.isnot(None),
                ChangeEvent.after_tile_id.isnot(None),
                before_exists,
                after_exists,
            )
        )
        .order_by(ChangeEvent.detected_at.desc())
        .offset(skip)
        .limit(limit)
    )
    result = await db.scalars(query)
    events = result.all()
    return events
