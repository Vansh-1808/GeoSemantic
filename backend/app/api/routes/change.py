import uuid
from typing import List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.db.database import get_session
from app.models.change_event import ChangeEvent
from app.schemas.change import (
    ChangeAnalyzeRequest,
    ChangeAnalyzeResponse,
    ChangeEventResponse,
    ConfidenceConfigResponse,
    ConfidenceConfigUpdateRequest,
    ConfidenceEvaluationRequest,
    ConfidenceEvaluationResponse,
)
from app.services.change_detection import ChangeDetectionService
from app.services.confidence_engine import confidence_engine

router = APIRouter(tags=["change"])


@router.post("/analyze", response_model=ChangeAnalyzeResponse)
async def analyze_change(
    request: ChangeAnalyzeRequest,
    db: AsyncSession = Depends(get_session)
):
    """Run the multi-temporal change detection pipeline for a given AOI and date range."""
    # H2: Service no longer holds db as instance state; db is passed per-call.
    service = ChangeDetectionService()
    candidates_found, events = await service.analyze_change(request, db)

    return ChangeAnalyzeResponse(
        message=f"Change detection analysis complete. Found {candidates_found} candidates.",
        candidates_found=candidates_found,
        events=events
    )


# ── Multi-Factor Change Confidence Engine Endpoints ────────────────────────────

@router.get("/confidence/config", response_model=ConfidenceConfigResponse)
async def get_confidence_config():
    """Retrieve current configurable weights and decision thresholds for the confidence engine."""
    return confidence_engine.get_config()


@router.put("/confidence/config", response_model=ConfidenceConfigResponse)
async def update_confidence_config(request: ConfidenceConfigUpdateRequest):
    """Dynamically update weights or thresholds for the confidence engine at runtime."""
    confidence_engine.update_config(
        weights=request.weights,
        thresholds=request.thresholds,
    )
    return confidence_engine.get_config()


@router.post("/confidence/evaluate", response_model=ConfidenceEvaluationResponse)
async def evaluate_confidence(request: ConfidenceEvaluationRequest):
    """
    Directly evaluate multi-factor confidence from raw signals.
    Provides complete transparency into weights applied, penalties, tier, and suppression reasons.
    """
    # If a temporary weight override is supplied in the request, evaluate with it
    active_engine = confidence_engine
    if request.weights_override:
        from app.services.confidence_engine import MultiFactorConfidenceEngine, ConfidenceWeights
        custom_weights = ConfidenceWeights(**{
            **confidence_engine.weights.model_dump(),
            **request.weights_override,
        })
        active_engine = MultiFactorConfidenceEngine(
            weights=custom_weights,
            thresholds=confidence_engine.thresholds,
        )

    res = active_engine.evaluate_signals(
        visual_change=request.visual_change,
        semantic_change=request.semantic_change,
        image_quality=request.image_quality,
        cloud_score=request.cloud_score,
        shadow_score=request.shadow_score,
        registration_confidence=request.registration_confidence,
        sensor_compatibility=request.sensor_compatibility,
        seasonal_compatibility=request.seasonal_compatibility,
        spectral_evidence=request.spectral_evidence,
        spatial_consistency=request.spatial_consistency,
        is_seasonal_vegetation=request.is_seasonal_vegetation,
        change_type=request.change_type,
        query_match_score=request.query_match_score,
    )

    return ConfidenceEvaluationResponse(
        final_confidence=res.final_confidence,
        confidence_tier=res.confidence_tier,
        is_suppressed=res.is_suppressed,
        reasons=res.reasons,
        positive_factors=res.positive_factors,
        factor_breakdown={k: v.model_dump() for k, v in res.factor_breakdown.items()},
        penalties_applied=res.penalties_applied,
        raw_signals=res.raw_signals,
        weights_applied=res.weights_applied,
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
