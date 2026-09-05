"""
Pydantic schemas for tile embedding generation, status reporting, and vector index statistics.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class EmbeddingGenerateRequest(BaseModel):
    """Configuration for triggering tile vector embedding."""
    force_reembed: bool = Field(
        default=False,
        description="If True, re-embeds all tiles even if they already have vector IDs.",
    )
    batch_size: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Number of tiles to embed in parallel per model forward pass.",
    )


class EmbeddingJobResponse(BaseModel):
    """Response returned when an embedding generation job is created or completed."""
    job_id: uuid.UUID
    scene_id: uuid.UUID
    status: Literal["PENDING", "RUNNING", "COMPLETED", "FAILED"] = "COMPLETED"
    total_tiles: int
    embedded_count: int
    skipped_count: int
    duration_seconds: Optional[float] = None
    message: str


class EmbeddingStatusResponse(BaseModel):
    """Real-time embedding progress and completion status for a specific scene."""
    scene_id: uuid.UUID
    total_tiles: int
    embedded_tiles: int
    pending_tiles: int
    progress_pct: float
    status: Literal["NOT_STARTED", "IN_PROGRESS", "COMPLETED", "PARTIAL", "FAILED"]
    last_embedded_at: Optional[datetime] = None
    models: List[str] = Field(default_factory=lambda: ["RemoteCLIP", "DINOv2"])
    collections: List[str] = Field(default_factory=lambda: ["remoteclip_tiles", "dino_tiles"])
    active_job_id: Optional[uuid.UUID] = None


class VectorCollectionStats(BaseModel):
    """Individual telemetry for a Qdrant collection."""
    collection_name: str
    exists: bool
    points_count: int
    vectors_count: int
    vector_size: int
    distance: str
    status: str


class VectorStatisticsResponse(BaseModel):
    """Global vector index statistics and disk storage telemetry."""
    total_vectors: int
    total_points: int
    collections: Dict[str, VectorCollectionStats]
    storage_usage_bytes: int
    storage_usage_mb: float
    storage_path: str
    qdrant_mode: str
    timestamp: str
