"""
ORM model for provenance tracking.
Every processing operation is recorded here.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class ProvenanceRecord(Base):
    """
    Immutable audit log for any processing operation on any entity.
    Provides full traceability: source → tile → embedding → analysis → decision.
    """

    __tablename__ = "provenance_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.uuid_generate_v4()
    )

    # ── Entity Reference ──────────────────────────────────────
    entity_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    # scene | tile | change_event | analyst_decision

    # ── Operation ─────────────────────────────────────────────
    operation: Mapped[str] = mapped_column(String(128), nullable=False)
    # ingestion | tiling | embedding | change_detection | analyst_review | export

    # ── Details ───────────────────────────────────────────────
    details: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    # Contains: model name, model version, model checksum,
    #   preprocessing steps, parameters, input hash, etc.

    operator: Mapped[str] = mapped_column(String(128), default="system", nullable=False)

    # ── Timestamps ───────────────────────────────────────────
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_provenance_entity", "entity_id", "entity_type"),
        Index("ix_provenance_operation", "operation"),
        Index("ix_provenance_recorded_at", "recorded_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<ProvenanceRecord entity={self.entity_type}/{self.entity_id} "
            f"op={self.operation}>"
        )


class ProcessingJob(Base):
    """
    Tracks async processing jobs (ingestion, embedding runs, change analysis).
    Enables progress reporting and retry logic.
    """

    __tablename__ = "processing_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.uuid_generate_v4()
    )
    job_type: Mapped[str] = mapped_column(String(64), nullable=False)
    # ingestion | embedding | change_analysis | clustering

    status: Mapped[str] = mapped_column(String(32), default="PENDING", nullable=False)
    # PENDING | RUNNING | COMPLETED | FAILED | CANCELLED

    related_entity_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    related_entity_type: Mapped[str | None] = mapped_column(String(64), nullable=True)

    progress_pct: Mapped[float] = mapped_column(default=0.0, nullable=False)
    progress_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    params: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_job_status", "status"),
        Index("ix_job_type", "job_type"),
    )
