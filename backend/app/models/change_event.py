"""
ORM model for change detection events.
Each event represents a detected (or suppressed) change between two tiles.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from geoalchemy2 import Geometry
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class ChangeEvent(Base):
    """
    Represents a potential change detected between two observations of the same area.
    Contains the full multi-factor confidence breakdown.
    """

    __tablename__ = "change_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.uuid_generate_v4()
    )

    # ── Tile Pair ─────────────────────────────────────────────
    before_tile_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tiles.id", ondelete="SET NULL"),
        nullable=True,
    )
    after_tile_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tiles.id", ondelete="SET NULL"),
        nullable=True,
    )

    # ── Temporal ──────────────────────────────────────────────
    before_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    after_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    earliest_supported_observation: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    temporal_persistence_count: Mapped[int] = mapped_column(default=0, nullable=False)

    # ── Geography ─────────────────────────────────────────────
    aoi_footprint: Mapped[object | None] = mapped_column(
        Geometry("POLYGON", srid=4326), nullable=True
    )
    center_lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    center_lat: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Classification ────────────────────────────────────────
    change_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # E.g.: construction, clearance, road_development, water_increase,
    #       water_decrease, vegetation_change, demolition, unknown

    # ── Confidence Engine ─────────────────────────────────────
    final_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Individual factor scores
    visual_change_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    semantic_change_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    spectral_change_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    registration_quality: Mapped[float | None] = mapped_column(Float, nullable=True)
    temporal_persistence_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Penalty factors
    cloud_penalty: Mapped[float | None] = mapped_column(Float, nullable=True)
    shadow_penalty: Mapped[float | None] = mapped_column(Float, nullable=True)
    seasonal_penalty: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Full factor breakdown for UI display
    confidence_breakdown: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # ── False-Alarm Suppression ───────────────────────────────
    is_suppressed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    suppression_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Evidence ──────────────────────────────────────────────
    evidence: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    # Contains: structural changes, texture changes, spectral indicators,
    # registration details, temporal series, etc.

    # ── Analyst Decision ──────────────────────────────────────
    review_status: Mapped[str] = mapped_column(
        String(32), default="PENDING", nullable=False
    )

    # ── Timestamps ───────────────────────────────────────────
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # ── Relationships ─────────────────────────────────────────
    before_tile: Mapped["Tile"] = relationship(  # noqa: F821
        "Tile", foreign_keys=[before_tile_id]
    )
    after_tile: Mapped["Tile"] = relationship(  # noqa: F821
        "Tile", foreign_keys=[after_tile_id]
    )
    analyst_decisions: Mapped[list["AnalystDecision"]] = relationship(  # noqa: F821
        "AnalystDecision", back_populates="change_event", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(
            "review_status IN ('PENDING','CONFIRMED','REJECTED','NEEDS_REVIEW')",
            name="ck_change_review_status",
        ),
        CheckConstraint(
            "final_confidence IS NULL OR (final_confidence >= 0 AND final_confidence <= 1)",
            name="ck_confidence_range",
        ),
        Index("ix_change_review_status", "review_status"),
        Index("ix_change_type", "change_type"),
        Index("ix_change_confidence", "final_confidence"),
        Index("ix_change_aoi", "aoi_footprint", postgresql_using="gist"),
        Index("ix_change_before_date", "before_date"),
        Index("ix_change_after_date", "after_date"),
    )

    def __repr__(self) -> str:
        return (
            f"<ChangeEvent id={self.id} type={self.change_type!r} "
            f"confidence={self.final_confidence:.2f if self.final_confidence else 'N/A'} "
            f"suppressed={self.is_suppressed}>"
        )
