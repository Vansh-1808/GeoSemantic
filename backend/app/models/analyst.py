"""
ORM model for analyst review decisions.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class AnalystDecision(Base):
    """
    Records an analyst's decision on a ChangeEvent.
    Supports confirm, reject, needs-review, and optional type correction.
    """

    __tablename__ = "analyst_decisions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.uuid_generate_v4()
    )
    change_event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("change_events.id", ondelete="CASCADE"),
        nullable=False,
    )

    # ── Decision ──────────────────────────────────────────────
    decision: Mapped[str] = mapped_column(String(32), nullable=False)
    # CONFIRMED | REJECTED | NEEDS_REVIEW

    corrected_change_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Analyst Identity ──────────────────────────────────────
    analyst_id: Mapped[str] = mapped_column(String(128), default="default", nullable=False)

    # ── Timestamp ─────────────────────────────────────────────
    decided_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ── Relationships ─────────────────────────────────────────
    change_event: Mapped["ChangeEvent"] = relationship(  # noqa: F821
        "ChangeEvent", back_populates="analyst_decisions"
    )

    __table_args__ = (
        CheckConstraint(
            "decision IN ('CONFIRMED','REJECTED','NEEDS_REVIEW')",
            name="ck_analyst_decision",
        ),
    )

    def __repr__(self) -> str:
        return f"<AnalystDecision change={self.change_event_id} decision={self.decision}>"


class FeedbackSignal(Base):
    """
    Stores analyst feedback signals for reranking.
    Positive signals boost similar results; negative signals penalize them.
    """

    __tablename__ = "feedback_signals"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.uuid_generate_v4()
    )
    tile_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tiles.id", ondelete="SET NULL"),
        nullable=True,
    )
    signal_type: Mapped[str] = mapped_column(String(32), nullable=False)
    # POSITIVE | NEGATIVE

    context: Mapped[str | None] = mapped_column(String(256), nullable=True)
    # E.g., query text that led to this result

    analyst_id: Mapped[str] = mapped_column(String(128), default="default", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        CheckConstraint(
            "signal_type IN ('POSITIVE','NEGATIVE')", name="ck_signal_type"
        ),
    )
