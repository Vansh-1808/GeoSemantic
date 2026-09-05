"""
ORM model for image tiles.
Each tile is a geographic sub-region of a parent Scene.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from geoalchemy2 import Geometry
from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class Tile(Base):
    """
    Represents a 256×256 (or configurable) geographic tile extracted from a Scene.
    Every tile maintains full traceability to its source.
    """

    __tablename__ = "tiles"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.uuid_generate_v4()
    )

    # ── Source ────────────────────────────────────────────────
    scene_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("scenes.id", ondelete="CASCADE"),
        nullable=False,
    )

    # ── Grid Position ─────────────────────────────────────────
    tile_col: Mapped[int] = mapped_column(Integer, nullable=False)
    tile_row: Mapped[int] = mapped_column(Integer, nullable=False)
    tile_size: Mapped[int] = mapped_column(Integer, nullable=False, default=256)

    # ── Pixel Coordinates in Source Raster ───────────────────
    pixel_x_off: Mapped[int] = mapped_column(Integer, nullable=False)
    pixel_y_off: Mapped[int] = mapped_column(Integer, nullable=False)
    pixel_width: Mapped[int] = mapped_column(Integer, nullable=False)
    pixel_height: Mapped[int] = mapped_column(Integer, nullable=False)

    # ── Geographic Footprint (EPSG:4326) ─────────────────────
    footprint: Mapped[object | None] = mapped_column(
        Geometry("POLYGON", srid=4326), nullable=True
    )
    center_lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    center_lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_west: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_south: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_east: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_north: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── File References ───────────────────────────────────────
    tile_path: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    thumbnail_path: Mapped[str | None] = mapped_column(String(2048), nullable=True)

    # ── Inherited Metadata (denormalized for query speed) ──────
    acquisition_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    sensor: Mapped[str | None] = mapped_column(String(128), nullable=True)
    resolution_m: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Quality ───────────────────────────────────────────────
    cloud_cover_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    nodata_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_valid: Mapped[bool] = mapped_column(default=True, nullable=False)

    # ── Embedding Status ──────────────────────────────────────
    remoteclip_vector_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    dino_vector_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # ── Processing History ────────────────────────────────────
    processing_history: Mapped[list | None] = mapped_column(JSONB, nullable=True)

    # ── Timestamps ───────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ── Relationships ─────────────────────────────────────────
    scene: Mapped["Scene"] = relationship("Scene", back_populates="tiles")  # noqa: F821

    __table_args__ = (
        Index("ix_tile_scene_id", "scene_id"),
        Index("ix_tile_acquisition_date", "acquisition_date"),
        Index("ix_tile_sensor", "sensor"),
        Index("ix_tile_quality", "quality_score"),
        Index("ix_tile_footprint", "footprint", postgresql_using="gist"),
        Index("ix_tile_embedded", "remoteclip_vector_id"),
    )

    def __repr__(self) -> str:
        return f"<Tile id={self.id} scene={self.scene_id} col={self.tile_col} row={self.tile_row}>"
