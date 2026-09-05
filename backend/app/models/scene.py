"""
ORM model for satellite scenes.
A scene represents one ingested GeoTIFF source file.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from geoalchemy2 import Geometry
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class Scene(Base):
    """
    Represents one ingested satellite imagery source file.
    All tile records reference back to a Scene.
    """

    __tablename__ = "scenes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.uuid_generate_v4()
    )
    # ── Source file ────────────────────────────────────────────
    source_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_checksum: Mapped[str | None] = mapped_column(String(64), nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # ── Sensor / Platform ─────────────────────────────────────
    sensor: Mapped[str | None] = mapped_column(String(128), nullable=True)
    platform: Mapped[str | None] = mapped_column(String(128), nullable=True)
    product_type: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # ── Temporal ──────────────────────────────────────────────
    acquisition_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    acquisition_date_str: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # ── Raster Properties ────────────────────────────────────
    crs_wkt: Mapped[str | None] = mapped_column(Text, nullable=True)
    crs_epsg: Mapped[int | None] = mapped_column(Integer, nullable=True)
    width_px: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height_px: Mapped[int | None] = mapped_column(Integer, nullable=True)
    num_bands: Mapped[int | None] = mapped_column(Integer, nullable=True)
    band_descriptions: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    resolution_x_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    resolution_y_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    nodata_value: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Geographic Footprint (always EPSG:4326 WGS84) ─────────
    # Stored as a polygon for spatial querying
    footprint: Mapped[object | None] = mapped_column(
        Geometry("POLYGON", srid=4326), nullable=True
    )
    # Bounding box corners in WGS84 degrees
    bbox_west: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_south: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_east: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_north: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Quality ───────────────────────────────────────────────
    cloud_cover_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_details: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # ── Processing Status ────────────────────────────────────
    ingestion_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="PENDING",
        server_default="PENDING"
    )
    ingestion_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    tile_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    embedded_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # ── Raw Metadata ──────────────────────────────────────────
    raw_metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # ── Timestamps ───────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # ── Relationships ─────────────────────────────────────────
    tiles: Mapped[list["Tile"]] = relationship(  # noqa: F821
        "Tile", back_populates="scene", cascade="all, delete-orphan"
    )
    provenance_records: Mapped[list["ProvenanceRecord"]] = relationship(  # noqa: F821
        "ProvenanceRecord",
        primaryjoin="and_(ProvenanceRecord.entity_type=='scene', "
                    "foreign(ProvenanceRecord.entity_id)==Scene.id)",
        viewonly=True,
    )

    __table_args__ = (
        CheckConstraint(
            "ingestion_status IN ('PENDING','PROCESSING','COMPLETED','FAILED')",
            name="ck_scene_status",
        ),
        CheckConstraint(
            "cloud_cover_pct IS NULL OR (cloud_cover_pct >= 0 AND cloud_cover_pct <= 100)",
            name="ck_cloud_cover_range",
        ),
        Index("ix_scene_acquisition_date", "acquisition_date"),
        Index("ix_scene_sensor", "sensor"),
        Index("ix_scene_status", "ingestion_status"),
        Index("ix_scene_footprint", "footprint", postgresql_using="gist"),
    )

    def __repr__(self) -> str:
        return f"<Scene id={self.id} filename={self.filename!r} status={self.ingestion_status}>"
