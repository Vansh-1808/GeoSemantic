"""
Pydantic schemas for Scene and Tile — request/response models.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Scene Schemas ─────────────────────────────────────────────

class SceneBase(BaseModel):
    filename: str
    sensor: str | None = None
    platform: str | None = None
    acquisition_date: datetime | None = None


class SceneCreate(SceneBase):
    source_path: str


class SceneInDB(SceneBase):
    id: uuid.UUID
    source_path: str
    file_size_bytes: int | None = None
    file_checksum: str | None = None
    crs_epsg: int | None = None
    width_px: int | None = None
    height_px: int | None = None
    num_bands: int | None = None
    resolution_x_m: float | None = None
    resolution_y_m: float | None = None
    bbox_west: float | None = None
    bbox_south: float | None = None
    bbox_east: float | None = None
    bbox_north: float | None = None
    cloud_cover_pct: float | None = None
    quality_score: float | None = None
    quality_details: dict | None = None
    ingestion_status: str
    ingestion_error: str | None = None
    tile_count: int
    embedded_count: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class SceneSummary(BaseModel):
    """Lightweight scene card for list views."""
    id: uuid.UUID
    filename: str
    sensor: str | None = None
    acquisition_date: datetime | None = None
    tile_count: int
    embedded_count: int
    quality_score: float | None = None
    cloud_cover_pct: float | None = None
    ingestion_status: str
    bbox_west: float | None = None
    bbox_south: float | None = None
    bbox_east: float | None = None
    bbox_north: float | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Tile Schemas ──────────────────────────────────────────────

class TileSummary(BaseModel):
    """Tile card used in search results and spatial views."""
    id: uuid.UUID
    scene_id: uuid.UUID
    tile_col: int
    tile_row: int
    tile_size: int | None = 512
    pixel_x_off: int | None = None
    pixel_y_off: int | None = None
    pixel_width: int | None = None
    pixel_height: int | None = None
    center_lon: float | None = None
    center_lat: float | None = None
    bbox_west: float | None = None
    bbox_south: float | None = None
    bbox_east: float | None = None
    bbox_north: float | None = None
    acquisition_date: datetime | None = None
    sensor: str | None = None
    quality_score: float | None = None
    cloud_cover_pct: float | None = None
    thumbnail_url: str | None = None
    preview_url: str | None = None
    is_valid: bool = True

    model_config = {"from_attributes": True}


class TileDetail(TileSummary):
    """Full tile detail including provenance and pixel offsets."""
    tile_path: str | None = None
    thumbnail_path: str | None = None
    resolution_m: float | None = None
    nodata_ratio: float | None = None
    remoteclip_vector_id: str | None = None
    dino_vector_id: str | None = None
    embedding_model: str | None = None
    embedded_at: datetime | None = None
    processing_history: list[Any] | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class TileListResponse(BaseModel):
    """Paginated list of tiles."""
    tiles: list[TileSummary]
    total: int
    page: int
    page_size: int


# ── Ingestion & Processing Schemas ────────────────────────────

class ProcessSceneRequest(BaseModel):
    """Configuration for scene tiling and preprocessing."""
    tile_size: int = Field(default=512, ge=64, le=4096, description="Tile size in pixels")
    overlap_px: int = Field(default=0, ge=0, le=2048, description="Tile overlap in pixels")
    normalize: bool = Field(default=True, description="Apply robust 2%-98% percentile stretch normalization")
    min_valid_pixel_ratio: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum ratio of non-nodata pixels")
    generate_preview: bool = Field(default=True, description="Generate high-resolution RGB preview image")


class ProcessSceneResponse(BaseModel):
    """Result of triggering scene tiling."""
    scene_id: uuid.UUID
    job_id: uuid.UUID
    status: str
    tile_count: int | None = None
    processing_time_seconds: float | None = None
    message: str


class IngestionRequest(BaseModel):
    """Response returned when a scene ingestion is started."""
    scene_id: uuid.UUID
    job_id: uuid.UUID
    message: str
    status: str = "PROCESSING"


class IngestionStatus(BaseModel):
    """Real-time ingestion progress."""
    scene_id: uuid.UUID
    job_id: uuid.UUID
    status: str
    progress_pct: float = 0.0
    message: str | None = None
    error: str | None = None
    tile_count: int = 0
    embedded_count: int = 0
    processing_time_seconds: float | None = None


class SceneListResponse(BaseModel):
    scenes: list[SceneSummary]
    total: int
    page: int
    page_size: int
