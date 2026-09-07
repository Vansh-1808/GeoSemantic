"""
Pydantic schemas for natural language semantic search and retrieval.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class SemanticSearchRequest(BaseModel):
    """Payload for natural language semantic text-to-satellite image retrieval with GeoSemantic fusion."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural language query string describing satellite scene (e.g., 'commercial harbor with docked ships')",
    )
    top_k: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of ranked results to return",
    )
    start_date: Optional[datetime] = Field(
        default=None,
        description="Earliest acquisition date filter",
    )
    end_date: Optional[datetime] = Field(
        default=None,
        description="Latest acquisition date filter",
    )
    sensor: Optional[str] = Field(
        default=None,
        description="Single satellite sensor filter (e.g., 'Sentinel-2B')",
    )
    sensors: Optional[List[str]] = Field(
        default=None,
        description="Multiple satellite sensor filter (e.g., ['Sentinel-2A', 'Sentinel-2B'])",
    )
    min_quality: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable tile quality score threshold (0.0 - 1.0)",
    )
    max_cloud_cover: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Maximum acceptable tile cloud cover percentage (0.0 - 100.0)",
    )
    aoi: Optional[List[float]] = Field(
        default=None,
        description="Area of Interest bounding box [west, south, east, north] in EPSG:4326 degrees",
    )
    aoi_bbox: Optional[List[float]] = Field(
        default=None,
        description="Explicit Area of Interest bounding box [west, south, east, north]",
    )
    aoi_polygon: Optional[List[List[float]]] = Field(
        default=None,
        description="Area of Interest polygon coordinates [[lon, lat], [lon, lat], ...] in EPSG:4326 degrees",
    )
    spatial_filter_mode: Literal["intersects", "within"] = Field(
        default="intersects",
        description="Spatial relationship mode with AOI polygon: 'intersects' or 'within'",
    )
    scene_id: Optional[uuid.UUID] = Field(
        default=None,
        description="Optional filter restricting results to a single source scene",
    )


class SemanticSearchResultItem(BaseModel):
    """Individual retrieved satellite tile ranking result."""
    tile_id: uuid.UUID
    similarity_score: float = Field(..., description="Cosine similarity score between query and tile (0.0 to 1.0)")
    rank: int = Field(..., description="1-based relevance ranking")
    scene_id: uuid.UUID
    scene_name: str
    sensor: Optional[str] = None
    acquisition_date: Optional[datetime] = None
    quality_score: Optional[float] = None
    cloud_cover_pct: Optional[float] = None
    tile_col: int
    tile_row: int
    center_coordinates: Dict[str, float] = Field(
        ..., description="Center coordinates: {'lon': float, 'lat': float}"
    )
    bbox: Dict[str, float] = Field(
        ..., description="Bounding box: {'west': float, 'south': float, 'east': float, 'north': float}"
    )
    footprint_geojson: Optional[Dict[str, Any]] = Field(
        default=None, description="GeoJSON polygon coordinates for spatial map rendering"
    )
    thumbnail_url: str = Field(..., description="URL to access 128px PNG thumbnail")
    preview_url: str = Field(..., description="URL to access 512px RGB visualization preview")
    landcover: Optional[Dict[str, float]] = Field(
        default=None, description="Physical optical landcover breakdown: water_pct, veg_pct, urban_pct"
    )
    provenance: Optional[Dict[str, Any]] = Field(
        default=None, description="Full audit traceability record from PostgreSQL"
    )


class SemanticSearchResponse(BaseModel):
    """Consolidated semantic search results and telemetry."""
    query: str
    total_found: int
    execution_time_ms: float
    text_embedding_time_ms: float
    vector_search_time_ms: float
    spatial_filter_time_ms: Optional[float] = 0.0
    device: str
    model_used: str
    results: List[SemanticSearchResultItem]
    filters_applied: Dict[str, Any]


class SearchFiltersResponse(BaseModel):
    """Metadata boundaries for populating frontend filter selectors."""
    sensors: List[str]
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    total_indexed_tiles: int
    min_quality: Optional[float] = 0.0
    max_cloud_cover: Optional[float] = 100.0
    sensor_counts: Optional[Dict[str, int]] = None


class VisualSearchRequest(BaseModel):
    """Filters for image-to-image or tile-based visual similarity retrieval."""
    top_k: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of visually similar tiles to return",
    )
    sensor: Optional[str] = Field(
        default=None,
        description="Optional filter by satellite sensor (e.g. 'Sentinel-2B')",
    )
    sensors: Optional[List[str]] = Field(
        default=None,
        description="Optional filter by multiple satellite sensors",
    )
    min_quality: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable tile quality threshold (0.0 to 1.0)",
    )
    max_cloud_cover: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Maximum acceptable tile cloud cover percentage (0.0 to 100.0)",
    )
    start_date: Optional[datetime] = Field(
        default=None,
        description="Earliest acquisition date filter",
    )
    end_date: Optional[datetime] = Field(
        default=None,
        description="Latest acquisition date filter",
    )
    aoi: Optional[List[float]] = Field(
        default=None,
        description="Area of Interest bounding box [west, south, east, north] in EPSG:4326 degrees",
    )
    aoi_bbox: Optional[List[float]] = Field(
        default=None,
        description="Explicit Area of Interest bounding box [west, south, east, north]",
    )
    aoi_polygon: Optional[List[List[float]]] = Field(
        default=None,
        description="Area of Interest polygon coordinates [[lon, lat], [lon, lat], ...] in EPSG:4326 degrees",
    )
    spatial_filter_mode: Literal["intersects", "within"] = Field(
        default="intersects",
        description="Spatial relationship mode with AOI polygon: 'intersects' or 'within'",
    )
    scene_id: Optional[uuid.UUID] = Field(
        default=None,
        description="Optional filter restricting results to a single source scene",
    )


class VisualSearchResponse(BaseModel):
    """Consolidated visual similarity search response."""
    query_type: str = Field(..., description="'tile_id' or 'uploaded_image'")
    reference_tile_id: Optional[uuid.UUID] = None
    reference_thumbnail_url: Optional[str] = None
    total_found: int
    execution_time_ms: float
    visual_embedding_time_ms: float
    vector_search_time_ms: float
    spatial_filter_time_ms: Optional[float] = 0.0
    device: str
    model_used: str = Field(default="DINOv2 (vit_small_patch14_dinov2.lvd142m)")
    embedding_dimension: int = Field(default=384)
    results: List[SemanticSearchResultItem]
    filters_applied: Dict[str, Any]

