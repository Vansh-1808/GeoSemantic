"""
Pydantic schemas for Image Quality Assessment and Confounding Factors.
"""
from __future__ import annotations

import uuid
from typing import Any, List

from pydantic import BaseModel, Field


class TileQualityResponse(BaseModel):
    """Quality metrics and suitability for a single tile."""
    tile_id: uuid.UUID
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Composite physical quality score [0.0, 1.0]")
    usable_for_change_analysis: bool = Field(..., description="Whether tile meets criteria for robust change detection")
    suitability: str = Field(..., description="'Suitable', 'Caution', or 'Poor Quality'")
    cloud_score: float = Field(..., ge=0.0, le=1.0, description="Fraction of tile occluded by cloud or dense haze")
    shadow_score: float = Field(..., ge=0.0, le=1.0, description="Fraction of land surface occluded by shadows")
    nodata_score: float = Field(..., ge=0.0, le=1.0, description="Fraction of tile lacking raster data (masked/black borders)")
    noise_score: float = Field(..., ge=0.0, le=1.0, description="Sensor banding or high-frequency row-to-row noise coefficient")
    saturation_score: float = Field(..., ge=0.0, le=1.0, description="Overexposed or sensor-saturated pixel fraction")
    cloud_cover_pct: float = Field(..., ge=0.0, le=100.0, description="Cloud cover percentage (0-100%)")
    nodata_ratio: float = Field(..., ge=0.0, le=1.0, description="No-data pixel ratio (0.0-1.0)")
    analysis_method: str = Field(default="deterministic_rgb_photometry", description="Assessment algorithm employed")
    satellite_mask_available: bool = Field(default=False, description="Whether official satellite quality mask (SCL/QA60) was used")
    caveats: List[str] = Field(default_factory=list, description="Honest scientific limitations of the RGB assessment")


class SceneQualityResponse(BaseModel):
    """Aggregate quality evaluation across all tiles within a scene."""
    scene_id: uuid.UUID
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Weighted mean tile quality score")
    suitability: str = Field(..., description="'Suitable', 'Caution', or 'Poor Quality'")
    tile_count: int = Field(..., ge=0, description="Total tiles assessed")
    tiles_suitable: int = Field(..., ge=0, description="Count of tiles classified as Suitable")
    tiles_caution: int = Field(..., ge=0, description="Count of tiles classified as Caution")
    tiles_poor: int = Field(..., ge=0, description="Count of tiles classified as Poor Quality")
    avg_cloud_cover_pct: float = Field(..., ge=0.0, le=100.0, description="Average cloud cover across scene")
    avg_nodata_ratio: float = Field(..., ge=0.0, le=1.0, description="Average nodata ratio across scene")
    usable_for_change_analysis: bool = Field(..., description="Whether scene has sufficient high-quality area for change detection")
    quality_details: dict[str, Any] = Field(default_factory=dict, description="Detailed per-factor distributions")


class SceneComputeQualityResponse(BaseModel):
    """Response returned upon computing and persisting quality metrics to the database."""
    scene_id: uuid.UUID
    tiles_computed: int
    quality_score: float
    suitability: str
    summary: SceneQualityResponse
