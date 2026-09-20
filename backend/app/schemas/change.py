from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChangeAnalyzeRequest(BaseModel):
    """Request to analyze changes in a specific AOI (or across entire archive) and time range."""
    aoi_wkt: Optional[str] = Field(None, description="Optional WKT polygon. If omitted, analyzes all overlapping imagery across the whole archive.")
    start_date: Optional[datetime] = Field(None, description="Optional start datetime. Defaults to beginning of archive.")
    end_date: Optional[datetime] = Field(None, description="Optional end datetime. Defaults to end of archive.")
    limit: int = Field(default=50, description="Max number of candidate pairs to process")
    query: Optional[str] = Field(None, description="Optional natural language query describing targeted change, e.g. 'New construction near water bodies'")


class ChangeEventResponse(BaseModel):
    """Response model for a change event with complete multi-factor confidence data."""
    id: uuid.UUID
    before_tile_id: Optional[uuid.UUID] = None
    after_tile_id: Optional[uuid.UUID] = None
    before_date: Optional[datetime] = None
    after_date: Optional[datetime] = None
    
    center_lon: Optional[float] = None
    center_lat: Optional[float] = None
    
    change_type: Optional[str] = None
    final_confidence: Optional[float] = None
    confidence_tier: Optional[str] = None
    
    visual_change_score: Optional[float] = None
    semantic_change_score: Optional[float] = None
    registration_quality: Optional[float] = None
    
    is_suppressed: bool = False
    suppression_reason: Optional[str] = None
    suppression_reasons: Optional[List[str]] = None
    positive_factors: Optional[List[str]] = None
    
    review_status: str
    
    detected_at: datetime
    updated_at: datetime
    
    evidence: Optional[Dict[str, Any]] = None
    confidence_breakdown: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(from_attributes=True)

    @model_validator(mode="wrap")
    @classmethod
    def populate_confidence_metadata(cls, data: Any, handler) -> "ChangeEventResponse":
        res = handler(data)
        if res.confidence_breakdown:
            cb = res.confidence_breakdown
            if not res.confidence_tier:
                res.confidence_tier = cb.get("confidence_tier")
            if not res.suppression_reasons:
                res.suppression_reasons = cb.get("reasons")
            if not res.positive_factors:
                res.positive_factors = cb.get("positive_factors")
            if not res.suppression_reason and res.suppression_reasons:
                res.suppression_reason = "; ".join(res.suppression_reasons)
        return res


class ChangeAnalyzeResponse(BaseModel):
    """Response for a change analysis run."""
    message: str
    candidates_found: int
    events: List[ChangeEventResponse]


class ConfidenceWeightsSchema(BaseModel):
    """Configurable weights for the multi-factor scoring engine."""
    weight_visual: float = 0.25
    weight_semantic: float = 0.25
    weight_spectral: float = 0.15
    weight_spatial: float = 0.15
    weight_quality: float = 0.08
    weight_registration: float = 0.08
    weight_sensor: float = 0.02
    weight_seasonal: float = 0.02
    cloud_penalty_multiplier: float = 1.2
    shadow_penalty_multiplier: float = 0.8
    misalignment_penalty_multiplier: float = 1.5
    seasonal_veg_penalty_multiplier: float = 1.2


class ConfidenceThresholdsSchema(BaseModel):
    """Configurable decision thresholds for confidence tiers and suppression."""
    high_threshold: float = 0.70
    medium_threshold: float = 0.45
    low_threshold: float = 0.25
    cloud_suppression_limit: float = 0.25
    registration_suppression_limit: float = 0.40
    seasonal_veg_ratio_limit: float = 0.60
    shadow_suppression_limit: float = 0.40


class ConfidenceConfigResponse(BaseModel):
    """Response returning current engine weights and thresholds."""
    weights: ConfidenceWeightsSchema
    thresholds: ConfidenceThresholdsSchema


class ConfidenceConfigUpdateRequest(BaseModel):
    """Request to update weights or thresholds dynamically."""
    weights: Optional[Dict[str, float]] = None
    thresholds: Optional[Dict[str, float]] = None


class ConfidenceEvaluationRequest(BaseModel):
    """Request to evaluate multi-factor confidence directly from 10 input signals or tile pair IDs."""
    visual_change: float = Field(0.5, ge=0.0, le=1.0)
    semantic_change: float = Field(0.5, ge=0.0, le=1.0)
    image_quality: float = Field(0.8, ge=0.0, le=1.0)
    cloud_score: float = Field(0.0, ge=0.0, le=1.0)
    shadow_score: float = Field(0.0, ge=0.0, le=1.0)
    registration_confidence: float = Field(0.9, ge=0.0, le=1.0)
    sensor_compatibility: float = Field(1.0, ge=0.0, le=1.0)
    seasonal_compatibility: float = Field(1.0, ge=0.0, le=1.0)
    spectral_evidence: float = Field(0.5, ge=0.0, le=1.0)
    spatial_consistency: float = Field(0.5, ge=0.0, le=1.0)
    is_seasonal_vegetation: bool = False
    change_type: Optional[str] = None
    query_match_score: Optional[float] = None
    weights_override: Optional[Dict[str, float]] = None


class ConfidenceEvaluationResponse(BaseModel):
    """Complete transparent confidence evaluation response."""
    final_confidence: float
    confidence_tier: str
    is_suppressed: bool
    reasons: List[str]
    positive_factors: List[str]
    factor_breakdown: Dict[str, Any]
    penalties_applied: Dict[str, float]
    raw_signals: Dict[str, float]
    weights_applied: Dict[str, float]
