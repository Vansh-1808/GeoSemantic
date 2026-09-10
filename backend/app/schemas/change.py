from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid
from pydantic import BaseModel, ConfigDict, Field

class ChangeAnalyzeRequest(BaseModel):
    """Request to analyze changes in a specific AOI (or across entire archive) and time range."""
    aoi_wkt: Optional[str] = Field(None, description="Optional WKT polygon. If omitted, analyzes all overlapping imagery across the whole archive.")
    start_date: Optional[datetime] = Field(None, description="Optional start datetime. Defaults to beginning of archive.")
    end_date: Optional[datetime] = Field(None, description="Optional end datetime. Defaults to end of archive.")
    limit: int = Field(default=50, description="Max number of candidate pairs to process")
    query: Optional[str] = Field(None, description="Optional natural language query describing targeted change, e.g. 'New construction near water bodies'")

class ChangeEventResponse(BaseModel):
    """Response model for a change event."""
    id: uuid.UUID
    before_tile_id: Optional[uuid.UUID] = None
    after_tile_id: Optional[uuid.UUID] = None
    before_date: Optional[datetime] = None
    after_date: Optional[datetime] = None
    
    center_lon: Optional[float] = None
    center_lat: Optional[float] = None
    
    change_type: Optional[str] = None
    final_confidence: Optional[float] = None
    
    visual_change_score: Optional[float] = None
    semantic_change_score: Optional[float] = None
    registration_quality: Optional[float] = None
    
    is_suppressed: bool = False
    review_status: str
    
    detected_at: datetime
    updated_at: datetime
    
    evidence: Optional[Dict[str, Any]] = None
    confidence_breakdown: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(from_attributes=True)

class ChangeAnalyzeResponse(BaseModel):
    """Response for a change analysis run."""
    message: str
    candidates_found: int
    events: List[ChangeEventResponse]
