"""
Pydantic schemas for Phase 14: Local LLM Foundation.

Defines API request/response types for:
  - Local AI status and model listing
  - General text generation
  - Structured analyst query parsing (AnalystQuery)
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ── AI Runtime Status ─────────────────────────────────────────────────────────

class AIModelItem(BaseModel):
    """Metadata for a single model available in the local inference runtime."""
    name: str
    model: str
    parameter_size: Optional[str] = None
    quantization_level: Optional[str] = None
    family: Optional[str] = None
    size_bytes: Optional[int] = None
    capabilities: List[str] = Field(default_factory=list)


class AIStatusResponse(BaseModel):
    """Response for GET /api/ai/status — reflects the real local runtime state."""
    available: bool
    provider: str = "ollama"
    model: str
    local: bool = True
    models_available: List[str] = Field(default_factory=list)
    latency_ms: Optional[float] = None
    error: Optional[str] = None


class AIModelsResponse(BaseModel):
    """Response for GET /api/ai/models — real models installed in the local runtime."""
    provider: str = "ollama"
    models: List[AIModelItem] = Field(default_factory=list)
    total: int = 0


# ── General Generation ────────────────────────────────────────────────────────

class AIGenerateRequest(BaseModel):
    """Request body for POST /api/ai/generate — general local LLM text generation."""
    prompt: str
    system: Optional[str] = None
    model: Optional[str] = None  # defaults to settings.llm_model
    format: Optional[Literal["json"]] = None
    temperature: Optional[float] = None


class AIGenerateResponse(BaseModel):
    """Response from the local LLM for general generation."""
    response: str
    model: str
    provider: str = "ollama"
    duration_ms: float
    done: bool = True


# ── Structured Analyst Query (Core Phase 14 Contract) ────────────────────────

class AnalystQuery(BaseModel):
    """
    Structured representation of an analyst query produced by the local LLM.

    This is the primary contract between the Language Intelligence layer and
    the rest of the platform (search, change detection, reporting).

    Fields labelled Optional may be absent for simple semantic queries.
    All fields are validated by Pydantic after LLM generation.
    The LLM is NOT permitted to invent satellite observations — it only
    interprets the analyst's textual intent.
    """
    # Intent classification
    intent: Literal["change_analysis", "semantic_search", "temporal_search", "general_query"] = "semantic_search"

    # Geographic entity (resolved later by GeographicGazetteer)
    location: Optional[str] = None
    location_type: Optional[str] = None  # "state" | "city" | "country" | "region"

    # Visual semantic concepts (sent to RemoteCLIP)
    concepts: List[str] = Field(default_factory=list)

    # Primary monitoring target
    target: Optional[str] = None  # e.g. "construction", "water", "road", "vegetation"

    # Change characterisation (only for change_analysis intent)
    change_type: Optional[str] = None  # e.g. "expansion", "construction", "increase", "loss"

    # Spatial relationships between features
    relationships: List[str] = Field(default_factory=list)  # e.g. ["near_road", "near_water"]

    # Temporal constraints
    start_date: Optional[str] = None  # e.g. "2023", "2022-06"
    end_date: Optional[str] = None    # e.g. "2025"

    # Optional satellite constraints
    sensor: Optional[str] = None               # e.g. "Sentinel-2", "Landsat"
    confidence_requirement: Optional[str] = None  # e.g. "high", "medium"

    # Derived landcover indicators (consistent with Phase 12/13 ParsedQuery)
    requires_water: bool = False
    requires_vegetation: bool = False
    requires_urban: bool = False

    # Provenance fields — carry through the query pipeline without mutation
    raw_query: str = ""
    corrected_query: Optional[str] = None  # set if Phase 13 spelling correction applied


class AnalystQueryParseRequest(BaseModel):
    """Request for POST /api/ai/parse-query."""
    query: str
    use_llm: bool = True  # allow callers to bypass LLM and use deterministic fallback


class AnalystQueryParseResponse(BaseModel):
    """
    Response from POST /api/ai/parse-query.
    Contains the structured AnalystQuery plus inference metadata.
    """
    analyst_query: AnalystQuery
    llm_used: bool
    llm_model: Optional[str] = None
    inference_duration_ms: Optional[float] = None
    validation_passed: bool
    fallback_reason: Optional[str] = None   # set if LLM was bypassed
    extra: Dict[str, Any] = Field(default_factory=dict)
