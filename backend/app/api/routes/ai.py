"""
Phase 14: Local AI API Routes
==============================
Exposes the Local LLM service via FastAPI endpoints:

  GET  /api/ai/status        — real-time Ollama runtime availability
  GET  /api/ai/models        — models physically installed in Ollama
  POST /api/ai/generate      — general local text generation
  POST /api/ai/parse-query   — structured analyst query extraction

All endpoints:
  * Return real data — never fake successful responses.
  * Fail gracefully — LLM unavailability is reported, not raised.
  * Never substitute cloud LLMs.
"""
from __future__ import annotations

from app.core.logging import get_logger
from app.schemas.ai import (
    AIGenerateRequest,
    AIGenerateResponse,
    AIModelsResponse,
    AIStatusResponse,
    AnalystQueryParseRequest,
    AnalystQueryParseResponse,
)
from app.services.llm_service import llm_service
from fastapi import APIRouter

router = APIRouter()
logger = get_logger(__name__)


@router.get("/status", response_model=AIStatusResponse, summary="Local AI runtime status")
async def ai_status_endpoint() -> AIStatusResponse:
    """
    Check whether the local LLM runtime (Ollama) is reachable and
    report which model is configured.

    Returns actual latency to the local runtime.
    Never fakes a successful response if Ollama is stopped.
    """
    return await llm_service.check_status()


@router.get("/models", response_model=AIModelsResponse, summary="List locally installed AI models")
async def ai_models_endpoint() -> AIModelsResponse:
    """
    Return the models actually available in the local Ollama runtime.
    Model names, sizes, and quantization levels come directly from Ollama —
    nothing is hardcoded.
    """
    return await llm_service.list_models()


@router.post("/generate", response_model=AIGenerateResponse, summary="Local LLM text generation")
async def ai_generate_endpoint(req: AIGenerateRequest) -> AIGenerateResponse:
    """
    General-purpose local LLM generation endpoint for testing and development.

    Accepts an optional system prompt and an optional output format.
    If `format="json"` is specified, Ollama enforces JSON-valid output.

    Measurement: inference duration_ms is always reported.
    """
    return await llm_service.generate(
        prompt=req.prompt,
        system=req.system,
        model=req.model,
        format=req.format,
    )


@router.post(
    "/parse-query",
    response_model=AnalystQueryParseResponse,
    summary="Parse analyst query into structured representation via local LLM",
)
async def ai_parse_query_endpoint(req: AnalystQueryParseRequest) -> AnalystQueryParseResponse:
    """
    Parse a natural-language analyst query into a validated AnalystQuery
    using the local LLM (Phase 14 pipeline: normalise → correct → LLM → validate).

    Strict grounding rule: the LLM only interprets query *intent*, it does
    NOT claim to have observed anything in satellite imagery.

    If `use_llm=false` or the runtime is unavailable, falls back to a minimal
    valid response with `llm_used=false` and a `fallback_reason`.
    """
    if not req.use_llm:
        from app.schemas.ai import AnalystQuery

        fallback_aq = AnalystQuery(
            intent="semantic_search",
            concepts=[req.query.strip()],
            raw_query=req.query,
        )
        return AnalystQueryParseResponse(
            analyst_query=fallback_aq,
            llm_used=False,
            validation_passed=False,
            fallback_reason="use_llm=false — LLM bypassed by caller.",
        )

    return await llm_service.parse_analyst_query(raw_query=req.query)
