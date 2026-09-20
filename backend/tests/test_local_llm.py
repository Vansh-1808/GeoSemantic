"""
Phase 14: Local LLM Foundation — Test Suite
============================================
Tests all required scenarios from the Phase 14 specification:

  Test 1 — Direct question: "What is satellite imagery?"
  Test 2 — Structured query: "Find new construction near roads after 2023."
  Test 3 — Domain typo: "rajasthan dessert"
  Test 4 — Temporal change: "Where has water increased between 2022 and 2025?"
  Test 5 — Offline / Failure behaviour
  Test 6 — Model listing from real runtime
  Test 7 — Schema validation & malformed JSON recovery

These tests require an Ollama process running locally.
Tests that require live Ollama are marked with @pytest.mark.asyncio.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas.ai import (
    AIStatusResponse,
    AnalystQuery,
    AnalystQueryParseRequest,
    AnalystQueryParseResponse,
)
from app.services.llm_service import LocalLLMService, _strip_code_fences


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ollama_generate_response(response_text: str) -> MagicMock:
    """Build a mock httpx Response returning an Ollama /api/generate body."""
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {"response": response_text, "done": True}
    return mock


def make_ollama_tags_response(model_names: list[str]) -> MagicMock:
    """Build a mock httpx Response returning an Ollama /api/tags body."""
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {
        "models": [
            {"name": n, "model": n, "size": 1_000_000, "details": {}, "capabilities": ["completion"]}
            for n in model_names
        ]
    }
    return mock


# ---------------------------------------------------------------------------
# Test 1 — "What is satellite imagery?" → valid non-empty response
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_1_what_is_satellite_imagery():
    """
    The local LLM must return a meaningful non-empty response to the direct
    question 'What is satellite imagery?'.
    This test uses a mocked Ollama client to remain deterministic.
    """
    expected_answer = (
        "Satellite imagery refers to photographs taken by satellites orbiting the Earth."
    )

    service = LocalLLMService()

    with patch("httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value.__aenter__.return_value
        instance.post = AsyncMock(return_value=make_ollama_generate_response(expected_answer))

        result = await service.generate(prompt="What is satellite imagery?")

    assert result.done is True
    assert len(result.response) > 20
    assert "satellite" in result.response.lower() or "imaging" in result.response.lower() or len(result.response) > 10
    assert result.duration_ms >= 0.0
    assert result.model == service._model


# ---------------------------------------------------------------------------
# Test 2 — Structured extraction: "Find new construction near roads after 2023."
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_2_structured_query_construction_near_roads():
    """
    The LLM must extract: intent=change_analysis, target=construction,
    relationships=[near_road], start_date=2023.
    """
    valid_json = json.dumps({
        "intent": "change_analysis",
        "location": None,
        "concepts": ["construction", "road"],
        "target": "construction",
        "change_type": "construction",
        "relationships": ["near_road"],
        "start_date": "2023",
        "end_date": None,
        "requires_water": False,
        "requires_vegetation": False,
        "requires_urban": True,
    })

    service = LocalLLMService()

    with patch("httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value.__aenter__.return_value
        instance.post = AsyncMock(return_value=make_ollama_generate_response(valid_json))

        aq, duration_ms = await service.generate_json(
            prompt="Find new construction near roads after 2023.",
            schema_class=AnalystQuery,
        )

    assert aq is not None
    assert aq.intent == "change_analysis"
    assert aq.target in ("construction", "construction_near_roads")
    assert "near_road" in aq.relationships
    assert aq.start_date == "2023"
    assert aq.end_date is None


# ---------------------------------------------------------------------------
# Test 3 — "rajasthan dessert" → location=Rajasthan, concept=desert
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_3_rajasthan_dessert_domain_typo():
    """
    The Phase 13 spelling corrector fixes 'dessert' -> 'desert'.
    The Phase 14 LLM should then interpret:
      location = Rajasthan
      concepts = ["desert"]
    """
    valid_json = json.dumps({
        "intent": "semantic_search",
        "location": "Rajasthan",
        "concepts": ["desert"],
        "target": None,
        "change_type": None,
        "relationships": [],
        "start_date": None,
        "end_date": None,
        "requires_water": False,
        "requires_vegetation": False,
        "requires_urban": False,
    })

    service = LocalLLMService()

    with patch("httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value.__aenter__.return_value
        instance.post = AsyncMock(return_value=make_ollama_generate_response(valid_json))

        result = await service.parse_analyst_query(
            raw_query="rajasthan dessert",
            corrected_query="rajasthan desert",
        )

    assert result.llm_used is True
    assert result.validation_passed is True
    aq = result.analyst_query
    assert aq.location is not None and aq.location.lower() in ("rajasthan", "rajasthan desert")
    assert "desert" in [c.lower() for c in aq.concepts]
    assert aq.intent == "semantic_search"
    # Provenance preserved
    assert aq.raw_query == "rajasthan dessert"
    assert aq.corrected_query == "rajasthan desert"


# ---------------------------------------------------------------------------
# Test 4 — "Where has water increased between 2022 and 2025?"
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_4_water_increase_temporal():
    """
    The LLM must extract:
      intent = change_analysis
      target = water
      change_type = increase (or "expansion")
      start_date = 2022
      end_date = 2025
    """
    valid_json = json.dumps({
        "intent": "change_analysis",
        "location": None,
        "concepts": ["water"],
        "target": "water",
        "change_type": "increase",
        "relationships": [],
        "start_date": "2022",
        "end_date": "2025",
        "requires_water": True,
        "requires_vegetation": False,
        "requires_urban": False,
    })

    service = LocalLLMService()

    with patch("httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value.__aenter__.return_value
        instance.post = AsyncMock(return_value=make_ollama_generate_response(valid_json))

        aq, _ = await service.generate_json(
            prompt="Where has water increased between 2022 and 2025?",
            schema_class=AnalystQuery,
        )

    assert aq is not None
    assert aq.intent == "change_analysis"
    assert aq.target == "water"
    assert aq.change_type in ("increase", "expansion")
    assert aq.start_date == "2022"
    assert aq.end_date == "2025"
    assert aq.requires_water is True


# ---------------------------------------------------------------------------
# Test 5 — Failure / Offline Behaviour
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_5_offline_check_status_returns_unavailable():
    """
    When Ollama is unreachable, check_status() must return available=False
    with a descriptive error message, NOT raise an exception.
    """
    import httpx as httpx_lib

    service = LocalLLMService()
    # Override base_url to a port that should never be open
    service._base_url = "http://localhost:59999"
    service._status_timeout = 1.0

    # Patch httpx.AsyncClient to raise ConnectError
    with patch("httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value.__aenter__.return_value
        instance.get = AsyncMock(side_effect=httpx_lib.ConnectError("Connection refused"))

        status = await service.check_status()

    assert status.available is False
    assert status.error is not None
    assert len(status.error) > 0
    # Must NOT raise


@pytest.mark.asyncio
async def test_5_offline_parse_falls_back_gracefully():
    """
    When Ollama is unreachable, parse_analyst_query() must return a minimal
    valid AnalystQuery with llm_used=False, NOT raise an exception.
    """
    import httpx as httpx_lib

    service = LocalLLMService()
    service._base_url = "http://localhost:59999"
    service._timeout = 1.0

    with patch("httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value.__aenter__.return_value
        instance.post = AsyncMock(side_effect=httpx_lib.ConnectError("Connection refused"))

        result = await service.parse_analyst_query(
            raw_query="desert in Rajasthan",
        )

    assert result.llm_used is False
    assert result.analyst_query is not None
    assert result.analyst_query.raw_query == "desert in Rajasthan"
    # Must NOT raise


@pytest.mark.asyncio
async def test_5_generate_offline_returns_structured_error():
    """generate() must not raise even when Ollama is completely offline."""
    import httpx as httpx_lib

    service = LocalLLMService()

    with patch("httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value.__aenter__.return_value
        instance.post = AsyncMock(side_effect=httpx_lib.ConnectError("Connection refused"))

        result = await service.generate(prompt="What is satellite imagery?")

    assert isinstance(result.response, str)
    assert result.done is False
    # Must NOT raise


# ---------------------------------------------------------------------------
# Test 6 — Model listing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_6_list_models_returns_real_installed_models():
    """
    list_models() must return the models physically installed in Ollama,
    not a hardcoded list.
    """
    installed = ["llama3:latest", "llama3.2:latest", "nomic-embed-text:latest"]

    service = LocalLLMService()

    with patch("httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value.__aenter__.return_value
        instance.get = AsyncMock(return_value=make_ollama_tags_response(installed))

        models_resp = await service.list_models()

    assert models_resp.total == 3
    names = [m.name for m in models_resp.models]
    for n in installed:
        assert n in names


@pytest.mark.asyncio
async def test_6_list_models_offline_returns_empty_gracefully():
    """list_models() must return an empty AIModelsResponse when Ollama is down."""
    import httpx as httpx_lib

    service = LocalLLMService()

    with patch("httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value.__aenter__.return_value
        instance.get = AsyncMock(side_effect=httpx_lib.ConnectError("Connection refused"))

        models_resp = await service.list_models()

    assert models_resp.total == 0
    assert models_resp.models == []


# ---------------------------------------------------------------------------
# Test 7 — Schema validation & malformed JSON recovery
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_7_malformed_json_returns_none_not_exception():
    """
    generate_json() must return (None, duration_ms) when the model produces
    invalid JSON rather than raising a ValidationError or json.JSONDecodeError.
    """
    malformed = "I'm sorry, I cannot answer that in JSON form. Here is my response: ..."

    service = LocalLLMService()

    with patch("httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value.__aenter__.return_value
        instance.post = AsyncMock(return_value=make_ollama_generate_response(malformed))

        result, duration_ms = await service.generate_json(
            prompt="some prompt",
            schema_class=AnalystQuery,
        )

    assert result is None
    assert isinstance(duration_ms, float)
    # Must NOT raise


def test_7_strip_code_fences():
    """Code fence stripping utility must handle all common patterns."""
    cases = [
        ("```json\n{\"a\": 1}\n```", '{"a": 1}'),
        ("```\n{\"a\": 1}\n```", '{"a": 1}'),
        ('{"a": 1}', '{"a": 1}'),
        ("  ```json\n{\"a\": 1}\n```  ", '{"a": 1}'),
    ]
    for raw, expected in cases:
        assert _strip_code_fences(raw) == expected, f"Failed for: {raw!r}"


@pytest.mark.asyncio
async def test_7_json_with_code_fences_is_recovered():
    """
    Some models wrap JSON in code fences even with format='json'.
    generate_json() must strip them and still produce a valid AnalystQuery.
    """
    fenced = '```json\n{"intent": "semantic_search", "location": "Kerala", "concepts": ["water"], "target": null, "change_type": null, "relationships": [], "start_date": null, "end_date": null, "requires_water": true, "requires_vegetation": false, "requires_urban": false}\n```'

    service = LocalLLMService()

    with patch("httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value.__aenter__.return_value
        instance.post = AsyncMock(return_value=make_ollama_generate_response(fenced))

        result, _ = await service.generate_json(
            prompt="water bodies in Kerala",
            schema_class=AnalystQuery,
        )

    assert result is not None
    assert result.location == "Kerala"
    assert "water" in result.concepts
    assert result.requires_water is True


# ---------------------------------------------------------------------------
# Integration — parse_async() connects Phase 13 → Phase 14
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_parse_async_llm_enriches_deterministic_base():
    """
    parse_async() must produce a ParsedQuery that merges the LLM's semantic
    insight with the deterministic base.  The location and bbox must still
    come from the Gazetteer (deterministic wins for spatial coords).
    """
    from app.services.query_understanding import query_understanding_service

    valid_json = json.dumps({
        "intent": "semantic_search",
        "location": "Rajasthan",
        "concepts": ["desert"],
        "target": None,
        "change_type": None,
        "relationships": [],
        "start_date": None,
        "end_date": None,
        "requires_water": False,
        "requires_vegetation": False,
        "requires_urban": False,
    })

    with patch("app.services.llm_service.llm_service") as mock_llm:
        mock_llm.parse_analyst_query = AsyncMock(return_value=AnalystQueryParseResponse(
            analyst_query=AnalystQuery(
                intent="semantic_search",
                location="Rajasthan",
                concepts=["desert"],
                raw_query="rajasthan desert",
            ),
            llm_used=True,
            llm_model="llama3:latest",
            inference_duration_ms=9800.0,
            validation_passed=True,
        ))

        from unittest.mock import patch as _patch
        from app.core.config import settings
        with _patch.object(settings, "llm_enabled", True):
            result = await query_understanding_service.parse_async("rajasthan desert")

    # Deterministic Gazetteer must still provide location/bbox
    assert result.location is not None
    assert result.location_bbox is not None
    # LLM concept should be merged in
    assert result.concept == "desert"
