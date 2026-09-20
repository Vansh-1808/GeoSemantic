"""
Phase 14: Local LLM Service
============================
Provides a clean async interface to the locally running Ollama inference
runtime.  The service is a singleton and follows the same design conventions
as the other services in this package (embedding_service, vector_store, etc.).

Key guarantees:
  * NEVER crashes the wider application — all failures are caught, logged,
    and reported as structured error states.
  * NEVER substitutes a cloud LLM when Ollama is unavailable.
  * NEVER fabricates satellite observations or makes up field values.
  * Fully configurable through `settings` (LLM_MODEL, LLM_BASE_URL, etc.).
  * Produces structured AnalystQuery output validated by Pydantic.
"""
from __future__ import annotations

import json
import re
import time
from typing import Optional, Tuple, Type, TypeVar

import httpx

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.ai import (
    AIGenerateResponse,
    AIModelItem,
    AIModelsResponse,
    AIStatusResponse,
    AnalystQuery,
    AnalystQueryParseResponse,
)

logger = get_logger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Satellite analyst system prompt — enforces strict grounding rule
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a satellite imagery analyst assistant embedded in the GeoSemantic platform.
Your role is to interpret analyst natural-language queries and extract structured parameters.
You work with satellite imagery, geographic regions, land cover, infrastructure, vegetation, water, and temporal observations.

STRICT GROUNDING RULE:
- Do NOT claim to have directly observed anything in an image.
- Only interpret what the analyst's query describes.
- Distinguish between "analyst intent" and "observed evidence".
- Never invent military facilities, weapons, coordinates, causes, events, or dates not present in the query.

Extract the following fields as a JSON object:
{
  "intent": "change_analysis" | "semantic_search",
  "location": <string: geographic entity name only, e.g. "Rajasthan", "Chennai", "Kerala", or null>,
  "concepts": [<strings: visual landcover features, e.g. "desert", "water", "construction", "road">],
  "target": <string: primary feature being searched/monitored, or null>,
  "change_type": <string: e.g. "expansion", "increase", "decrease", "construction", "loss", "clearance", or null>,
  "relationships": [<strings: spatial relationships, e.g. "near_road", "near_water", "in_forest">],
  "start_date": <string: e.g. "2023", "2022-06", or null>,
  "end_date": <string: e.g. "2025", or null>,
  "requires_water": <boolean>,
  "requires_vegetation": <boolean>,
  "requires_urban": <boolean>
}

Rules:
- Set intent to "change_analysis" if the query asks about changes, growth, loss, new features, or temporal comparisons.
- Set intent to "semantic_search" for visual similarity queries without temporal change.
- location must be ONLY a geographic proper noun (state, city, country, region) — NOT a visual concept like "water" or "near roads".
- concepts should list visual landcover features extracted from the query — exclude the geographic name.
- relationships uses values: near_road, near_water, in_forest, near_airport, near_urban.
- Output ONLY valid JSON with no surrounding text or code fences.

Example 1:
Query: "Find new construction near roads in Rajasthan after 2023."
JSON: {"intent": "change_analysis", "location": "Rajasthan", "concepts": ["construction", "road"], "target": "construction", "change_type": "construction", "relationships": ["near_road"], "start_date": "2023", "end_date": null, "requires_water": false, "requires_vegetation": false, "requires_urban": true}

Example 2:
Query: "Rajasthan desert"
JSON: {"intent": "semantic_search", "location": "Rajasthan", "concepts": ["desert"], "target": null, "change_type": null, "relationships": [], "start_date": null, "end_date": null, "requires_water": false, "requires_vegetation": false, "requires_urban": false}

Example 3:
Query: "Where has water increased between 2022 and 2025?"
JSON: {"intent": "change_analysis", "location": null, "concepts": ["water"], "target": "water", "change_type": "increase", "relationships": [], "start_date": "2022", "end_date": "2025", "requires_water": true, "requires_vegetation": false, "requires_urban": false}"""


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences that some models add around JSON output."""
    # Strip ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text.strip())
    return text.strip()


class LocalLLMService:
    """
    Async client wrapper around the locally running Ollama HTTP API.

    Design:
    - Uses a shared httpx.AsyncClient for connection pooling.
    - All public methods catch exceptions and return structured error states
      rather than propagating raw exceptions to API routes.
    - Singleton exported as `llm_service` at module level.
    """

    def __init__(self) -> None:
        self._base_url = settings.llm_base_url
        self._model = settings.llm_model
        self._fallback_model = settings.llm_fallback_model
        self._timeout = settings.llm_timeout_seconds
        self._status_timeout = settings.llm_status_timeout_seconds
        self._temperature = settings.llm_temperature
        self._enabled = settings.llm_enabled
        # H3: Shared httpx client for connection pooling — created lazily on first use.
        self._http_client: Optional[httpx.AsyncClient] = None
        logger.info(
            "llm_service_initialized",
            provider=settings.llm_provider,
            model=self._model,
            base_url=self._base_url,
            enabled=self._enabled,
        )

    # ── Connection Management ─────────────────────────────────────────────────

    def _get_client(self, timeout: float) -> httpx.AsyncClient:
        """
        Return the shared httpx.AsyncClient, creating it if it doesn't exist yet.
        H3: Reuses a single connection pool across all Ollama calls instead of
        opening/closing a new TCP connection per request.
        """
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=timeout)
        return self._http_client

    async def close(self) -> None:
        """Close the shared httpx client. Called from the FastAPI lifespan shutdown."""
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None
            logger.info("llm_service_http_client_closed")

    # ── Status & Discovery ────────────────────────────────────────────────────

    async def check_status(self) -> AIStatusResponse:
        """
        Ping the Ollama runtime to determine availability.
        Returns a structured status object — never raises.
        """
        if not self._enabled:
            return AIStatusResponse(
                available=False,
                model=self._model,
                error="LLM disabled via LLM_ENABLED=false",
            )

        t0 = time.perf_counter()
        try:
            client = self._get_client(timeout=self._status_timeout)
            res = await client.get(f"{self._base_url}/api/tags")
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)

            if res.status_code != 200:
                return AIStatusResponse(
                    available=False,
                    model=self._model,
                    error=f"Ollama returned HTTP {res.status_code}",
                    latency_ms=latency_ms,
                )

            data = res.json()
            model_names = [m["name"] for m in data.get("models", [])]
            return AIStatusResponse(
                available=True,
                model=self._model,
                models_available=model_names,
                latency_ms=latency_ms,
            )

        except httpx.ConnectError:
            return AIStatusResponse(
                available=False,
                model=self._model,
                error="Local model runtime unavailable — is Ollama running?",
            )
        except Exception as exc:
            logger.warning("llm_status_check_failed", error=str(exc))
            return AIStatusResponse(
                available=False,
                model=self._model,
                error=f"Status check failed: {type(exc).__name__}",
            )

    async def list_models(self) -> AIModelsResponse:
        """
        Return real models installed in the local Ollama runtime.
        Never hardcodes model availability.
        """
        try:
            async with httpx.AsyncClient(timeout=self._status_timeout) as client:
                res = await client.get(f"{self._base_url}/api/tags")

            if res.status_code != 200:
                return AIModelsResponse()

            raw_models = res.json().get("models", [])
            items = []
            for m in raw_models:
                details = m.get("details", {})
                items.append(
                    AIModelItem(
                        name=m.get("name", ""),
                        model=m.get("model", ""),
                        parameter_size=details.get("parameter_size"),
                        quantization_level=details.get("quantization_level"),
                        family=details.get("family"),
                        size_bytes=m.get("size"),
                        capabilities=m.get("capabilities", []),
                    )
                )
            return AIModelsResponse(models=items, total=len(items))

        except Exception as exc:
            logger.warning("llm_list_models_failed", error=str(exc))
            return AIModelsResponse()

    # ── Text Generation ───────────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        format: Optional[str] = None,
    ) -> AIGenerateResponse:
        """
        Send a generation request to the local Ollama model.
        Returns an AIGenerateResponse — never raises on model/runtime errors.
        """
        if not self._enabled:
            return AIGenerateResponse(
                response="LLM is disabled.",
                model=self._model,
                duration_ms=0.0,
                done=False,
            )

        chosen_model = model or self._model
        payload: dict = {
            "model": chosen_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self._temperature},
        }
        if system:
            payload["system"] = system
        if format:
            payload["format"] = format

        t0 = time.perf_counter()
        try:
            client = self._get_client(timeout=self._timeout)
            res = await client.post(f"{self._base_url}/api/generate", json=payload)
            duration_ms = round((time.perf_counter() - t0) * 1000, 1)

            if res.status_code != 200:
                logger.warning("llm_generate_http_error", status=res.status_code, model=chosen_model)
                return AIGenerateResponse(
                    response=f"Ollama returned HTTP {res.status_code}",
                    model=chosen_model,
                    duration_ms=duration_ms,
                    done=False,
                )

            data = res.json()
            return AIGenerateResponse(
                response=data.get("response", ""),
                model=chosen_model,
                duration_ms=duration_ms,
                done=data.get("done", True),
            )

        except httpx.ConnectError:
            logger.warning("llm_generate_connect_error", model=chosen_model)
            return AIGenerateResponse(
                response="Local model runtime unavailable.",
                model=chosen_model,
                duration_ms=0.0,
                done=False,
            )
        except httpx.TimeoutException:
            duration_ms = round((time.perf_counter() - t0) * 1000, 1)
            logger.warning("llm_generate_timeout", model=chosen_model, duration_ms=duration_ms)
            return AIGenerateResponse(
                response=f"Inference timed out after {self._timeout}s.",
                model=chosen_model,
                duration_ms=duration_ms,
                done=False,
            )
        except Exception as exc:
            logger.warning("llm_generate_error", error=str(exc), model=chosen_model)
            return AIGenerateResponse(
                response=f"Generation error: {type(exc).__name__}",
                model=chosen_model,
                duration_ms=0.0,
                done=False,
            )

    async def generate_json(
        self,
        prompt: str,
        schema_class: Type[T],
        system: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Tuple[Optional[T], float]:
        """
        Generate a structured JSON response and validate it with the given Pydantic schema.

        Returns:
          (parsed_model, duration_ms) on success.
          (None, duration_ms) if JSON is invalid or validation fails.

        Never silently fabricates fields.
        """
        result = await self.generate(
            prompt=prompt,
            system=system,
            model=model,
            format="json",
        )
        duration_ms = result.duration_ms

        raw = result.response.strip()
        if not raw:
            logger.warning("llm_json_empty_response", model=result.model)
            return None, duration_ms

        # Strip code fences that some models may add even with format="json"
        raw = _strip_code_fences(raw)

        try:
            parsed_obj = schema_class.model_validate_json(raw)  # type: ignore[attr-defined]
            return parsed_obj, duration_ms
        except Exception as first_err:
            # Attempt recovery: parse as plain dict then re-validate
            try:
                data = json.loads(raw)
                parsed_obj = schema_class.model_validate(data)  # type: ignore[attr-defined]
                return parsed_obj, duration_ms
            except Exception:
                logger.warning(
                    "llm_json_validation_failed",
                    model=result.model,
                    error=str(first_err),
                    raw_preview=raw[:200],
                )
                return None, duration_ms

    # ── Analyst Query Parsing ─────────────────────────────────────────────────

    async def parse_analyst_query(
        self,
        raw_query: str,
        corrected_query: Optional[str] = None,
        model: Optional[str] = None,
    ) -> AnalystQueryParseResponse:
        """
        Parse a natural-language analyst query into a structured AnalystQuery.

        Pipeline:
          Corrected query (from Phase 13) → Local LLM (structured JSON) → Pydantic validation

        Observability: logs original query, model, duration, and output.

        Failure behaviour:
          If the LLM is unavailable or produces invalid output, returns a minimal
          AnalystQuery with llm_used=False and fallback_reason set.
        """
        effective_query = corrected_query or raw_query

        logger.debug(
            "llm_parse_analyst_query",
            raw_query=raw_query,
            corrected_query=corrected_query,
            effective_query=effective_query,
        )

        t0 = time.perf_counter()
        parsed_aq, duration_ms = await self.generate_json(
            prompt=effective_query,
            schema_class=AnalystQuery,
            system=_SYSTEM_PROMPT,
            model=model,
        )
        total_ms = round((time.perf_counter() - t0) * 1000, 1)

        chosen_model = model or self._model

        if parsed_aq is not None:
            # Inject provenance fields from outside Pydantic so the LLM never
            # needs to repeat them (reduces prompt length and hallucination risk)
            parsed_aq = parsed_aq.model_copy(
                update={"raw_query": raw_query, "corrected_query": corrected_query}
            )

            logger.info(
                "llm_analyst_query_parsed",
                model=chosen_model,
                intent=parsed_aq.intent,
                location=parsed_aq.location,
                concepts=parsed_aq.concepts,
                target=parsed_aq.target,
                duration_ms=total_ms,
            )

            return AnalystQueryParseResponse(
                analyst_query=parsed_aq,
                llm_used=True,
                llm_model=chosen_model,
                inference_duration_ms=total_ms,
                validation_passed=True,
            )

        # Fallback: return a minimal valid AnalystQuery
        logger.warning(
            "llm_analyst_query_fallback",
            model=chosen_model,
            query=raw_query,
            duration_ms=total_ms,
        )
        fallback_aq = AnalystQuery(
            intent="semantic_search",
            concepts=[effective_query.strip()],
            raw_query=raw_query,
            corrected_query=corrected_query,
        )
        return AnalystQueryParseResponse(
            analyst_query=fallback_aq,
            llm_used=False,
            llm_model=chosen_model,
            inference_duration_ms=total_ms,
            validation_passed=False,
            fallback_reason="LLM unavailable or produced invalid JSON — using minimal fallback.",
        )


# Module-level singleton — mirrors the pattern used by all other services
llm_service = LocalLLMService()
