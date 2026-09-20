# GeoSemantic Satellite Intelligence Platform — Forensic Engineering Audit

**Date:** September 2026
**Scope:** Full-stack read-only audit (SIH 2026 — Phase 12–14)
**Method:** Direct source inspection, line-level evidence, no assumptions
**Verdict format:** ✅ Verified | ⚠️ Warning | ❌ Defect | 🔴 Critical

---

## 1. Architecture Overview

```
Frontend (Next.js 16.3.4 + React 19) → [Next.js rewrites] → FastAPI 0.115.5
FastAPI → PostgreSQL/PostGIS  (sqlalchemy 2.0 asyncpg)
FastAPI → Qdrant (local embedded, qdrant-client 1.12.1)
FastAPI → Local Filesystem (GeoTIFF tiles, thumbnails, previews)
FastAPI → Ollama (localhost:11434, optional, HTTP-based)
```

**Data flow for semantic search:**
```
Query text → QueryNormalizer → SpellingCorrectionEngine → QueryUnderstandingEngine
→ EmbeddingService (RemoteCLIP ViT-B-32 / 512-dim)
→ Qdrant cosine similarity search
→ PostGIS spatial verification (ST_Intersects)
→ Spectral fusion scoring
→ PostgreSQL provenance enrichment
→ Response
```

---

## 2. Frontend

### 2.1 Framework and Routing

✅ **Next.js 16.3.4 with App Router** confirmed at `frontend/package.json:27`.
✅ **React 19.2.8** — very cutting edge; this is prerelease territory in 2026.
✅ **TypeScript** enabled (`tsconfig.json` present).

**Routes confirmed from `frontend/app/` directory:**

| Route | Directory | Purpose |
|-------|-----------|---------|
| `/` | `app/page.tsx` | Mission Dashboard |
| `/search` | `app/search/` | Semantic text search |
| `/image-search` | `app/image-search/` | Visual image-to-image search |
| `/change` | `app/change/` | Multi-temporal change detection |
| `/clusters` | `app/clusters/` | Spatial clustering explorer |
| `/review` | `app/review/` | Analyst review queue |
| `/ingest` | `app/ingest/` | GeoTIFF upload |
| `/datasets` | `app/datasets/` | Dataset management |
| `/system` | `app/system/` | System health |
| `/timeline` | `app/timeline/` | Temporal timeline |

✅ All routes listed in the dashboard Quick Actions have corresponding directories.

### 2.2 API Communication

✅ `frontend/lib/api.ts` (900 lines) is a typed Axios client.
✅ `next.config.ts` configures rewrites: all `/api/*` proxied to `http://localhost:8000/api/*`.

⚠️ **Hardcoded localhost in rewrites.** The `destination` for all rewrites is `http://localhost:8000/*`. There is no environment variable to override the backend URL. For staging/production, `next.config.ts` must be manually edited.

⚠️ **CORS double-registration in FastAPI.** `main.py:130-148` registers `scenes.router`, `tiles.router`, `models.router`, `embedding.router`, `search.router`, `quality.router`, and `change.router` **twice each** — once under `/api/...` and once under the root `/...`. This doubles the OpenAPI schema endpoint count and may confuse generated clients, though functionally harmless.

### 2.3 State Management

✅ **Zustand** (`store.ts`) for local UI state.
✅ **TanStack React Query v5** for server-state caching and refetching.

### 2.4 Frontend Defects

⚠️ **Hardcoded model size strings in `page.tsx`.** Lines 217 and 228 contain:
```typescript
: "577.2 MB (Ready)"
: "84.2 MB (Ready)"
```
These literals are rendered when `size_mb` is null but `present` is true. They are stale magic constants and do not reflect actual on-disk sizes of locally installed models.

⚠️ **"Reviews Pending" stat card is always `"—"`** (`page.tsx:110`). There is no API query backing the review count. If the review queue feature is real, this is incomplete.

---

## 3. Backend

### 3.1 Framework and Entry Point

✅ **FastAPI 0.115.5** with async lifespan context manager.
✅ `app/main.py` correctly uses `@asynccontextmanager` lifespan — deprecated `startup`/`shutdown` events are not used.
✅ **CORS** configured with explicit origins and a regex `r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"`. The regex is sound.

🔴 **SECRET_KEY default is `"change-me"`** (`config.py:32`). The `.env` file sets it to `"change-me-to-a-random-string-in-production"` — also a weak placeholder. There is **no enforcement** that prevents the application from starting with the default key.

### 3.2 Database Layer

✅ **SQLAlchemy 2.0 async engine** with `asyncpg` driver.
✅ `NullPool` used for `APP_ENV=test` to prevent connection leaks.
✅ `pool_size=10, max_overflow=20` for production — appropriate for a desktop analyst tool.
✅ `expire_on_commit=False` set correctly for async sessions.
✅ `check_database_connection()` issues a live `SELECT 1` rather than just ping.

⚠️ **Session auto-commit on success.** `database.py:59` calls `await session.commit()` inside the `get_session` generator. This means every GET request also commits a transaction. While harmless for reads, it is non-standard.

### 3.3 ORM Models

✅ **`Scene` model** (`scene.py`): Correct PostGIS `Geometry("POLYGON", srid=4326)` column. GiST index on footprint at line 135. `CHECK` constraints validated for status enum and cloud cover range.
✅ **`Tile` model** (`tile.py`): Foreign key to `scenes.id` with `CASCADE` delete. GiST spatial index on `footprint`. `remoteclip_vector_id` and `dino_vector_id` tracked.
✅ **`TYPE_CHECKING`** guards in `analyst.py`, `change_event.py`, `scene.py`, `tile.py` — prevents circular import issues at runtime.

⚠️ **`Tile.tile_size` default is 256 in the model** (`tile.py:53`), but `config.py:54` sets `default_tile_size=512`. These differ. The model's column default will silently record 256 if the value is ever omitted by the application code — a data integrity inconsistency.

### 3.4 Alembic Migrations

✅ `alembic.ini` present, `alembic/` directory present.

⚠️ **Audit could not verify that migration history matches current model definitions** (read-only audit). `Tile.resolution_m` and `processing_history` columns should be verified in alembic heads.

---

## 4. AI / ML Models

### 4.1 RemoteCLIP (Phase 12+)

✅ **RemoteCLIP ViT-B-32** — text and image embeddings at 512 dimensions.
✅ Offline guarantee enforced: `model_manager.py:20-21` sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` at module load time. This is correct.
✅ **Prompt ensembling in `embedding_service.py:61-73`** — four template variants averaged before L2 normalization. Known technique for improving CLIP text-image alignment in remote sensing.

⚠️ **`INFERENCE_DEVICE=cpu` hardcoded in `.env`**. Users with NVIDIA GPUs must manually change this.

### 4.2 DINOv2 (Visual Search)

✅ **DINOv2 ViT-S/14** at 384 dimensions — loaded locally from `../data/models/dino_cache/`.
✅ Image-to-image visual search uses the `dino_tiles` collection in Qdrant.

### 4.3 Local LLM (Phase 14 — Ollama)

✅ **Ollama HTTP API** at `http://localhost:11434` — not bundled, installed separately.
✅ **Model**: `llama3:latest` (8B parameters, ~4.7 GB at Q4_K_M) with fallback `llama3.2:latest` (3B parameters, ~2.0 GB) — `config.py:84-85`.

✅ **Graceful fallback**: If Ollama is unreachable or JSON parsing fails, `llm_service.py:413-426` returns `llm_used=False` and `fallback_reason`. The wider application is never broken.
✅ **Strict grounding rule** enforced by system prompt (`llm_service.py:44-87`): "Do NOT claim to have directly observed anything in an image."
✅ **Code fence stripping** in `_strip_code_fences()` handles models that wrap JSON in markdown fences.
✅ **Temperature=0.1** for determinism — appropriate for structured extraction tasks.

⚠️ **`llm_status_timeout_seconds=3.0`** (`config.py:88`) is very tight. If Ollama is busy loading a model from disk, 3 seconds causes false "unavailable" status. Recommend 5-8 seconds.

⚠️ **No retry logic in `generate()`**: `llm_service.py:244-246` makes a single HTTP POST. A transient I/O spike causes timeout and falls back to deterministic mode.

❌ **No connection pooling to Ollama.** `LocalLLMService` uses `httpx.AsyncClient` via `async with` — a new client per call. Each query opens and closes a new TCP connection to localhost:11434. A shared `httpx.AsyncClient` should be created at init and closed in the lifespan hook.

### 4.4 LLM Integration in Query Pipeline

✅ Merge strategy in `query_understanding.py:689-750` is correct: Gazetteer wins for location/bbox, regex wins for temporal, LLM wins for intent/concept/target.
✅ `parse_async()` always calls deterministic `parse()` first — guaranteed non-null base result.

⚠️ **`semantic_search.py:63` calls synchronous `parse()`**, not `parse_async()`. This means the LLM enhancement is **never invoked from the main `/api/search/semantic` endpoint**. The LLM pipeline is only accessible via `POST /api/ai/parse-query`. Phase 14 is not actually wired into live search results.

---

## 5. Query Understanding (Phases 12-13)

### 5.1 Query Normalization (Phase 12)

✅ **`QueryNormalizer.normalize()`** — four-stage pipeline: cleanup → punctuation normalization → synonym substitution → lemmatization.
✅ **54 synonym mappings** covering energy, aviation, transportation, water, vegetation, arid, and urban domains.
✅ **43 domain lemma mappings** for satellite-specific plural-to-singular normalization.
✅ **Punctuation normalization** correctly preserves hyphens in dates, decimal points in coordinates, and compound words.

### 5.2 Spelling Correction (Phase 13)

✅ **Damerau-Levenshtein implementation** in `spelling_correction.py:146-187` — full OSA algorithm supporting insertions, deletions, substitutions, and transpositions.
✅ **`dessert → desert` confusable** mapped at highest precedence (confidence 0.96).
✅ **116+ domain vocabulary words** as correction target candidates.
✅ **Max edit distance of 2** for words ≥5 chars; 1 for shorter words.
✅ **First-letter constraint for distance-2 corrections** (`spelling_correction.py:251`) prevents `farms → dams` false replacements.

⚠️ **`correct_query()` is O(n × |vocabulary|)** per token — fast now at 116 words, but will slow down if vocabulary grows past ~1000 terms.

⚠️ **`_format_suggestion()` uses a hardcoded proper-name table** (25 entries). Geographic names not in this list will not receive proper capitalization in "Did you mean" suggestions.

### 5.3 Geographic Entity Understanding (Gazetteer)

✅ **29 geographic entities** registered: 1 country, 11 states/UTs, 12 cities, 5 regions — all with EPSG:4326 bounding boxes.
✅ **WKT polygon property** generates valid PostGIS polygon string.
✅ **Alias matching** with length-priority sort ensures "Greater Noida" matches before "Noida".
✅ **`strip_entity_mentions()`** correctly removes location from embedding text.

⚠️ **Bounding boxes are axis-aligned rectangles.** For oddly-shaped states (e.g., Kerala elongated N-S), this generates a wide bbox matching tiles outside the true state boundary. False positives are possible.

⚠️ **`KNOWN_LOCATIONS` in `query_understanding.py:28-53` is a partial duplicate** of `GAZETTEER_ENTITIES` in `gazetteer.py`. If a bbox is updated in one place, it must be updated in both — maintenance risk.

❌ **Gazetteer has no mechanism for dynamic region addition.** Adding Telangana, Odisha, or Assam requires editing two source files separately. No admin API or database-backed entity store exists.

---

## 6. Semantic Search Pipeline

✅ **Full pipeline**: text → embedding → Qdrant filter → PostGIS spatial verification → spectral fusion → provenance enrichment.
✅ **Fetch limit**: `min(1000, max(top_k * 20, 500))` — over-fetches to allow post-filtering.

⚠️ **Qdrant spatial filter uses `center_lon` / `center_lat` points**, not tile footprint polygons. Tiles whose center is outside the bbox but whose edge overlaps are missed by the Qdrant pre-filter. The subsequent PostGIS step corrects this, but only for tiles that survived the center-point pre-filter.

⚠️ **`spectral_analysis_service.analyze_image(thumb_path)` is called per tile** (`semantic_search.py:348`). For 500 fetched tiles, this results in 500 JPEG thumbnail disk reads per query — a significant performance bottleneck.

⚠️ **No in-memory cache for spectral results.** The same thumbnail may be analyzed on every query that matches that tile.

---

## 7. Change Detection

✅ **`ChangeDetectionService`** uses `ST_Intersects` for PostGIS spatial tile selection.
✅ **`MultiFactorConfidenceEngine`** implements 10 independent physical signals.
✅ **OpenCV ECC sub-pixel registration** confirmed.
✅ **Cloud/shadow suppression** with hard thresholds and configurable multipliers.
✅ Seasonal phenology flag suppresses false alarms from agricultural greening cycles.
✅ **Config endpoints** at `PUT /api/change/confidence/config` allow runtime weight adjustment.

⚠️ **`ChangeDetectionService.__init__(self, db)` takes `AsyncSession` as a constructor argument** (`change_detection.py:42`). A new service instance is created per API request (`change.py:31`). This is an anti-pattern compared to the singleton services. It prevents service-level state caching.

❌ **Change detection logger uses `logging.getLogger(__name__)`** (`change_detection.py:22`) while all other services use `from app.core.logging import get_logger`. Change detection logs are not captured by the platform's `structlog` pipeline and will not carry context fields.

---

## 8. Vector Store (Qdrant)

✅ **Qdrant embedded mode** — no server process required.
✅ **Singleton pattern** prevents multiple `QdrantClient` instances fighting over the same local file lock (critical on Windows).
✅ **`prune_orphaned_vectors()`** correctly reconciles Qdrant vs. PostgreSQL tile IDs.
✅ **`vector_store.close()`** called in lifespan shutdown to release file lock.

❌ **`ensure_collections()` called in `__init__`** (`vector_store.py:48`) which runs at module import time via the module-level singleton at line 318. If the Qdrant store directory does not exist or is locked, **the FastAPI app import itself fails**. This should be wrapped in try/except with a degraded startup mode.

---

## 9. Ingestion Pipeline

✅ Pipeline stages: validate → extract metadata → quality → tile → thumbnails → PostGIS → provenance.
✅ Per-stage progress tracking in `ProcessingJob` record.
✅ `pyproj` PROJ path initialization correctly handles PROJ data directory on Windows.

⚠️ **`min_valid_pixel_ratio=0.3`** — tiles with more than 70% nodata are filtered. This is aggressive for coastal scenes.

---

## 10. Security Audit

🔴 **No authentication or authorization anywhere.** No API keys, JWT tokens, or user sessions on any FastAPI endpoint. Any process that can reach `localhost:8000` can delete scenes, trigger ingestion of arbitrary files, or modify confidence engine weights.

⚠️ **`SECRET_KEY` defaults to `"change-me"`** (`config.py:32`). Not currently used for signing (so immediate risk is zero), but any future feature inherits a weak key if the developer forgets to change it.

⚠️ **Static file mounts expose all thumbnails without access control** (`main.py:100-113`). Any browser reaching the backend can enumerate and download satellite thumbnails.

✅ **`.env` is in `.gitignore`** — credentials correctly excluded from version control.

---

## 11. Dependency Audit

| Package | Version | Risk |
|---------|---------|------|
| `react` | `19.2.8` | Pre-GA release — ecosystem compatibility risk |
| `next` | `16.3.4` | Very new — possible undocumented breaking changes |
| `torch` | `2.5.1` | Stable |
| `rasterio` | `1.4.3` | Stable |
| `qdrant-client` | `1.12.1` | Stable |
| `open-clip-torch` | `2.26.1` | Stable |
| `opencv-python-headless` | `4.10.0.84` | Stable |
| `lucide-react` | `^1.40.0` | Very new — breaking changes possible |
| `maplibre-gl` | `^6.7.0` | Recent major version |

⚠️ **`httpx` listed twice in `requirements.txt`** (lines 44 and 59). Both `0.28.0`, no conflict, but sloppy.

⚠️ **Frontend uses `^` version ranges** for all dependencies. In an offline-first platform, this is dangerous — `npm install` requires internet if `node_modules` is missing. The `package-lock.json` is present which is the saving grace.

---

## 12. Test Coverage

17 test files in `backend/tests/`:

| File | Size | Notes |
|------|------|-------|
| `test_local_llm.py` | 16.0 KB | Largest — good coverage |
| `test_embedding_pipeline.py` | 14.3 KB | Good coverage |
| `test_semantic_search.py` | 13.1 KB | Good coverage |
| `test_visual_search.py` | 10.4 KB | Good coverage |
| `test_confidence_engine.py` | 10.2 KB | Good coverage |
| `test_models.py` | 9.2 KB | Good coverage |
| `test_geographic_understanding.py` | 8.0 KB | Good coverage |
| `test_query_understanding.py` | 8.2 KB | Good coverage |
| `test_tiling.py` | 7.2 KB | Good coverage |
| `test_spelling_correction.py` | 6.9 KB | Good coverage |
| `test_quality.py` | 6.1 KB | Good coverage |
| `test_ingestion.py` | 5.8 KB | Good coverage |
| `test_search_parse_api.py` | 3.3 KB | Acceptable |
| `test_change_detection.py` | **2.4 KB** | **⚠️ Critically thin for a 761-line service** |
| `test_api_health.py` | 1.5 KB | Acceptable |

✅ All 5 core intelligence pillars have dedicated test files.
⚠️ `test_change_detection.py` at 2.4 KB is disproportionately thin for the largest service (761 lines, most complex business logic).
⚠️ No pytest-based end-to-end integration test. `test_end_to_end.py` in `scripts/` is a script, not a pytest test.

---

## 13. Logging

✅ **`structlog`** used throughout via `from app.core.logging import get_logger`.
✅ Log entries carry named fields (`query=`, `total=`, `duration_ms=`).

❌ **`change_detection.py` uses `logging.getLogger(__name__)`** (stdlib, not structlog). Logs from this module appear in raw format, mixed with structured output. This is a consistency defect that breaks log parsing.

---

## 14. Configuration Management

✅ **`pydantic-settings`** with `BaseSettings` — clean, type-safe.
✅ Loads `.env` from two locations: backend-relative and CWD.
✅ **`resolve_paths()`** validator converts all relative paths to absolute at startup.
✅ `find_local_checkpoint()` searches three candidate locations for model files.

⚠️ **No LLM settings in `.env` template.** `LLM_ENABLED`, `LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL` are defined in `config.py` but absent from `.env`. Users who want to change the LLM model have no guidance from the template.

---

## 15. Summary of Defects by Severity

### 🔴 Critical

| # | Issue | File | Line |
|---|-------|------|------|
| C1 | No authentication on any API endpoint | `app/main.py` | All routes |
| C2 | `VectorStoreService.__init__` crashes startup if Qdrant path fails | `vector_store.py` | 41–53 |
| C3 | LLM never invoked from live search — `parse()` called instead of `parse_async()` | `semantic_search.py` | 63 |

### ❌ High Severity

| # | Issue | File | Line |
|---|-------|------|------|
| H1 | `change_detection.py` uses stdlib logger, not structlog | `change_detection.py` | 22 |
| H2 | `ChangeDetectionService` instantiated per request (anti-pattern) | `change.py` | 31 |
| H3 | No httpx connection pool in `LocalLLMService` | `llm_service.py` | 141, 182, 244 |
| H4 | Tile size mismatch: model default 256 vs. config default 512 | `tile.py:53`, `config.py:54` | — |

### ⚠️ Medium

| # | Issue | File |
|---|-------|------|
| M1 | Qdrant spatial filter uses center point, not footprint | `semantic_search.py:141-180` |
| M2 | Spectral thumbnail reads not cached — O(n) disk I/O per search | `semantic_search.py:348` |
| M3 | Duplicate bbox data in `KNOWN_LOCATIONS` and `GAZETTEER_ENTITIES` | `query_understanding.py:28-53` |
| M4 | Gazetteer uses axis-aligned bbox, not true state boundaries | `gazetteer.py:36-39` |
| M5 | `llm_status_timeout_seconds=3.0` too tight | `config.py:88` |
| M6 | No LLM env vars in `.env` template | `.env` |
| M7 | Hardcoded localhost backend URL in Next.js rewrites | `next.config.ts:21` |
| M8 | Double-registration of all routers | `main.py:133-148` |
| M9 | Session commits on every GET request | `database.py:59` |
| M10 | Change detection test coverage critically thin | `tests/test_change_detection.py` |
| M11 | `httpx` listed twice in requirements | `requirements.txt:44,59` |
| M12 | Frontend hardcodes model size strings | `page.tsx:217,228` |

---

## 16. What Is Genuinely Working Well

1. **Offline guarantee is real.** `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are enforced at module load time. The platform cannot accidentally download weights at runtime.

2. **The confidence engine is the strongest component.** 10-signal weighted fusion with configurable weights, four distinct suppression triggers, transparent factor breakdown, and seasonal phenology detection is architecturally correct and novel.

3. **The spelling corrector is a proper implementation**, not a word list lookup. Damerau-Levenshtein with domain-weight bonuses and first-letter constraints is correct approach for a specialized vocabulary.

4. **The gazetteer/query pipeline cleanly separates visual concept from location.** "Desert in Rajasthan" does not embed "Rajasthan" as a visual feature — this is the correct semantic decomposition.

5. **The LLM fallback chain is robust.** The platform cannot break because Ollama is offline. Every call is wrapped in try/except, the fallback is immediate, and the caller always gets a valid `ParsedQuery`.

6. **Provenance tracking via `ProvenanceRecord`** records the model name, version, timestamp, and operator for every tile embedding — full data lineage.

7. **PostgreSQL+PostGIS spatial pipeline is correct.** `ST_Intersects`, `ST_MakeEnvelope`, `ST_GeomFromText`, and `ST_GeomFromWKT` are used properly with SRID 4326 throughout.

---

## 17. SIH 2026 Evaluation Readiness

| Criterion | Status | Notes |
|-----------|--------|-------|
| Semantic retrieval | ✅ Ready | RemoteCLIP + Qdrant operational |
| Multi-temporal change analysis | ✅ Ready | 10-signal confidence engine |
| Offline operation | ✅ Ready | Enforced at code level |
| Geographic query understanding | ✅ Ready | Gazetteer + PostGIS |
| Spelling robustness | ✅ Ready | Damerau-Levenshtein |
| Local LLM integration | ⚠️ Partial | LLM not wired to main search endpoint (C3) |
| Authentication | ❌ Missing | No auth on any endpoint (C1) |
| Production deployment | ❌ Not ready | Hardcoded localhost URLs everywhere |
| Change detection test coverage | ⚠️ Thin | 761-line service, 2.4 KB test file |

---

*Audit conducted by read-only source inspection. No files were modified.*
