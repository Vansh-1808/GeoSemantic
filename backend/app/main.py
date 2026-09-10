"""
FastAPI application entry point.
Mounts all routers, configures middleware, and manages startup/shutdown lifecycle.
"""
# Reload trigger: change routes updated without duplicate prefix
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import os
try:
    import pyproj
    proj_dir = pyproj.datadir.get_data_dir()
    os.environ["PROJ_LIB"] = proj_dir
    os.environ["PROJ_DATA"] = proj_dir
except Exception:
    pass

from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.db.database import check_database_connection

# Configure structured logging before anything else
configure_logging(settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    # ── Startup ───────────────────────────────────────────────
    logger.info("geosemantic_starting", env=settings.app_env)

    # Ensure data directories exist
    settings.ensure_directories()

    # Verify database connectivity
    db_ok = await check_database_connection()
    if not db_ok:
        logger.error("database_unavailable_on_startup")
    else:
        logger.info("database_connected")

    # Lazy-import and initialize services (so startup isn't blocked by model loading)
    # Models are loaded on first inference request to keep startup fast
    from app.api.routes import health  # noqa: F401 — registers routes

    logger.info("geosemantic_ready", port=settings.app_port)

    yield

    # ── Shutdown ──────────────────────────────────────────────
    try:
        from app.services.vector_store import vector_store
        vector_store.close()
    except Exception:
        pass
    logger.info("geosemantic_shutting_down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="GeoSemantic Satellite Intelligence Platform",
        description=(
            "Offline-first satellite imagery retrieval, change detection, "
            "and analyst review system. SIH 2026."
        ),
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            settings.frontend_url,
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3001",
        ],
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Static files (thumbnails, previews, exports) ──────────
    settings.thumbnails_dir.mkdir(parents=True, exist_ok=True)
    settings.previews_dir.mkdir(parents=True, exist_ok=True)
    settings.exports_dir.mkdir(parents=True, exist_ok=True)

    app.mount(
        "/thumbnails",
        StaticFiles(directory=str(settings.thumbnails_dir)),
        name="thumbnails",
    )
    app.mount(
        "/previews",
        StaticFiles(directory=str(settings.previews_dir)),
        name="previews",
    )
    app.mount(
        "/exports",
        StaticFiles(directory=str(settings.exports_dir)),
        name="exports",
    )

    # ── Routers ───────────────────────────────────────────────
    from app.api.routes import (  # noqa: F401
        embedding,
        health,
        ingest,
        models,
        quality,
        scenes,
        search,
        tiles,
        change,
    )

    app.include_router(health.router, prefix="/api", tags=["health"])
    app.include_router(ingest.router, prefix="/api/ingest", tags=["ingestion"])
    app.include_router(scenes.router, prefix="/api/scenes", tags=["scenes"])
    app.include_router(scenes.router, prefix="/scenes", tags=["scenes-root"])
    app.include_router(tiles.router, prefix="/api/tiles", tags=["tiles"])
    app.include_router(tiles.router, prefix="/tiles", tags=["tiles-root"])
    app.include_router(models.router, prefix="/api/models", tags=["models"])
    app.include_router(models.router, prefix="/models", tags=["models-root"])
    app.include_router(embedding.router, prefix="/api/embedding", tags=["embedding"])
    app.include_router(embedding.router, prefix="/api/embeddings", tags=["embeddings-alias"])
    app.include_router(embedding.router, prefix="/embedding", tags=["embedding-root"])
    app.include_router(embedding.router, prefix="/api/vector", tags=["vector"])
    app.include_router(embedding.router, prefix="/vector", tags=["vector-root"])
    app.include_router(search.router, prefix="/api/search", tags=["search"])
    app.include_router(search.router, prefix="/search", tags=["search-root"])
    app.include_router(quality.router, prefix="/api/quality", tags=["quality"])
    app.include_router(quality.router, prefix="/quality", tags=["quality-root"])
    app.include_router(change.router, prefix="/api/change", tags=["change"])
    app.include_router(change.router, prefix="/change", tags=["change-root"])

    return app


app = create_app()
