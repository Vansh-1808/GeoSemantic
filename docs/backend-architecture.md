# Backend Architecture

## Module Structure

```
backend/
├── app/
│   ├── main.py              FastAPI app factory, lifecycle hooks
│   ├── api/
│   │   └── routes/
│   │       ├── health.py    GET /api/health, /api/health/detailed
│   │       ├── ingest.py    POST /api/ingest/scene, GET status/scenes
│   │       ├── tiles.py     GET /api/tiles/{id}, /thumbnails
│   │       └── search.py    POST /api/search/text, /image (Phase 5)
│   ├── core/
│   │   ├── config.py        Pydantic settings (reads .env)
│   │   └── logging.py       Structlog configuration
│   ├── db/
│   │   └── database.py      Async SQLAlchemy engine, session factory
│   ├── models/              SQLAlchemy ORM models
│   │   ├── scene.py         scenes table
│   │   ├── tile.py          tiles table
│   │   ├── change_event.py  change_events table
│   │   ├── analyst.py       analyst_decisions, feedback_signals
│   │   └── provenance.py    provenance_records, processing_jobs
│   ├── schemas/             Pydantic request/response models
│   │   └── scene.py
│   └── services/            Business logic (no FastAPI dependencies)
│       └── ingestion.py     Full ingestion pipeline
├── alembic/                 Database migrations
├── tests/                   Pytest tests
├── requirements.txt
├── pyproject.toml
└── alembic.ini
```

## Database Schema

### scenes
Core scene record. One row per ingested GeoTIFF.
- Geographic footprint stored as PostGIS POLYGON (EPSG:4326)
- Original CRS preserved in `crs_wkt`
- Status: PENDING → PROCESSING → COMPLETED | FAILED

### tiles
256×256 (configurable) tiles extracted from scenes.
- Maintains full geospatial context (footprint, coordinates)
- Links to Qdrant vector IDs for embedding retrieval
- Inherits acquisition metadata from parent scene

### change_events
Detected changes between tile pairs.
- Full multi-factor confidence breakdown in JSONB
- Evidence stored as structured JSONB for UI display
- False-alarm suppression flags with human-readable reasons

### analyst_decisions
Audit trail of analyst confirm/reject/flag actions.
- Links to change_events
- Stores corrected change type and notes
- Used for feedback-based reranking

### feedback_signals
Positive/negative signals from analyst actions.
- Used by retrieval service to rerank results
- Linked to tiles, not change events

### provenance_records
Immutable audit log for all processing operations.
- entity_type: scene | tile | change_event | analyst_decision
- details: JSONB with model name, version, checksum, parameters

### processing_jobs
Tracks async background operations.
- Status: PENDING → RUNNING → COMPLETED | FAILED | CANCELLED
- Progress percentage and message for UI polling

## Services (implemented per phase)

| Service | Phase | Purpose |
|---------|-------|---------|
| ingestion.py | 2 | GeoTIFF → tiles → thumbnails → PostGIS |
| embedding_service.py | 3 | RemoteCLIP + DINOv2 embedding generation |
| vector_store.py | 4 | Qdrant collection management |
| retrieval_service.py | 5 | Text/image search + feedback reranking |
| change_detection_service.py | 6 | Multi-factor change analysis |
| temporal_service.py | 7 | Persistence analysis, earliest observation |
| confidence_engine.py | 8 | Interpretable multi-factor confidence |
| clustering_service.py | 10 | HDBSCAN + UMAP cluster discovery |
| provenance_service.py | 13 | Export, audit trail generation |
