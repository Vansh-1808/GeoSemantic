# GeoSemantic Satellite Intelligence Platform

**SIH 2026 — Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery**

> An offline-first, production-quality satellite imagery intelligence platform. Search archives with natural language, run image similarity queries, detect meaningful changes over time, and explore clusters of similar locations — all running locally on your machine.

---

## Quick Start

### Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.11+ | 3.12 recommended |
| Node.js | 20+ | LTS |
| PostgreSQL + PostGIS | 15–18 | Local installation |
| Git | Any | — |

### 1. Clone & configure

```bash
git clone <repo-url>
cd SIH2026
cp backend/.env.example backend/.env
# Edit backend/.env — set your PostgreSQL credentials
```

### 2. Backend setup

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt

# Create the database
python scripts/setup_db.py

# Run migrations
alembic upgrade head
```

### 3. Download AI models (one-time, requires internet)

```bash
python scripts/download_models.py
```

### 4. Frontend setup

```bash
cd frontend
npm install
```

### 5. Start services

**Terminal 1 — Backend:**
```bash
cd backend
.venv\Scripts\activate
uvicorn app.main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```

Open **http://localhost:3000**

### 6. Fetch Satellite Imagery (Any Location / City)

Download real Sentinel-2 true-color satellite GeoTIFFs (10m resolution, 100% free, zero login or API key required) using Microsoft Planetary Computer's open STAC API:

**Download by City Name:**
```bash
# Activate backend environment
.venv\Scripts\activate

# Fetch imagery for any city (Delhi, Mumbai, Bengaluru, Chennai, Jaipur, Kolkata, Hyderabad, etc.)
python scripts/fetch_my_area.py --city "Delhi"
python scripts/fetch_my_area.py --city "Mumbai"
python scripts/fetch_my_area.py --city "Bengaluru"
```

**Download by Exact Coordinates (Latitude / Longitude):**
```bash
python scripts/fetch_my_area.py --lat 28.5562 --lon 77.1000 --name "delhi_airport" --size 2048
```

**Download Curated Public Satellite Samples:**
```bash
python scripts/download_real_samples.py
```

*Downloaded GeoTIFF scenes are saved to `backend/data/raw_scenes/`. You can drag-and-drop or ingest them directly through the web UI at `http://localhost:3000`.*

---

## Architecture

```
Frontend (Next.js 15)  ←→  Backend (FastAPI)  ←→  PostgreSQL+PostGIS
                                                ←→  Qdrant (embedded)
                                                ←→  Local File Store
                                                ←→  AI Models (local)
```

## Project Structure

```
SIH2026/
├── backend/          FastAPI application
├── frontend/         Next.js 15 analyst dashboard  
├── data/             Local storage (not committed)
│   ├── imagery/      Raw GeoTIFF scenes
│   ├── tiles/        Processed tile files
│   ├── thumbnails/   PNG previews
│   ├── models/       AI model weights
│   ├── qdrant_store/ Vector database storage
│   ├── exports/      Analysis exports
│   └── maps/         Offline basemap tiles
├── scripts/          Utility scripts
└── docs/             Documentation
```

## Offline Operation

After initial setup, all functionality works without internet:

- AI inference uses locally stored model weights
- Vector search uses embedded Qdrant (local files)
- Maps use locally cached tiles
- No external API calls of any kind

The dashboard shows an **OFFLINE MODE** indicator when network is unavailable.

---

## Key Features

| Feature | Status |
|---------|--------|
| GeoTIFF ingestion with metadata extraction | Phase 2 |
| Natural language semantic search | Phase 5 |
| Image-to-image visual similarity search | Phase 5 |
| Multi-temporal change detection | Phase 6 |
| Multi-factor confidence engine | Phase 8 |
| Temporal persistence analysis | Phase 7 |
| False-alarm suppression | Phase 6 |
| HDBSCAN cluster discovery | Phase 10 |
| Analyst review workflow | Phase 11 |
| Feedback-based reranking | Phase 12 |
| Complete provenance tracking | Phase 13 |
| Incremental ingestion | Phase 14 |
| Full offline operation | Phase 15 |

## License

SIH 2026 prototype. See individual model licenses in `data/models/MODEL_MANIFEST.json`.
