# GeoSemantic Satellite Intelligence Platform — Complete User Manual & System Guide

Welcome to the **GeoSemantic Satellite Intelligence Platform**. This document provides an exhaustive, end-to-end guide explaining everything you can do with the application, how the system works under the hood, how to upload and ingest datasets, and how to execute semantic text searches, visual similarity searches, and multi-filter GeoSemantic queries.

---

## Table of Contents
1. [Platform Overview & Core Philosophy](#1-platform-overview--core-philosophy)
2. [What You Can Do With the App (Features & Capabilities)](#2-what-you-can-do-with-the-app-features--capabilities)
3. [System Architecture & Technology Stack](#3-system-architecture--technology-stack)
4. [How to Upload & Ingest Satellite Datasets](#4-how-to-upload--ingest-satellite-datasets)
   - [Method 1: Web Interface Upload (Drag & Drop)](#method-1-web-interface-upload-drag--drop)
   - [Method 2: REST API Ingestion](#method-2-rest-api-ingestion)
   - [Method 3: Automated CLI & Demo Data Generation](#method-3-automated-cli--demo-data-generation)
5. [Step-by-Step Guide to Key Features](#5-step-by-step-guide-to-key-features)
   - [A. Semantic Text-to-Satellite Retrieval](#a-semantic-text-to-satellite-retrieval)
   - [B. Multi-Filter Search & GeoSemantic Fusion (Phase 8)](#b-multi-filter-search--geosemantic-fusion-phase-8)
   - [C. Visual & Image-to-Image Similarity Search](#c-visual--image-to-image-similarity-search)
   - [D. Tile Inspection & Provenance Audit](#d-tile-inspection--provenance-audit)
   - [E. System Telemetry & Vector Health](#e-system-telemetry--vector-health)
6. [How to Start the Platform Locally](#6-how-to-start-the-platform-locally)
7. [API Reference & Endpoints](#7-api-reference--endpoints)

---

## 1. Platform Overview & Core Philosophy

The **GeoSemantic Platform** is designed for high-precision satellite imagery intelligence. It enables analysts, geospatial researchers, and defense/remote-sensing teams to search through large archives of satellite imagery using **natural language descriptions** (e.g., *"industrial facilities near water"* or *"commercial airport runway & aprons"*) or **sample images** to find visually similar locations.

### Key Tenets
1. **100% Offline & Sovereign**: All neural network inference, text embeddings, visual similarity extractions, raster tiling, vector indexing, and spatial queries execute locally on your machine. **Zero data or telemetry ever leaves your environment.**
2. **Two-Tier Spatial Fusion**: Combines fast Qdrant vector filtering (using geographic bounding envelopes and metadata) with exact **PostGIS GiST** spatial verification (`ST_Intersects` and `ST_Within`).
3. **Rigorous Geospatial Provenance**: Every tile and vector maintains an immutable audit trail in PostgreSQL, recording acquisition timestamps, satellite sensor, quality score, cloud cover percentage, exact WGS-84 footprint, and AI model checkpoints.

---

## 2. What You Can Do With the App (Features & Capabilities)

| Capability | What It Does | Where in the App |
| :--- | :--- | :--- |
| **Ingest Satellite Scenes** | Upload GeoTIFF/TIFF/PNG/JPEG satellite files. The system validates CRS, extracts metadata, computes quality/cloud scores, tiles the scene into 512x512 tiles, and creates thumbnails. | `/ingest` or `POST /api/scenes/ingest` |
| **Explore Dataset Archive** | View all ingested satellite scenes, inspect their geographic footprints, view tile counts, resolution, cloud cover, and launch embedding jobs. | `/datasets` |
| **Semantic Text Search** | Enter natural language queries. The query is embedded via **RemoteCLIP ViT-B-32** (512-dim) and matched against the Qdrant vector index. | `/search` |
| **Multi-Filter & GeoSemantic Fusion** | Restrict searches by Area of Interest (AOI) polygon, bounding box, date ranges, sensors (`Sentinel-2`, `Demo-MSI`), min quality score, and max cloud cover. | `/search` (Expand Filters) |
| **Interactive Map Polygon Drawing** | Click directly on the interactive coverage map to draw custom multi-point polygons or click-and-drag bounding boxes. | `/search` (Spatial Canvas) |
| **Tile Footprint Overlays** | View the exact geographic vector footprint (`Polygon` GeoJSON) of every matching tile drawn directly on the spatial map with similarity color gradients. | `/search` & `/image-search` |
| **Image-to-Image Visual Search** | Upload a query satellite/drone image, or click **"Find Similar Locations"** on any existing tile, to retrieve visually similar terrain using **DINOv2** (384-dim). | `/image-search` |
| **Tile Inspection & Provenance Audit** | Inspect full-resolution imagery, pixel bounds, sensor specifications, and complete operator/pipeline provenance history. | Any tile card -> "Inspect & Audit" |
| **System & Vector Storage Telemetry** | Inspect database status, disk usage, model checkpoint sizes, Qdrant collection point counts, and device status (CPU/CUDA). | `/system` |

---

## 3. System Architecture & Technology Stack

```
                                    +----------------------------------------+
                                    |     Next.js 16 Modern Web Interface    |
                                    |    (React 19, Tailwind CSS, Lucide)    |
                                    +----------------------------------------+
                                                        |
                                          HTTP / REST (Next.js Rewrites)
                                                        |
                                                        v
+--------------------------------------------------------------------------------------------------------------------+
|                                              FastAPI Backend (Python 3.12)                                         |
|                                                                                                                    |
|  +---------------------------+  +--------------------------+  +--------------------------+  +-------------------+  |
|  |     Ingestion Service     |  |      Tiling Service      |  |     Embedding Service    |  |   Search Service  |  |
|  | (Rasterio, Geospatial Val)|  |   (512x512, Overlap)     |  | (RemoteCLIP + DINOv2)    |  |  (Fusion Engine)  |  |
|  +---------------------------+  +--------------------------+  +--------------------------+  +-------------------+  |
|                |                              |                             |                             |        |
+----------------|------------------------------|-----------------------------|-----------------------------|--------+
                 |                              |                             |                             |
                 v                              v                             v                             v
+------------------------------------+  +--------------------+  +---------------------------+  +---------------------+
|        PostgreSQL 17 + PostGIS     |  |    Local Storage   |  |   Qdrant Vector Engine    |  |   Local AI Models   |
|   (Tables: scenes, tiles, jobs,    |  |  data/raw_scenes/  |  |     (Embedded disk store) |  |   RemoteCLIP (512d) |
|    provenance, GiST spatial idx)   |  |  data/tiles/       |  | Collections:              |  |   DINOv2 (384d)     |
|                                    |  |  data/thumbnails/  |  | - remoteclip_tiles        |  |                     |
|                                    |  |                    |  | - dino_tiles              |  |                     |
+------------------------------------+  +--------------------+  +---------------------------+  +---------------------+
```

---

## 4. How to Upload & Ingest Satellite Datasets

There are three ways to upload satellite datasets into the platform:

### Method 1: Web Interface Upload (Drag & Drop)
1. Open your browser and navigate to **`http://localhost:3000/ingest`**.
2. Drag and drop any satellite raster file into the upload zone (supported formats: `.tif`, `.tiff`, `.png`, `.jpg`).
3. The system automatically:
   - Validates the raster coordinate reference system (CRS) and geographic bounds.
   - Calculates the quality score and cloud cover percentage.
   - Computes optimum grid divisions (512x512 analysis tiles).
   - Generates an RGB thumbnail preview.
4. Click **"Ingest & Process Scene"**.
5. Once ingested, click **"Generate Embeddings"** to process the tiles through the offline RemoteCLIP and DINOv2 models into Qdrant.

### Method 2: REST API Ingestion
You can ingest scenes programmatically or via `curl`:
```bash
curl -X POST "http://localhost:8000/api/scenes/ingest" \
     -F "file=@/path/to/satellite_scene.tif" \
     -F "sensor=Sentinel-2B" \
     -F "acquisition_date=2024-03-15T10:30:00Z"
```
The response returns the created `scene_id`, file metadata, resolution, and calculated bounds.

To trigger embedding generation for all tiles in that scene:
```bash
curl -X POST "http://localhost:8000/api/embedding/generate/{scene_id}?sync=true"
```

### Method 3: Automated CLI & Demo Data Generation
If you want to quickly populate the platform with realistic synthetic satellite imagery across multiple dates and sensors:
```powershell
# From the repository root:
backend\.venv\Scripts\python scripts/generate_demo_data.py
```
This script generates synthetic multispectral scenes (urban, industrial, agricultural, river corridors), processes tiles, and indexes vectors into Qdrant.

To run a complete end-to-end verification of the entire pipeline:
```powershell
backend\.venv\Scripts\python scripts/test_end_to_end.py
```

---

## 5. Step-by-Step Guide to Key Features

### A. Semantic Text-to-Satellite Retrieval
1. Navigate to **`http://localhost:3000/search`**.
2. Type any natural language prompt into the search bar:
   - *"commercial airport runway & aprons"*
   - *"dense residential buildings and road grid"*
   - *"river channel with highway bridge crossing"*
   - *"industrial chemical plant with storage tanks"*
   - *"harbor docks with cargo container vessels"*
3. Click **Search Archive** (or press `Enter`).
4. **Behind the scenes**:
   - The query text is encoded using the offline **RemoteCLIP ViT-B-32** vision-language model into a 512-dimensional vector.
   - Qdrant computes cosine similarity against all tile image vectors in milliseconds.
   - Matching tiles are ranked and returned with similarity scores, thumbnails, sensor tags, and acquisition dates.

---

### B. Multi-Filter Search & GeoSemantic Fusion (Phase 8)
To combine semantic retrieval with geospatial and metadata constraints:
1. On the **`/search`** page, click the **"Filters"** button to open the filter drawer.
2. **Sensor Selection**: Click one or more sensor pills (e.g. `Sentinel-2`, `Demo-MSI`).
3. **Date Range**: Set the Start Date and End Date to restrict results chronologically.
4. **Quality & Atmosphere Sliders**:
   - Set **Min Quality Score** (e.g. `80%`) to filter out degraded or artifacted tiles.
   - Set **Max Cloud Cover** (e.g. `20%`) to guarantee cloud-free imagery.
5. **Area of Interest (AOI) Polygon Drawing**:
   - In the right-hand map header, click **"Draw Polygon"**.
   - Click anywhere on the map grid to place vertices around your region of interest.
   - Click **"Complete (N pts)"** to finalize the polygon.
6. **Bounding Box Mode**:
   - Alternatively, click **"Drag BBox"** and drag a rectangle across the canvas, or enter coordinates in the West, South, East, North input fields.
7. **Spatial Operator Toggle**:
   - Toggle between **`ST_Intersects`** (tiles that intersect the drawn area) and **`ST_Within`** (tiles strictly inside).
8. Click **Search Archive**.
9. **Two-Tier Fusion in Action**:
   - Tier 1: Qdrant uses the polygon envelope and metadata to prune candidates instantly.
   - Tier 2: PostGIS verifies exact tile boundary intersections using GiST indexes in < 10ms.
   - The telemetry bar displays: Total time, Text Embed time, Qdrant search time, and PostGIS GiST time.

---

### C. Visual & Image-to-Image Similarity Search
1. Navigate to **`http://localhost:3000/image-search`**.
2. You can initiate visual searches in two ways:
   - **Upload an Image**: Drag & drop any aerial/satellite photo or local screenshot into the upload box.
   - **Archive Tile Selector**: Select any previously ingested tile from the visual drawer.
   - **"Find Similar Locations" Button**: On any tile detail modal across the app, click this button to automatically jump to visual search with that tile as the reference.
3. The offline **DINOv2** visual model extracts 384-dimensional features.
4. Qdrant scans the `dino_tiles` collection and returns visually similar satellite terrain, automatically excluding the reference tile from its own results.

---

### D. Tile Inspection & Provenance Audit
1. Click **"Inspect & Audit"** on any search result tile or scene card.
2. The modal displays:
   - High-resolution imagery preview.
   - Geographic centroid (`[latitude, longitude]`) and bounding box.
   - Quality score, cloud cover %, and pixel dimensions.
   - **Full Provenance Audit Trail**: Ingestion timestamp, tiling pipeline version, RemoteCLIP vector ID, DINOv2 vector ID, and operator identity.

---

### E. System Telemetry & Vector Health
1. Navigate to **`http://localhost:3000/system`**.
2. Inspect:
   - **PostGIS Connection**: Live ping and spatial table verification.
   - **Offline AI Models**: Checkpoint presence, file size, device (`CPU` or `CUDA`), and embedding dimensions.
   - **Qdrant Vector Collections**: Point counts, vector counts, and storage usage in MB.
   - **Local Storage Health**: Storage paths for raw scenes, tiles, thumbnails, and model weights.

---

## 6. How to Start the Platform Locally

The platform is designed to run entirely offline on Windows/Linux/macOS with two lightweight processes:

### Terminal 1 — Backend (FastAPI + Uvicorn)
```powershell
cd backend
.venv\Scripts\uvicorn app.main:app --reload --port 8000
```
- API Base: `http://localhost:8000`
- Interactive Swagger Docs: `http://localhost:8000/api/docs`

### Terminal 2 — Frontend (Next.js 16)
```powershell
cd frontend
npm run dev
```
- Web Application: `http://localhost:3000`

---

## 7. API Reference & Endpoints

| Method | Route | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | High-level system health check. |
| `GET` | `/api/health/detailed` | Detailed status of PostGIS, models, directories, and offline status. |
| `GET` | `/api/models/status` | Reports loaded status, dimensions, and versions for RemoteCLIP & DINOv2. |
| `POST` | `/api/scenes/ingest` | Uploads and registers a new satellite scene raster. |
| `GET` | `/api/scenes` | Lists ingested scenes with pagination and metadata. |
| `POST` | `/api/embedding/generate/{scene_id}` | Runs RemoteCLIP & DINOv2 embedding on all tiles for a scene. |
| `GET` | `/api/embedding/status/{scene_id}` | Tracks real-time embedding progress percentage. |
| `GET` | `/api/vector/statistics` | Qdrant point counts, collection statuses, and disk storage usage. |
| `POST` | `/api/search/semantic` | Natural language text-to-satellite retrieval with multi-filter fusion. |
| `GET` | `/api/search/filters` | Dynamic metadata filter values (sensors, date bounds, tile counts). |
| `POST` | `/api/search/image` | Image-to-image similarity search using an uploaded file. |
| `POST` | `/api/search/similar/{tile_id}` | Visual similarity search using an existing tile ID. |
| `GET` | `/api/tiles/{tile_id}` | Detailed tile metadata and provenance audit trail. |

---

*GeoSemantic Platform — Built for Autonomous, 100% Offline Satellite Intelligence.*
