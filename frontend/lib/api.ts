/**
 * Typed API client for the GeoSemantic FastAPI backend.
 * All requests go through Next.js rewrites (/api → localhost:8000/api).
 */

import axios, { AxiosInstance } from "axios";

// ── Types ─────────────────────────────────────────────────────

export interface HealthStatus {
  status: "healthy" | "degraded";
  timestamp: string;
  offline_mode: boolean;
  components: {
    database: { ok: boolean; message: string };
    models: {
      remoteclip: { present: boolean; path: string; size_mb: number | null };
      dino: { present: boolean; note?: string; path?: string; size_mb?: number | null };
    };
    directories: Record<string, boolean>;
  };
  system: { platform: string; python: string; hostname: string };
}

export interface ModelStatusItem {
  model_name: string;
  version: string;
  loaded: boolean;
  loaded_status: boolean;
  local_path: string;
  local_path_valid: boolean;
  device: string;
  embedding_dimension: number;
  embedding_dim: number;
  modalities: string[];
  status: "LOADED" | "UNLOADED" | "CHECKPOINT_MISSING" | "ERROR";
  details?: {
    model_id?: string;
    has_weights_file?: boolean;
    last_error?: string | null;
  };
}

export interface ModelsStatusResponse {
  models: ModelStatusItem[];
  device: string;
  cuda_available: boolean;
  offline_mode: boolean;
  timestamp: string;
}

export interface TextEmbeddingResponse {
  model_name: string;
  text: string;
  embedding: number[];
  dimension: number;
  device: string;
}

export interface EmbeddingJobResponse {
  job_id: string;
  scene_id: string;
  status: "PENDING" | "RUNNING" | "COMPLETED" | "FAILED";
  total_tiles: number;
  embedded_count: number;
  skipped_count: number;
  duration_seconds: number | null;
  message: string;
}

export interface EmbeddingStatusResponse {
  scene_id: string;
  total_tiles: number;
  embedded_tiles: number;
  pending_tiles: number;
  progress_pct: number;
  status: "NOT_STARTED" | "IN_PROGRESS" | "COMPLETED" | "PARTIAL" | "FAILED";
  last_embedded_at: string | null;
  models: string[];
  collections: string[];
  active_job_id: string | null;
}

export interface VectorCollectionStats {
  collection_name: string;
  exists: boolean;
  points_count: number;
  vectors_count: number;
  vector_size: number;
  distance: string;
  status: string;
}

export interface VectorStatisticsResponse {
  total_vectors: number;
  total_points: number;
  collections: Record<string, VectorCollectionStats>;
  storage_usage_bytes: number;
  storage_usage_mb: number;
  storage_path: string;
  qdrant_mode: string;
  timestamp: string;
}

export interface SceneSummary {
  id: string;
  filename: string;
  sensor: string | null;
  acquisition_date: string | null;
  tile_count: number;
  embedded_count: number;
  quality_score: number | null;
  cloud_cover_pct: number | null;
  ingestion_status: "PENDING" | "PROCESSING" | "COMPLETED" | "FAILED";
  bbox_west: number | null;
  bbox_south: number | null;
  bbox_east: number | null;
  bbox_north: number | null;
  created_at: string;
}

export interface SceneListResponse {
  scenes: SceneSummary[];
  total: number;
  page: number;
  page_size: number;
}

export interface IngestionRequest {
  scene_id: string;
  job_id: string;
  message: string;
  status: string;
}

export interface IngestionStatus {
  scene_id: string;
  job_id: string;
  status: "PENDING" | "RUNNING" | "COMPLETED" | "FAILED" | "CANCELLED";
  progress_pct: number;
  message: string | null;
  error: string | null;
  tile_count: number;
  embedded_count: number;
}

export interface TileSummary {
  id: string;
  scene_id: string;
  tile_col: number;
  tile_row: number;
  tile_size?: number | null;
  pixel_x_off?: number | null;
  pixel_y_off?: number | null;
  pixel_width?: number | null;
  pixel_height?: number | null;
  center_lon: number | null;
  center_lat: number | null;
  bbox_west: number | null;
  bbox_south: number | null;
  bbox_east: number | null;
  bbox_north: number | null;
  acquisition_date: string | null;
  sensor: string | null;
  quality_score: number | null;
  cloud_cover_pct: number | null;
  thumbnail_url: string | null;
  preview_url: string | null;
  is_valid: boolean;
}

export interface TileDetail extends TileSummary {
  tile_path?: string | null;
  thumbnail_path?: string | null;
  resolution_m?: number | null;
  nodata_ratio?: number | null;
  remoteclip_vector_id?: string | null;
  dino_vector_id?: string | null;
  embedding_model?: string | null;
  embedded_at?: string | null;
  processing_history?: any[] | null;
  created_at: string;
}

export interface TileListResponse {
  tiles: TileSummary[];
  total: number;
  page: number;
  page_size: number;
}

// ── Quality Assessment Schemas ───────────────────────────────

export interface TileQuality {
  tile_id: string;
  quality_score: number;
  usable_for_change_analysis: boolean;
  suitability: "Suitable" | "Caution" | "Poor Quality";
  cloud_score: number;
  shadow_score: number;
  nodata_score: number;
  noise_score: number;
  saturation_score: number;
  cloud_cover_pct: number;
  nodata_ratio: number;
  analysis_method: string;
  satellite_mask_available: boolean;
  caveats: string[];
}

export interface SceneQuality {
  scene_id: string;
  quality_score: number;
  suitability: "Suitable" | "Caution" | "Poor Quality";
  tile_count: number;
  tiles_suitable: number;
  tiles_caution: number;
  tiles_poor: number;
  avg_cloud_cover_pct: number;
  avg_nodata_ratio: number;
  usable_for_change_analysis: boolean;
  quality_details: {
    suitable_ratio?: number;
    avg_shadow_score?: number;
    avg_noise_score?: number;
    avg_saturation_score?: number;
    [key: string]: any;
  };
}

export interface SceneComputeQualityResponse {
  scene_id: string;
  tiles_computed: number;
  quality_score: number;
  suitability: string;
  summary: SceneQuality;
}

// ── Change Detection Schemas ────────────────────────────────

export interface ChangeAnalyzeRequest {
  aoi_wkt?: string | null;
  start_date: string;
  end_date: string;
  limit?: number;
  query?: string;
}

export interface ChangeEventResponse {
  id: string;
  before_tile_id: string | null;
  after_tile_id: string | null;
  before_date: string | null;
  after_date: string | null;
  center_lon: number | null;
  center_lat: number | null;
  change_type: string | null;
  final_confidence: number | null;
  visual_change_score: number | null;
  semantic_change_score: number | null;
  registration_quality: number | null;
  is_suppressed: boolean;
  review_status: string;
  detected_at: string;
  updated_at: string;
  evidence?: Record<string, any> | null;
  confidence_breakdown?: Record<string, any> | null;
}

export interface ChangeAnalyzeResponse {
  message: string;
  candidates_found: number;
  events: ChangeEventResponse[];
}



export interface ProcessSceneRequest {
  tile_size?: number;
  overlap_px?: number;
  normalize?: boolean;
  min_valid_pixel_ratio?: number;
  generate_preview?: boolean;
}

export interface ProcessSceneResponse {
  scene_id: string;
  job_id: string;
  status: string;
  tile_count?: number | null;
  processing_time_seconds?: number | null;
  message: string;
}

export interface SemanticSearchRequest {
  query: string;
  top_k?: number;
  start_date?: string | null;
  end_date?: string | null;
  sensor?: string | null;
  sensors?: string[] | null;
  min_quality?: number | null;
  max_cloud_cover?: number | null;
  aoi?: [number, number, number, number] | null; // [west, south, east, north]
  aoi_bbox?: [number, number, number, number] | null;
  aoi_polygon?: [number, number][] | null; // [[lon, lat], ...]
  spatial_filter_mode?: "intersects" | "within";
  scene_id?: string | null;
}

export interface SemanticSearchResultItem {
  tile_id: string;
  similarity_score: number;
  rank: number;
  scene_id: string;
  scene_name: string;
  sensor: string | null;
  acquisition_date: string | null;
  quality_score: number | null;
  cloud_cover_pct: number | null;
  tile_col: number;
  tile_row: number;
  center_coordinates: { lon: number; lat: number };
  bbox: { west: number; south: number; east: number; north: number };
  footprint_geojson?: {
    type: string;
    coordinates: number[][][];
  } | null;
  thumbnail_url: string;
  preview_url: string;
  landcover?: {
    water_pct?: number;
    veg_pct?: number;
    urban_pct?: number;
  } | null;
  provenance: {
    operation?: string;
    model_name?: string;
    model_version?: string;
    embedding_dimension?: number;
    recorded_at?: string;
    operator?: string;
    details?: Record<string, any>;
    [key: string]: any;
  } | null;
}

export interface SemanticSearchResponse {
  query: string;
  total_found: number;
  execution_time_ms: number;
  text_embedding_time_ms: number;
  vector_search_time_ms: number;
  spatial_filter_time_ms?: number;
  device: string;
  model_used: string;
  results: SemanticSearchResultItem[];
  filters_applied: Record<string, any>;
}

export interface SearchFiltersResponse {
  sensors: string[];
  sensor_counts?: Record<string, number>;
  min_date: string | null;
  max_date: string | null;
  min_quality?: number | null;
  max_cloud_cover?: number | null;
  total_indexed_tiles: number;
}

export interface VisualSearchRequest {
  top_k?: number;
  sensor?: string | null;
  sensors?: string[] | null;
  min_quality?: number | null;
  max_cloud_cover?: number | null;
  start_date?: string | null;
  end_date?: string | null;
  aoi?: [number, number, number, number] | null;
  aoi_bbox?: [number, number, number, number] | null;
  aoi_polygon?: [number, number][] | null;
  spatial_filter_mode?: "intersects" | "within";
  scene_id?: string | null;
}

export interface VisualSearchResponse {
  query_type: "tile_id" | "uploaded_image";
  reference_tile_id: string | null;
  reference_thumbnail_url: string | null;
  total_found: number;
  execution_time_ms: number;
  visual_embedding_time_ms: number;
  vector_search_time_ms: number;
  spatial_filter_time_ms?: number;
  device: string;
  model_used: string;
  embedding_dimension: number;
  results: SemanticSearchResultItem[];
  filters_applied: Record<string, any>;
}

// ── API Client ────────────────────────────────────────────────

class GeoSemanticAPI {
  private client: AxiosInstance;

  constructor() {
    const isBrowser = typeof window !== "undefined";
    const isLocalhost =
      isBrowser &&
      (window.location.hostname === "localhost" ||
        window.location.hostname === "127.0.0.1");

    const baseURL =
      process.env.NEXT_PUBLIC_API_URL ||
      (isLocalhost ? "http://localhost:8000/api" : "/api");

    this.client = axios.create({
      baseURL,
      timeout: 300_000,
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
    });
  }

  // ── Health ────────────────────────────────────────────────

  async getHealth(): Promise<{ status: string; timestamp: string }> {
    const res = await this.client.get("/health");
    return res.data;
  }

  async getDetailedHealth(): Promise<HealthStatus> {
    const res = await this.client.get("/health/detailed");
    return res.data;
  }

  // ── Ingestion & Scenes ─────────────────────────────────────

  async ingestScene(file: File): Promise<IngestionRequest> {
    const form = new FormData();
    form.append("file", file);
    const res = await this.client.post("/ingest/scene", form, {
      timeout: 300_000, // 5 min for large files
    });
    return res.data;
  }

  async getIngestionStatus(jobId: string): Promise<IngestionStatus> {
    const res = await this.client.get(`/ingest/status/${jobId}`);
    return res.data;
  }

  async listScenes(page = 1, pageSize = 20, status?: string): Promise<SceneListResponse> {
    const params: Record<string, string | number> = { page, page_size: pageSize };
    if (status) params.status = status;
    const res = await this.client.get("/ingest/scenes", { params });
    return res.data;
  }

  async getScene(sceneId: string): Promise<SceneSummary> {
    const res = await this.client.get(`/ingest/scenes/${sceneId}`);
    return res.data;
  }

  async deleteScene(sceneId: string): Promise<void> {
    await this.client.delete(`/ingest/scenes/${sceneId}`);
  }

  async processScene(
    sceneId: string,
    req?: ProcessSceneRequest,
    sync = false
  ): Promise<ProcessSceneResponse> {
    const res = await this.client.post(`/scenes/${sceneId}/process`, req ?? {}, {
      params: { sync },
      timeout: 120_000,
    });
    return res.data;
  }

  // ── Tiles ─────────────────────────────────────────────────

  async listTiles(params?: {
    scene_id?: string;
    is_valid?: boolean;
    min_quality?: number;
    page?: number;
    page_size?: number;
  }): Promise<TileListResponse> {
    const res = await this.client.get("/tiles", { params });
    return res.data;
  }

  async getTile(tileId: string): Promise<TileDetail> {
    const res = await this.client.get(`/tiles/${tileId}`);
    return res.data;
  }

  async getTilesForScene(sceneId: string, limit = 200): Promise<TileSummary[]> {
    const res = await this.client.get(`/tiles/scene/${sceneId}`, {
      params: { limit },
    });
    return res.data;
  }

  getThumbnailUrl(tileId: string): string {
    const base = this.client.defaults.baseURL || "/api";
    return `${base}/tiles/${tileId}/thumbnail`;
  }

  getPreviewUrl(tileId: string): string {
    const base = this.client.defaults.baseURL || "/api";
    return `${base}/tiles/${tileId}/preview`;
  }

  // ── AI Models ─────────────────────────────────────────────

  async getModelsStatus(): Promise<ModelsStatusResponse> {
    const res = await this.client.get("/models/status");
    return res.data;
  }

  async loadModel(name: string): Promise<ModelStatusItem> {
    const res = await this.client.post(`/models/${name}/load`);
    return res.data;
  }

  async unloadModel(name: string): Promise<ModelStatusItem> {
    const res = await this.client.post(`/models/${name}/unload`);
    return res.data;
  }

  async embedText(text: string, model_name = "RemoteCLIP"): Promise<TextEmbeddingResponse> {
    const res = await this.client.post("/models/embed/text", { text, model_name });
    return res.data;
  }

  // ── Vector Store & Embedding Pipeline ────────────────────

  async generateEmbeddings(
    sceneId: string,
    options?: { force_reembed?: boolean; sync?: boolean; batch_size?: number }
  ): Promise<EmbeddingJobResponse> {
    const res = await this.client.post(`/embedding/generate/${sceneId}`, null, {
      params: {
        force_reembed: options?.force_reembed ?? false,
        sync: options?.sync ?? false,
        batch_size: options?.batch_size ?? 8,
      },
      timeout: 120_000,
    });
    return res.data;
  }

  async getEmbeddingStatus(sceneId: string): Promise<EmbeddingStatusResponse> {
    const res = await this.client.get(`/embedding/status/${sceneId}`);
    return res.data;
  }

  async getVectorStatistics(): Promise<VectorStatisticsResponse> {
    const res = await this.client.get("/vector/statistics");
    return res.data;
  }

  // ── Semantic & Visual Search ──────────────────────────────

  async semanticSearch(req: SemanticSearchRequest): Promise<SemanticSearchResponse> {
    const res = await this.client.post("/search/semantic", req, {
      timeout: 60_000,
    });
    return res.data;
  }

  async getSearchFilters(): Promise<SearchFiltersResponse> {
    const res = await this.client.get("/search/filters");
    return res.data;
  }

  async searchSimilarByTile(
    tileId: string,
    req?: VisualSearchRequest
  ): Promise<VisualSearchResponse> {
    const res = await this.client.post(`/search/similar/${tileId}`, req ?? {}, {
      timeout: 60_000,
    });
    return res.data;
  }

  async searchByUploadedImage(
    file: File,
    options?: VisualSearchRequest
  ): Promise<VisualSearchResponse> {
    const form = new FormData();
    form.append("file", file);

    const params: Record<string, any> = {};
    if (options?.top_k) params.top_k = options.top_k;
    if (options?.sensor) params.sensor = options.sensor;
    if (options?.sensors && options.sensors.length > 0) {
      params.sensors = options.sensors;
    }
    if (options?.min_quality != null) params.min_quality = options.min_quality;
    if (options?.max_cloud_cover != null) params.max_cloud_cover = options.max_cloud_cover;
    if (options?.spatial_filter_mode) params.spatial_filter_mode = options.spatial_filter_mode;
    if (options?.start_date) params.start_date = options.start_date;
    if (options?.end_date) params.end_date = options.end_date;
    const bbox = options?.aoi_bbox || options?.aoi;
    if (bbox && bbox.length === 4) {
      params.west = bbox[0];
      params.south = bbox[1];
      params.east = bbox[2];
      params.north = bbox[3];
    }

    const res = await this.client.post("/search/image", form, {
      params,
      timeout: 60_000,
    });
    return res.data;
  }

  // ── Quality Assessment & Intelligence ─────────────────────

  async getQualityForTile(tileId: string): Promise<TileQuality> {
    const res = await this.client.get(`/quality/tile/${tileId}`);
    return res.data;
  }

  async computeQualityForTile(tileId: string): Promise<TileQuality> {
    const res = await this.client.post(`/quality/tile/${tileId}/compute`);
    return res.data;
  }

  async getQualityForScene(sceneId: string): Promise<SceneQuality> {
    const res = await this.client.get(`/quality/scene/${sceneId}`);
    return res.data;
  }

  async computeQualityForScene(sceneId: string): Promise<SceneComputeQualityResponse> {
    const res = await this.client.post(`/quality/scene/${sceneId}/compute`);
    return res.data;
  }

  // ── Change Detection ──────────────────────────────────────

  async analyzeChange(req: ChangeAnalyzeRequest): Promise<ChangeAnalyzeResponse> {
    const res = await this.client.post("/change/analyze", req, {
      timeout: 120_000,
    });
    return res.data;
  }

  async getChangeEvent(changeId: string): Promise<ChangeEventResponse> {
    const res = await this.client.get(`/change/${changeId}`);
    return res.data;
  }

  async listChangeEvents(
    limit = 50,
    skip = 0,
    min_confidence = 0.0
  ): Promise<ChangeEventResponse[]> {
    const res = await this.client.get("/change", {
      params: { limit, skip, min_confidence },
    });
    return res.data;
  }
}

export const api = new GeoSemanticAPI();
