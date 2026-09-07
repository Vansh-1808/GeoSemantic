"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import {
  api,
  type SearchFiltersResponse,
  type SemanticSearchResultItem,
  type TileDetail,
  type TileSummary,
  type VisualSearchRequest,
  type VisualSearchResponse,
} from "@/lib/api";
import { SearchCoverageMap } from "@/components/search/SearchCoverageMap";
import { TileDetailModal } from "@/components/search/TileDetailModal";
import { formatDate, formatCoord, qualityColor } from "@/lib/utils";
import {
  AlertCircle,
  Calendar,
  Camera,
  Check,
  Clock,
  Compass,
  Database,
  Eye,
  FileImage,
  Filter,
  Grid,
  Image as ImageIcon,
  Layers,
  MapPin,
  RefreshCw,
  RotateCcw,
  Search,
  Sliders,
  Sparkles,
  UploadCloud,
  Zap,
} from "lucide-react";

export default function ImageSearchPage() {
  return (
    <Suspense fallback={<div className="p-8 text-gray-500 text-xs">Loading Visual Search Console...</div>}>
      <ImageSearchContent />
    </Suspense>
  );
}

function ImageSearchContent() {
  const searchParams = useSearchParams();
  const initialTileId = searchParams.get("tile_id");

  // Input Mode: "upload" vs "archive"
  const [inputMode, setInputMode] = useState<"upload" | "archive">("upload");

  // Upload state
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadedPreviewUrl, setUploadedPreviewUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Selected archive tile state
  const [selectedArchiveTileId, setSelectedArchiveTileId] = useState<string | null>(initialTileId);
  const [selectedArchiveTile, setSelectedArchiveTile] = useState<TileDetail | null>(null);

  // Search parameters & filters
  const [topK, setTopK] = useState(20);
  const [showFilters, setShowFilters] = useState(false);
  const [selectedSensor, setSelectedSensor] = useState<string>("");
  const [startDate, setStartDate] = useState<string>("");
  const [endDate, setEndDate] = useState<string>("");
  const [minQuality, setMinQuality] = useState<number>(0);
  const [aoiWest, setAoiWest] = useState<string>("");
  const [aoiSouth, setAoiSouth] = useState<string>("");
  const [aoiEast, setAoiEast] = useState<string>("");
  const [aoiNorth, setAoiNorth] = useState<string>("");

  // Results & Interaction state
  const [searchResponse, setSearchResponse] = useState<VisualSearchResponse | null>(null);
  const [selectedResultTileId, setSelectedResultTileId] = useState<string | null>(null);
  const [inspectingTile, setInspectingTile] = useState<SemanticSearchResultItem | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);

  // Fetch filter boundaries
  const { data: filtersData } = useQuery<SearchFiltersResponse>({
    queryKey: ["searchFilters"],
    queryFn: () => api.getSearchFilters(),
  });

  // Fetch candidate tiles for the archive picker
  const { data: candidateTiles, isLoading: isLoadingTiles } = useQuery<{ tiles: TileSummary[] }>({
    queryKey: ["candidateTiles"],
    queryFn: () => api.listTiles({ page_size: 24 }),
  });

  // Load initial tile if provided via URL
  useEffect(() => {
    if (initialTileId) {
      setInputMode("archive");
      setSelectedArchiveTileId(initialTileId);
      api.getTile(initialTileId).then((tile) => {
        setSelectedArchiveTile(tile);
        executeTileSearch(initialTileId);
      }).catch(() => {
        // Silently ignore if not found
      });
    }
  }, [initialTileId]);

  const activeFiltersCount = [
    selectedSensor ? 1 : 0,
    startDate ? 1 : 0,
    endDate ? 1 : 0,
    minQuality > 0 ? 1 : 0,
    aoiWest && aoiSouth && aoiEast && aoiNorth ? 1 : 0,
  ].reduce((a, b) => a + b, 0);

  const resetFilters = () => {
    setSelectedSensor("");
    setStartDate("");
    setEndDate("");
    setMinQuality(0);
    setAoiWest("");
    setAoiSouth("");
    setAoiEast("");
    setAoiNorth("");
  };

  const buildFilterPayload = (): VisualSearchRequest => {
    let aoiCoords: [number, number, number, number] | null = null;
    if (aoiWest && aoiSouth && aoiEast && aoiNorth) {
      const w = parseFloat(aoiWest);
      const s = parseFloat(aoiSouth);
      const e = parseFloat(aoiEast);
      const n = parseFloat(aoiNorth);
      if (!isNaN(w) && !isNaN(s) && !isNaN(e) && !isNaN(n)) {
        aoiCoords = [w, s, e, n];
      }
    }

    return {
      top_k: topK,
      sensor: selectedSensor || null,
      start_date: startDate ? new Date(startDate).toISOString() : null,
      end_date: endDate ? new Date(endDate).toISOString() : null,
      min_quality: minQuality > 0 ? minQuality / 100 : null,
      aoi: aoiCoords,
    };
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setUploadedFile(file);
      const url = URL.createObjectURL(file);
      setUploadedPreviewUrl(url);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) {
      setUploadedFile(file);
      const url = URL.createObjectURL(file);
      setUploadedPreviewUrl(url);
    }
  };

  const executeTileSearch = async (tileId: string) => {
    setIsSearching(true);
    setSearchError(null);
    try {
      const filters = buildFilterPayload();
      const resp = await api.searchSimilarByTile(tileId, filters);
      setSearchResponse(resp);
      if (resp.results.length > 0) {
        setSelectedResultTileId(resp.results[0].tile_id);
      }
    } catch (err: any) {
      setSearchError(
        err.response?.data?.detail || err.message || "Visual similarity search failed."
      );
    } finally {
      setIsSearching(false);
    }
  };

  const executeImageSearch = async () => {
    if (!uploadedFile) return;
    setIsSearching(true);
    setSearchError(null);
    try {
      const filters = buildFilterPayload();
      const resp = await api.searchByUploadedImage(uploadedFile, filters);
      setSearchResponse(resp);
      if (resp.results.length > 0) {
        setSelectedResultTileId(resp.results[0].tile_id);
      }
    } catch (err: any) {
      setSearchError(
        err.response?.data?.detail || err.message || "Uploaded image visual search failed."
      );
    } finally {
      setIsSearching(false);
    }
  };

  const handleExecuteSearch = () => {
    if (inputMode === "upload") {
      executeImageSearch();
    } else if (selectedArchiveTileId) {
      executeTileSearch(selectedArchiveTileId);
    }
  };

  const selectedResultTile =
    searchResponse?.results.find((r) => r.tile_id === selectedResultTileId) || null;

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-4 md:p-8 space-y-6">
      {/* Top Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-gray-800 pb-5">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-white tracking-tight">
              Visual & Image-to-Image Search
            </h1>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-950 text-indigo-300 border border-indigo-800/80 flex items-center gap-1.5">
              <Sparkles className="h-3 w-3" />
              DINOv2 384-dim Visual Embeddings
            </span>
          </div>
          <p className="text-sm text-gray-400 mt-1">
            Discover visually similar satellite terrain, infrastructure, and facilities using deep vision representations.
          </p>
        </div>

        {/* Telemetry pill */}
        {filtersData && (
          <div className="flex items-center gap-3 bg-gray-900/80 border border-gray-800 rounded-xl px-4 py-2 text-xs font-mono">
            <div>
              <span className="text-gray-500">Indexed Tiles:</span>{" "}
              <span className="text-indigo-400 font-semibold">{filtersData.total_indexed_tiles}</span>
            </div>
            <span className="text-gray-700">•</span>
            <div>
              <span className="text-gray-500">Collection:</span>{" "}
              <span className="text-emerald-400 font-semibold">dino_tiles</span>
            </div>
          </div>
        )}
      </div>

      {/* Input Selection Mode Tabs */}
      <div className="bg-gray-900/60 border border-gray-800 rounded-2xl p-5 space-y-4 shadow-xl">
        <div className="flex items-center justify-between border-b border-gray-800 pb-3">
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setInputMode("upload")}
              className={`px-4 py-2 rounded-xl text-xs font-semibold flex items-center gap-2 transition ${
                inputMode === "upload"
                  ? "bg-indigo-600 text-white shadow-lg shadow-indigo-950/60"
                  : "bg-gray-950 text-gray-400 hover:text-white border border-gray-800"
              }`}
            >
              <UploadCloud className="h-4 w-4" />
              <span>Upload Local Satellite Image</span>
            </button>

            <button
              type="button"
              onClick={() => setInputMode("archive")}
              className={`px-4 py-2 rounded-xl text-xs font-semibold flex items-center gap-2 transition ${
                inputMode === "archive"
                  ? "bg-indigo-600 text-white shadow-lg shadow-indigo-950/60"
                  : "bg-gray-950 text-gray-400 hover:text-white border border-gray-800"
              }`}
            >
              <Grid className="h-4 w-4" />
              <span>Select Existing Archive Tile</span>
            </button>
          </div>

          <div className="flex items-center gap-2">
            <select
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="bg-gray-950 border border-gray-800 rounded-xl px-3 py-2 text-xs text-gray-300 focus:outline-none"
            >
              <option value={10}>Top 10</option>
              <option value={20}>Top 20</option>
              <option value={50}>Top 50</option>
            </select>

            <button
              type="button"
              onClick={() => setShowFilters(!showFilters)}
              className={`px-3 py-2 rounded-xl text-xs font-semibold flex items-center gap-1.5 border transition ${
                activeFiltersCount > 0
                  ? "bg-indigo-950 text-indigo-300 border-indigo-700"
                  : "bg-gray-950 text-gray-400 border-gray-800 hover:bg-gray-800"
              }`}
            >
              <Filter className="h-3.5 w-3.5" />
              <span>Filters</span>
              {activeFiltersCount > 0 && (
                <span className="h-4 w-4 rounded-full bg-indigo-500 text-gray-950 font-bold text-[9px] flex items-center justify-center">
                  {activeFiltersCount}
                </span>
              )}
            </button>
          </div>
        </div>

        {/* Upload Mode Dropzone */}
        {inputMode === "upload" && (
          <div className="space-y-4">
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-gray-700 hover:border-indigo-500/80 rounded-2xl p-6 transition-all duration-200 cursor-pointer bg-gray-950/60 flex flex-col items-center justify-center text-center group"
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*,.tif,.tiff,.geotiff,.png,.jpg,.jpeg"
                onChange={handleFileChange}
                className="hidden"
              />

              {uploadedPreviewUrl ? (
                <div className="flex flex-col sm:flex-row items-center gap-6">
                  <div className="relative w-32 h-32 rounded-xl overflow-hidden border border-indigo-500/50 shadow-lg group-hover:scale-105 transition duration-200 bg-gray-900 flex items-center justify-center">
                    {uploadedFile?.name.toLowerCase().endsWith(".tif") || uploadedFile?.name.toLowerCase().endsWith(".tiff") ? (
                      <div className="flex flex-col items-center justify-center p-2 text-center text-indigo-300">
                        <ImageIcon className="h-10 w-10 text-indigo-400 mb-1" />
                        <span className="text-[10px] font-mono font-semibold uppercase tracking-wider">GeoTIFF Raster</span>
                      </div>
                    ) : (
                      <img
                        src={uploadedPreviewUrl}
                        alt="Uploaded query preview"
                        className="w-full h-full object-cover"
                        onError={(e) => {
                          e.currentTarget.style.display = "none";
                        }}
                      />
                    )}
                    <div className="absolute top-1 right-1 bg-indigo-950/80 px-1.5 py-0.5 rounded text-[9px] font-mono text-indigo-300">
                      Query
                    </div>
                  </div>
                  <div className="text-left space-y-1">
                    <p className="text-sm font-semibold text-white">{uploadedFile?.name}</p>
                    <p className="text-xs text-gray-400 font-mono">
                      Size: {uploadedFile ? `${(uploadedFile.size / 1024).toFixed(1)} KB` : "—"}
                    </p>
                    <p className="text-xs text-indigo-400">
                      Click or drop another image to replace.
                    </p>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center">
                  <div className="p-3 bg-indigo-950/60 rounded-2xl border border-indigo-800/60 text-indigo-400 mb-2 group-hover:scale-110 transition">
                    <UploadCloud className="h-6 w-6" />
                  </div>
                  <p className="text-sm font-medium text-white">
                    Drop satellite image here, or <span className="text-indigo-400 underline">browse</span>
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    Supports PNG, JPEG, and GeoTIFF satellite tiles.
                  </p>
                </div>
              )}
            </div>

            <div className="flex justify-end">
              <button
                type="button"
                onClick={handleExecuteSearch}
                disabled={isSearching || !uploadedFile}
                className="px-6 py-3 bg-gradient-to-r from-indigo-600 to-blue-600 hover:from-indigo-500 hover:to-blue-500 text-white rounded-xl text-xs font-semibold shadow-lg shadow-indigo-950/50 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition"
              >
                {isSearching ? (
                  <>
                    <div className="h-4 w-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    <span>Extracting DINOv2 Features...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4" />
                    <span>Find Similar Locations</span>
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* Archive Tile Selector Mode */}
        {inputMode === "archive" && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <p className="text-xs text-gray-400">
                Select any existing tile from the archive to instantly search visual neighbors:
              </p>
              {selectedArchiveTileId && (
                <span className="text-xs font-mono text-indigo-400 bg-indigo-950 px-2 py-0.5 rounded border border-indigo-800">
                  Selected: {selectedArchiveTileId.slice(0, 8)}...
                </span>
              )}
            </div>

            {isLoadingTiles ? (
              <div className="py-12 text-center text-xs text-gray-500">
                Loading archive tiles...
              </div>
            ) : (
              <div className="grid grid-cols-3 sm:grid-cols-6 md:grid-cols-8 gap-2.5 max-h-48 overflow-y-auto p-1 bg-gray-950 rounded-xl border border-gray-800">
                {candidateTiles?.tiles.map((tile) => {
                  const isSelected = selectedArchiveTileId === tile.id;
                  return (
                    <div
                      key={tile.id}
                      onClick={() => {
                        setSelectedArchiveTileId(tile.id);
                        api.getTile(tile.id).then(setSelectedArchiveTile);
                        executeTileSearch(tile.id);
                      }}
                      className={`relative aspect-square rounded-lg overflow-hidden cursor-pointer border transition-all ${
                        isSelected
                          ? "border-indigo-400 ring-2 ring-indigo-400 shadow-md shadow-indigo-500/40 scale-95"
                          : "border-gray-800 hover:border-gray-600 opacity-70 hover:opacity-100"
                      }`}
                    >
                      <img
                        src={tile.thumbnail_url || api.getThumbnailUrl(tile.id)}
                        alt={`Tile ${tile.tile_col},${tile.tile_row}`}
                        className="w-full h-full object-cover"
                        onError={(e) => {
                          e.currentTarget.onerror = null;
                          e.currentTarget.src = api.getThumbnailUrl(tile.id);
                        }}
                      />
                      <div className="absolute bottom-0 inset-x-0 bg-black/70 text-[8px] font-mono text-gray-300 text-center py-0.5">
                        {tile.tile_col},{tile.tile_row}
                      </div>
                      {isSelected && (
                        <div className="absolute top-1 right-1 bg-indigo-600 rounded-full p-0.5">
                          <Check className="h-2.5 w-2.5 text-white" />
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}

            <div className="flex items-center justify-between pt-2">
              <span className="text-xs text-gray-500">
                Reference tile will be queried against <code>dino_tiles</code> and excluded from self-results.
              </span>
              <button
                type="button"
                onClick={handleExecuteSearch}
                disabled={isSearching || !selectedArchiveTileId}
                className="px-6 py-3 bg-gradient-to-r from-indigo-600 to-blue-600 hover:from-indigo-500 hover:to-blue-500 text-white rounded-xl text-xs font-semibold shadow-lg shadow-indigo-950/50 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition"
              >
                {isSearching ? (
                  <>
                    <div className="h-4 w-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    <span>Searching Qdrant...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4" />
                    <span>Search Similar Sites</span>
                  </>
                )}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Expandable Multi-Factor Filter Panel */}
      {showFilters && (
        <div className="bg-gray-900/70 border border-gray-800 rounded-2xl p-5 space-y-4 animate-in slide-in-from-top duration-200">
          <div className="flex items-center justify-between border-b border-gray-800 pb-3">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-300 flex items-center gap-2">
              <Sliders className="h-4 w-4 text-indigo-400" />
              Visual Retrieval Constraints & Geographic Filters
            </h3>
            {activeFiltersCount > 0 && (
              <button
                type="button"
                onClick={resetFilters}
                className="text-xs text-gray-400 hover:text-indigo-400 flex items-center gap-1"
              >
                <RotateCcw className="h-3 w-3" /> Reset Filters
              </button>
            )}
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
            <div>
              <label className="block text-gray-400 font-medium mb-1.5">Satellite Sensor</label>
              <select
                value={selectedSensor}
                onChange={(e) => setSelectedSensor(e.target.value)}
                className="w-full bg-gray-950 border border-gray-800 rounded-xl px-3 py-2 text-gray-200 focus:outline-none"
              >
                <option value="">All Sensors</option>
                {filtersData?.sensors.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-gray-400 font-medium mb-1.5">Acquisition Start Date</label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="w-full bg-gray-950 border border-gray-800 rounded-xl px-3 py-2 text-gray-200 focus:outline-none"
              />
            </div>

            <div>
              <label className="block text-gray-400 font-medium mb-1.5">Acquisition End Date</label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="w-full bg-gray-950 border border-gray-800 rounded-xl px-3 py-2 text-gray-200 focus:outline-none"
              />
            </div>

            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-gray-400 font-medium">Min Quality Threshold</label>
                <span className="font-mono text-emerald-400 font-semibold">{minQuality}%</span>
              </div>
              <input
                type="range"
                min={0}
                max={100}
                step={5}
                value={minQuality}
                onChange={(e) => setMinQuality(Number(e.target.value))}
                className="w-full h-2 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-emerald-400"
              />
            </div>
          </div>

          <div className="border-t border-gray-800/80 pt-3">
            <label className="block text-gray-400 font-medium mb-2 flex items-center gap-1.5">
              <Compass className="h-3.5 w-3.5 text-indigo-400" />
              Area of Interest (AOI) Bounding Box [EPSG:4326]
            </label>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
              <input
                type="number"
                step="any"
                placeholder="West Lon"
                value={aoiWest}
                onChange={(e) => setAoiWest(e.target.value)}
                className="w-full bg-gray-950 border border-gray-800 rounded-lg px-2.5 py-1.5 font-mono text-gray-200"
              />
              <input
                type="number"
                step="any"
                placeholder="South Lat"
                value={aoiSouth}
                onChange={(e) => setAoiSouth(e.target.value)}
                className="w-full bg-gray-950 border border-gray-800 rounded-lg px-2.5 py-1.5 font-mono text-gray-200"
              />
              <input
                type="number"
                step="any"
                placeholder="East Lon"
                value={aoiEast}
                onChange={(e) => setAoiEast(e.target.value)}
                className="w-full bg-gray-950 border border-gray-800 rounded-lg px-2.5 py-1.5 font-mono text-gray-200"
              />
              <input
                type="number"
                step="any"
                placeholder="North Lat"
                value={aoiNorth}
                onChange={(e) => setAoiNorth(e.target.value)}
                className="w-full bg-gray-950 border border-gray-800 rounded-lg px-2.5 py-1.5 font-mono text-gray-200"
              />
            </div>
          </div>
        </div>
      )}

      {/* Error Message */}
      {searchError && (
        <div className="bg-red-950/60 border border-red-800 rounded-xl p-4 flex items-center gap-3 text-red-300 text-xs">
          <AlertCircle className="h-5 w-5 text-red-400 shrink-0" />
          <span>{searchError}</span>
        </div>
      )}

      {/* Telemetry Bar */}
      {searchResponse && (
        <div className="bg-gray-900/60 border border-gray-800 rounded-xl p-3.5 flex flex-wrap items-center justify-between gap-3 text-xs">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-white">Visual Matches:</span>
            <span className="font-mono text-indigo-300 font-bold">
              {searchResponse.total_found} tiles
            </span>
            <span className="text-gray-600">via</span>
            <span className="font-medium text-gray-300">
              {searchResponse.query_type === "uploaded_image" ? "Uploaded Image" : `Tile ID ${searchResponse.reference_tile_id?.slice(0, 8)}...`}
            </span>
          </div>

          <div className="flex items-center gap-4 text-gray-400 font-mono text-[11px]">
            <div className="flex items-center gap-1">
              <Clock className="h-3.5 w-3.5 text-indigo-400" />
              <span>Total: {searchResponse.execution_time_ms}ms</span>
            </div>
            <span className="text-gray-700">•</span>
            <div>DINOv2 Embed: {searchResponse.visual_embedding_time_ms}ms</div>
            <span className="text-gray-700">•</span>
            <div>Qdrant Search: {searchResponse.vector_search_time_ms}ms</div>
            <span className="text-gray-700">•</span>
            <div className="text-indigo-400 font-semibold">{searchResponse.embedding_dimension}-dim</div>
          </div>
        </div>
      )}

      {/* Two-Column Analyst Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Ranked Similar Results */}
        <div className="lg:col-span-7 space-y-4">
          {!searchResponse && !isSearching && (
            <div className="bg-gray-900/40 border border-gray-800/80 rounded-2xl p-10 text-center flex flex-col items-center justify-center space-y-3">
              <div className="p-3 bg-indigo-950/60 rounded-2xl border border-indigo-800/80 text-indigo-400">
                <ImageIcon className="h-8 w-8" />
              </div>
              <h3 className="text-base font-semibold text-white">Visual Similarity Engine Ready</h3>
              <p className="text-xs text-gray-400 max-w-md">
                Upload a local satellite image or select an existing archive tile above.
                Features are extracted using offline DINOv2 (384-dim) to retrieve visually analogous locations.
              </p>
            </div>
          )}

          {isSearching && (
            <div className="space-y-3">
              {[1, 2, 3, 4].map((i) => (
                <div
                  key={i}
                  className="bg-gray-900/40 border border-gray-800 rounded-xl p-4 flex gap-4 animate-pulse"
                >
                  <div className="w-24 h-24 bg-gray-800 rounded-lg shrink-0" />
                  <div className="flex-1 space-y-2 py-1">
                    <div className="h-4 bg-gray-800 rounded w-1/3" />
                    <div className="h-3 bg-gray-800 rounded w-1/2" />
                    <div className="h-3 bg-gray-800 rounded w-2/3" />
                  </div>
                </div>
              ))}
            </div>
          )}

          {searchResponse && searchResponse.results.length === 0 && !isSearching && (
            <div className="bg-gray-900/40 border border-gray-800 rounded-2xl p-8 text-center text-gray-400 text-xs">
              <p className="font-semibold text-gray-300">No visually similar tiles found.</p>
              <p className="mt-1 text-gray-500">
                Try relaxing the quality threshold or expanding the temporal/spatial filters.
              </p>
            </div>
          )}

          {searchResponse && searchResponse.results.length > 0 && (
            <div className="space-y-3">
              {searchResponse.results.map((item) => {
                const isSelected = selectedResultTileId === item.tile_id;
                const scorePct = Math.min(100, Math.max(0, item.similarity_score * 100)).toFixed(1);

                return (
                  <div
                    key={item.tile_id}
                    onClick={() => setSelectedResultTileId(item.tile_id)}
                    className={`bg-gray-900/50 hover:bg-gray-900 border rounded-2xl p-4 transition-all duration-150 cursor-pointer flex flex-col sm:flex-row gap-4 ${
                      isSelected
                        ? "border-indigo-400 ring-1 ring-indigo-400/80 shadow-lg shadow-indigo-950/40 bg-gray-900/80"
                        : "border-gray-800 hover:border-gray-700"
                    }`}
                  >
                    {/* Thumbnail Image */}
                    <div className="relative w-24 h-24 sm:w-28 sm:h-28 bg-gray-950 rounded-xl overflow-hidden shrink-0 border border-gray-800 flex items-center justify-center group">
                      <img
                        src={item.thumbnail_url}
                        alt={`Tile ${item.tile_col},${item.tile_row}`}
                        className="w-full h-full object-cover transition duration-300 group-hover:scale-105"
                        onError={(e) => {
                          e.currentTarget.onerror = null;
                          e.currentTarget.src = api.getThumbnailUrl(item.tile_id);
                        }}
                      />
                      <div className="absolute top-1 left-1 bg-gray-950/80 px-1.5 py-0.5 rounded text-[9px] font-mono text-gray-300 border border-gray-800">
                        #{item.rank}
                      </div>
                    </div>

                    {/* Metadata & Details */}
                    <div className="flex-1 flex flex-col justify-between space-y-2">
                      <div>
                        <div className="flex items-center justify-between gap-2">
                          <div className="flex items-center gap-2">
                            <span className="font-semibold text-white text-sm">
                              Tile [{item.tile_col}, {item.tile_row}]
                            </span>
                            <span className="text-xs text-gray-400 font-mono truncate max-w-[160px]">
                              {item.scene_name}
                            </span>
                          </div>

                          {/* Visual Match Badge */}
                          <div className="flex items-center gap-1 px-2.5 py-1 rounded-full bg-indigo-950/90 border border-indigo-700/80 text-indigo-300 font-mono font-bold text-xs">
                            <Zap className="h-3 w-3 text-indigo-400" />
                            <span>{scorePct}% Match</span>
                          </div>
                        </div>

                        {/* Metadata Tag Row */}
                        <div className="flex flex-wrap items-center gap-2 mt-1.5 text-[11px] font-mono text-gray-400">
                          <span className="px-1.5 py-0.5 rounded bg-gray-800 text-indigo-300">
                            {item.sensor || "Sentinel-2"}
                          </span>
                          <span>•</span>
                          <span>{formatDate(item.acquisition_date)}</span>
                          <span>•</span>
                          <span className={qualityColor(item.quality_score)}>
                            Q: {item.quality_score != null ? `${(item.quality_score * 100).toFixed(0)}%` : "—"}
                          </span>
                          {/* Landcover Badges */}
                          {item.landcover?.water_pct != null && item.landcover.water_pct >= 1.0 && (
                            <>
                              <span>•</span>
                              <span className="px-1.5 py-0.5 rounded bg-cyan-950/90 text-cyan-300 border border-cyan-800/70 font-semibold flex items-center gap-0.5">
                                💧 {item.landcover.water_pct.toFixed(1)}% Water
                              </span>
                            </>
                          )}
                          {item.landcover?.veg_pct != null && item.landcover.veg_pct >= 2.0 && (
                            <>
                              <span>•</span>
                              <span className="px-1.5 py-0.5 rounded bg-emerald-950/90 text-emerald-300 border border-emerald-800/70 font-semibold flex items-center gap-0.5">
                                🌿 {item.landcover.veg_pct.toFixed(1)}% Veg
                              </span>
                            </>
                          )}
                        </div>
                      </div>

                      {/* Coordinates & Actions */}
                      <div className="flex items-center justify-between pt-1 border-t border-gray-800/80 text-xs">
                        <div className="flex items-center gap-1 font-mono text-[11px] text-gray-400">
                          <MapPin className="h-3 w-3 text-indigo-400" />
                          <span>
                            [{formatCoord(item.center_coordinates.lat, 4)}, {formatCoord(item.center_coordinates.lon, 4)}]
                          </span>
                        </div>

                        <div className="flex items-center gap-2">
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              setInspectingTile(item);
                            }}
                            className="px-2.5 py-1 bg-gray-800 hover:bg-indigo-900/60 hover:text-indigo-300 text-gray-300 rounded-lg text-xs font-medium flex items-center gap-1 transition border border-gray-700 hover:border-indigo-700"
                          >
                            <Eye className="h-3.5 w-3.5" />
                            <span>Inspect & Audit</span>
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Right Column: Spatial Heatmap & Query Reference */}
        <div className="lg:col-span-5 space-y-4">
          <div className="sticky top-6 space-y-4">
            {/* Interactive Coverage Heatmap */}
            <SearchCoverageMap
              results={searchResponse?.results || []}
              selectedTileId={selectedResultTileId}
              onSelectTile={(tile) => setSelectedResultTileId(tile.tile_id)}
              onInspectTile={(tile) => setInspectingTile(tile)}
            />

            {/* Reference Query Preview Card */}
            <div className="bg-gray-900/60 border border-gray-800 rounded-2xl p-4 space-y-3 shadow-xl">
              <div className="flex items-center justify-between border-b border-gray-800 pb-2">
                <span className="font-semibold text-white text-xs flex items-center gap-1.5">
                  <Camera className="h-3.5 w-3.5 text-indigo-400" />
                  Query Reference Template
                </span>
                <span className="text-[10px] font-mono text-indigo-300 bg-indigo-950/80 px-2 py-0.5 rounded border border-indigo-800/80">
                  {searchResponse?.model_used || "DINOv2 (384-dim)"}
                </span>
              </div>

              {inputMode === "upload" && uploadedPreviewUrl ? (
                <div className="flex items-center gap-3">
                  <img
                    src={uploadedPreviewUrl}
                    alt="Query reference"
                    className="w-16 h-16 rounded-xl object-cover border border-gray-800 bg-gray-950 shrink-0"
                  />
                  <div className="text-xs space-y-0.5 overflow-hidden">
                    <p className="font-medium text-gray-200 truncate">{uploadedFile?.name}</p>
                    <p className="text-gray-400 font-mono text-[11px]">Mode: Uploaded Image</p>
                    <p className="text-emerald-400 font-mono text-[11px]">Vector Extracted</p>
                  </div>
                </div>
              ) : selectedArchiveTile ? (
                <div className="flex items-center gap-3">
                  <img
                    src={selectedArchiveTile.thumbnail_url || api.getThumbnailUrl(selectedArchiveTile.id)}
                    alt="Query reference tile"
                    className="w-16 h-16 rounded-xl object-cover border border-gray-800 bg-gray-950 shrink-0"
                  />
                  <div className="text-xs space-y-0.5 overflow-hidden">
                    <p className="font-medium text-gray-200">
                      Tile [{selectedArchiveTile.tile_col}, {selectedArchiveTile.tile_row}]
                    </p>
                    <p className="text-gray-400 font-mono text-[11px]">
                      Sensor: {selectedArchiveTile.sensor || "Sentinel-2B"}
                    </p>
                    <p className="text-gray-400 font-mono text-[11px]">
                      Center: [{formatCoord(selectedArchiveTile.center_lat, 4)}, {formatCoord(selectedArchiveTile.center_lon, 4)}]
                    </p>
                  </div>
                </div>
              ) : (
                <div className="text-xs text-gray-500 text-center py-2">
                  No reference image selected yet.
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Modal Inspector */}
      <TileDetailModal
        tile={inspectingTile}
        onClose={() => setInspectingTile(null)}
      />
    </div>
  );
}
