"use client";

import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  api,
  type SearchFiltersResponse,
  type SemanticSearchRequest,
  type SemanticSearchResponse,
  type SemanticSearchResultItem,
} from "@/lib/api";
import { SearchCoverageMap } from "@/components/search/SearchCoverageMap";
import { TileDetailModal } from "@/components/search/TileDetailModal";
import { formatDate, formatCoord, qualityColor } from "@/lib/utils";
import {
  AlertCircle,
  Calendar,
  Clock,
  Cloud,
  Compass,
  Cpu,
  Database,
  Eye,
  Filter,
  Layers,
  MapPin,
  Maximize2,
  RotateCcw,
  Search,
  Sliders,
  Sparkles,
  Zap,
  X,
  ShieldCheck,
} from "lucide-react";

const QUERY_SUGGESTIONS = [
  "Commercial airport runway & aprons",
  "Dense residential buildings and road grid",
  "Harbor docks with cargo container vessels",
  "Agricultural farmland and irrigation channels",
  "Industrial chemical plant with storage tanks",
  "River channel with highway bridge crossing",
];

export default function SearchPage() {
  // Search input state
  const [queryText, setQueryText] = useState("");
  const [topK, setTopK] = useState(20);

  // Filters state
  const [showFilters, setShowFilters] = useState(false);
  const [selectedSensors, setSelectedSensors] = useState<string[]>([]);
  const [startDate, setStartDate] = useState<string>("");
  const [endDate, setEndDate] = useState<string>("");
  const [minQuality, setMinQuality] = useState<number>(0);
  const [maxCloudCover, setMaxCloudCover] = useState<number>(100);

  // Spatial AOI state (Polygon vs Bounding Box)
  const [aoiPolygon, setAoiPolygon] = useState<[number, number][] | null>(null);
  const [aoiBbox, setAoiBbox] = useState<[number, number, number, number] | null>(null);
  const [spatialFilterMode, setSpatialFilterMode] = useState<"intersects" | "within">("intersects");

  // BBox manual input coordinates
  const [manualWest, setManualWest] = useState<string>("");
  const [manualSouth, setManualSouth] = useState<string>("");
  const [manualEast, setManualEast] = useState<string>("");
  const [manualNorth, setManualNorth] = useState<string>("");

  // Result interaction state
  const [searchResponse, setSearchResponse] = useState<SemanticSearchResponse | null>(null);
  const [selectedTileId, setSelectedTileId] = useState<string | null>(null);
  const [inspectingTile, setInspectingTile] = useState<SemanticSearchResultItem | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);

  // Fetch dynamic metadata filters (available sensors, date boundaries, total index count)
  const { data: filtersData } = useQuery<SearchFiltersResponse>({
    queryKey: ["searchFilters"],
    queryFn: () => api.getSearchFilters(),
  });

  // Calculate active filter count
  const activeFiltersCount = [
    selectedSensors.length > 0 ? 1 : 0,
    startDate || endDate ? 1 : 0,
    minQuality > 0 ? 1 : 0,
    maxCloudCover < 100 ? 1 : 0,
    aoiPolygon && aoiPolygon.length >= 3 ? 1 : 0,
    aoiBbox && aoiBbox.length === 4 ? 1 : 0,
  ].reduce((a, b) => a + b, 0);

  // Sync manual bbox inputs when map updates aoiBbox
  useEffect(() => {
    if (aoiBbox && aoiBbox.length === 4) {
      setManualWest(aoiBbox[0].toFixed(4));
      setManualSouth(aoiBbox[1].toFixed(4));
      setManualEast(aoiBbox[2].toFixed(4));
      setManualNorth(aoiBbox[3].toFixed(4));
    }
  }, [aoiBbox]);

  const toggleSensor = (sensorName: string) => {
    setSelectedSensors((prev) =>
      prev.includes(sensorName)
        ? prev.filter((s) => s !== sensorName)
        : [...prev, sensorName]
    );
  };

  const handleManualBboxChange = (wStr: string, sStr: string, eStr: string, nStr: string) => {
    setManualWest(wStr);
    setManualSouth(sStr);
    setManualEast(eStr);
    setManualNorth(nStr);

    const w = parseFloat(wStr);
    const s = parseFloat(sStr);
    const e = parseFloat(eStr);
    const n = parseFloat(nStr);

    if (!isNaN(w) && !isNaN(s) && !isNaN(e) && !isNaN(n) && e > w && n > s) {
      setAoiBbox([w, s, e, n]);
      setAoiPolygon(null);
    }
  };

  const resetFilters = () => {
    setSelectedSensors([]);
    setStartDate("");
    setEndDate("");
    setMinQuality(0);
    setMaxCloudCover(100);
    setAoiPolygon(null);
    setAoiBbox(null);
    setManualWest("");
    setManualSouth("");
    setManualEast("");
    setManualNorth("");
    setSpatialFilterMode("intersects");
  };

  const handleSearch = async (overrideQuery?: string) => {
    const q = (overrideQuery ?? queryText).trim();
    if (!q) return;

    if (overrideQuery) {
      setQueryText(overrideQuery);
    }

    setIsSearching(true);
    setSearchError(null);

    // Build payload
    const payload: SemanticSearchRequest = {
      query: q,
      top_k: topK,
      sensors: selectedSensors.length > 0 ? selectedSensors : null,
      sensor: selectedSensors.length === 1 ? selectedSensors[0] : null,
      start_date: startDate ? new Date(startDate).toISOString() : null,
      end_date: endDate ? new Date(endDate).toISOString() : null,
      min_quality: minQuality > 0 ? minQuality / 100 : null,
      max_cloud_cover: maxCloudCover < 100 ? maxCloudCover : null,
      aoi_bbox: aoiBbox,
      aoi_polygon: aoiPolygon,
      spatial_filter_mode: spatialFilterMode,
    };

    try {
      const resp = await api.semanticSearch(payload);
      setSearchResponse(resp);
      if (resp.results.length > 0) {
        setSelectedTileId(resp.results[0].tile_id);
      } else {
        setSelectedTileId(null);
      }
    } catch (err: any) {
      setSearchError(
        err.response?.data?.detail || err.message || "Failed to execute semantic search."
      );
    } finally {
      setIsSearching(false);
    }
  };

  const selectedTile = searchResponse?.results.find((r) => r.tile_id === selectedTileId) || null;

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-4 md:p-8 space-y-6">
      {/* Top Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-gray-800 pb-5">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-white tracking-tight">
              GeoSemantic Multi-Filter Search
            </h1>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-cyan-950 text-cyan-300 border border-cyan-800/80 flex items-center gap-1.5">
              <Sparkles className="h-3 w-3" />
              RemoteCLIP + PostGIS GiST
            </span>
          </div>
          <p className="text-sm text-gray-400 mt-1">
            Two-tier spatial filtering: Qdrant envelope pre-filtering fused with PostGIS exact polygon intersection.
          </p>
        </div>

        {/* Index Statistics Pill */}
        {filtersData && (
          <div className="flex items-center gap-3 bg-gray-900/80 border border-gray-800 rounded-xl px-4 py-2 text-xs font-mono">
            <div>
              <span className="text-gray-500">Indexed Tiles:</span>{" "}
              <span className="text-cyan-400 font-semibold">{filtersData.total_indexed_tiles}</span>
            </div>
            <span className="text-gray-700">•</span>
            <div>
              <span className="text-gray-500">Sensors:</span>{" "}
              <span className="text-emerald-400 font-semibold">
                {filtersData.sensors.length > 0 ? filtersData.sensors.join(", ") : "None detected"}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Natural Language Search Box & Prompt Suggestions */}
      <div className="space-y-3">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSearch();
          }}
          className="flex flex-col sm:flex-row gap-2"
        >
          <div className="relative flex-1">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-cyan-400" />
            <input
              type="text"
              value={queryText}
              onChange={(e) => setQueryText(e.target.value)}
              placeholder='Describe what to search for (e.g. "industrial facilities near water" or "commercial airport runway")'
              className="w-full bg-gray-900 border border-gray-700/80 rounded-xl pl-12 pr-10 py-3.5 text-sm text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/80 focus:border-transparent transition shadow-lg"
            />
            {queryText && (
              <button
                type="button"
                onClick={() => setQueryText("")}
                className="absolute right-3 top-1/2 -translate-y-1/2 p-1 text-gray-400 hover:text-white rounded-md"
              >
                ✕
              </button>
            )}
          </div>

          <div className="flex items-center gap-2">
            {/* Top-K Selector */}
            <select
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="bg-gray-900 border border-gray-700/80 rounded-xl px-3 py-3.5 text-xs text-gray-200 focus:outline-none focus:ring-2 focus:ring-cyan-500"
              title="Maximum results to return"
            >
              <option value={10}>Top 10</option>
              <option value={20}>Top 20</option>
              <option value={50}>Top 50</option>
            </select>

            {/* Filter Toggle Button */}
            <button
              type="button"
              onClick={() => setShowFilters(!showFilters)}
              className={`px-4 py-3.5 rounded-xl text-xs font-semibold flex items-center gap-2 border transition ${
                activeFiltersCount > 0
                  ? "bg-cyan-950/90 text-cyan-300 border-cyan-700"
                  : "bg-gray-900 text-gray-300 border-gray-700/80 hover:bg-gray-800"
              }`}
            >
              <Filter className="h-4 w-4" />
              <span>Filters</span>
              {activeFiltersCount > 0 && (
                <span className="h-5 w-5 rounded-full bg-cyan-500 text-gray-950 font-bold text-[10px] flex items-center justify-center">
                  {activeFiltersCount}
                </span>
              )}
            </button>

            {/* Execute Search Button */}
            <button
              type="submit"
              disabled={isSearching || !queryText.trim()}
              className="px-6 py-3.5 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white rounded-xl text-sm font-semibold shadow-lg shadow-cyan-950/50 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition"
            >
              {isSearching ? (
                <>
                  <div className="h-4 w-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  <span>Retrieving...</span>
                </>
              ) : (
                <>
                  <Search className="h-4 w-4" />
                  <span>Search Archive</span>
                </>
              )}
            </button>
          </div>
        </form>

        {/* Quick Suggestion Pills */}
        <div className="flex items-center gap-2 flex-wrap text-xs">
          <span className="text-gray-500 flex items-center gap-1 font-mono text-[11px]">
            <Sparkles className="h-3 w-3 text-cyan-400" />
            Suggestions:
          </span>
          {QUERY_SUGGESTIONS.map((sug) => (
            <button
              key={sug}
              type="button"
              onClick={() => handleSearch(sug)}
              className="px-2.5 py-1 rounded-lg bg-gray-900/60 hover:bg-gray-800 border border-gray-800 text-gray-300 hover:text-cyan-300 transition text-[11px]"
            >
              {sug}
            </button>
          ))}
        </div>
      </div>

      {/* Expandable Multi-Factor Filter Panel */}
      {showFilters && (
        <div className="bg-gray-900/70 border border-gray-800 rounded-2xl p-5 space-y-5 animate-in slide-in-from-top duration-200 shadow-xl">
          <div className="flex items-center justify-between border-b border-gray-800 pb-3">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-300 flex items-center gap-2">
              <Sliders className="h-4 w-4 text-cyan-400" />
              GeoSemantic Multi-Filter Composition
            </h3>
            {activeFiltersCount > 0 && (
              <button
                type="button"
                onClick={resetFilters}
                className="text-xs text-gray-400 hover:text-cyan-400 flex items-center gap-1 transition"
              >
                <RotateCcw className="h-3 w-3" /> Reset All Filters
              </button>
            )}
          </div>

          {/* Filter Grid: Sensors, Dates, Quality & Cloud Cover */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5 text-xs">
            {/* 1. Multi-Sensor Selection Pills */}
            <div className="space-y-2">
              <label className="block text-gray-400 font-medium">
                Sensors & Platforms ({selectedSensors.length === 0 ? "All" : selectedSensors.length})
              </label>
              <div className="flex flex-wrap gap-1.5">
                {filtersData?.sensors && filtersData.sensors.length > 0 ? (
                  filtersData.sensors.map((s) => {
                    const isSelected = selectedSensors.includes(s);
                    const count = filtersData.sensor_counts?.[s];
                    return (
                      <button
                        key={s}
                        type="button"
                        onClick={() => toggleSensor(s)}
                        className={`px-2.5 py-1 rounded-lg border text-xs font-medium transition flex items-center gap-1.5 ${
                          isSelected
                            ? "bg-cyan-950 text-cyan-300 border-cyan-700"
                            : "bg-gray-950/80 text-gray-400 border-gray-800 hover:text-gray-200"
                        }`}
                      >
                        <span>{s}</span>
                        {count !== undefined && (
                          <span className="text-[10px] font-mono opacity-70">({count})</span>
                        )}
                      </button>
                    );
                  })
                ) : (
                  <span className="text-gray-500 italic">No indexed sensors detected.</span>
                )}
              </div>
            </div>

            {/* 2. Temporal Date Range */}
            <div className="space-y-2">
              <label className="block text-gray-400 font-medium flex items-center gap-1">
                <Calendar className="h-3 w-3 text-cyan-400" />
                Acquisition Date Range
              </label>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <span className="text-[10px] text-gray-500 block mb-0.5">From</span>
                  <input
                    type="date"
                    value={startDate}
                    onChange={(e) => setStartDate(e.target.value)}
                    className="w-full bg-gray-950 border border-gray-800 rounded-lg px-2.5 py-1.5 text-gray-200 focus:outline-none focus:ring-1 focus:ring-cyan-500 text-xs"
                  />
                </div>
                <div>
                  <span className="text-[10px] text-gray-500 block mb-0.5">To</span>
                  <input
                    type="date"
                    value={endDate}
                    onChange={(e) => setEndDate(e.target.value)}
                    className="w-full bg-gray-950 border border-gray-800 rounded-lg px-2.5 py-1.5 text-gray-200 focus:outline-none focus:ring-1 focus:ring-cyan-500 text-xs"
                  />
                </div>
              </div>
              {(startDate || endDate) && (
                <button
                  type="button"
                  onClick={() => {
                    setStartDate("");
                    setEndDate("");
                  }}
                  className="text-[10px] text-cyan-400 hover:underline"
                >
                  Clear dates
                </button>
              )}
            </div>

            {/* 3. Minimum Quality Slider */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-gray-400 font-medium flex items-center gap-1">
                  <ShieldCheck className="h-3 w-3 text-emerald-400" />
                  Min Quality Score
                </label>
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
              <div className="flex justify-between text-[10px] text-gray-500">
                <span>0% (All)</span>
                <span>50%</span>
                <span>80% (Clean)</span>
              </div>
            </div>

            {/* 4. Maximum Cloud Cover Slider */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-gray-400 font-medium flex items-center gap-1">
                  <Cloud className="h-3 w-3 text-blue-400" />
                  Max Cloud Cover
                </label>
                <span className="font-mono text-blue-400 font-semibold">{maxCloudCover}%</span>
              </div>
              <input
                type="range"
                min={0}
                max={100}
                step={5}
                value={maxCloudCover}
                onChange={(e) => setMaxCloudCover(Number(e.target.value))}
                className="w-full h-2 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-blue-400"
              />
              <div className="flex justify-between text-[10px] text-gray-500">
                <span>0% (Cloudless)</span>
                <span>20%</span>
                <span>100% (Any)</span>
              </div>
            </div>
          </div>

          {/* Spatial AOI Section: Active Filter Status & Manual BBox Coordinates */}
          <div className="border-t border-gray-800 pt-4 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <Compass className="h-4 w-4 text-cyan-400" />
                <span className="font-semibold text-gray-300 text-xs">
                  Area of Interest (AOI) & Spatial Verification
                </span>
                {aoiPolygon && aoiPolygon.length >= 3 && (
                  <span className="px-2 py-0.5 bg-amber-950/80 border border-amber-800 text-amber-300 rounded text-[11px] font-mono">
                    Polygon: {aoiPolygon.length} vertices
                  </span>
                )}
                {aoiBbox && aoiBbox.length === 4 && (
                  <span className="px-2 py-0.5 bg-blue-950/80 border border-blue-800 text-blue-300 rounded text-[11px] font-mono">
                    BBox: [{aoiBbox[0].toFixed(2)}, {aoiBbox[1].toFixed(2)}, {aoiBbox[2].toFixed(2)}, {aoiBbox[3].toFixed(2)}]
                  </span>
                )}
              </div>

              {/* Spatial Mode Selector */}
              <div className="flex items-center gap-2 text-xs font-mono">
                <span className="text-gray-500">PostGIS Operator:</span>
                <button
                  type="button"
                  onClick={() => setSpatialFilterMode("intersects")}
                  className={`px-2 py-1 rounded-lg border text-xs transition ${
                    spatialFilterMode === "intersects"
                      ? "bg-cyan-950 text-cyan-300 border-cyan-700 font-bold"
                      : "bg-gray-950 text-gray-400 border-gray-800"
                  }`}
                >
                  ST_Intersects
                </button>
                <button
                  type="button"
                  onClick={() => setSpatialFilterMode("within")}
                  className={`px-2 py-1 rounded-lg border text-xs transition ${
                    spatialFilterMode === "within"
                      ? "bg-cyan-950 text-cyan-300 border-cyan-700 font-bold"
                      : "bg-gray-950 text-gray-400 border-gray-800"
                  }`}
                >
                  ST_Within
                </button>
                {(aoiPolygon || aoiBbox) && (
                  <button
                    type="button"
                    onClick={() => {
                      setAoiPolygon(null);
                      setAoiBbox(null);
                      setManualWest("");
                      setManualSouth("");
                      setManualEast("");
                      setManualNorth("");
                    }}
                    className="ml-2 text-xs text-red-400 hover:underline"
                  >
                    Clear AOI
                  </button>
                )}
              </div>
            </div>

            {/* Manual Bounding Box Coordinates Input */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
              <div>
                <span className="text-[10px] text-gray-500 block mb-1">West Lon (°)</span>
                <input
                  type="number"
                  step="any"
                  placeholder="e.g. 77.0"
                  value={manualWest}
                  onChange={(e) =>
                    handleManualBboxChange(e.target.value, manualSouth, manualEast, manualNorth)
                  }
                  className="w-full bg-gray-950 border border-gray-800 rounded-lg px-2.5 py-1.5 font-mono text-gray-200 text-xs focus:ring-1 focus:ring-cyan-500 focus:outline-none"
                />
              </div>
              <div>
                <span className="text-[10px] text-gray-500 block mb-1">South Lat (°)</span>
                <input
                  type="number"
                  step="any"
                  placeholder="e.g. 28.0"
                  value={manualSouth}
                  onChange={(e) =>
                    handleManualBboxChange(manualWest, e.target.value, manualEast, manualNorth)
                  }
                  className="w-full bg-gray-950 border border-gray-800 rounded-lg px-2.5 py-1.5 font-mono text-gray-200 text-xs focus:ring-1 focus:ring-cyan-500 focus:outline-none"
                />
              </div>
              <div>
                <span className="text-[10px] text-gray-500 block mb-1">East Lon (°)</span>
                <input
                  type="number"
                  step="any"
                  placeholder="e.g. 77.5"
                  value={manualEast}
                  onChange={(e) =>
                    handleManualBboxChange(manualWest, manualSouth, e.target.value, manualNorth)
                  }
                  className="w-full bg-gray-950 border border-gray-800 rounded-lg px-2.5 py-1.5 font-mono text-gray-200 text-xs focus:ring-1 focus:ring-cyan-500 focus:outline-none"
                />
              </div>
              <div>
                <span className="text-[10px] text-gray-500 block mb-1">North Lat (°)</span>
                <input
                  type="number"
                  step="any"
                  placeholder="e.g. 28.5"
                  value={manualNorth}
                  onChange={(e) =>
                    handleManualBboxChange(manualWest, manualSouth, manualEast, e.target.value)
                  }
                  className="w-full bg-gray-950 border border-gray-800 rounded-lg px-2.5 py-1.5 font-mono text-gray-200 text-xs focus:ring-1 focus:ring-cyan-500 focus:outline-none"
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Error Banner */}
      {searchError && (
        <div className="bg-red-950/60 border border-red-800 rounded-xl p-4 flex items-center gap-3 text-red-300 text-xs">
          <AlertCircle className="h-5 w-5 text-red-400 shrink-0" />
          <span>{searchError}</span>
        </div>
      )}

      {/* Search Telemetry Bar */}
      {searchResponse && (
        <div className="bg-gray-900/60 border border-gray-800 rounded-xl p-3.5 flex flex-wrap items-center justify-between gap-3 text-xs shadow">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-white">Results:</span>
            <span className="font-mono text-cyan-300 font-bold">
              {searchResponse.total_found} tiles
            </span>
            <span className="text-gray-600">for</span>
            <span className="font-medium text-gray-300 italic">"{searchResponse.query}"</span>
          </div>

          <div className="flex flex-wrap items-center gap-3 text-gray-400 font-mono text-[11px]">
            <div className="flex items-center gap-1 text-white font-semibold">
              <Clock className="h-3.5 w-3.5 text-cyan-400" />
              <span>Total: {searchResponse.execution_time_ms}ms</span>
            </div>
            <span className="text-gray-700">•</span>
            <div>Text Embed: {searchResponse.text_embedding_time_ms}ms</div>
            <span className="text-gray-700">•</span>
            <div>Qdrant Search: {searchResponse.vector_search_time_ms}ms</div>
            {searchResponse.spatial_filter_time_ms !== undefined && (
              <>
                <span className="text-gray-700">•</span>
                <div className="text-cyan-300">
                  PostGIS GiST: {searchResponse.spatial_filter_time_ms}ms
                </div>
              </>
            )}
            <span className="text-gray-700">•</span>
            <div className="text-emerald-400 font-semibold">{searchResponse.device.toUpperCase()}</div>
          </div>
        </div>
      )}

      {/* Main Two-Column Analyst Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Ranked Results List */}
        <div className="lg:col-span-7 space-y-4">
          {!searchResponse && !isSearching && (
            <div className="bg-gray-900/40 border border-gray-800/80 rounded-2xl p-10 text-center flex flex-col items-center justify-center space-y-3">
              <div className="p-3 bg-cyan-950/60 rounded-2xl border border-cyan-800/80 text-cyan-400">
                <Search className="h-8 w-8" />
              </div>
              <h3 className="text-base font-semibold text-white">GeoSemantic Retrieval Ready</h3>
              <p className="text-xs text-gray-400 max-w-md">
                Enter any natural language query above or select a suggestion pill. You can combine it with
                an Area of Interest polygon, date constraints, quality thresholds, and satellite sensors.
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

          {/* Zero-Result Informative Handling State */}
          {searchResponse && searchResponse.results.length === 0 && !isSearching && (
            <div className="bg-gray-900/40 border border-gray-800 rounded-2xl p-8 space-y-4 text-center">
              <div className="p-3 bg-amber-950/40 rounded-2xl border border-amber-800/60 text-amber-400 w-12 h-12 mx-auto flex items-center justify-center">
                <AlertCircle className="h-6 w-6" />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-gray-200">No Matching Satellite Tiles Found</h3>
                <p className="text-xs text-gray-400 mt-1 max-w-md mx-auto">
                  No tiles matched the combined semantic query and active geospatial / metadata filters simultaneously.
                </p>
              </div>

              {/* Active Filter Summary Badges */}
              <div className="bg-gray-950/70 border border-gray-800 rounded-xl p-3 max-w-lg mx-auto text-xs space-y-2 text-left">
                <span className="text-[11px] font-semibold text-gray-400 block border-b border-gray-800 pb-1">
                  Active Filter Configuration:
                </span>
                <div className="flex flex-wrap gap-2 text-[11px] font-mono">
                  <span className="px-2 py-0.5 bg-gray-800 text-gray-300 rounded">
                    Query: "{searchResponse.query}"
                  </span>
                  {selectedSensors.length > 0 && (
                    <span className="px-2 py-0.5 bg-cyan-950 text-cyan-300 border border-cyan-800 rounded">
                      Sensors: {selectedSensors.join(", ")}
                    </span>
                  )}
                  {(startDate || endDate) && (
                    <span className="px-2 py-0.5 bg-blue-950 text-blue-300 border border-blue-800 rounded">
                      Date: {startDate || "earliest"} → {endDate || "latest"}
                    </span>
                  )}
                  {minQuality > 0 && (
                    <span className="px-2 py-0.5 bg-emerald-950 text-emerald-300 border border-emerald-800 rounded">
                      Min Quality: {minQuality}%
                    </span>
                  )}
                  {maxCloudCover < 100 && (
                    <span className="px-2 py-0.5 bg-indigo-950 text-indigo-300 border border-indigo-800 rounded">
                      Max Cloud: {maxCloudCover}%
                    </span>
                  )}
                  {aoiPolygon && (
                    <span className="px-2 py-0.5 bg-amber-950 text-amber-300 border border-amber-800 rounded">
                      AOI Polygon ({aoiPolygon.length} pts)
                    </span>
                  )}
                  {aoiBbox && (
                    <span className="px-2 py-0.5 bg-amber-950 text-amber-300 border border-amber-800 rounded">
                      AOI BBox
                    </span>
                  )}
                </div>
              </div>

              {/* Action Buttons to Relax Filters */}
              <div className="flex flex-wrap items-center justify-center gap-2 pt-2">
                {(aoiPolygon || aoiBbox) && (
                  <button
                    type="button"
                    onClick={() => {
                      setAoiPolygon(null);
                      setAoiBbox(null);
                      setManualWest("");
                      setManualSouth("");
                      setManualEast("");
                      setManualNorth("");
                      handleSearch();
                    }}
                    className="px-3 py-1.5 bg-gray-800 hover:bg-gray-700 text-cyan-300 rounded-lg text-xs font-medium transition"
                  >
                    Clear AOI & Re-Search
                  </button>
                )}
                {(minQuality > 0 || maxCloudCover < 100) && (
                  <button
                    type="button"
                    onClick={() => {
                      setMinQuality(0);
                      setMaxCloudCover(100);
                      handleSearch();
                    }}
                    className="px-3 py-1.5 bg-gray-800 hover:bg-gray-700 text-emerald-300 rounded-lg text-xs font-medium transition"
                  >
                    Relax Quality Thresholds
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => {
                    resetFilters();
                    handleSearch();
                  }}
                  className="px-3 py-1.5 bg-cyan-900/60 hover:bg-cyan-800 text-cyan-200 rounded-lg text-xs font-medium transition border border-cyan-700/60"
                >
                  Reset All Filters & Re-Search
                </button>
              </div>
            </div>
          )}

          {/* Results List */}
          {searchResponse && searchResponse.results.length > 0 && (
            <div className="space-y-3">
              {searchResponse.results.map((item) => {
                const isSelected = selectedTileId === item.tile_id;
                const scorePct = Math.min(100, Math.max(0, item.similarity_score * 100)).toFixed(1);

                return (
                  <div
                    key={item.tile_id}
                    onClick={() => setSelectedTileId(item.tile_id)}
                    className={`bg-gray-900/50 hover:bg-gray-900 border rounded-2xl p-4 transition-all duration-150 cursor-pointer flex flex-col sm:flex-row gap-4 ${
                      isSelected
                        ? "border-cyan-400 ring-1 ring-cyan-400/80 shadow-lg shadow-cyan-950/40 bg-gray-900/80"
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
                          e.currentTarget.style.display = "none";
                          const fallback = e.currentTarget.nextElementSibling;
                          if (fallback) fallback.classList.remove("hidden");
                        }}
                      />
                      <div className="hidden flex flex-col items-center justify-center p-2 text-center text-gray-500">
                        <Layers className="h-6 w-6 text-gray-600 mb-1" />
                        <span className="text-[9px] font-mono">No Image</span>
                      </div>
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

                          {/* Similarity Score Pill */}
                          <div className="flex items-center gap-1 px-2.5 py-1 rounded-full bg-emerald-950/90 border border-emerald-700/80 text-emerald-400 font-mono font-bold text-xs">
                            <Zap className="h-3 w-3" />
                            <span>{scorePct}%</span>
                          </div>
                        </div>

                        {/* Metadata Tag Row */}
                        <div className="flex flex-wrap items-center gap-2 mt-1.5 text-[11px] font-mono text-gray-400">
                          <span className="px-1.5 py-0.5 rounded bg-gray-800 text-cyan-300">
                            {item.sensor || "Sentinel-2"}
                          </span>
                          <span>•</span>
                          <span>{formatDate(item.acquisition_date)}</span>
                          <span>•</span>
                          <span className={qualityColor(item.quality_score)}>
                            Q: {item.quality_score != null ? `${(item.quality_score * 100).toFixed(0)}%` : "—"}
                          </span>
                          {item.cloud_cover_pct != null && (
                            <>
                              <span>•</span>
                              <span className="text-blue-300 flex items-center gap-0.5">
                                <Cloud className="h-2.5 w-2.5" />
                                {item.cloud_cover_pct.toFixed(0)}%
                              </span>
                            </>
                          )}
                          {/* Landcover Badges (Water & Vegetation) */}
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
                          <MapPin className="h-3 w-3 text-cyan-400" />
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
                            className="px-2.5 py-1 bg-gray-800 hover:bg-cyan-900/60 hover:text-cyan-300 text-gray-300 rounded-lg text-xs font-medium flex items-center gap-1 transition border border-gray-700 hover:border-cyan-700"
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

        {/* Right Column: Spatial Heatmap & Selected Tile Card */}
        <div className="lg:col-span-5 space-y-4">
          <div className="sticky top-6 space-y-4">
            {/* Interactive Coverage Heatmap with Polygon & BBox Drawing */}
            <SearchCoverageMap
              results={searchResponse?.results || []}
              selectedTileId={selectedTileId}
              onSelectTile={(tile) => setSelectedTileId(tile.tile_id)}
              onInspectTile={(tile) => setInspectingTile(tile)}
              aoiPolygon={aoiPolygon}
              onAoiPolygonChange={(poly) => {
                setAoiPolygon(poly);
                if (poly && poly.length >= 3) {
                  setAoiBbox(null);
                }
              }}
              aoiBbox={aoiBbox}
              onAoiBboxChange={(bbox) => {
                setAoiBbox(bbox);
                if (bbox && bbox.length === 4) {
                  setAoiPolygon(null);
                }
              }}
              spatialFilterMode={spatialFilterMode}
              onSpatialFilterModeChange={(mode) => setSpatialFilterMode(mode)}
            />

            {/* Selected Tile Inspector Card */}
            {selectedTile ? (
              <div className="bg-gray-900/60 border border-gray-800 rounded-2xl p-4 space-y-3 shadow-xl">
                <div className="flex items-center justify-between border-b border-gray-800 pb-2.5">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-white text-xs">Selected Candidate</span>
                    <span className="text-cyan-400 font-mono font-bold text-xs">
                      Rank #{selectedTile.rank}
                    </span>
                  </div>
                  <span className="px-2 py-0.5 rounded-full text-[11px] font-mono font-semibold bg-emerald-950 text-emerald-300 border border-emerald-800">
                    {Math.min(100, Math.max(0, selectedTile.similarity_score * 100)).toFixed(1)}% Match
                  </span>
                </div>

                <div className="flex items-center gap-3">
                  <img
                    src={selectedTile.thumbnail_url}
                    alt="Tile"
                    className="w-16 h-16 rounded-xl object-cover border border-gray-800 bg-gray-950 shrink-0"
                  />
                  <div className="text-xs space-y-1 overflow-hidden">
                    <p className="font-medium text-gray-200 truncate">{selectedTile.scene_name}</p>
                    <p className="text-gray-400 font-mono text-[11px]">
                      Sensor: {selectedTile.sensor || "Sentinel-2"}
                    </p>
                    {selectedTile.landcover && (
                      <div className="flex items-center gap-2 font-mono text-[10px] text-cyan-300">
                        <span>💧 {selectedTile.landcover.water_pct?.toFixed(1) ?? 0}% Water</span>
                        <span>•</span>
                        <span className="text-emerald-300">🌿 {selectedTile.landcover.veg_pct?.toFixed(1) ?? 0}% Veg</span>
                      </div>
                    )}
                    <p className="text-gray-400 font-mono text-[11px]">
                      Center: [{formatCoord(selectedTile.center_coordinates.lat, 4)}, {formatCoord(selectedTile.center_coordinates.lon, 4)}]
                    </p>
                  </div>
                </div>

                <button
                  type="button"
                  onClick={() => setInspectingTile(selectedTile)}
                  className="w-full py-2 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white rounded-xl text-xs font-semibold flex items-center justify-center gap-1.5 transition shadow"
                >
                  <Eye className="h-3.5 w-3.5" />
                  <span>Inspect Full Tile & Audit Provenance</span>
                </button>
              </div>
            ) : (
              <div className="bg-gray-900/30 border border-gray-800/60 rounded-2xl p-6 text-center text-xs text-gray-500">
                <Layers className="h-6 w-6 mx-auto mb-2 text-gray-600 opacity-60" />
                <span>Select any tile from the search results or map to inspect details.</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* High-Resolution Tile & Provenance Modal */}
      <TileDetailModal
        tile={inspectingTile}
        onClose={() => setInspectingTile(null)}
      />
    </div>
  );
}
