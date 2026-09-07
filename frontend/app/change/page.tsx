"use client";

import React, { useState, useEffect } from "react";
import { api, ChangeAnalyzeRequest, ChangeEventResponse, SceneSummary } from "@/lib/api";
import { ChangeCandidateCard } from "@/components/change/ChangeCandidateCard";
import { 
  Search, Loader2, AlertCircle, Sparkles, Globe, Calendar, 
  Layers, Database, Filter, RefreshCw, ChevronRight, CheckCircle2 
} from "lucide-react";

export default function ChangeAnalysisPage() {
  const [loading, setLoading] = useState(false);
  const [scenesLoading, setScenesLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Dynamic scenes retrieved from database
  const [scenes, setScenes] = useState<SceneSummary[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>("ALL");

  // Form State
  const [query, setQuery] = useState("");
  const [aoiWkt, setAoiWkt] = useState<string>("");
  const [startDate, setStartDate] = useState("2018-01-01");
  const [endDate, setEndDate] = useState("2026-12-31");
  const [limit, setLimit] = useState(25);
  
  const [results, setResults] = useState<ChangeEventResponse[]>([]);
  const [recentEvents, setRecentEvents] = useState<ChangeEventResponse[]>([]);

  // Load real scenes and recent events on mount
  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    setScenesLoading(true);
    try {
      const [sceneRes, eventRes] = await Promise.allSettled([
        api.listScenes(1, 100),
        api.listChangeEvents(20),
      ]);

      if (sceneRes.status === "fulfilled") {
        const fetchedScenes = sceneRes.value.scenes;
        setScenes(fetchedScenes);

        // Auto-select Noida Jewar Airport dataset for immediate demo showcase
        const jewarScene = fetchedScenes.find(s => s.filename.toLowerCase().includes("jewar") || s.filename.toLowerCase().includes("noida"));
        if (jewarScene && jewarScene.bbox_west != null && jewarScene.bbox_south != null && jewarScene.bbox_east != null && jewarScene.bbox_north != null) {
          setSelectedDatasetId(jewarScene.id);
          const w = jewarScene.bbox_west.toFixed(4);
          const s = jewarScene.bbox_south.toFixed(4);
          const e = jewarScene.bbox_east.toFixed(4);
          const n = jewarScene.bbox_north.toFixed(4);
          setAoiWkt(`POLYGON((${w} ${s}, ${e} ${s}, ${e} ${n}, ${w} ${n}, ${w} ${s}))`);
          setStartDate("2020-01-01");
          setEndDate("2025-01-01");
          setQuery("Airport runway, passenger terminal and road expansion");
        }
      }
      if (eventRes.status === "fulfilled") {
        setRecentEvents(eventRes.value);
      }
    } catch (err) {
      console.error("Failed to load initial data:", err);
    } finally {
      setScenesLoading(false);
    }
  };

  // Handle dynamic dataset selection
  const handleDatasetChange = (datasetId: string) => {
    setSelectedDatasetId(datasetId);

    if (datasetId === "ALL") {
      setAoiWkt("");
      return;
    }

    if (datasetId === "CUSTOM") {
      return;
    }

    const scene = scenes.find(s => s.id === datasetId);
    if (scene && scene.bbox_west != null && scene.bbox_south != null && scene.bbox_east != null && scene.bbox_north != null) {
      // Build polygon from real database bounding box coordinates
      const w = scene.bbox_west.toFixed(4);
      const s = scene.bbox_south.toFixed(4);
      const e = scene.bbox_east.toFixed(4);
      const n = scene.bbox_north.toFixed(4);
      setAoiWkt(`POLYGON((${w} ${s}, ${e} ${s}, ${e} ${n}, ${w} ${n}, ${w} ${s}))`);

      if (scene.acquisition_date) {
        // Expand date window around acquisition
        const d = new Date(scene.acquisition_date);
        const startY = Math.max(2015, d.getFullYear() - 4);
        const endY = d.getFullYear() + 2;
        setStartDate(`${startY}-01-01`);
        setEndDate(`${endY}-12-31`);
      }
    }
  };

  // Semantic query helper tags (set query text only, without altering coordinates or dates)
  const semanticTags = [
    { label: "✈️ Airport & Runway Expansion", text: "Airport runway, terminal and infrastructure expansion" },
    { label: "🏗️ New Construction & Urban", text: "New building construction, urban development and paved structures" },
    { label: "🌊 Water Body Changes", text: "Water body expansion, reservoir changes and shoreline alterations" },
    { label: "🌳 Vegetation Clearance", text: "Vegetation clearance, deforestation and tree removal" },
  ];

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResults([]);
    
    try {
      const cleanWkt = aoiWkt.trim();
      const req: ChangeAnalyzeRequest = {
        aoi_wkt: cleanWkt && cleanWkt.toUpperCase() !== "ALL" ? cleanWkt : null,
        start_date: new Date(startDate).toISOString(),
        end_date: new Date(endDate).toISOString(),
        limit,
        query: query.trim() || undefined,
      };
      
      const res = await api.analyzeChange(req);
      setResults(res.events);
      if (res.candidates_found === 0) {
        setError("No overlapping multi-temporal observations found for this time frame and geographic area.");
      }
      
      // Refresh recent list
      const updatedRecent = await api.listChangeEvents(20);
      setRecentEvents(updatedRecent);
    } catch (err: any) {
      console.error(err);
      setError(err?.response?.data?.detail || err?.message || "Change detection analysis failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-6 md:p-8">
      {/* Header */}
      <div className="max-w-7xl mx-auto mb-8">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-gray-800/80 pb-6">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="px-2.5 py-0.5 rounded-full bg-blue-500/10 border border-blue-500/30 text-blue-400 text-xs font-semibold tracking-wider uppercase">
                Phase 10 Intelligence
              </span>
              <span className="text-gray-500 text-xs">•</span>
              <span className="text-gray-400 text-xs font-medium">Multi-Temporal Change Engine</span>
            </div>
            <h1 className="text-3xl font-bold text-white tracking-tight">
              Satellite Change Detection
            </h1>
            <p className="text-sm text-gray-400 mt-1.5 max-w-3xl leading-relaxed">
              Detect physical ground changes across any multi-temporal satellite dataset with OpenCV ECC sub-pixel registration, 
              deterministic spectral landcover verification, and RemoteCLIP vision-language semantic alignment.
            </p>
          </div>

          <button
            onClick={loadInitialData}
            disabled={scenesLoading}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gray-900 border border-gray-800 text-xs text-gray-300 hover:text-white hover:border-gray-700 transition-colors"
            title="Refresh Ingested Datasets"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${scenesLoading ? "animate-spin text-blue-400" : ""}`} />
            <span>Refresh Archives</span>
          </button>
        </div>
      </div>

      {/* Main Grid */}
      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Form: Analysis Configuration */}
        <div className="lg:col-span-4 space-y-6">
          <div className="bg-gray-900/80 border border-gray-800 rounded-2xl p-6 shadow-xl backdrop-blur-sm">
            <h2 className="text-base font-semibold text-white mb-5 flex items-center gap-2 border-b border-gray-800 pb-3">
              <Filter className="w-4 h-4 text-blue-400" />
              Analysis Configuration
            </h2>

            <form onSubmit={handleAnalyze} className="space-y-5">
              {/* Natural Language Query */}
              <div>
                <div className="flex justify-between items-center mb-1.5">
                  <label className="text-xs font-medium text-gray-300">
                    Natural Language Query <span className="text-gray-500 font-normal">(Optional)</span>
                  </label>
                  {query && (
                    <button
                      type="button"
                      onClick={() => setQuery("")}
                      className="text-[11px] text-gray-400 hover:text-red-400 transition-colors"
                    >
                      Clear
                    </button>
                  )}
                </div>
                <div className="relative">
                  <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="e.g. Airport runway expansion, New buildings..."
                    className="w-full bg-gray-950 border border-gray-800 rounded-xl px-3.5 py-2.5 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-blue-500 transition-colors"
                  />
                  <Sparkles className="w-4 h-4 text-blue-400 absolute right-3 top-3 pointer-events-none opacity-60" />
                </div>

                {/* Semantic Query Helper Pills */}
                <div className="flex flex-wrap gap-1.5 mt-2">
                  {semanticTags.map((tag) => (
                    <button
                      key={tag.label}
                      type="button"
                      onClick={() => setQuery(tag.text)}
                      className="text-[11px] px-2 py-1 rounded-md bg-gray-800/80 hover:bg-blue-600/20 hover:text-blue-300 border border-gray-700/60 text-gray-400 transition-colors text-left"
                    >
                      {tag.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Ingested Dataset / Region Selector */}
              <div>
                <label className="text-xs font-medium text-gray-300 block mb-1.5 flex items-center justify-between">
                  <span>Target Dataset / Region</span>
                  <span className="text-gray-500 font-normal text-[11px]">
                    {scenes.length} scene{scenes.length === 1 ? "" : "s"} indexed
                  </span>
                </label>
                <select
                  value={selectedDatasetId}
                  onChange={(e) => handleDatasetChange(e.target.value)}
                  className="w-full bg-gray-950 border border-gray-800 rounded-xl px-3.5 py-2.5 text-sm text-gray-100 focus:outline-none focus:border-blue-500 transition-colors cursor-pointer"
                >
                  <option value="ALL">🌐 Whole Archive (Compare Any Overlapping Scenes)</option>
                  <optgroup label="Ingested Satellite Scenes">
                    {scenes.map((s) => (
                      <option key={s.id} value={s.id}>
                        📍 {s.filename} ({s.acquisition_date ? s.acquisition_date.slice(0, 10) : "No Date"}, {s.tile_count} tiles)
                      </option>
                    ))}
                  </optgroup>
                  <option value="CUSTOM">✏️ Custom Polygon Area (WKT Input)</option>
                </select>
              </div>

              {/* AOI WKT Coordinates */}
              <div>
                <div className="flex justify-between items-center mb-1.5">
                  <label className="text-xs font-medium text-gray-300">
                    Area of Interest (WKT)
                  </label>
                  {!aoiWkt && (
                    <span className="text-[11px] text-blue-400 font-mono">
                      Whole Archive Mode
                    </span>
                  )}
                </div>
                <textarea
                  value={aoiWkt}
                  onChange={(e) => {
                    setAoiWkt(e.target.value);
                    setSelectedDatasetId("CUSTOM");
                  }}
                  placeholder="Leave empty to analyze all overlapping imagery across the entire archive, or enter POLYGON((...))"
                  rows={3}
                  className="w-full bg-gray-950 border border-gray-800 rounded-xl p-3 text-xs font-mono text-gray-200 placeholder-gray-600 focus:outline-none focus:border-blue-500 transition-colors resize-none"
                />
              </div>

              {/* Date Range */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs font-medium text-gray-300 block mb-1.5">
                    Start Date
                  </label>
                  <input
                    type="date"
                    value={startDate}
                    onChange={(e) => setStartDate(e.target.value)}
                    className="w-full bg-gray-950 border border-gray-800 rounded-xl px-3 py-2 text-xs text-gray-200 focus:outline-none focus:border-blue-500 transition-colors"
                  />
                </div>
                <div>
                  <label className="text-xs font-medium text-gray-300 block mb-1.5">
                    End Date
                  </label>
                  <input
                    type="date"
                    value={endDate}
                    onChange={(e) => setEndDate(e.target.value)}
                    className="w-full bg-gray-950 border border-gray-800 rounded-xl px-3 py-2 text-xs text-gray-200 focus:outline-none focus:border-blue-500 transition-colors"
                  />
                </div>
              </div>

              {/* Max Candidates */}
              <div>
                <div className="flex justify-between items-center text-xs mb-1.5">
                  <span className="text-gray-300 font-medium">Candidate Pair Limit</span>
                  <span className="text-blue-400 font-mono font-semibold">{limit} pairs</span>
                </div>
                <input
                  type="range"
                  min="5"
                  max="100"
                  step="5"
                  value={limit}
                  onChange={(e) => setLimit(Number(e.target.value))}
                  className="w-full accent-blue-500 cursor-pointer"
                />
              </div>

              {/* Run Button */}
              <button
                type="submit"
                disabled={loading}
                className="w-full py-3 px-4 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white text-sm font-semibold rounded-xl shadow-lg shadow-blue-500/20 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-all"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Executing Registration & Semantic Inference...</span>
                  </>
                ) : (
                  <>
                    <Search className="w-4 h-4" />
                    <span>Run Change Analysis</span>
                  </>
                )}
              </button>
            </form>
          </div>
        </div>

        {/* Right Content: Analysis Results */}
        <div className="lg:col-span-8 space-y-6">
          {/* Active Error Banner */}
          {error && (
            <div className="p-4 bg-red-950/40 border border-red-800/60 rounded-xl text-red-200 text-sm flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
              <div>
                <span className="font-semibold block mb-0.5">Analysis Notice</span>
                <span>{error}</span>
              </div>
            </div>
          )}

          {/* Results Section */}
          <div>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <h2 className="text-xl font-bold text-white">
                  {results.length > 0 ? "Analysis Results" : "Recent Change Candidates"}
                </h2>
                <span className="px-2.5 py-0.5 rounded-full bg-gray-800 text-xs font-semibold text-gray-300">
                  {results.length > 0 ? `${results.length} candidate${results.length === 1 ? "" : "s"}` : `${recentEvents.length} recorded`}
                </span>
              </div>
            </div>

            {/* Candidate Cards Grid */}
            {results.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                {results.map((ev) => (
                  <ChangeCandidateCard key={ev.id} event={ev} />
                ))}
              </div>
            ) : recentEvents.length > 0 ? (
              <div className="space-y-4">
                <p className="text-xs text-gray-400">
                  Showing recently computed change events from your archive. Run an analysis above to generate new candidate comparisons.
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                  {recentEvents.map((ev) => (
                    <ChangeCandidateCard key={ev.id} event={ev} />
                  ))}
                </div>
              </div>
            ) : (
              <div className="bg-gray-900/40 border border-gray-800/60 rounded-2xl p-12 text-center text-gray-500">
                <Database className="w-10 h-10 mx-auto mb-3 opacity-40 text-blue-400" />
                <h3 className="text-base font-medium text-gray-300 mb-1">No Changes Detected Yet</h3>
                <p className="text-xs text-gray-500 max-w-md mx-auto">
                  Select a target dataset or specify a date range, enter an optional query, and click &quot;Run Change Analysis&quot; to inspect real multi-temporal changes.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
