"use client";

import React, { useState, useEffect } from "react";
import { api, ChangeAnalyzeRequest, ChangeEventResponse, SceneSummary } from "@/lib/api";
import { ChangeCandidateCard } from "@/components/change/ChangeCandidateCard";
import { 
  Search, Loader2, AlertCircle, Sparkles, Globe, Calendar, 
  Layers, Database, Filter, RefreshCw, ChevronRight, ChevronDown, CheckCircle2, ArrowRight
} from "lucide-react";

export default function ChangeAnalysisPage() {
  const [loading, setLoading] = useState(false);
  const [scenesLoading, setScenesLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Dynamic scenes retrieved from database
  const [scenes, setScenes] = useState<SceneSummary[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>("ALL");

  // Natural Language Search State (No coordinates required!)
  const [query, setQuery] = useState("Airport runway, passenger terminal and road expansion");
  
  // Optional Advanced Filters (Hidden by default)
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [aoiWkt, setAoiWkt] = useState<string>("");
  const [startDate, setStartDate] = useState("2016-01-01");
  const [endDate, setEndDate] = useState("2026-12-31");
  const [limit, setLimit] = useState(25);
  
  const [results, setResults] = useState<ChangeEventResponse[]>([]);
  const [recentEvents, setRecentEvents] = useState<ChangeEventResponse[]>([]);

  // Semantic query helper tags for one-click demonstration
  const semanticTags = [
    { 
      label: "✈️ Airport & Runway Expansion", 
      text: "Airport runway, passenger terminal and road expansion" 
    },
    { 
      label: "☀️ Solar Farm (Bhadla, Rajasthan)", 
      text: "Solar farm installation and solar panel development in Rajasthan" 
    },
    { 
      label: "🏗️ New Construction & Urban", 
      text: "New building construction, urban development and paved structures" 
    },
    { 
      label: "🌊 Water Body & Coastline Shifts", 
      text: "Water body changes, reservoir shoreline shifts and coastal alterations" 
    },
    { 
      label: "🌲 Vegetation Clearance", 
      text: "Vegetation clearance, deforestation and tree removal" 
    },
    { 
      label: "🌐 All Detected Changes", 
      text: "" 
    },
  ];

  // Load initial data and run initial analysis on mount
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
        setScenes(sceneRes.value.scenes);
      }
      if (eventRes.status === "fulfilled") {
        setRecentEvents(eventRes.value);
      }

      // Automatically trigger initial analysis across the whole archive
      executeAnalysis("Airport runway, passenger terminal and road expansion");
    } catch (err) {
      console.error("Failed to load initial data:", err);
    } finally {
      setScenesLoading(false);
    }
  };

  const executeAnalysis = async (searchQuery: string, customWkt?: string) => {
    setLoading(true);
    setError(null);
    setResults([]);

    try {
      const cleanWkt = (customWkt !== undefined ? customWkt : aoiWkt).trim();
      const req: ChangeAnalyzeRequest = {
        aoi_wkt: cleanWkt && cleanWkt.toUpperCase() !== "ALL" ? cleanWkt : null,
        start_date: startDate ? new Date(startDate).toISOString() : undefined,
        end_date: endDate ? new Date(endDate).toISOString() : undefined,
        limit,
        query: searchQuery.trim() || undefined,
      };

      const res = await api.analyzeChange(req);
      setResults(res.events);
      if (res.candidates_found === 0) {
        setError("No overlapping multi-temporal observations found matching this query.");
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

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    executeAnalysis(query);
  };

  const handleTagClick = (tagText: string) => {
    setQuery(tagText);
    executeAnalysis(tagText);
  };

  // Handle optional dynamic dataset selection in Advanced Filters
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
      const w = scene.bbox_west.toFixed(4);
      const s = scene.bbox_south.toFixed(4);
      const e = scene.bbox_east.toFixed(4);
      const n = scene.bbox_north.toFixed(4);
      setAoiWkt(`POLYGON((${w} ${s}, ${e} ${s}, ${e} ${n}, ${w} ${n}, ${w} ${s}))`);
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
              Ask in natural language to discover where physical ground changes have occurred across any satellite dataset. 
              Zero coordinates required — powered by OpenCV ECC sub-pixel registration, deterministic optical landcover verification, and RemoteCLIP vision-language alignment.
            </p>
          </div>

          <button
            onClick={loadInitialData}
            disabled={scenesLoading || loading}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gray-900 border border-gray-800 text-xs text-gray-300 hover:text-white hover:border-gray-700 transition-colors"
            title="Refresh Ingested Datasets"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${scenesLoading ? "animate-spin text-blue-400" : ""}`} />
            <span>Refresh Archives</span>
          </button>
        </div>
      </div>

      {/* Hero Natural Language Search Bar */}
      <div className="max-w-7xl mx-auto mb-8">
        <div className="bg-gray-900/90 border border-gray-800 rounded-2xl p-6 shadow-2xl backdrop-blur-md">
          <form onSubmit={handleSearchSubmit} className="space-y-4">
            <div className="flex flex-col md:flex-row gap-3">
              <div className="relative flex-1">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                  <Search className="h-5 w-5 text-blue-400" />
                </div>
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Ask in natural language: e.g. 'Airport runway expansion', 'Solar farm in Rajasthan', 'New construction near water'..."
                  className="w-full bg-gray-950/80 border border-gray-700/80 rounded-xl pl-11 pr-10 py-3.5 text-base text-gray-100 placeholder-gray-500 focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all"
                />
                {query && (
                  <button
                    type="button"
                    onClick={() => setQuery("")}
                    className="absolute inset-y-0 right-0 pr-3.5 flex items-center text-xs text-gray-400 hover:text-red-400"
                  >
                    Clear
                  </button>
                )}
              </div>

              <button
                type="submit"
                disabled={loading}
                className="flex items-center justify-center gap-2 px-6 py-3.5 bg-blue-600 hover:bg-blue-500 disabled:bg-blue-800/60 text-white text-sm font-semibold rounded-xl shadow-lg shadow-blue-600/20 transition-all cursor-pointer whitespace-nowrap"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Analyzing Archive...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    <span>Analyze Changes</span>
                  </>
                )}
              </button>
            </div>

            {/* Quick-Click Semantic Query Chips */}
            <div className="flex flex-wrap items-center gap-2 pt-1">
              <span className="text-xs font-medium text-gray-400 mr-1 flex items-center gap-1">
                <Sparkles className="w-3 h-3 text-blue-400" /> Demo queries:
              </span>
              {semanticTags.map((tag) => (
                <button
                  key={tag.label}
                  type="button"
                  onClick={() => handleTagClick(tag.text)}
                  className={`text-xs px-3 py-1.5 rounded-lg border transition-all text-left flex items-center gap-1.5 ${
                    query === tag.text
                      ? "bg-blue-600/20 border-blue-500/60 text-blue-300 font-medium"
                      : "bg-gray-950/60 hover:bg-gray-800 border-gray-800 text-gray-300 hover:text-white"
                  }`}
                >
                  <span>{tag.label}</span>
                </button>
              ))}
            </div>

            {/* Collapsible Advanced Filters (Optional - zero coordinates required) */}
            <div className="pt-2 border-t border-gray-800/80">
              <button
                type="button"
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="flex items-center gap-2 text-xs font-medium text-gray-400 hover:text-gray-200 transition-colors"
              >
                <Filter className="w-3.5 h-3.5 text-blue-400" />
                <span>Advanced Geographic & Date Filters</span>
                <span className="text-[11px] text-gray-500 font-normal">(Optional)</span>
                {showAdvanced ? (
                  <ChevronDown className="w-3.5 h-3.5 text-gray-400" />
                ) : (
                  <ChevronRight className="w-3.5 h-3.5 text-gray-400" />
                )}
              </button>

              {showAdvanced && (
                <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4 p-4 rounded-xl bg-gray-950/60 border border-gray-800/80 animate-in fade-in duration-200">
                  {/* Region Filter */}
                  <div>
                    <label className="text-xs font-medium text-gray-300 block mb-1.5">
                      Target Dataset / Region
                    </label>
                    <select
                      value={selectedDatasetId}
                      onChange={(e) => handleDatasetChange(e.target.value)}
                      className="w-full bg-gray-900 border border-gray-800 rounded-lg px-3 py-2 text-xs text-gray-200 focus:outline-none focus:border-blue-500"
                    >
                      <option value="ALL">🌐 Whole Archive (Compare Any Overlapping Scenes)</option>
                      {scenes.map((s) => (
                        <option key={s.id} value={s.id}>
                          📍 {s.filename} ({s.acquisition_date ? s.acquisition_date.slice(0, 10) : "No Date"})
                        </option>
                      ))}
                      <option value="CUSTOM">✏️ Custom Polygon Area (WKT Input)</option>
                    </select>
                  </div>

                  {/* Date Range */}
                  <div>
                    <label className="text-xs font-medium text-gray-300 block mb-1.5">
                      Observation Time Frame
                    </label>
                    <div className="grid grid-cols-2 gap-2">
                      <input
                        type="date"
                        value={startDate}
                        onChange={(e) => setStartDate(e.target.value)}
                        className="w-full bg-gray-900 border border-gray-800 rounded-lg px-2.5 py-1.5 text-xs text-gray-200 focus:outline-none focus:border-blue-500"
                      />
                      <input
                        type="date"
                        value={endDate}
                        onChange={(e) => setEndDate(e.target.value)}
                        className="w-full bg-gray-900 border border-gray-800 rounded-lg px-2.5 py-1.5 text-xs text-gray-200 focus:outline-none focus:border-blue-500"
                      />
                    </div>
                  </div>

                  {/* Candidate Limit */}
                  <div>
                    <div className="flex justify-between items-center text-xs mb-1.5">
                      <span className="text-gray-300 font-medium">Candidate Limit</span>
                      <span className="text-blue-400 font-mono font-semibold">{limit} pairs</span>
                    </div>
                    <input
                      type="range"
                      min="5"
                      max="100"
                      step="5"
                      value={limit}
                      onChange={(e) => setLimit(Number(e.target.value))}
                      className="w-full accent-blue-500 cursor-pointer mt-2"
                    />
                  </div>

                  {/* Custom WKT Polygon (if selected) */}
                  {selectedDatasetId === "CUSTOM" && (
                    <div className="md:col-span-3 mt-2">
                      <label className="text-xs font-medium text-gray-300 block mb-1.5">
                        Area of Interest WKT Polygon
                      </label>
                      <input
                        type="text"
                        value={aoiWkt}
                        onChange={(e) => setAoiWkt(e.target.value)}
                        placeholder="POLYGON((west south, east south, east north, west north, west south))"
                        className="w-full bg-gray-900 border border-gray-800 rounded-lg px-3 py-2 text-xs font-mono text-gray-200 focus:outline-none focus:border-blue-500"
                      />
                    </div>
                  )}
                </div>
              )}
            </div>
          </form>
        </div>
      </div>

      {/* Results Section */}
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Results Header */}
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-semibold text-white flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-blue-400" />
              Detected Change Candidates
            </h2>
            {results.length > 0 && (
              <span className="px-2.5 py-0.5 rounded-full bg-blue-500/20 border border-blue-500/40 text-blue-300 text-xs font-mono font-semibold">
                {results.length} pair{results.length === 1 ? "" : "s"} identified
              </span>
            )}
          </div>

          <div className="text-xs text-gray-400 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-green-400 inline-block animate-pulse"></span>
            <span>Real OpenCV ECC Registration + Multi-Modal Alignment</span>
          </div>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="p-4 rounded-xl bg-red-900/20 border border-red-800/60 text-red-300 text-sm flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold">Analysis Notice</p>
              <p className="mt-0.5 text-xs text-red-300/80">{error}</p>
            </div>
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 py-6">
            {[1, 2, 3].map((i) => (
              <div key={i} className="bg-gray-900/60 border border-gray-800 rounded-xl p-4 h-96 animate-pulse flex flex-col justify-between">
                <div className="flex gap-2">
                  <div className="flex-1 aspect-square bg-gray-800/80 rounded-lg"></div>
                  <div className="flex-1 aspect-square bg-gray-800/80 rounded-lg"></div>
                </div>
                <div className="space-y-3 mt-4">
                  <div className="h-4 bg-gray-800 rounded w-2/3"></div>
                  <div className="h-3 bg-gray-800/60 rounded w-full"></div>
                  <div className="h-8 bg-gray-800/40 rounded-lg"></div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Results Grid */}
        {!loading && results.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {results.map((event) => (
              <ChangeCandidateCard key={event.id} event={event} />
            ))}
          </div>
        )}

        {/* Empty State */}
        {!loading && results.length === 0 && !error && (
          <div className="text-center py-16 bg-gray-900/40 border border-gray-800/60 rounded-2xl p-8">
            <Globe className="w-12 h-12 text-gray-600 mx-auto mb-3" />
            <h3 className="text-base font-semibold text-gray-300">Ready to Detect Ground Changes</h3>
            <p className="text-xs text-gray-500 max-w-md mx-auto mt-1 mb-6">
              Enter any query above or select a demo tag like &quot;Airport runway expansion&quot; or &quot;Solar farm installation&quot; to inspect real multi-temporal changes.
            </p>
          </div>
        )}

        {/* Recent Archive Events Section */}
        {recentEvents.length > 0 && (
          <div className="mt-12 pt-8 border-t border-gray-800/80">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
                <Database className="w-4 h-4 text-gray-400" />
                Archived Multi-Temporal Events in Database
              </h3>
              <span className="text-xs text-gray-500 font-mono">
                {recentEvents.length} persisted records
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {recentEvents.slice(0, 4).map((ev) => (
                <div key={ev.id} className="bg-gray-900/60 border border-gray-800 rounded-xl p-3.5 hover:border-blue-500/40 transition-colors flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="text-[10px] font-semibold text-blue-400 uppercase tracking-wider">
                        {ev.change_type?.replace(/_/g, " ") || "Change"}
                      </span>
                      <span className="text-[10px] font-mono text-gray-400">
                        {Math.round((ev.final_confidence || 0) * 100)}% conf
                      </span>
                    </div>
                    <p className="text-xs text-gray-300 line-clamp-2">
                      {ev.evidence?.explanation || "Physical surface shift detected."}
                    </p>
                  </div>
                  <div className="text-[10px] text-gray-500 mt-2 flex justify-between">
                    <span>Before: {ev.before_date?.slice(0, 10) || "T1"}</span>
                    <span>After: {ev.after_date?.slice(0, 10) || "T2"}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
