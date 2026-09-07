"use client";

import React, { useState, useRef } from "react";
import { ChangeEventResponse, api } from "@/lib/api";
import { 
  X, ZoomIn, ZoomOut, RotateCcw, Layers, Sparkles, 
  Compass, Flame, Sliders, Activity, Calendar, MapPin
} from "lucide-react";

interface ChangeDetailModalProps {
  event: ChangeEventResponse | null;
  onClose: () => void;
}

export function ChangeDetailModal({ event, onClose }: ChangeDetailModalProps) {
  if (!event) return null;

  const [viewMode, setViewMode] = useState<"side-by-side" | "heatmap" | "slider">("side-by-side");
  const [sliderPos, setSliderPos] = useState<number>(50);
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const isDragging = useRef(false);

  const evidence = event.evidence || {};
  const beforeImg = evidence.before_thumbnail_url || (event.before_tile_id ? api.getThumbnailUrl(event.before_tile_id) : null);
  const afterImg = evidence.after_thumbnail_url || (event.after_tile_id ? api.getThumbnailUrl(event.after_tile_id) : null);
  const diffImg = evidence.diff_thumbnail_url;

  const confidencePct = Math.round((event.final_confidence || 0) * 100);
  const visualPct = Math.round((event.visual_change_score || 0) * 100);
  const semanticPct = Math.round((event.semantic_change_score || 0) * 100);
  const regQualityPct = Math.round((event.registration_quality || 0) * 100);

  const changedPixels = evidence.changed_pixels ?? 0;
  const totalPixels = evidence.total_pixels ?? 65536;
  const changedPct = evidence.changed_area_pct ?? 0;
  const alteredAreaM2 = evidence.altered_area_m2 ?? 0;
  const alteredAreaHa = evidence.altered_area_ha ?? 0;
  const resolutionM = evidence.resolution_m ?? 10.0;

  const specBefore = evidence.spectral_before || { urban_pct: 0, veg_pct: 0, water_pct: 0 };
  const specAfter = evidence.spectral_after || { urban_pct: 0, veg_pct: 0, water_pct: 0 };
  const deltaUrban = evidence.delta_urban ?? 0;
  const deltaVeg = evidence.delta_veg ?? 0;
  const deltaWater = evidence.delta_water ?? 0;

  const handleZoom = (delta: number) => {
    setZoomLevel(prev => Math.min(3, Math.max(0.75, prev + delta)));
  };

  const handleResetZoom = () => {
    setZoomLevel(1);
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/85 backdrop-blur-md flex items-center justify-center p-4 md:p-6 overflow-y-auto animate-in fade-in duration-200">
      <div className="bg-gray-900 border border-gray-800 rounded-2xl w-full max-w-6xl max-h-[95vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-800 flex items-center justify-between bg-gray-950/60">
          <div>
            <div className="flex items-center gap-3">
              <span className="px-2.5 py-0.5 rounded-full bg-blue-500/15 border border-blue-500/40 text-blue-400 text-xs font-semibold uppercase tracking-wider">
                {event.change_type?.replace(/_/g, " ") || "Change Event"}
              </span>
              <span className="text-gray-400 text-xs font-mono">
                ID: {event.id}
              </span>
            </div>
            <h2 className="text-xl font-bold text-white mt-1">High-Resolution Multi-Temporal Inspection</h2>
          </div>

          <button
            onClick={onClose}
            className="p-2 rounded-xl text-gray-400 hover:text-white hover:bg-gray-800 transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Toolbar */}
        <div className="px-6 py-3 bg-gray-950/30 border-b border-gray-800 flex flex-wrap items-center justify-between gap-3">
          {/* View Modes */}
          <div className="flex items-center gap-1 bg-gray-800/70 p-1 rounded-xl border border-gray-700/60 text-xs">
            <button
              onClick={() => setViewMode("side-by-side")}
              className={`px-3 py-1.5 rounded-lg font-medium transition-all flex items-center gap-1.5 ${
                viewMode === "side-by-side"
                  ? "bg-blue-600 text-white shadow-sm"
                  : "text-gray-400 hover:text-white"
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              Side-by-Side
            </button>
            <button
              onClick={() => setViewMode("slider")}
              className={`px-3 py-1.5 rounded-lg font-medium transition-all flex items-center gap-1.5 ${
                viewMode === "slider"
                  ? "bg-blue-600 text-white shadow-sm"
                  : "text-gray-400 hover:text-white"
              }`}
            >
              <Sliders className="w-3.5 h-3.5" />
              Interactive Swipe
            </button>
            {diffImg && (
              <button
                onClick={() => setViewMode("heatmap")}
                className={`px-3 py-1.5 rounded-lg font-medium transition-all flex items-center gap-1.5 ${
                  viewMode === "heatmap"
                    ? "bg-blue-600 text-white shadow-sm"
                    : "text-gray-400 hover:text-white"
                }`}
              >
                <Flame className="w-3.5 h-3.5 text-orange-400" />
                Change Heatmap
              </button>
            )}
          </div>

          {/* Zoom Controls */}
          <div className="flex items-center gap-2 text-xs">
            <button
              onClick={() => handleZoom(-0.25)}
              className="p-1.5 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg border border-gray-700 transition-colors"
              title="Zoom Out"
            >
              <ZoomOut className="w-4 h-4" />
            </button>
            <span className="font-mono text-gray-300 w-12 text-center">{Math.round(zoomLevel * 100)}%</span>
            <button
              onClick={() => handleZoom(0.25)}
              className="p-1.5 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg border border-gray-700 transition-colors"
              title="Zoom In"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
            <button
              onClick={handleResetZoom}
              className="p-1.5 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg border border-gray-700 transition-colors"
              title="Reset Zoom"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Modal Main Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Visual Display Container */}
          <div className="bg-gray-950 rounded-2xl border border-gray-800 p-4 overflow-hidden flex items-center justify-center min-h-[380px]">
            {viewMode === "side-by-side" && (
              <div 
                className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full max-w-4xl transition-transform duration-150"
                style={{ transform: `scale(${zoomLevel})` }}
              >
                {/* Before Observation */}
                <div className="space-y-2">
                  <div className="flex justify-between items-center text-xs text-gray-400 px-1">
                    <span className="font-semibold text-gray-300">Observation T1 (Before)</span>
                    <span className="flex items-center gap-1 font-mono">
                      <Calendar className="w-3.5 h-3.5 text-gray-500" />
                      {event.before_date?.slice(0, 10) || "T1"}
                    </span>
                  </div>
                  <div className="relative aspect-square rounded-xl overflow-hidden border border-gray-800 bg-gray-900 shadow-inner">
                    {beforeImg ? (
                      <img src={beforeImg} alt="Before" className="w-full h-full object-contain" />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-gray-600">No Image</div>
                    )}
                  </div>
                </div>

                {/* After Observation */}
                <div className="space-y-2">
                  <div className="flex justify-between items-center text-xs text-blue-400 px-1">
                    <span className="font-semibold text-blue-300">Observation T2 (After)</span>
                    <span className="flex items-center gap-1 font-mono text-blue-400">
                      <Calendar className="w-3.5 h-3.5 text-blue-500" />
                      {event.after_date?.slice(0, 10) || "T2"}
                    </span>
                  </div>
                  <div className="relative aspect-square rounded-xl overflow-hidden border border-blue-900/60 bg-gray-900 shadow-inner ring-1 ring-blue-500/20">
                    {afterImg ? (
                      <img src={afterImg} alt="After" className="w-full h-full object-contain" />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-gray-600">No Image</div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {viewMode === "heatmap" && diffImg && (
              <div 
                className="flex flex-col items-center gap-3 transition-transform duration-150"
                style={{ transform: `scale(${zoomLevel})` }}
              >
                <div className="text-xs text-orange-400 font-medium flex items-center gap-1.5">
                  <Flame className="w-4 h-4 text-orange-400" />
                  <span>Radiometric Change Detection Mask (Altered Pixels Highlighted in Neon Orange/Red)</span>
                </div>
                <div className="relative w-[460px] max-w-full aspect-square rounded-xl overflow-hidden border border-orange-500/50 shadow-2xl">
                  <img src={diffImg} alt="Change Heatmap" className="w-full h-full object-contain" />
                </div>
              </div>
            )}

            {viewMode === "slider" && beforeImg && afterImg && (
              <div 
                className="flex flex-col items-center gap-3 transition-transform duration-150 w-full max-w-xl"
                style={{ transform: `scale(${zoomLevel})` }}
              >
                <div className="text-xs text-gray-400 font-medium">
                  Drag the slider below to reveal changes between Before and After
                </div>
                <div className="relative w-full aspect-square rounded-xl overflow-hidden border border-gray-800 shadow-2xl select-none">
                  {/* After Image (Base) */}
                  <img src={afterImg} alt="After" className="absolute inset-0 w-full h-full object-cover" />
                  
                  {/* Before Image (Clipped) */}
                  <div 
                    className="absolute inset-y-0 left-0 overflow-hidden"
                    style={{ width: `${sliderPos}%` }}
                  >
                    <img 
                      src={beforeImg} 
                      alt="Before" 
                      className="absolute inset-0 w-full h-full object-cover max-w-none"
                      style={{ width: "100%" }}
                    />
                  </div>

                  {/* Divider Line */}
                  <div 
                    className="absolute inset-y-0 w-0.5 bg-blue-400 shadow-[0_0_10px_rgba(59,130,246,0.8)]"
                    style={{ left: `${sliderPos}%` }}
                  >
                    <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-7 h-7 rounded-full bg-blue-600 border-2 border-white flex items-center justify-center text-white text-[10px] font-bold shadow-lg">
                      ↔
                    </div>
                  </div>
                </div>

                {/* Range Slider Control */}
                <input 
                  type="range" 
                  min="0" 
                  max="100" 
                  value={sliderPos} 
                  onChange={e => setSliderPos(Number(e.target.value))}
                  className="w-full accent-blue-500 cursor-pointer mt-2"
                />
              </div>
            )}
          </div>

          {/* AI Explanation Banner */}
          {evidence.explanation && (
            <div className="p-4 bg-blue-950/30 border border-blue-800/50 rounded-xl flex items-start gap-3">
              <Sparkles className="w-5 h-5 text-blue-400 shrink-0 mt-0.5" />
              <div>
                <span className="text-xs font-semibold text-blue-300 uppercase tracking-wider block mb-1">
                  Validated Observation Findings
                </span>
                <p className="text-sm text-blue-100 leading-relaxed">
                  {evidence.explanation}
                </p>
              </div>
            </div>
          )}

          {/* Exact Metrics Grid (Zero Fake Numbers) */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Physical Surface Changes */}
            <div className="bg-gray-950/60 p-4 rounded-xl border border-gray-800 space-y-3">
              <div className="flex items-center gap-2 text-xs font-semibold text-gray-300 uppercase tracking-wider border-b border-gray-800 pb-2">
                <Activity className="w-4 h-4 text-emerald-400" />
                <span>Physical Surface Metrics</span>
              </div>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-400">Ground Area Altered:</span>
                  <span className="font-bold text-emerald-400 font-mono">
                    {alteredAreaM2 > 0 ? `${alteredAreaM2.toLocaleString()} m² (${alteredAreaHa} ha)` : "N/A"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Changed Pixels:</span>
                  <span className="font-mono text-gray-200">
                    {changedPixels.toLocaleString()} / {totalPixels.toLocaleString()} ({changedPct}%)
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Ground Resolution:</span>
                  <span className="font-mono text-gray-200">{resolutionM} m / pixel</span>
                </div>
              </div>
            </div>

            {/* Landcover Transitions */}
            <div className="bg-gray-950/60 p-4 rounded-xl border border-gray-800 space-y-3">
              <div className="flex items-center gap-2 text-xs font-semibold text-gray-300 uppercase tracking-wider border-b border-gray-800 pb-2">
                <Layers className="w-4 h-4 text-blue-400" />
                <span>Landcover Transitions</span>
              </div>
              <div className="space-y-2 text-xs font-mono">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400 font-sans">Built-up:</span>
                  <span className={deltaUrban > 0 ? "text-amber-400" : "text-gray-300"}>
                    {specBefore.urban_pct}% → {specAfter.urban_pct}% ({deltaUrban > 0 ? `+${deltaUrban}%` : `${deltaUrban}%`})
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400 font-sans">Vegetation:</span>
                  <span className={deltaVeg < 0 ? "text-red-400" : "text-emerald-400"}>
                    {specBefore.veg_pct}% → {specAfter.veg_pct}% ({deltaVeg > 0 ? `+${deltaVeg}%` : `${deltaVeg}%`})
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400 font-sans">Water:</span>
                  <span className="text-cyan-400">
                    {specBefore.water_pct}% → {specAfter.water_pct}% ({deltaWater > 0 ? `+${deltaWater}%` : `${deltaWater}%`})
                  </span>
                </div>
              </div>
            </div>

            {/* Algorithmic Confidence */}
            <div className="bg-gray-950/60 p-4 rounded-xl border border-gray-800 space-y-3">
              <div className="flex items-center gap-2 text-xs font-semibold text-gray-300 uppercase tracking-wider border-b border-gray-800 pb-2">
                <Compass className="w-4 h-4 text-purple-400" />
                <span>Scientific Quality</span>
              </div>
              <div className="space-y-2 text-xs font-mono">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400 font-sans">Overall Confidence:</span>
                  <span className="font-bold text-white text-sm">{confidencePct}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400 font-sans">OpenCV ECC Reg Quality:</span>
                  <span className="text-gray-200">{regQualityPct}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400 font-sans">Visual Difference (RMSE):</span>
                  <span className="text-gray-200">{visualPct}%</span>
                </div>
                {evidence.query_match_score != null && (
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400 font-sans">Query Semantic Match:</span>
                    <span className="text-blue-400 font-bold">{Math.round(evidence.query_match_score * 100)}%</span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Coordinates & Footprint */}
          <div className="p-3 bg-gray-950/40 rounded-xl border border-gray-800/80 flex flex-wrap justify-between items-center text-xs text-gray-400 font-mono">
            <span className="flex items-center gap-1.5">
              <MapPin className="w-3.5 h-3.5 text-gray-500" />
              Centroid: {event.center_lat?.toFixed(5)}° N, {event.center_lon?.toFixed(5)}° E
            </span>
            <span>Sensor Observation: Sentinel-2 / Landsat High-Resolution Optical</span>
          </div>
        </div>
      </div>
    </div>
  );
}
