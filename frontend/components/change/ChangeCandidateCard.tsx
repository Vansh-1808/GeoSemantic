import React, { useState } from "react";
import { ChangeEventResponse } from "@/lib/api";
import { api } from "@/lib/api";
import { Sparkles, Compass, Layers, ArrowUpRight, Activity, Maximize2, Flame } from "lucide-react";
import { ChangeDetailModal } from "./ChangeDetailModal";

interface ChangeCandidateCardProps {
  event: ChangeEventResponse;
}

export function ChangeCandidateCard({ event }: ChangeCandidateCardProps) {
  const [imgErrorBefore, setImgErrorBefore] = useState(false);
  const [imgErrorAfter, setImgErrorAfter] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const evidence = event.evidence || {};
  const beforeImg = evidence.before_thumbnail_url || (event.before_tile_id ? api.getThumbnailUrl(event.before_tile_id) : null);
  const afterImg = evidence.after_thumbnail_url || (event.after_tile_id ? api.getThumbnailUrl(event.after_tile_id) : null);
  const diffImg = evidence.diff_thumbnail_url;

  const confidencePct = Math.round((event.final_confidence || 0) * 100);
  const visualPct = Math.round((event.visual_change_score || 0) * 100);
  const semanticPct = Math.round((event.semantic_change_score || 0) * 100);
  const regQualityPct = Math.round((event.registration_quality || 0) * 100);

  const queryMatchScore = evidence.query_match_score != null ? Math.round(evidence.query_match_score * 100) : null;
  const explanation = evidence.explanation;
  const deltaUrban = evidence.delta_urban;
  const deltaWater = evidence.delta_water;
  const deltaVeg = evidence.delta_veg;
  const alteredAreaHa = evidence.altered_area_ha;
  const changedPixels = evidence.changed_pixels;

  return (
    <>
      <div className="bg-gray-900/90 rounded-xl overflow-hidden border border-gray-800 hover:border-blue-500/50 transition-all duration-200 shadow-xl flex flex-col justify-between group/card">
        {/* Visual Header Comparison - Click to open High-Res Zoom */}
        <div>
          <div 
            onClick={() => setIsModalOpen(true)}
            className="flex gap-2 p-3 bg-gray-950 border-b border-gray-800 cursor-pointer relative group"
            title="Click to Zoom In & Compare High-Resolution Observations"
          >
            {/* Before Tile */}
            <div className="flex-1 relative aspect-square bg-gray-800 rounded-lg overflow-hidden border border-gray-800">
              {beforeImg && !imgErrorBefore ? (
                <img
                  src={beforeImg}
                  alt="Before Observation"
                  onError={() => setImgErrorBefore(true)}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                />
              ) : (
                <div className="w-full h-full flex flex-col items-center justify-center text-gray-500 text-xs p-2 text-center">
                  <Layers className="w-6 h-6 mb-1 opacity-50" />
                  <span>Before Tile</span>
                </div>
              )}
              <div className="absolute top-2 left-2 bg-black/75 backdrop-blur-sm text-gray-200 text-[10px] font-semibold tracking-wider uppercase px-2 py-0.5 rounded border border-gray-700">
                Before: {event.before_date?.slice(0, 10) || "T1"}
              </div>
            </div>

            {/* After Tile */}
            <div className="flex-1 relative aspect-square bg-gray-800 rounded-lg overflow-hidden border border-gray-800">
              {afterImg && !imgErrorAfter ? (
                <img
                  src={afterImg}
                  alt="After Observation"
                  onError={() => setImgErrorAfter(true)}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                />
              ) : (
                <div className="w-full h-full flex flex-col items-center justify-center text-gray-500 text-xs p-2 text-center">
                  <Layers className="w-6 h-6 mb-1 opacity-50" />
                  <span>After Tile</span>
                </div>
              )}
              <div className="absolute top-2 left-2 bg-blue-900/80 backdrop-blur-sm text-blue-200 text-[10px] font-semibold tracking-wider uppercase px-2 py-0.5 rounded border border-blue-700">
                After: {event.after_date?.slice(0, 10) || "T2"}
              </div>
            </div>

            {/* Hover overlay hint */}
            <div className="absolute inset-0 bg-blue-900/10 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity backdrop-blur-[1px]">
              <span className="bg-gray-900/90 text-white text-xs font-medium px-3 py-1.5 rounded-lg border border-blue-500/50 shadow-lg flex items-center gap-1.5">
                <Maximize2 className="w-3.5 h-3.5 text-blue-400" />
                Zoom & Compare
              </span>
            </div>
          </div>

        {/* Content Body */}
        <div className="p-4 space-y-3.5">
          {/* Change Type & Query Match Badge */}
          <div className="flex items-start justify-between gap-2">
            <div>
              <span className="text-[11px] font-medium text-gray-400 uppercase tracking-wider block">Classification</span>
              <span className="text-white font-semibold text-sm capitalize">
                {event.change_type?.replace(/_/g, " ") || "Detected Change"}
              </span>
            </div>

            {queryMatchScore !== null && (
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-blue-500/15 border border-blue-500/40 text-blue-400 text-xs font-bold shadow-sm">
                <Sparkles className="w-3.5 h-3.5" />
                <span>{queryMatchScore}% Match</span>
              </div>
            )}
          </div>

          {/* AI Explanation Banner */}
          {explanation && (
            <div className="p-2.5 bg-blue-950/25 border border-blue-900/40 rounded-lg text-xs text-blue-200/90 leading-relaxed flex items-start gap-2">
              <Sparkles className="w-4 h-4 text-blue-400 shrink-0 mt-0.5" />
              <span>{explanation}</span>
            </div>
          )}

          {/* Overall Confidence Bar */}
          <div>
            <div className="flex justify-between items-center text-xs mb-1.5">
              <span className="text-gray-400 flex items-center gap-1">
                <Activity className="w-3.5 h-3.5 text-blue-400" />
                Confidence Score
              </span>
              <span className="text-white font-bold font-mono">{confidencePct}%</span>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-2 overflow-hidden">
              <div
                className={`h-2 rounded-full transition-all duration-500 ${
                  confidencePct > 65
                    ? "bg-gradient-to-r from-emerald-500 to-green-400"
                    : confidencePct > 35
                    ? "bg-gradient-to-r from-blue-500 to-cyan-400"
                    : "bg-gray-600"
                }`}
                style={{ width: `${Math.max(5, confidencePct)}%` }}
              ></div>
            </div>
          </div>

          {/* Exact Observation Metrics (Area & Pixels) */}
          {(alteredAreaHa > 0 || (changedPixels != null && changedPixels > 0)) && (
            <div className="flex items-center justify-between text-[11px] px-2.5 py-1.5 bg-gray-950/70 rounded-lg border border-gray-800/80 font-mono">
              <span className="text-gray-400">Altered Surface:</span>
              <span className="text-emerald-400 font-bold">
                {alteredAreaHa > 0 ? `${alteredAreaHa} ha` : ""} {changedPixels != null ? `(${changedPixels.toLocaleString()} px)` : ""}
              </span>
            </div>
          )}

          {/* Spectral Landcover Shift Badges */}
          {(deltaUrban != null || deltaWater != null || deltaVeg != null) && (
            <div className="flex flex-wrap gap-1.5 pt-1">
              {deltaUrban != null && (
                <span className={`text-[11px] px-2 py-0.5 rounded font-medium ${
                  deltaUrban > 0 ? "bg-amber-950/50 text-amber-300 border border-amber-800/40" : "bg-gray-800 text-gray-400"
                }`}>
                  Built-up: {deltaUrban > 0 ? `+${deltaUrban}%` : `${deltaUrban}%`}
                </span>
              )}
              {deltaWater != null && (
                <span className={`text-[11px] px-2 py-0.5 rounded font-medium ${
                  Math.abs(deltaWater) > 1 ? "bg-blue-950/50 text-blue-300 border border-blue-800/40" : "bg-gray-800 text-gray-400"
                }`}>
                  Water: {deltaWater > 0 ? `+${deltaWater}%` : `${deltaWater}%`}
                </span>
              )}
              {deltaVeg != null && (
                <span className={`text-[11px] px-2 py-0.5 rounded font-medium ${
                  deltaVeg < 0 ? "bg-red-950/40 text-red-300 border border-red-800/40" : "bg-emerald-950/40 text-emerald-300 border border-emerald-800/40"
                }`}>
                  Veg: {deltaVeg > 0 ? `+${deltaVeg}%` : `${deltaVeg}%`}
                </span>
              )}
            </div>
          )}

          {/* Factor Breakdown Grid */}
          <div className="grid grid-cols-3 gap-2 pt-2.5 border-t border-gray-800 text-xs">
            <div className="bg-gray-800/40 p-2 rounded border border-gray-800">
              <div className="text-gray-400 text-[10px] uppercase font-medium">Visual Shift</div>
              <div className="text-gray-200 font-mono font-semibold mt-0.5">{visualPct}%</div>
            </div>
            <div className="bg-gray-800/40 p-2 rounded border border-gray-800">
              <div className="text-gray-400 text-[10px] uppercase font-medium">Semantic</div>
              <div className="text-gray-200 font-mono font-semibold mt-0.5">{semanticPct}%</div>
            </div>
            <div className="bg-gray-800/40 p-2 rounded border border-gray-800">
              <div className="text-gray-400 text-[10px] uppercase font-medium">Registration</div>
              <div className="text-gray-200 font-mono font-semibold mt-0.5">{regQualityPct}%</div>
            </div>
          </div>
        </div>
      </div>

      {/* Card Footer with Inspect Action */}
      <div className="px-4 py-2.5 bg-gray-950/80 border-t border-gray-800/80 flex justify-between items-center text-[11px] text-gray-400 font-mono">
        <span className="flex items-center gap-1">
          <Compass className="w-3 h-3 text-gray-500" />
          {event.center_lat?.toFixed(4)}, {event.center_lon?.toFixed(4)}
        </span>
        <button
          onClick={() => setIsModalOpen(true)}
          className="text-blue-400 hover:text-blue-300 font-sans font-medium flex items-center gap-1 transition-colors"
        >
          <Maximize2 className="w-3 h-3" />
          <span>Inspect</span>
        </button>
      </div>
    </div>

    {/* High-Resolution Zoom Preview Modal */}
    {isModalOpen && (
      <ChangeDetailModal event={event} onClose={() => setIsModalOpen(false)} />
    )}
  </>
  );
}
