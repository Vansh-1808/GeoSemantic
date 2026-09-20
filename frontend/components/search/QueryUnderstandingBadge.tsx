"use client";

import { useState } from "react";
import type { ParsedQuery } from "@/lib/api";
import {
  Sparkles,
  Search,
  TrendingUp,
  MapPin,
  Calendar,
  Cloud,
  Layers,
  Compass,
  Building2,
  Trees,
  Droplets,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
  Terminal,
  Wand2,
  Globe,
} from "lucide-react";

interface QueryUnderstandingBadgeProps {
  parsed: ParsedQuery;
  onAcceptSuggestion?: (suggestion: string) => void;
}

export function QueryUnderstandingBadge({ parsed, onAcceptSuggestion }: QueryUnderstandingBadgeProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const isChangeSearch = parsed.intent === "change_search";
  const hasInferredFilters =
    Boolean(parsed.start_date) ||
    Boolean(parsed.end_date) ||
    parsed.max_cloud_cover !== null && parsed.max_cloud_cover !== undefined ||
    Boolean(parsed.location_bbox);

  return (
    <div className="bg-gradient-to-r from-gray-900/90 via-slate-900/80 to-gray-900/90 border border-cyan-900/40 rounded-2xl p-4 shadow-xl backdrop-blur-md space-y-3 transition-all">
      {/* Header Row: Title, Intent Badge, Expand Button */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <div className="p-1.5 rounded-lg bg-cyan-950/80 border border-cyan-700/60 text-cyan-400">
            <Sparkles className="h-4 w-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold uppercase tracking-wider text-gray-200">
                Query &amp; Geographic Understanding
              </span>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-950 text-emerald-300 border border-emerald-800/80 font-mono">
                Phase 14
              </span>
            </div>
            <p className="text-[11px] text-gray-400">
              Offline spelling correction, gazetteer grounding &amp; PostGIS candidate filtering
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Intent Badge */}
          {isChangeSearch ? (
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-amber-950/80 border border-amber-600 text-amber-300 shadow-sm shadow-amber-950/50">
              <TrendingUp className="h-3.5 w-3.5 text-amber-400" />
              <span>Change Search</span>
            </span>
          ) : (
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-cyan-950/80 border border-cyan-600 text-cyan-300 shadow-sm shadow-cyan-950/50">
              <Search className="h-3.5 w-3.5 text-cyan-400" />
              <span>Semantic Search</span>
            </span>
          )}

          {/* Expand Details Toggle */}
          <button
            type="button"
            onClick={() => setIsExpanded(!isExpanded)}
            className="px-2.5 py-1 rounded-lg text-xs font-medium text-gray-300 hover:text-white bg-gray-800/80 hover:bg-gray-700/80 border border-gray-700/70 transition flex items-center gap-1"
          >
            <span>{isExpanded ? "Collapse" : "Details"}</span>
            {isExpanded ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
          </button>
        </div>
      </div>

      {/* Phase 13 "Did you mean: ..." Suggestion Banner */}
      {parsed.did_you_mean && (
        <div className="bg-gradient-to-r from-amber-950/60 to-purple-950/60 border border-amber-500/50 rounded-xl p-3 flex flex-wrap items-center justify-between gap-3 text-xs shadow-md">
          <div className="flex items-center gap-2.5">
            <div className="p-1 rounded-md bg-amber-500/20 text-amber-300">
              <Wand2 className="h-4 w-4" />
            </div>
            <div>
              <span className="text-white font-semibold">{parsed.did_you_mean}</span>
              <p className="text-[11px] text-amber-200/70">
                Spelling correction applied offline. Showing results for the corrected concept.
              </p>
            </div>
          </div>
          {onAcceptSuggestion && parsed.corrected_query && (
            <button
              type="button"
              onClick={() => onAcceptSuggestion(parsed.corrected_query || "")}
              className="px-3 py-1.5 bg-amber-500 hover:bg-amber-400 text-gray-950 font-bold rounded-lg text-xs shadow transition flex items-center gap-1"
            >
              <span>Search &quot;{parsed.corrected_query}&quot;</span>
            </button>
          )}
        </div>
      )}

      {/* Primary Extracted Chips Bar */}
      <div className="flex flex-wrap items-center gap-2 text-xs">
        {/* Spelling Correction Chips */}
        {parsed.corrections && parsed.corrections.map((corr, idx) => (
          <span
            key={`corr-${idx}`}
            className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-fuchsia-950/50 border border-fuchsia-700/70 text-fuchsia-200"
            title={`Damerau-Levenshtein distance: ${corr.edit_distance}, Confidence: ${Math.round(corr.confidence * 100)}%`}
          >
            <Wand2 className="h-3 w-3 text-fuchsia-400" />
            <span className="line-through text-fuchsia-400/80 font-mono">{corr.original_word}</span>
            <span className="text-gray-400">→</span>
            <span className="font-bold text-white font-mono">{corr.corrected_word}</span>
            <span className="text-[10px] font-mono text-fuchsia-300 opacity-80">({Math.round(corr.confidence * 100)}%)</span>
          </span>
        ))}

        {/* Concept / Target */}
        {parsed.target && (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-amber-950/40 border border-amber-800/60 text-amber-300">
            <span className="text-[10px] uppercase font-bold text-amber-500">Target:</span>
            <span className="font-medium">{parsed.target.replace(/_/g, " ")}</span>
          </span>
        )}

        {parsed.concept && (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-cyan-950/40 border border-cyan-800/60 text-cyan-300">
            <span className="text-[10px] uppercase font-bold text-cyan-500">Concept:</span>
            <span className="font-medium">{parsed.concept.replace(/_/g, " ")}</span>
          </span>
        )}

        {/* Spatial Relationship */}
        {parsed.relationship && (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-indigo-950/40 border border-indigo-800/60 text-indigo-300">
            <Compass className="h-3 w-3 text-indigo-400" />
            <span className="text-[10px] uppercase font-bold text-indigo-400">Rel:</span>
            <span className="font-medium">{parsed.relationship.replace(/_/g, " ")}</span>
          </span>
        )}

        {/* Location & Geographic Entity (Phase 14) */}
        {parsed.location && (
          <span
            className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-emerald-950/50 border border-emerald-700/70 text-emerald-200"
            title={
              parsed.location_bbox
                ? `PostGIS Spatial Bounds: [${parsed.location_bbox.join(", ")}]`
                : undefined
            }
          >
            <MapPin className="h-3 w-3 text-emerald-400 shrink-0" />
            {parsed.location_type && (
              <span className="text-[10px] uppercase font-bold px-1.5 py-0.5 rounded bg-emerald-900/80 text-emerald-300 border border-emerald-700/60 font-mono">
                {parsed.location_type === "state" && "🏛️ State"}
                {parsed.location_type === "city" && "🏙️ City"}
                {parsed.location_type === "country" && "🌍 Country"}
                {parsed.location_type === "region" && "🗺️ Region"}
                {!["state", "city", "country", "region"].includes(parsed.location_type) && parsed.location_type}
              </span>
            )}
            <span className="font-semibold text-white">{parsed.location}</span>
            {parsed.location_bbox && (
              <span className="text-[10px] text-emerald-400 font-mono bg-emerald-950/80 px-1.5 py-0.5 rounded border border-emerald-800/60">
                PostGIS BBox
              </span>
            )}
          </span>
        )}

        {/* Temporal Bounds */}
        {(parsed.start_date || parsed.end_date || parsed.time_anchor) && (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-purple-950/40 border border-purple-800/60 text-purple-300">
            <Calendar className="h-3 w-3 text-purple-400" />
            <span className="font-medium">
              {parsed.start_date && parsed.end_date
                ? `${new Date(parsed.start_date).getFullYear()} – ${new Date(parsed.end_date).getFullYear()}`
                : parsed.start_date
                ? `After ${new Date(parsed.start_date).getFullYear()}`
                : parsed.end_date
                ? `Before ${new Date(parsed.end_date).getFullYear()}`
                : parsed.time_anchor}
            </span>
          </span>
        )}

        {/* Cloud Cover Threshold */}
        {parsed.max_cloud_cover !== null && parsed.max_cloud_cover !== undefined && (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-sky-950/40 border border-sky-800/60 text-sky-300">
            <Cloud className="h-3 w-3 text-sky-400" />
            <span>Cloud ≤ {parsed.max_cloud_cover}%</span>
          </span>
        )}

        {/* Satellite Sensor Constraints */}
        {parsed.sensor_constraints && parsed.sensor_constraints.length > 0 && (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-violet-950/40 border border-violet-800/60 text-violet-300">
            <Layers className="h-3 w-3 text-violet-400" />
            <span>{parsed.sensor_constraints.join(", ")}</span>
          </span>
        )}

        {/* Landcover Indicators */}
        {parsed.requires_water && (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-blue-950/60 text-blue-300 border border-blue-800/60 text-[11px]">
            <Droplets className="h-2.5 w-2.5" /> Water
          </span>
        )}
        {parsed.requires_vegetation && (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-emerald-950/60 text-emerald-300 border border-emerald-800/60 text-[11px]">
            <Trees className="h-2.5 w-2.5" /> Vegetation
          </span>
        )}
        {parsed.requires_urban && (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-amber-950/60 text-amber-300 border border-amber-800/60 text-[11px]">
            <Building2 className="h-2.5 w-2.5" /> Built-up
          </span>
        )}
      </div>

      {/* Canonical Embedding String Callout */}
      <div className="bg-gray-950/80 border border-gray-800 rounded-xl px-3 py-2 flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
        <div className="flex items-center gap-2 min-w-0">
          <Terminal className="h-3.5 w-3.5 text-cyan-400 shrink-0" />
          <span className="text-gray-400 shrink-0">RemoteCLIP Prompt:</span>
          <span className="text-cyan-300 font-bold truncate">
            &quot;{parsed.canonical_embedding_text}&quot;
          </span>
        </div>
        <div className="flex items-center gap-2 text-[11px] text-gray-400 shrink-0">
          <span className="inline-flex items-center gap-1 text-emerald-400">
            <CheckCircle2 className="h-3 w-3" />
            <span>Geo Decoupled</span>
          </span>
          <span className="text-gray-600">|</span>
          <span className="text-gray-400">Prepositions Stripped</span>
        </div>
      </div>

      {/* Expanded Details Breakdown */}
      {isExpanded && (
        <div className="pt-2 border-t border-gray-800/80 grid grid-cols-1 md:grid-cols-2 gap-3 text-xs animate-in fade-in duration-200">
          {/* Normalization Compare */}
          <div className="space-y-1.5 bg-gray-950/60 border border-gray-800/60 rounded-xl p-3">
            <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider block">
              Query Normalization Pipeline
            </span>
            <div className="space-y-1 font-mono text-[11px]">
              <div className="flex items-start gap-2">
                <span className="text-gray-500 w-16 shrink-0">Raw:</span>
                <span className="text-gray-300 break-all">&quot;{parsed.raw_query}&quot;</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-cyan-500 w-16 shrink-0">Cleaned:</span>
                <span className="text-cyan-200 break-all">&quot;{parsed.normalized_query}&quot;</span>
              </div>
            </div>
          </div>

          {/* Inferred Auto-Filters */}
          <div className="space-y-1.5 bg-gray-950/60 border border-gray-800/60 rounded-xl p-3">
            <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider block">
              Automatically Injected Constraints
            </span>
            <div className="text-[11px] text-gray-300 space-y-1">
              {hasInferredFilters ? (
                <ul className="list-disc list-inside space-y-0.5 text-gray-300">
                  {parsed.location && parsed.location_bbox && (
                    <li>
                      Spatial BBox for <strong className="text-white">{parsed.location}</strong>:{" "}
                      <span className="font-mono text-[10px] text-emerald-300">
                        [{parsed.location_bbox.join(", ")}]
                      </span>
                    </li>
                  )}
                  {parsed.start_date && (
                    <li>
                      Start date filter: <strong className="text-white">{new Date(parsed.start_date).toLocaleDateString()}</strong>
                    </li>
                  )}
                  {parsed.end_date && (
                    <li>
                      End date filter: <strong className="text-white">{new Date(parsed.end_date).toLocaleDateString()}</strong>
                    </li>
                  )}
                  {parsed.max_cloud_cover !== null && parsed.max_cloud_cover !== undefined && (
                    <li>
                      Maximum cloud filter: <strong className="text-white">{parsed.max_cloud_cover}%</strong>
                    </li>
                  )}
                </ul>
              ) : (
                <span className="text-gray-500 italic">
                  No automated constraint filters injected; full catalog searched via semantic embedding.
                </span>
              )}
            </div>
          </div>

          {/* Spelling Corrections Detail */}
          {parsed.corrections && parsed.corrections.length > 0 && (
            <div className="md:col-span-2 space-y-1.5 bg-fuchsia-950/30 border border-fuchsia-800/40 rounded-xl p-3">
              <span className="text-[10px] font-semibold text-fuchsia-300 uppercase tracking-wider block">
                Offline Spelling Corrections Applied (Damerau-Levenshtein)
              </span>
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2">
                {parsed.corrections.map((corr, idx) => (
                  <div key={idx} className="bg-gray-950/60 rounded-lg p-2 border border-gray-800 text-[11px] flex justify-between items-center">
                    <div>
                      <span className="line-through text-gray-500 font-mono">{corr.original_word}</span>
                      <span className="mx-1 text-gray-400">→</span>
                      <strong className="text-white font-mono">{corr.corrected_word}</strong>
                    </div>
                    <div className="text-[10px] font-mono text-fuchsia-400">
                      dist: {corr.edit_distance} | {Math.round(corr.confidence * 100)}%
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Phase 14: Geographic Entity Understanding & Spatial Filtering */}
          {parsed.location && (
            <div className="md:col-span-2 space-y-1.5 bg-emerald-950/30 border border-emerald-800/40 rounded-xl p-3">
              <span className="text-[10px] font-semibold text-emerald-300 uppercase tracking-wider flex items-center gap-1.5">
                <Globe className="h-3.5 w-3.5 text-emerald-400" />
                <span>Geographic Entity Resolution &amp; Spatial Containment (Phase 14)</span>
              </span>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 pt-1 font-mono text-[11px]">
                <div className="bg-gray-950/70 rounded-lg p-2 border border-gray-800">
                  <span className="text-gray-500 text-[10px] block uppercase">Entity &amp; Admin Level</span>
                  <div className="flex items-center gap-1.5 mt-0.5">
                    <span className="text-white font-bold">{parsed.location}</span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-900/60 text-emerald-300 border border-emerald-700/40">
                      {parsed.location_type || "jurisdiction"}
                    </span>
                  </div>
                </div>
                <div className="bg-gray-950/70 rounded-lg p-2 border border-gray-800">
                  <span className="text-gray-500 text-[10px] block uppercase">PostGIS Bounding Box</span>
                  <span className="text-emerald-300 text-[10px] break-all block mt-0.5">
                    {parsed.location_bbox ? `[${parsed.location_bbox.join(", ")}]` : "N/A"}
                  </span>
                </div>
                <div className="bg-gray-950/70 rounded-lg p-2 border border-gray-800">
                  <span className="text-gray-500 text-[10px] block uppercase">Decoupled Execution</span>
                  <span className="text-gray-300 text-[10px] block mt-0.5">
                    Spatial SQL filter (<span className="text-cyan-300">ST_Intersects</span>) + Vector ranking inside area
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
