"use client";

import React from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import {
  ShieldCheck,
  ShieldAlert,
  ShieldX,
  Cloud,
  Moon,
  Ban,
  Radio,
  Sun,
  AlertTriangle,
  RotateCw,
  Info,
  CheckCircle2,
  XCircle,
} from "lucide-react";
import { api, type TileQuality } from "@/lib/api";
import { QualityBadge } from "./QualityBadge";

interface TileQualityPanelProps {
  tileId: string;
}

export function TileQualityPanel({ tileId }: TileQualityPanelProps) {
  const queryClient = useQueryClient();

  const {
    data: quality,
    isLoading,
    isError,
    refetch,
  } = useQuery<TileQuality>({
    queryKey: ["tile-quality", tileId],
    queryFn: () => api.getQualityForTile(tileId),
    staleTime: 60_000,
  });

  const computeMutation = useMutation({
    mutationFn: () => api.computeQualityForTile(tileId),
    onSuccess: (updated) => {
      queryClient.setQueryData(["tile-quality", tileId], updated);
      queryClient.invalidateQueries({ queryKey: ["tile-detail", tileId] });
      queryClient.invalidateQueries({ queryKey: ["tiles"] });
    },
  });

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center p-12 text-center">
        <RotateCw className="h-8 w-8 text-blue-400 animate-spin mb-3" />
        <p className="text-sm text-gray-300 font-medium">Analyzing optical quality...</p>
        <p className="text-xs text-gray-500 mt-1">
          Evaluating cloud, shadow, sensor noise, and boundary nodata
        </p>
      </div>
    );
  }

  if (isError || !quality) {
    return (
      <div className="p-8 text-center bg-rose-950/20 border border-rose-800/40 rounded-xl m-4">
        <AlertTriangle className="h-8 w-8 text-rose-400 mx-auto mb-2" />
        <p className="text-sm font-semibold text-rose-300">Quality Assessment Failed</p>
        <p className="text-xs text-gray-400 mt-1 mb-4">
          Could not compute photometric quality for tile {tileId}.
        </p>
        <button
          onClick={() => refetch()}
          className="px-3 py-1.5 text-xs bg-rose-900/60 hover:bg-rose-800 text-white rounded-lg border border-rose-700 transition-colors"
        >
          Retry Analysis
        </button>
      </div>
    );
  }

  const scorePct = Math.round(quality.quality_score * 100);
  const isSuitable = quality.suitability === "Suitable";
  const isCaution = quality.suitability === "Caution";
  const isPoor = quality.suitability === "Poor Quality";

  const factors = [
    {
      name: "Cloud / Dense Haze",
      score: quality.cloud_score,
      pct: `${quality.cloud_cover_pct.toFixed(1)}%`,
      icon: Cloud,
      description: "Occludes ground features; prevents optical change detection",
      color: quality.cloud_score > 0.2 ? "bg-rose-500" : quality.cloud_score > 0.05 ? "bg-amber-500" : "bg-emerald-500",
      penaltyLevel: quality.cloud_score > 0.2 ? "Severe" : quality.cloud_score > 0.05 ? "Moderate" : "Nominal",
    },
    {
      name: "Surface Shadows",
      score: quality.shadow_score,
      pct: `${(quality.shadow_score * 100).toFixed(1)}%`,
      icon: Moon,
      description: "Low-reflectance land pixels (dark ocean separated via NDWI)",
      color: quality.shadow_score > 0.2 ? "bg-amber-500" : "bg-emerald-500",
      penaltyLevel: quality.shadow_score > 0.2 ? "Moderate" : "Nominal",
    },
    {
      name: "No-Data / Invalid Pixels",
      score: quality.nodata_score,
      pct: `${(quality.nodata_ratio * 100).toFixed(1)}%`,
      icon: Ban,
      description: "Unmapped border margins or masked raster sensor pixels",
      color: quality.nodata_score > 0.15 ? "bg-rose-500" : quality.nodata_score > 0.02 ? "bg-amber-500" : "bg-emerald-500",
      penaltyLevel: quality.nodata_score > 0.15 ? "Severe" : quality.nodata_score > 0.02 ? "Moderate" : "Nominal",
    },
    {
      name: "Sensor Noise / Banding",
      score: quality.noise_score,
      pct: `${(quality.noise_score * 100).toFixed(1)}%`,
      icon: Radio,
      description: "High-frequency row-to-row sensor jitter & striping artifacts",
      color: quality.noise_score > 0.1 ? "bg-amber-500" : "bg-emerald-500",
      penaltyLevel: quality.noise_score > 0.1 ? "Noticeable" : "Clean",
    },
    {
      name: "Sensor Saturation / Burn",
      score: quality.saturation_score,
      pct: `${(quality.saturation_score * 100).toFixed(1)}%`,
      icon: Sun,
      description: "Blown-out overexposed land pixels exceeding detector range",
      color: quality.saturation_score > 0.05 ? "bg-amber-500" : "bg-emerald-500",
      penaltyLevel: quality.saturation_score > 0.05 ? "Moderate" : "Minimal",
    },
  ];

  return (
    <div className="p-6 space-y-6 overflow-y-auto max-h-[calc(90vh-140px)]">
      {/* Top Banner: Composite Score + Change Suitability */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Composite Score Card */}
        <div className="bg-gray-950/80 border border-gray-800 rounded-2xl p-5 flex flex-col items-center justify-center text-center relative overflow-hidden">
          <div className="text-xs uppercase tracking-wider font-semibold text-gray-400 mb-2">
            Composite Quality
          </div>
          <div className="relative flex items-center justify-center">
            <span
              className={`text-4xl font-extrabold font-mono tracking-tight ${
                isSuitable ? "text-emerald-400" : isCaution ? "text-amber-400" : "text-rose-400"
              }`}
            >
              {scorePct}%
            </span>
          </div>
          <div className="mt-2">
            <QualityBadge
              suitability={quality.suitability}
              qualityScore={quality.quality_score}
              size="sm"
            />
          </div>
          <div className="text-[11px] text-gray-500 mt-2 font-mono">
            Score: {quality.quality_score.toFixed(4)} / 1.0000
          </div>
        </div>

        {/* Change Suitability Verdict */}
        <div className="md:col-span-2 bg-gray-950/80 border border-gray-800 rounded-2xl p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold uppercase tracking-wider text-gray-400">
                Change Detection Suitability
              </span>
              <span
                className={`text-xs px-2.5 py-0.5 rounded-full font-semibold border flex items-center gap-1.5 ${
                  quality.usable_for_change_analysis
                    ? "bg-emerald-950/70 text-emerald-300 border-emerald-500/40"
                    : "bg-rose-950/70 text-rose-300 border-rose-500/40"
                }`}
              >
                {quality.usable_for_change_analysis ? (
                  <>
                    <CheckCircle2 className="h-3.5 w-3.5" />
                    Verified Usable
                  </>
                ) : (
                  <>
                    <XCircle className="h-3.5 w-3.5" />
                    Caution / Unusable
                  </>
                )}
              </span>
            </div>

            <p className="text-sm text-gray-200 leading-relaxed">
              {isSuitable
                ? "This tile possesses clean optical fidelity, minimal cloud occlusion, and intact raster coverage. It is fully qualified for automatic cross-sensor change detection without confounding false alarms."
                : isCaution
                ? "Confounding factors (haze, shadow, or minor boundary nodata) are detected. Change detection algorithms should apply high-threshold confidence filtering."
                : "Severe occlusion or missing data pixels detected. This tile should be excluded from automated change analysis pipelines to avoid corrupting analytics."}
            </p>
          </div>

          <div className="flex items-center justify-between pt-3 border-t border-gray-800/80 mt-3 text-xs text-gray-400">
            <span className="font-mono text-[11px]">
              Engine: <span className="text-gray-300">{quality.analysis_method}</span>
            </span>
            <button
              onClick={() => computeMutation.mutate()}
              disabled={computeMutation.isPending}
              className="inline-flex items-center gap-1.5 px-3 py-1 bg-gray-900 hover:bg-gray-800 text-gray-200 rounded-lg border border-gray-700 transition-colors text-xs font-medium disabled:opacity-50"
            >
              <RotateCw
                className={`h-3 w-3 ${computeMutation.isPending ? "animate-spin text-blue-400" : ""}`}
              />
              {computeMutation.isPending ? "Computing..." : "Recompute Quality"}
            </button>
          </div>
        </div>
      </div>

      {/* Factor Breakdown */}
      <div className="bg-gray-950/60 border border-gray-800 rounded-2xl p-5">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-4">
          Confounding Factors Breakdown
        </h4>

        <div className="space-y-4">
          {factors.map((f) => (
            <div key={f.name} className="space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-2">
                  <f.icon className="h-4 w-4 text-gray-400" />
                  <span className="font-medium text-gray-200">{f.name}</span>
                  <span className="text-[10px] text-gray-500 hidden sm:inline font-normal">
                    — {f.description}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[11px] px-1.5 py-0.2 rounded text-gray-400 bg-gray-900 border border-gray-800">
                    {f.penaltyLevel}
                  </span>
                  <span className="font-mono font-semibold text-gray-200 w-12 text-right">
                    {f.pct}
                  </span>
                </div>
              </div>

              {/* Progress Bar */}
              <div className="h-1.5 w-full bg-gray-900 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.min(100, Math.max(2, f.score * 100))}%` }}
                  transition={{ duration: 0.5, ease: "easeOut" }}
                  className={`h-full rounded-full ${f.color}`}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Scientific Transparency & Limitations */}
      <div className="bg-blue-950/20 border border-blue-900/40 rounded-2xl p-4 text-xs">
        <div className="flex items-center gap-2 font-semibold text-blue-300 mb-2">
          <Info className="h-4 w-4 text-blue-400 flex-shrink-0" />
          <span>Scientific Metadata & Assessment Limitations</span>
        </div>
        <div className="text-gray-300 space-y-1 pl-6 list-disc">
          {quality.caveats.map((c, idx) => (
            <p key={idx} className="leading-relaxed">
              {c}
            </p>
          ))}
          <p className="text-gray-400 pt-1">
            Satellite QA Mask Available:{" "}
            <span className="font-mono text-gray-300">
              {quality.satellite_mask_available ? "YES (SCL/QA60)" : "NO (Deterministic RGB Photometry)"}
            </span>
          </p>
        </div>
      </div>
    </div>
  );
}
