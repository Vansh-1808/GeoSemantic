"use client";

import React from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  ShieldCheck,
  ShieldAlert,
  ShieldX,
  RotateCw,
  Cloud,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  BarChart3,
  Layers,
} from "lucide-react";
import { api, type SceneQuality } from "@/lib/api";
import { QualityBadge } from "./QualityBadge";

interface SceneQualityWidgetProps {
  sceneId: string;
  onQualityUpdated?: () => void;
}

export function SceneQualityWidget({ sceneId, onQualityUpdated }: SceneQualityWidgetProps) {
  const queryClient = useQueryClient();

  const {
    data: quality,
    isLoading,
    isError,
    refetch,
  } = useQuery<SceneQuality>({
    queryKey: ["scene-quality", sceneId],
    queryFn: () => api.getQualityForScene(sceneId),
    enabled: !!sceneId,
    staleTime: 60_000,
  });

  const computeMutation = useMutation({
    mutationFn: () => api.computeQualityForScene(sceneId),
    onSuccess: (data) => {
      queryClient.setQueryData(["scene-quality", sceneId], data.summary);
      queryClient.invalidateQueries({ queryKey: ["scene-detail", sceneId] });
      queryClient.invalidateQueries({ queryKey: ["scenes"] });
      queryClient.invalidateQueries({ queryKey: ["tiles", sceneId] });
      if (onQualityUpdated) onQualityUpdated();
    },
  });

  if (isLoading) {
    return (
      <div className="bg-gray-900/60 border border-gray-800 rounded-xl p-4 flex items-center justify-center gap-3 text-xs text-gray-400">
        <RotateCw className="h-4 w-4 animate-spin text-blue-400" />
        <span>Loading scene quality intelligence...</span>
      </div>
    );
  }

  if (isError || !quality) {
    return (
      <div className="bg-gray-900/60 border border-gray-800 rounded-xl p-4 flex items-center justify-between">
        <div className="flex items-center gap-2 text-xs text-gray-400">
          <AlertTriangle className="h-4 w-4 text-amber-400" />
          <span>Quality metrics uncomputed for this scene</span>
        </div>
        <button
          onClick={() => computeMutation.mutate()}
          disabled={computeMutation.isPending}
          className="px-3 py-1 bg-blue-600/20 hover:bg-blue-600/30 text-blue-400 text-xs font-medium rounded-lg border border-blue-500/30 transition-colors flex items-center gap-1.5"
        >
          <RotateCw className={`h-3 w-3 ${computeMutation.isPending ? "animate-spin" : ""}`} />
          {computeMutation.isPending ? "Assessing..." : "Compute Quality"}
        </button>
      </div>
    );
  }

  const scorePct = Math.round(quality.quality_score * 100);
  const totalTiles = quality.tile_count || 1;
  const suitablePct = Math.round((quality.tiles_suitable / totalTiles) * 100);
  const cautionPct = Math.round((quality.tiles_caution / totalTiles) * 100);
  const poorPct = Math.round((quality.tiles_poor / totalTiles) * 100);

  return (
    <div className="bg-gray-900/80 border border-gray-800/80 rounded-xl p-4 space-y-3">
      {/* Top row: Header, Badge, and Action button */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2.5">
          <div className="p-1.5 bg-blue-500/10 rounded-lg border border-blue-500/20 text-blue-400">
            <BarChart3 className="h-4 w-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold text-white uppercase tracking-wider">
                Scene Quality Layer
              </span>
              <QualityBadge
                suitability={quality.suitability}
                qualityScore={quality.quality_score}
                showScore={true}
                size="xs"
              />
            </div>
            <p className="text-[11px] text-gray-400">
              {quality.usable_for_change_analysis
                ? "Qualified for automated cross-temporal change detection"
                : "Caution: Confounding factors require analyst oversight"}
            </p>
          </div>
        </div>

        <button
          onClick={() => computeMutation.mutate()}
          disabled={computeMutation.isPending}
          className="inline-flex items-center gap-1.5 px-3 py-1 bg-gray-800 hover:bg-gray-700 text-gray-200 text-xs font-medium rounded-lg border border-gray-700 transition-colors disabled:opacity-50"
          title="Recompute deterministic quality for all tiles in this scene"
        >
          <RotateCw
            className={`h-3 w-3 ${computeMutation.isPending ? "animate-spin text-blue-400" : ""}`}
          />
          {computeMutation.isPending ? "Assessing Tiles..." : "Assess Scene"}
        </button>
      </div>

      {/* Distribution Progress Bar */}
      <div className="space-y-1">
        <div className="flex items-center justify-between text-[11px] text-gray-400">
          <span>Tile Quality Breakdown ({quality.tile_count} tiles)</span>
          <span className="font-mono text-gray-300">
            Avg Cloud: {quality.avg_cloud_cover_pct.toFixed(1)}% | Avg NoData:{" "}
            {(quality.avg_nodata_ratio * 100).toFixed(1)}%
          </span>
        </div>
        <div className="h-2 w-full bg-gray-950 rounded-full overflow-hidden flex border border-gray-800/80">
          <div
            style={{ width: `${suitablePct}%` }}
            className="bg-emerald-500 transition-all duration-500"
            title={`Suitable: ${quality.tiles_suitable} tiles (${suitablePct}%)`}
          />
          <div
            style={{ width: `${cautionPct}%` }}
            className="bg-amber-500 transition-all duration-500"
            title={`Caution: ${quality.tiles_caution} tiles (${cautionPct}%)`}
          />
          <div
            style={{ width: `${poorPct}%` }}
            className="bg-rose-500 transition-all duration-500"
            title={`Poor Quality: ${quality.tiles_poor} tiles (${poorPct}%)`}
          />
        </div>
      </div>

      {/* Counters pill bar */}
      <div className="grid grid-cols-3 gap-2 pt-1">
        <div className="bg-gray-950/60 border border-emerald-900/30 rounded-lg p-2 text-center">
          <div className="text-[10px] text-gray-400 font-medium flex items-center justify-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
            Suitable
          </div>
          <div className="text-sm font-bold font-mono text-emerald-400">
            {quality.tiles_suitable}{" "}
            <span className="text-[10px] font-normal text-gray-500">
              ({suitablePct}%)
            </span>
          </div>
        </div>

        <div className="bg-gray-950/60 border border-amber-900/30 rounded-lg p-2 text-center">
          <div className="text-[10px] text-gray-400 font-medium flex items-center justify-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-amber-400" />
            Caution
          </div>
          <div className="text-sm font-bold font-mono text-amber-400">
            {quality.tiles_caution}{" "}
            <span className="text-[10px] font-normal text-gray-500">
              ({cautionPct}%)
            </span>
          </div>
        </div>

        <div className="bg-gray-950/60 border border-rose-900/30 rounded-lg p-2 text-center">
          <div className="text-[10px] text-gray-400 font-medium flex items-center justify-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-rose-400" />
            Poor Quality
          </div>
          <div className="text-sm font-bold font-mono text-rose-400">
            {quality.tiles_poor}{" "}
            <span className="text-[10px] font-normal text-gray-500">
              ({poorPct}%)
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
