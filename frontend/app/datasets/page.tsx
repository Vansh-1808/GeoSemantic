"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, type SceneSummary, type TileSummary } from "@/lib/api";
import {
  formatDate,
  formatBytes,
  statusBadgeClass,
  qualityColor,
  formatCoord,
} from "@/lib/utils";
import {
  Satellite,
  Layers,
  Settings2,
  Clock,
  RefreshCw,
  Search,
  Eye,
  Sliders,
  Sparkles,
  ChevronRight,
  Database,
  Grid,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { ProcessSceneDialog } from "@/components/scenes/ProcessSceneDialog";
import { TilePreviewModal } from "@/components/tiles/TilePreviewModal";
import { SpatialCoverageMap } from "@/components/map/SpatialCoverageMap";

export default function DatasetsPage() {
  const [selectedSceneId, setSelectedSceneId] = useState<string | null>(null);
  const [inspectTileId, setInspectTileId] = useState<string | null>(null);
  const [isProcessDialogOpen, setIsProcessDialogOpen] = useState<boolean>(false);
  const [tileFilter, setTileFilter] = useState<"all" | "high-quality">("all");

  // ── Fetch scenes ─────────────────────────────────────────────
  const {
    data: scenesData,
    isLoading: isScenesLoading,
    refetch: refetchScenes,
  } = useQuery({
    queryKey: ["scenes-list"],
    queryFn: () => api.listScenes(1, 50),
    refetchInterval: 15_000,
  });

  const scenes = scenesData?.scenes ?? [];
  // Set default selected scene once loaded
  const activeScene =
    scenes.find((s) => s.id === selectedSceneId) || scenes[0] || null;

  // ── Fetch tiles for active scene ─────────────────────────────
  const {
    data: tiles = [],
    isLoading: isTilesLoading,
    refetch: refetchTiles,
  } = useQuery({
    queryKey: ["tiles-for-scene", activeScene?.id],
    queryFn: () => (activeScene ? api.getTilesForScene(activeScene.id, 300) : []),
    enabled: !!activeScene,
  });

  const filteredTiles = tiles.filter((t) => {
    if (tileFilter === "high-quality") {
      return (t.quality_score ?? 0) >= 0.7;
    }
    return true;
  });

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2.5">
            <Database className="h-6 w-6 text-blue-400" />
            Scene & Tile Explorer
          </h1>
          <p className="text-gray-400 text-sm mt-1">
            Explore satellite scenes, configure windowed tiling, inspect spatial coverage, and preview analysis tiles.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => {
              refetchScenes();
              refetchTiles();
            }}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-900 hover:bg-gray-800 border border-gray-800 hover:border-gray-700 rounded-xl text-xs text-gray-300 transition-colors shadow-sm"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Refresh
          </button>
          {activeScene && (
            <button
              id="btn-process-active-scene"
              onClick={() => setIsProcessDialogOpen(true)}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-xl text-xs font-semibold text-white shadow-lg shadow-blue-600/20 transition-all"
            >
              <Settings2 className="h-4 w-4" />
              Process Scene
            </button>
          )}
        </div>
      </div>

      {isScenesLoading ? (
        <div className="py-24 text-center text-gray-500 text-sm">
          Loading satellite scenes...
        </div>
      ) : scenes.length === 0 ? (
        <div className="text-center py-20 bg-gray-900/60 border border-gray-800 rounded-2xl p-8 max-w-md mx-auto">
          <Satellite className="h-12 w-12 text-gray-600 mx-auto mb-3" />
          <h3 className="text-base font-semibold text-white">No Scenes Ingested</h3>
          <p className="text-gray-400 text-xs mt-1 mb-4">
            Upload a GeoTIFF in the Ingestion tab to begin tiling and semantic analysis.
          </p>
          <a
            href="/ingest"
            className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-xl text-xs font-semibold text-white transition-all shadow-md"
          >
            Go to Ingestion
          </a>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left Column: Scene Selector Cards */}
          <div className="lg:col-span-4 space-y-3">
            <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider px-1">
              Ingested Scenes ({scenes.length})
            </h2>

            <div className="space-y-2 max-h-[680px] overflow-y-auto pr-1">
              {scenes.map((s) => {
                const isSelected = (activeScene?.id ?? "") === s.id;
                return (
                  <motion.div
                    key={s.id}
                    layout
                    onClick={() => setSelectedSceneId(s.id)}
                    className={`p-4 rounded-xl border cursor-pointer transition-all ${
                      isSelected
                        ? "bg-blue-950/20 border-blue-500/70 shadow-lg shadow-blue-950/40 ring-1 ring-blue-500/30"
                        : "bg-gray-900 border-gray-800/80 hover:border-gray-700 hover:bg-gray-900/80"
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-semibold text-white truncate">
                          {s.filename}
                        </p>
                        <p className="text-xs text-gray-500 mt-0.5">
                          {s.sensor ?? "Satellite Scene"} • {formatDate(s.acquisition_date)}
                        </p>
                      </div>
                      <span
                        className={`text-[10px] px-2 py-0.5 rounded font-mono font-medium border ${statusBadgeClass(
                          s.ingestion_status
                        )}`}
                      >
                        {s.ingestion_status}
                      </span>
                    </div>

                    <div className="flex items-center gap-4 mt-3 pt-3 border-t border-gray-800/60 text-[11px] text-gray-400">
                      <span className="flex items-center gap-1.5 font-mono">
                        <Layers className="h-3.5 w-3.5 text-blue-400" />
                        {s.tile_count} tiles
                      </span>
                      {s.quality_score != null && (
                        <span className={`font-medium ${qualityColor(s.quality_score)}`}>
                          Quality: {(s.quality_score * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </div>

          {/* Right Column: Active Scene Details, Spatial Coverage & Tile Grid */}
          <div className="lg:col-span-8 space-y-6">
            {activeScene && (
              <>
                {/* Active Scene Summary Banner */}
                <div className="p-5 bg-gray-900 border border-gray-800 rounded-2xl shadow-xl flex flex-col md:flex-row md:items-center justify-between gap-4">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-semibold text-blue-400 uppercase tracking-wider">
                        Active Scene
                      </span>
                      <span className="text-gray-600">•</span>
                      <span className="text-xs text-gray-400 font-mono">
                        {activeScene.id}
                      </span>
                    </div>
                    <h2 className="text-lg font-bold text-white">
                      {activeScene.filename}
                    </h2>
                    <div className="flex items-center gap-4 text-xs text-gray-400 pt-1">
                      <span>🛰️ {activeScene.sensor ?? "Optical"}</span>
                      <span>📅 {formatDate(activeScene.acquisition_date)}</span>
                      <span>🧩 {tiles.length} Analysis Tiles</span>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setIsProcessDialogOpen(true)}
                      className="px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-xl text-xs font-semibold border border-gray-700 flex items-center gap-2 transition-all shadow-sm"
                    >
                      <Settings2 className="h-4 w-4 text-blue-400" />
                      Configure Tiling
                    </button>
                  </div>
                </div>

                {/* Spatial Coverage Map Component */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between px-1">
                    <h3 className="text-xs font-semibold text-gray-300 uppercase tracking-wider">
                      Geographic Extent & Coverage
                    </h3>
                    <span className="text-xs text-gray-500 font-mono">
                      Center: [{formatCoord(((activeScene.bbox_west ?? 0) + (activeScene.bbox_east ?? 0)) / 2)},{" "}
                      {formatCoord(((activeScene.bbox_south ?? 0) + (activeScene.bbox_north ?? 0)) / 2)}]
                    </span>
                  </div>

                  <SpatialCoverageMap
                    scene={activeScene}
                    tiles={tiles}
                    selectedTileId={inspectTileId}
                    onTileSelect={(t) => setInspectTileId(t.id)}
                  />
                </div>

                {/* Tile Preview Grid */}
                <div className="space-y-3 pt-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Grid className="h-4 w-4 text-blue-400" />
                      <h3 className="text-xs font-semibold text-gray-300 uppercase tracking-wider">
                        Generated Tiles ({filteredTiles.length})
                      </h3>
                    </div>

                    <div className="flex items-center gap-1 bg-gray-950 p-1 rounded-xl border border-gray-800 text-xs">
                      <button
                        onClick={() => setTileFilter("all")}
                        className={`px-3 py-1 rounded-lg transition-all ${
                          tileFilter === "all"
                            ? "bg-blue-600 text-white font-medium"
                            : "text-gray-400 hover:text-white"
                        }`}
                      >
                        All ({tiles.length})
                      </button>
                      <button
                        onClick={() => setTileFilter("high-quality")}
                        className={`px-3 py-1 rounded-lg transition-all ${
                          tileFilter === "high-quality"
                            ? "bg-blue-600 text-white font-medium"
                            : "text-gray-400 hover:text-white"
                        }`}
                      >
                        High Quality (≥70%)
                      </button>
                    </div>
                  </div>

                  {isTilesLoading ? (
                    <div className="text-center py-16 text-gray-500 text-xs">
                      Loading tiles...
                    </div>
                  ) : filteredTiles.length === 0 ? (
                    <div className="text-center py-16 bg-gray-900 border border-gray-800 rounded-2xl p-6 text-gray-500 text-xs">
                      No tiles match the selected filter.
                    </div>
                  ) : (
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
                      {filteredTiles.map((t) => {
                        const previewUrl = t.preview_url || api.getPreviewUrl(t.id);
                        const thumbUrl = t.thumbnail_url || api.getThumbnailUrl(t.id);

                        return (
                          <motion.div
                            key={t.id}
                            whileHover={{ y: -2 }}
                            onClick={() => setInspectTileId(t.id)}
                            className="bg-gray-900 border border-gray-800 hover:border-blue-500/60 rounded-xl overflow-hidden cursor-pointer group shadow-sm transition-all flex flex-col"
                          >
                            {/* Thumbnail / Preview Image */}
                            <div className="relative aspect-square w-full bg-black flex items-center justify-center overflow-hidden">
                              <img
                                src={previewUrl}
                                alt={`Tile ${t.tile_col}, ${t.tile_row}`}
                                onError={(e) => {
                                  (e.target as HTMLImageElement).src = thumbUrl;
                                }}
                                className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                              />
                              <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                                <span className="p-2 bg-blue-600 rounded-lg text-white text-xs flex items-center gap-1 shadow-lg">
                                  <Eye className="h-3.5 w-3.5" />
                                  Inspect
                                </span>
                              </div>
                              <span className="absolute top-1.5 left-1.5 text-[10px] font-mono bg-black/70 backdrop-blur-sm text-gray-300 px-1.5 py-0.5 rounded border border-white/10">
                                {t.tile_col},{t.tile_row}
                              </span>
                            </div>

                            {/* Tile Footer */}
                            <div className="p-2.5 space-y-1 text-[11px] bg-gray-900">
                              <div className="flex items-center justify-between">
                                <span className="text-gray-400 font-mono text-[10px]">
                                  {t.tile_size ?? 512}×{t.tile_size ?? 512}px
                                </span>
                                {t.quality_score != null && (
                                  <span
                                    className={`font-semibold text-[10px] ${qualityColor(
                                      t.quality_score
                                    )}`}
                                  >
                                    {(t.quality_score * 100).toFixed(0)}%
                                  </span>
                                )}
                              </div>
                              <div className="text-[10px] text-gray-500 font-mono truncate">
                                [{formatCoord(t.center_lat, 3)}, {formatCoord(t.center_lon, 3)}]
                              </div>
                            </div>
                          </motion.div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* Process Scene Modal */}
      {activeScene && (
        <ProcessSceneDialog
          scene={activeScene}
          isOpen={isProcessDialogOpen}
          onClose={() => setIsProcessDialogOpen(false)}
          onSuccess={() => {
            refetchScenes();
            refetchTiles();
          }}
        />
      )}

      {/* Tile Inspection Modal */}
      <TilePreviewModal
        tileId={inspectTileId}
        isOpen={!!inspectTileId}
        onClose={() => setInspectTileId(null)}
      />
    </div>
  );
}
