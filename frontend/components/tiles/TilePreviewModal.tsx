"use client";

import { useState } from "react";
import { type TileSummary, type TileDetail, api } from "@/lib/api";
import { formatCoord, qualityColor, formatDate } from "@/lib/utils";
import {
  X,
  ExternalLink,
  MapPin,
  Calendar,
  Layers,
  Sparkles,
  CloudOff,
  Crosshair,
  FileCode,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useQuery } from "@tanstack/react-query";
import { TileQualityPanel } from "@/components/quality/TileQualityPanel";
import { QualityBadge } from "@/components/quality/QualityBadge";

interface TilePreviewModalProps {
  tileId: string | null;
  isOpen: boolean;
  onClose: () => void;
}

export function TilePreviewModal({
  tileId,
  isOpen,
  onClose,
}: TilePreviewModalProps) {
  const [activeTab, setActiveTab] = useState<"preview" | "quality" | "metadata" | "history">("preview");

  const { data: tile, isLoading } = useQuery<TileDetail>({
    queryKey: ["tile-detail", tileId],
    queryFn: () => (tileId ? api.getTile(tileId) : Promise.reject("No tile id")),
    enabled: isOpen && !!tileId,
  });

  if (!isOpen || !tileId) return null;

  const previewUrl = tile?.preview_url || api.getPreviewUrl(tileId);
  const thumbUrl = tile?.thumbnail_url || api.getThumbnailUrl(tileId);

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md">
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 12 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 12 }}
          className="relative w-full max-w-4xl bg-gray-900 border border-gray-800 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]"
        >
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-gray-800 bg-gray-950/70">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-500/10 rounded-lg border border-blue-500/20 text-blue-400">
                <Layers className="h-5 w-5" />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                  <span>
                    Tile [{tile?.tile_col ?? "—"}, {tile?.tile_row ?? "—"}]
                  </span>
                  {tile?.quality_score != null && (
                    <QualityBadge
                      qualityScore={tile.quality_score}
                      showScore={true}
                      size="xs"
                    />
                  )}
                </h3>
                <p className="text-[11px] text-gray-500 font-mono">
                  ID: {tileId}
                </p>
              </div>
            </div>

            {/* Navigation Tabs */}
            <div className="flex items-center gap-1 bg-gray-950 p-1 rounded-xl border border-gray-800">
              {(["preview", "quality", "metadata", "history"] as const).map((tab) => (
                <button
                  key={tab}
                  type="button"
                  onClick={() => setActiveTab(tab)}
                  className={`px-3 py-1 text-xs font-medium rounded-lg capitalize transition-all ${
                    activeTab === tab
                      ? "bg-blue-600 text-white shadow-sm"
                      : "text-gray-400 hover:text-gray-200"
                  }`}
                >
                  {tab}
                </button>
              ))}
            </div>

            <button
              onClick={onClose}
              className="p-1.5 text-gray-400 hover:text-white rounded-lg hover:bg-gray-800 transition-colors"
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          {/* Content Area */}
          <div className="flex-1 overflow-y-auto p-6">
            {isLoading ? (
              <div className="flex items-center justify-center py-20 text-gray-500 text-xs">
                Loading tile metadata...
              </div>
            ) : !tile ? (
              <div className="text-center py-20 text-gray-500 text-xs">
                Tile not found.
              </div>
            ) : (
              <>
                {/* PREVIEW TAB */}
                {activeTab === "preview" && (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-start">
                    {/* Image View */}
                    <div className="md:col-span-2 space-y-3">
                      <div className="relative aspect-square w-full rounded-xl overflow-hidden bg-black border border-gray-800 flex items-center justify-center shadow-inner group">
                        <img
                          src={previewUrl}
                          alt={`Tile ${tile.tile_col}, ${tile.tile_row}`}
                          onError={(e) => {
                            // Fallback to thumbnail
                            (e.target as HTMLImageElement).src = thumbUrl;
                          }}
                          className="w-full h-full object-contain"
                        />
                        <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                          <a
                            href={previewUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="p-1.5 bg-black/60 hover:bg-black/90 text-white rounded-lg text-xs flex items-center gap-1 border border-white/10"
                          >
                            <ExternalLink className="h-3.5 w-3.5" />
                            Full Res
                          </a>
                        </div>
                      </div>
                      <div className="flex items-center justify-between text-[11px] text-gray-500">
                        <span>
                          Dimensions: {tile.pixel_width ?? tile.tile_size ?? 512} ×{" "}
                          {tile.pixel_height ?? tile.tile_size ?? 512} px
                        </span>
                        <span>RGB 2%–98% Percentile Stretch Visualization</span>
                      </div>
                    </div>

                    {/* Quick Stats Panel */}
                    <div className="space-y-4 bg-gray-950/60 p-4 rounded-xl border border-gray-800/80 text-xs">
                      <div>
                        <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider block mb-2">
                          Geographic Center
                        </span>
                        <div className="flex items-center gap-2 text-gray-200 font-mono bg-gray-900/80 p-2.5 rounded-lg border border-gray-800">
                          <Crosshair className="h-4 w-4 text-blue-400 shrink-0" />
                          <span>
                            {formatCoord(tile.center_lat)},{" "}
                            {formatCoord(tile.center_lon)}
                          </span>
                        </div>
                      </div>

                      <div>
                        <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider block mb-2">
                          Source Coordinates (px)
                        </span>
                        <div className="space-y-1 bg-gray-900/80 p-2.5 rounded-lg border border-gray-800 font-mono text-[11px] text-gray-300">
                          <div className="flex justify-between">
                            <span className="text-gray-500">Col Offset (X):</span>
                            <span>{tile.pixel_x_off ?? "—"}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-500">Row Offset (Y):</span>
                            <span>{tile.pixel_y_off ?? "—"}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-500">Tile Size:</span>
                            <span>{tile.tile_size ?? 512} px</span>
                          </div>
                        </div>
                      </div>

                      <div>
                        <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider block mb-2">
                          Quality Indicators
                        </span>
                        <div className="space-y-1.5 bg-gray-900/80 p-2.5 rounded-lg border border-gray-800 text-[11px]">
                          <div className="flex justify-between items-center">
                            <span className="text-gray-400">Quality Score:</span>
                            <span className={`font-semibold ${qualityColor(tile.quality_score)}`}>
                              {tile.quality_score != null
                                ? (tile.quality_score * 100).toFixed(1) + "%"
                                : "—"}
                            </span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-gray-400">Cloud Estimate:</span>
                            <span className="text-gray-300 font-mono">
                              {tile.cloud_cover_pct != null
                                ? `${tile.cloud_cover_pct.toFixed(1)}%`
                                : "0.0%"}
                            </span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-gray-400">No-data Ratio:</span>
                            <span className="text-gray-300 font-mono">
                              {tile.nodata_ratio != null
                                ? `${(tile.nodata_ratio * 100).toFixed(1)}%`
                                : "0.0%"}
                            </span>
                          </div>
                        </div>
                      </div>

                      {tile.tile_path && (
                        <div className="pt-2">
                          <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider block mb-1">
                            GeoTIFF Tile Path
                          </span>
                          <p className="text-[10px] text-gray-500 font-mono truncate">
                            {tile.tile_path}
                          </p>
                        </div>
                      )}

                      <a
                        href={`/image-search?tile_id=${tile.id}`}
                        className="w-full mt-3 py-2.5 px-4 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white rounded-xl text-xs font-semibold flex items-center justify-center gap-2 transition shadow-lg shadow-indigo-950/50"
                      >
                        <Sparkles className="h-4 w-4" />
                        Find Similar Locations
                      </a>
                    </div>
                  </div>
                )}

                {/* QUALITY TAB */}
                {activeTab === "quality" && (
                  <TileQualityPanel tileId={tileId} />
                )}

                {/* METADATA TAB */}
                {activeTab === "metadata" && (
                  <div className="space-y-4 text-xs">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="bg-gray-950 p-4 rounded-xl border border-gray-800 space-y-2">
                        <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider block">
                          Spatial Bounding Box (WGS84)
                        </span>
                        <div className="space-y-1 font-mono text-[11px] text-gray-300">
                          <div className="flex justify-between">
                            <span className="text-gray-500">West (Min Lon):</span>
                            <span>{formatCoord(tile.bbox_west, 6)}°</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-500">South (Min Lat):</span>
                            <span>{formatCoord(tile.bbox_south, 6)}°</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-500">East (Max Lon):</span>
                            <span>{formatCoord(tile.bbox_east, 6)}°</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-500">North (Max Lat):</span>
                            <span>{formatCoord(tile.bbox_north, 6)}°</span>
                          </div>
                        </div>
                      </div>

                      <div className="bg-gray-950 p-4 rounded-xl border border-gray-800 space-y-2">
                        <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider block">
                          Sensor & Provenance
                        </span>
                        <div className="space-y-1 text-[11px] text-gray-300">
                          <div className="flex justify-between">
                            <span className="text-gray-500">Sensor:</span>
                            <span>{tile.sensor ?? "Unknown"}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-500">Acquisition:</span>
                            <span>{formatDate(tile.acquisition_date)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-500">Parent Scene ID:</span>
                            <span className="font-mono text-[10px]">
                              {tile.scene_id}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-500">Created:</span>
                            <span>{new Date(tile.created_at).toLocaleString()}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* HISTORY TAB */}
                {activeTab === "history" && (
                  <div className="space-y-3 text-xs">
                    <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider block">
                      Tile Processing Lineage
                    </span>
                    {tile.processing_history && tile.processing_history.length > 0 ? (
                      <div className="space-y-2">
                        {tile.processing_history.map((item, idx) => (
                          <div
                            key={idx}
                            className="bg-gray-950 p-3 rounded-xl border border-gray-800 font-mono text-[11px] text-gray-300"
                          >
                            <pre className="whitespace-pre-wrap text-gray-400">
                              {JSON.stringify(item, null, 2)}
                            </pre>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="p-4 bg-gray-950 rounded-xl border border-gray-800 text-gray-500">
                        Default Phase 3 ingestion pipeline.
                      </div>
                    )}
                  </div>
                )}
              </>
            )}
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
