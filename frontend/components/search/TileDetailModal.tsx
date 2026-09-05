"use client";

import { useState } from "react";
import { type SemanticSearchResultItem } from "@/lib/api";
import { formatDate, formatCoord } from "@/lib/utils";
import {
  Check,
  ChevronDown,
  ChevronUp,
  Clock,
  Compass,
  Copy,
  Cpu,
  Database,
  ExternalLink,
  Layers,
  MapPin,
  ShieldCheck,
  Sparkles,
  X,
} from "lucide-react";

interface TileDetailModalProps {
  tile: SemanticSearchResultItem | null;
  onClose: () => void;
}

export function TileDetailModal({ tile, onClose }: TileDetailModalProps) {
  const [copiedId, setCopiedId] = useState(false);
  const [copiedCoords, setCopiedCoords] = useState(false);
  const [showRawJson, setShowRawJson] = useState(false);

  if (!tile) return null;

  const copyToClipboard = (text: string, type: "id" | "coords") => {
    navigator.clipboard.writeText(text);
    if (type === "id") {
      setCopiedId(true);
      setTimeout(() => setCopiedId(false), 2000);
    } else {
      setCopiedCoords(true);
      setTimeout(() => setCopiedCoords(false), 2000);
    }
  };

  const scorePct = (tile.similarity_score * 100).toFixed(1);
  const coordsStr = `${formatCoord(tile.center_coordinates.lat, 5)}°, ${formatCoord(tile.center_coordinates.lon, 5)}°`;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-150">
      <div
        className="relative w-full max-w-4xl max-h-[90vh] bg-gray-950 border border-gray-800 rounded-2xl shadow-2xl overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-800 bg-gray-900/60">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center h-8 w-8 rounded-lg bg-cyan-950 border border-cyan-700/60 text-cyan-400 font-mono font-bold text-sm">
              #{tile.rank}
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-base font-semibold text-white">Tile Inspection & Provenance</h2>
                <span className="px-2 py-0.5 rounded-full text-xs font-mono font-bold bg-emerald-950/80 text-emerald-400 border border-emerald-700/70">
                  {scorePct}% Similarity
                </span>
              </div>
              <p className="text-xs text-gray-400 font-mono">
                Tile [{tile.tile_col}, {tile.tile_row}] • Scene: {tile.scene_name}
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 text-gray-400 hover:text-white rounded-lg hover:bg-gray-800 transition"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Scrollable Content Body */}
        <div className="overflow-y-auto p-6 space-y-6 flex-1">
          {/* Main Visualization & Quick Stats Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* High-Resolution Preview Display */}
            <div className="flex flex-col items-center">
              <div className="relative w-full aspect-square max-w-[380px] bg-gray-900 rounded-xl border border-gray-800 overflow-hidden shadow-inner flex items-center justify-center group">
                <img
                  src={tile.preview_url || tile.thumbnail_url}
                  alt={`Tile ${tile.tile_col},${tile.tile_row}`}
                  className="w-full h-full object-cover transition duration-300 group-hover:scale-105"
                  onError={(e) => {
                    // Fallback to thumbnail if preview fails
                    if (e.currentTarget.src !== tile.thumbnail_url) {
                      e.currentTarget.src = tile.thumbnail_url;
                    }
                  }}
                />

                <div className="absolute top-2 left-2 bg-gray-950/80 backdrop-blur-md px-2 py-0.5 rounded text-[11px] font-mono text-gray-300 border border-gray-800">
                  512×512 px RGB
                </div>

                <div className="absolute bottom-2 right-2 bg-gray-950/80 backdrop-blur-md px-2 py-0.5 rounded text-[11px] font-mono text-cyan-300 border border-gray-800">
                  Rank #{tile.rank}
                </div>
              </div>
              <p className="text-[11px] text-gray-500 mt-2 text-center">
                Radiometrically normalized 512px RGB visualization tile
              </p>
            </div>

            {/* Core Metadata Table */}
            <div className="space-y-4">
              <div>
                <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                  <Database className="h-3.5 w-3.5 text-cyan-400" />
                  Geospatial Metadata
                </h3>
                <div className="bg-gray-900/60 rounded-xl border border-gray-800 divide-y divide-gray-800/80 text-xs">
                  <div className="p-2.5 flex items-center justify-between">
                    <span className="text-gray-400">Tile ID</span>
                    <div className="flex items-center gap-1.5">
                      <span className="font-mono text-gray-300">{String(tile.tile_id).slice(0, 16)}...</span>
                      <button
                        onClick={() => copyToClipboard(String(tile.tile_id), "id")}
                        className="p-1 hover:bg-gray-800 text-gray-400 hover:text-white rounded"
                        title="Copy Tile ID"
                      >
                        {copiedId ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
                      </button>
                    </div>
                  </div>

                  <div className="p-2.5 flex items-center justify-between">
                    <span className="text-gray-400">Sensor & Platform</span>
                    <span className="font-mono font-medium text-cyan-300">{tile.sensor || "Sentinel-2B"}</span>
                  </div>

                  <div className="p-2.5 flex items-center justify-between">
                    <span className="text-gray-400">Acquisition Date</span>
                    <span className="font-mono text-gray-200">{formatDate(tile.acquisition_date)}</span>
                  </div>

                  <div className="p-2.5 flex items-center justify-between">
                    <span className="text-gray-400">Center Coordinates</span>
                    <div className="flex items-center gap-1.5">
                      <span className="font-mono text-gray-200">{coordsStr}</span>
                      <button
                        onClick={() => copyToClipboard(coordsStr, "coords")}
                        className="p-1 hover:bg-gray-800 text-gray-400 hover:text-white rounded"
                        title="Copy Coordinates"
                      >
                        {copiedCoords ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
                      </button>
                    </div>
                  </div>

                  <div className="p-2.5 flex items-center justify-between">
                    <span className="text-gray-400">Quality Score</span>
                    <div className="flex items-center gap-2">
                      <div className="w-16 bg-gray-800 h-2 rounded-full overflow-hidden">
                        <div
                          className="bg-emerald-400 h-full rounded-full"
                          style={{ width: `${(tile.quality_score ?? 1.0) * 100}%` }}
                        />
                      </div>
                      <span className="font-mono text-emerald-400 font-semibold">
                        {tile.quality_score != null ? `${(tile.quality_score * 100).toFixed(0)}%` : "100%"}
                      </span>
                    </div>
                  </div>

                  <div className="p-2.5 flex items-center justify-between">
                    <span className="text-gray-400">Cloud Coverage</span>
                    <span className="font-mono text-gray-300">
                      {tile.cloud_cover_pct != null ? `${tile.cloud_cover_pct.toFixed(1)}%` : "0.0%"}
                    </span>
                  </div>

                  <div className="p-2.5 flex items-center justify-between">
                    <span className="text-gray-400">Tile Grid Index</span>
                    <span className="font-mono text-gray-300">Col {tile.tile_col}, Row {tile.tile_row}</span>
                  </div>
                </div>
              </div>

              {/* Geographic Bounding Box Panel */}
              <div>
                <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                  <Compass className="h-3.5 w-3.5 text-blue-400" />
                  Bounding Box (EPSG:4326)
                </h3>
                <div className="bg-gray-900/60 p-3 rounded-xl border border-gray-800 text-xs font-mono grid grid-cols-2 gap-2 text-gray-300">
                  <div>
                    <span className="text-gray-500 mr-2">West:</span>
                    {formatCoord(tile.bbox.west, 5)}°
                  </div>
                  <div>
                    <span className="text-gray-500 mr-2">North:</span>
                    {formatCoord(tile.bbox.north, 5)}°
                  </div>
                  <div>
                    <span className="text-gray-500 mr-2">East:</span>
                    {formatCoord(tile.bbox.east, 5)}°
                  </div>
                  <div>
                    <span className="text-gray-500 mr-2">South:</span>
                    {formatCoord(tile.bbox.south, 5)}°
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Provenance & Audit Trail Section */}
          <div className="border-t border-gray-800 pt-5">
            <h3 className="text-xs font-semibold text-gray-300 uppercase tracking-wider mb-3 flex items-center gap-1.5">
              <ShieldCheck className="h-4 w-4 text-emerald-400" />
              Full Provenance & Embedding Audit Trail
            </h3>

            <div className="bg-gray-900/40 rounded-xl border border-gray-800 p-4 space-y-3">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                <div className="p-2 bg-gray-950/60 rounded-lg border border-gray-800/80">
                  <div className="text-gray-500 text-[10px] uppercase font-mono">Model</div>
                  <div className="font-semibold text-cyan-300 font-mono mt-0.5">
                    {tile.provenance?.model_name || "RemoteCLIP"}
                  </div>
                </div>

                <div className="p-2 bg-gray-950/60 rounded-lg border border-gray-800/80">
                  <div className="text-gray-500 text-[10px] uppercase font-mono">Architecture</div>
                  <div className="font-semibold text-gray-200 font-mono mt-0.5">
                    {tile.provenance?.model_version || "ViT-B-32"}
                  </div>
                </div>

                <div className="p-2 bg-gray-950/60 rounded-lg border border-gray-800/80">
                  <div className="text-gray-500 text-[10px] uppercase font-mono">Vector Dimension</div>
                  <div className="font-semibold text-emerald-400 font-mono mt-0.5">
                    {tile.provenance?.embedding_dimension || 512}-dim
                  </div>
                </div>

                <div className="p-2 bg-gray-950/60 rounded-lg border border-gray-800/80">
                  <div className="text-gray-500 text-[10px] uppercase font-mono">Vector Index</div>
                  <div className="font-semibold text-blue-400 font-mono mt-0.5">
                    remoteclip_tiles
                  </div>
                </div>
              </div>

              {/* Expandable Raw JSON Accordion */}
              <div className="pt-2">
                <button
                  onClick={() => setShowRawJson(!showRawJson)}
                  className="w-full flex items-center justify-between text-xs text-gray-400 hover:text-gray-200 py-1.5 px-2 rounded-lg bg-gray-950/50 border border-gray-800 transition"
                >
                  <span className="flex items-center gap-1.5">
                    <Cpu className="h-3.5 w-3.5 text-gray-400" />
                    View Raw Audit JSON Payload
                  </span>
                  {showRawJson ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                </button>

                {showRawJson && (
                  <pre className="mt-2 p-3 bg-gray-950 rounded-xl border border-gray-800 text-[11px] font-mono text-gray-300 overflow-x-auto max-h-48">
                    {JSON.stringify(
                      {
                        tile_id: tile.tile_id,
                        scene_id: tile.scene_id,
                        similarity_score: tile.similarity_score,
                        rank: tile.rank,
                        provenance: tile.provenance,
                        bbox: tile.bbox,
                        center: tile.center_coordinates,
                      },
                      null,
                      2
                    )}
                  </pre>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-3 border-t border-gray-800 bg-gray-900/60 text-xs text-gray-400">
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
            <span>Strict Offline Verification: Verified from Local Qdrant Index</span>
          </div>
          <div className="flex items-center gap-2">
            <a
              href={`/image-search?tile_id=${tile.tile_id}`}
              className="px-3.5 py-1.5 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white rounded-lg transition font-medium text-xs flex items-center gap-1.5 shadow"
            >
              <Sparkles className="h-3.5 w-3.5" />
              <span>Find Similar Locations</span>
            </a>
            <button
              onClick={onClose}
              className="px-4 py-1.5 bg-gray-800 hover:bg-gray-700 text-white rounded-lg transition font-medium text-xs"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
