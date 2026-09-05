"use client";

import { useEffect, useRef, useState, useMemo } from "react";
import { type SceneSummary, type TileSummary } from "@/lib/api";
import { formatCoord, qualityColor } from "@/lib/utils";
import {
  Layers,
  Maximize2,
  Minimize2,
  RotateCcw,
  Compass,
  Crosshair,
  Info,
} from "lucide-react";

interface SpatialCoverageMapProps {
  scene: SceneSummary;
  tiles: TileSummary[];
  selectedTileId?: string | null;
  onTileSelect?: (tile: TileSummary) => void;
}

export function SpatialCoverageMap({
  scene,
  tiles,
  selectedTileId,
  onTileSelect,
}: SpatialCoverageMapProps) {
  const [hoveredTile, setHoveredTile] = useState<TileSummary | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Geographic extents
  const bounds = useMemo(() => {
    const w = scene.bbox_west ?? 77.0;
    const s = scene.bbox_south ?? 28.0;
    const e = scene.bbox_east ?? 77.5;
    const n = scene.bbox_north ?? 28.5;
    return {
      west: Math.min(w, e),
      south: Math.min(s, n),
      east: Math.max(w, e),
      north: Math.max(s, n),
      widthDeg: Math.abs(e - w) || 0.01,
      heightDeg: Math.abs(n - s) || 0.01,
    };
  }, [scene]);

  // Coordinate projection helper: converts lon/lat to SVG percentage (0-100%)
  const projectToPercent = (lon: number, lat: number) => {
    const xPct = ((lon - bounds.west) / bounds.widthDeg) * 100;
    const yPct = ((bounds.north - lat) / bounds.heightDeg) * 100;
    return { x: Math.max(0, Math.min(100, xPct)), y: Math.max(0, Math.min(100, yPct)) };
  };

  return (
    <div
      ref={containerRef}
      className="relative w-full h-[400px] md:h-[480px] bg-gray-950 rounded-2xl border border-gray-800 overflow-hidden flex flex-col select-none shadow-2xl"
    >
      {/* Top Map Header / Overlay Controls */}
      <div className="absolute top-3 left-3 right-3 z-10 flex items-center justify-between pointer-events-none">
        <div className="flex items-center gap-2 bg-gray-900/80 backdrop-blur-md px-3 py-1.5 rounded-xl border border-gray-800 pointer-events-auto shadow-lg">
          <Layers className="h-4 w-4 text-blue-400" />
          <span className="text-xs font-semibold text-white">
            Spatial Coverage Grid
          </span>
          <span className="text-[10px] text-gray-400 font-mono bg-gray-800/80 px-1.5 py-0.5 rounded">
            {tiles.length} tiles
          </span>
        </div>

        <div className="flex items-center gap-1 bg-gray-900/80 backdrop-blur-md px-2.5 py-1 rounded-xl border border-gray-800 pointer-events-auto text-[11px] text-gray-400 font-mono">
          <Compass className="h-3.5 w-3.5 text-blue-400 mr-1" />
          <span>EPSG:4326</span>
        </div>
      </div>

      {/* Main Coordinate Grid & Tiles Display */}
      <div className="relative flex-1 w-full h-full p-8 flex items-center justify-center overflow-hidden">
        {/* Subtle Background Coordinate Crosshair Grid */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#1f2937_1px,transparent_1px),linear-gradient(to_bottom,#1f2937_1px,transparent_1px)] bg-[size:32px_32px] opacity-25" />

        {/* Scene Footprint Canvas Container */}
        <div className="relative w-full h-full max-w-2xl max-h-[360px] border-2 border-dashed border-blue-500/40 rounded-xl bg-blue-950/10 p-2 transition-all">
          {/* Scene Bounds Label */}
          <div className="absolute -top-3 left-4 bg-blue-950 border border-blue-800/80 text-[10px] text-blue-300 font-mono px-2 py-0.5 rounded shadow">
            Scene Extent [{formatCoord(bounds.west)}, {formatCoord(bounds.south)}] → [{formatCoord(bounds.east)}, {formatCoord(bounds.north)}]
          </div>

          {/* Tiles Layer */}
          <div className="relative w-full h-full">
            {tiles.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center text-gray-500 text-xs">
                <Crosshair className="h-8 w-8 mb-2 text-gray-600 animate-pulse" />
                <p>No analysis tiles generated yet.</p>
                <p className="text-[11px] text-gray-600 mt-0.5">
                  Click "Process Scene" to generate 512×512 tiles.
                </p>
              </div>
            ) : (
              tiles.map((tile) => {
                if (
                  tile.bbox_west == null ||
                  tile.bbox_south == null ||
                  tile.bbox_east == null ||
                  tile.bbox_north == null
                ) {
                  return null;
                }

                // Project tile bounds to relative percentages inside the scene
                const topLeft = projectToPercent(tile.bbox_west, tile.bbox_north);
                const bottomRight = projectToPercent(tile.bbox_east, tile.bbox_south);
                const widthPct = Math.max(2, bottomRight.x - topLeft.x);
                const heightPct = Math.max(2, bottomRight.y - topLeft.y);

                const isSelected = selectedTileId === tile.id;
                const isHovered = hoveredTile?.id === tile.id;

                return (
                  <div
                    key={tile.id}
                    onClick={() => onTileSelect && onTileSelect(tile)}
                    onMouseEnter={() => setHoveredTile(tile)}
                    onMouseLeave={() => setHoveredTile(null)}
                    style={{
                      left: `${topLeft.x}%`,
                      top: `${topLeft.y}%`,
                      width: `${widthPct}%`,
                      height: `${heightPct}%`,
                    }}
                    className={`absolute cursor-pointer transition-all duration-150 rounded border ${
                      isSelected
                        ? "bg-blue-500/40 border-blue-400 ring-2 ring-blue-400/80 z-20"
                        : isHovered
                        ? "bg-blue-600/30 border-blue-300 z-10 scale-[1.02]"
                        : "bg-gray-800/40 border-gray-700/80 hover:bg-gray-700/50"
                    }`}
                  >
                    {/* Tile Label */}
                    <div className="absolute inset-0 flex items-center justify-center p-1 overflow-hidden pointer-events-none">
                      <span className="text-[10px] font-mono font-medium text-gray-300 truncate opacity-80">
                        {tile.tile_col},{tile.tile_row}
                      </span>
                    </div>

                    {/* Quality Dot Indicator */}
                    <div
                      className={`absolute top-1 right-1 h-1.5 w-1.5 rounded-full ${
                        tile.quality_score != null && tile.quality_score >= 0.8
                          ? "bg-emerald-400"
                          : tile.quality_score != null && tile.quality_score >= 0.5
                          ? "bg-amber-400"
                          : "bg-red-400"
                      }`}
                    />
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>

      {/* Bottom Status / Hover Info Overlay */}
      <div className="px-4 py-2.5 bg-gray-950/80 border-t border-gray-800 flex items-center justify-between text-xs text-gray-400">
        <div className="flex items-center gap-3">
          {hoveredTile ? (
            <div className="flex items-center gap-2 font-mono text-[11px] text-gray-200">
              <span className="text-blue-400 font-semibold">
                Tile [{hoveredTile.tile_col}, {hoveredTile.tile_row}]
              </span>
              <span>•</span>
              <span>
                Center: [{formatCoord(hoveredTile.center_lat)}, {formatCoord(hoveredTile.center_lon)}]
              </span>
              <span>•</span>
              <span className={qualityColor(hoveredTile.quality_score)}>
                Quality: {hoveredTile.quality_score != null ? `${(hoveredTile.quality_score * 100).toFixed(0)}%` : "—"}
              </span>
            </div>
          ) : (
            <span className="text-gray-500 text-[11px]">
              Hover or click on any tile to inspect bounds and high-resolution imagery preview.
            </span>
          )}
        </div>

        <div className="flex items-center gap-3 text-[11px]">
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-emerald-400 inline-block" />
            <span className="text-gray-500">≥80%</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-amber-400 inline-block" />
            <span className="text-gray-500">50–79%</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-red-400 inline-block" />
            <span className="text-gray-500">&lt;50%</span>
          </div>
        </div>
      </div>
    </div>
  );
}
