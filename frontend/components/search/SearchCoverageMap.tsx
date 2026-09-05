"use client";

import { useMemo, useState, useRef, useCallback } from "react";
import { type SemanticSearchResultItem } from "@/lib/api";
import { formatCoord } from "@/lib/utils";
import {
  Compass,
  Crosshair,
  Eye,
  Layers,
  MapPin,
  Sparkles,
  Square,
  Trash2,
  Check,
  MousePointer,
  Maximize2,
} from "lucide-react";

interface SearchCoverageMapProps {
  results: SemanticSearchResultItem[];
  selectedTileId: string | null;
  onSelectTile: (tile: SemanticSearchResultItem) => void;
  onInspectTile?: (tile: SemanticSearchResultItem) => void;
  aoiPolygon?: [number, number][] | null; // [[lon, lat], ...]
  onAoiPolygonChange?: (polygon: [number, number][] | null) => void;
  aoiBbox?: [number, number, number, number] | null; // [west, south, east, north]
  onAoiBboxChange?: (bbox: [number, number, number, number] | null) => void;
  spatialFilterMode?: "intersects" | "within";
  onSpatialFilterModeChange?: (mode: "intersects" | "within") => void;
}

type DrawingTool = "none" | "polygon" | "bbox";

export function SearchCoverageMap({
  results,
  selectedTileId,
  onSelectTile,
  onInspectTile,
  aoiPolygon,
  onAoiPolygonChange,
  aoiBbox,
  onAoiBboxChange,
  spatialFilterMode = "intersects",
  onSpatialFilterModeChange,
}: SearchCoverageMapProps) {
  const [hoveredTile, setHoveredTile] = useState<SemanticSearchResultItem | null>(null);
  const [activeTool, setActiveTool] = useState<DrawingTool>("none");

  // In-progress polygon vertices while drawing
  const [draftPoints, setDraftPoints] = useState<[number, number][]>([]);
  // Cursor coordinate tracking
  const [cursorGeo, setCursorGeo] = useState<{ lon: number; lat: number } | null>(null);
  // BBox dragging state
  const [dragStart, setDragStart] = useState<{ lon: number; lat: number } | null>(null);
  const [dragCurrent, setDragCurrent] = useState<{ lon: number; lat: number } | null>(null);

  const canvasRef = useRef<HTMLDivElement>(null);

  // Compute dynamic geographic envelope encompassing results, polygon, and bbox
  const bounds = useMemo(() => {
    let minLon = Infinity;
    let minLat = Infinity;
    let maxLon = -Infinity;
    let maxLat = -Infinity;

    const includePoint = (lon: number, lat: number) => {
      if (lon < minLon) minLon = lon;
      if (lat < minLat) minLat = lat;
      if (lon > maxLon) maxLon = lon;
      if (lat > maxLat) maxLat = lat;
    };

    // 1. Check tile results
    for (const r of results) {
      includePoint(r.bbox.west, r.bbox.south);
      includePoint(r.bbox.east, r.bbox.north);
    }

    // 2. Check AOI Polygon
    if (aoiPolygon && aoiPolygon.length > 0) {
      for (const [lon, lat] of aoiPolygon) {
        includePoint(lon, lat);
      }
    }

    // 3. Check Draft Points
    for (const [lon, lat] of draftPoints) {
      includePoint(lon, lat);
    }

    // 4. Check AOI BBox
    if (aoiBbox && aoiBbox.length === 4) {
      includePoint(aoiBbox[0], aoiBbox[1]);
      includePoint(aoiBbox[2], aoiBbox[3]);
    }

    // Fallback if no geometry present
    if (minLon === Infinity) {
      minLon = 77.05;
      minLat = 28.45;
      maxLon = 77.45;
      maxLat = 28.75;
    }

    const rawW = maxLon - minLon || 0.05;
    const rawH = maxLat - minLat || 0.05;
    const padW = rawW * 0.12;
    const padH = rawH * 0.12;

    return {
      west: minLon - padW,
      south: minLat - padH,
      east: maxLon + padW,
      north: maxLat + padH,
      widthDeg: rawW + padW * 2,
      heightDeg: rawH + padH * 2,
    };
  }, [results, aoiPolygon, draftPoints, aoiBbox]);

  // Project geographic lon/lat to percentage (0-100%)
  const projectToPercent = useCallback(
    (lon: number, lat: number) => {
      const xPct = ((lon - bounds.west) / bounds.widthDeg) * 100;
      const yPct = ((bounds.north - lat) / bounds.heightDeg) * 100;
      return {
        x: Math.max(0, Math.min(100, xPct)),
        y: Math.max(0, Math.min(100, yPct)),
      };
    },
    [bounds]
  );

  // Unproject client mouse coordinate to geographic lon/lat
  const unprojectFromClient = useCallback(
    (clientX: number, clientY: number) => {
      if (!canvasRef.current) return { lon: bounds.west, lat: bounds.north };
      const rect = canvasRef.current.getBoundingClientRect();
      const xRel = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
      const yRel = Math.max(0, Math.min(1, (clientY - rect.top) / rect.height));

      const lon = bounds.west + xRel * bounds.widthDeg;
      const lat = bounds.north - yRel * bounds.heightDeg;
      return { lon, lat };
    },
    [bounds]
  );

  // Handle canvas mouse move
  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const geo = unprojectFromClient(e.clientX, e.clientY);
    setCursorGeo(geo);

    if (activeTool === "bbox" && dragStart) {
      setDragCurrent(geo);
    }
  };

  // Handle canvas click (for polygon vertices)
  const handleCanvasClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (activeTool !== "polygon") return;
    const geo = unprojectFromClient(e.clientX, e.clientY);
    setDraftPoints((prev) => [...prev, [geo.lon, geo.lat]]);
  };

  // Handle bbox mousedown / mouseup
  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (activeTool !== "bbox") return;
    const geo = unprojectFromClient(e.clientX, e.clientY);
    setDragStart(geo);
    setDragCurrent(geo);
  };

  const handleMouseUp = () => {
    if (activeTool === "bbox" && dragStart && dragCurrent) {
      const w = Math.min(dragStart.lon, dragCurrent.lon);
      const e = Math.max(dragStart.lon, dragCurrent.lon);
      const s = Math.min(dragStart.lat, dragCurrent.lat);
      const n = Math.max(dragStart.lat, dragCurrent.lat);

      if (e - w > 0.001 && n - s > 0.001) {
        onAoiBboxChange?.([w, s, e, n]);
        // Also clear polygon when bbox is drawn
        onAoiPolygonChange?.(null);
      }
      setDragStart(null);
      setDragCurrent(null);
      setActiveTool("none");
    }
  };

  // Complete polygon drawing
  const finishPolygon = () => {
    if (draftPoints.length >= 3) {
      onAoiPolygonChange?.(draftPoints);
      // Clear bbox when polygon is drawn
      onAoiBboxChange?.(null);
    }
    setDraftPoints([]);
    setActiveTool("none");
  };

  const cancelDrawing = () => {
    setDraftPoints([]);
    setDragStart(null);
    setDragCurrent(null);
    setActiveTool("none");
  };

  const clearAllAoi = () => {
    setDraftPoints([]);
    setDragStart(null);
    setDragCurrent(null);
    onAoiPolygonChange?.(null);
    onAoiBboxChange?.(null);
    setActiveTool("none");
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.8) {
      return {
        bg: "bg-emerald-500/25",
        border: "border-emerald-400",
        ring: "ring-emerald-400",
        fill: "rgba(16, 185, 129, 0.35)",
        stroke: "#34d399",
      };
    }
    if (score >= 0.65) {
      return {
        bg: "bg-cyan-500/25",
        border: "border-cyan-400",
        ring: "ring-cyan-400",
        fill: "rgba(6, 182, 212, 0.35)",
        stroke: "#22d3ee",
      };
    }
    if (score >= 0.5) {
      return {
        bg: "bg-blue-500/20",
        border: "border-blue-400",
        ring: "ring-blue-400",
        fill: "rgba(59, 130, 246, 0.30)",
        stroke: "#60a5fa",
      };
    }
    return {
      bg: "bg-amber-500/20",
      border: "border-amber-400",
      ring: "ring-amber-400",
      fill: "rgba(245, 158, 11, 0.25)",
      stroke: "#fbbf24",
    };
  };

  // SVG points string for active polygon
  const activePolygonPoints = useMemo(() => {
    const pts = aoiPolygon && aoiPolygon.length >= 3 ? aoiPolygon : draftPoints;
    if (pts.length === 0) return "";
    return pts
      .map(([lon, lat]) => {
        const { x, y } = projectToPercent(lon, lat);
        return `${x},${y}`;
      })
      .join(" ");
  }, [aoiPolygon, draftPoints, projectToPercent]);

  // Compute bbox rectangle percentage coordinates
  const activeBboxRect = useMemo(() => {
    if (activeTool === "bbox" && dragStart && dragCurrent) {
      const w = Math.min(dragStart.lon, dragCurrent.lon);
      const e = Math.max(dragStart.lon, dragCurrent.lon);
      const s = Math.min(dragStart.lat, dragCurrent.lat);
      const n = Math.max(dragStart.lat, dragCurrent.lat);
      const tl = projectToPercent(w, n);
      const br = projectToPercent(e, s);
      return {
        x: tl.x,
        y: tl.y,
        width: Math.max(0.5, br.x - tl.x),
        height: Math.max(0.5, br.y - tl.y),
      };
    }

    if (aoiBbox && aoiBbox.length === 4) {
      const tl = projectToPercent(aoiBbox[0], aoiBbox[3]);
      const br = projectToPercent(aoiBbox[2], aoiBbox[1]);
      return {
        x: tl.x,
        y: tl.y,
        width: Math.max(0.5, br.x - tl.x),
        height: Math.max(0.5, br.y - tl.y),
      };
    }

    return null;
  }, [activeTool, dragStart, dragCurrent, aoiBbox, projectToPercent]);

  const hasAoiFilter = (aoiPolygon && aoiPolygon.length >= 3) || (aoiBbox && aoiBbox.length === 4);

  return (
    <div className="relative w-full h-[400px] lg:h-[480px] bg-gray-950 rounded-2xl border border-gray-800 overflow-hidden flex flex-col select-none shadow-2xl">
      {/* Map Header Toolbar */}
      <div className="absolute top-3 left-3 right-3 z-20 flex flex-wrap items-center justify-between gap-2 pointer-events-none">
        {/* Left: Layer Info & Heatmap Stats */}
        <div className="flex items-center gap-2 bg-gray-900/90 backdrop-blur-md px-3 py-1.5 rounded-xl border border-gray-800 pointer-events-auto shadow-lg">
          <Layers className="h-4 w-4 text-cyan-400" />
          <span className="text-xs font-semibold text-white">GeoSpatial Canvas</span>
          <span className="text-[10px] text-cyan-300 font-mono bg-cyan-950/80 border border-cyan-800/80 px-1.5 py-0.5 rounded">
            {results.length} ranked
          </span>
          {hasAoiFilter && (
            <span className="text-[10px] text-amber-300 font-mono bg-amber-950/80 border border-amber-800/80 px-1.5 py-0.5 rounded flex items-center gap-1">
              AOI Filter Active
            </span>
          )}
        </div>

        {/* Right: Interactive Drawing Controls */}
        <div className="flex items-center gap-1.5 bg-gray-900/90 backdrop-blur-md p-1 rounded-xl border border-gray-800 pointer-events-auto text-xs shadow-lg">
          {/* Draw Polygon Button */}
          <button
            type="button"
            onClick={() => {
              if (activeTool === "polygon") cancelDrawing();
              else {
                setActiveTool("polygon");
                setDraftPoints([]);
              }
            }}
            className={`px-2.5 py-1 rounded-lg flex items-center gap-1.5 text-xs font-medium transition ${
              activeTool === "polygon"
                ? "bg-cyan-500 text-gray-950 font-bold"
                : "text-gray-300 hover:text-white hover:bg-gray-800"
            }`}
            title="Click to place vertices on map, then click Complete"
          >
            <Compass className="h-3.5 w-3.5" />
            <span>Draw Polygon</span>
          </button>

          {/* Draw Bounding Box Button */}
          <button
            type="button"
            onClick={() => {
              if (activeTool === "bbox") cancelDrawing();
              else {
                setActiveTool("bbox");
                setDragStart(null);
                setDragCurrent(null);
              }
            }}
            className={`px-2.5 py-1 rounded-lg flex items-center gap-1.5 text-xs font-medium transition ${
              activeTool === "bbox"
                ? "bg-blue-500 text-gray-950 font-bold"
                : "text-gray-300 hover:text-white hover:bg-gray-800"
            }`}
            title="Click and drag to define rectangular bounding envelope"
          >
            <Square className="h-3.5 w-3.5" />
            <span>Drag BBox</span>
          </button>

          {/* Complete Polygon Button (visible when drafting >= 3 pts) */}
          {activeTool === "polygon" && draftPoints.length >= 3 && (
            <button
              type="button"
              onClick={finishPolygon}
              className="px-2.5 py-1 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg flex items-center gap-1 text-xs font-semibold shadow"
            >
              <Check className="h-3.5 w-3.5" />
              <span>Complete ({draftPoints.length} pts)</span>
            </button>
          )}

          {/* Spatial Mode: Intersects vs Within */}
          {onSpatialFilterModeChange && hasAoiFilter && (
            <button
              type="button"
              onClick={() =>
                onSpatialFilterModeChange(spatialFilterMode === "intersects" ? "within" : "intersects")
              }
              className="px-2 py-1 bg-gray-800 hover:bg-gray-700 text-cyan-300 border border-cyan-800/80 rounded-lg text-[10px] font-mono"
              title="Toggle whether tiles must intersect or be completely within the AOI"
            >
              Mode: {spatialFilterMode.toUpperCase()}
            </button>
          )}

          {/* Clear AOI Filter Button */}
          {(hasAoiFilter || draftPoints.length > 0) && (
            <button
              type="button"
              onClick={clearAllAoi}
              className="p-1 text-gray-400 hover:text-red-400 rounded-lg hover:bg-gray-800 transition"
              title="Clear Area of Interest"
            >
              <Trash2 className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>

      {/* Drawing Mode Guidance Banner */}
      {activeTool !== "none" && (
        <div className="absolute top-16 left-1/2 -translate-x-1/2 z-20 bg-cyan-950/90 backdrop-blur-md border border-cyan-700 text-cyan-200 text-xs px-3.5 py-1.5 rounded-full shadow-xl flex items-center gap-2 pointer-events-none animate-pulse">
          <MousePointer className="h-3.5 w-3.5 text-cyan-400" />
          {activeTool === "polygon"
            ? `Click on map to place polygon vertices (${draftPoints.length} placed). Click "Complete" when done.`
            : "Click and drag across the map canvas to set rectangular bounding box."}
        </div>
      )}

      {/* Main Interactive Geographic Canvas */}
      <div
        ref={canvasRef}
        onClick={handleCanvasClick}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        className={`relative flex-1 w-full h-full p-6 flex items-center justify-center overflow-hidden ${
          activeTool === "polygon"
            ? "cursor-crosshair"
            : activeTool === "bbox"
            ? "cursor-crosshair"
            : "cursor-default"
        }`}
      >
        {/* Subtle Coordinate Grid Lines */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#1f2937_1px,transparent_1px),linear-gradient(to_bottom,#1f2937_1px,transparent_1px)] bg-[size:36px_36px] opacity-30 pointer-events-none" />

        {/* Dynamic Bounds Canvas Frame */}
        <div className="relative w-full h-full max-w-4xl max-h-[360px] border border-cyan-500/30 rounded-xl bg-cyan-950/10 p-2 transition-all shadow-inner">
          {/* Corner Coordinate Badges */}
          <div className="absolute -top-3 left-3 bg-gray-900 border border-gray-700 text-[10px] text-gray-400 font-mono px-2 py-0.5 rounded shadow z-10">
            N {formatCoord(bounds.north, 3)}° | W {formatCoord(bounds.west, 3)}°
          </div>
          <div className="absolute -bottom-3 right-3 bg-gray-900 border border-gray-700 text-[10px] text-gray-400 font-mono px-2 py-0.5 rounded shadow z-10">
            S {formatCoord(bounds.south, 3)}° | E {formatCoord(bounds.east, 3)}°
          </div>

          {/* SVG Vector Overlays: AOI Polygon, Bounding Box, and Result Footprints */}
          <svg className="absolute inset-0 w-full h-full pointer-events-none z-10" viewBox="0 0 100 100" preserveAspectRatio="none">
            {/* 1. Footprint Geometries for All Ranked Results */}
            {results.map((item) => {
              if (item.footprint_geojson && item.footprint_geojson.coordinates?.[0]) {
                const polyPoints = item.footprint_geojson.coordinates[0]
                  .map(([lon, lat]) => {
                    const { x, y } = projectToPercent(lon, lat);
                    return `${x},${y}`;
                  })
                  .join(" ");

                const color = getScoreColor(item.similarity_score);
                const isSelected = selectedTileId === item.tile_id;
                const isHovered = hoveredTile?.tile_id === item.tile_id;

                return (
                  <polygon
                    key={`footprint-${item.tile_id}`}
                    points={polyPoints}
                    fill={isSelected ? "rgba(6, 182, 212, 0.45)" : isHovered ? "rgba(6, 182, 212, 0.35)" : color.fill}
                    stroke={isSelected ? "#ffffff" : isHovered ? "#22d3ee" : color.stroke}
                    strokeWidth={isSelected ? 1.5 : 0.8}
                    className="transition-all duration-150"
                  />
                );
              }
              return null;
            })}

            {/* 2. Drawn AOI Polygon */}
            {activePolygonPoints && (
              <>
                <polygon
                  points={activePolygonPoints}
                  fill="rgba(245, 158, 11, 0.18)"
                  stroke="#f59e0b"
                  strokeWidth="1.2"
                  strokeDasharray={draftPoints.length > 0 ? "2,2" : undefined}
                />
              </>
            )}

            {/* 3. Drawn Bounding Box */}
            {activeBboxRect && (
              <rect
                x={activeBboxRect.x}
                y={activeBboxRect.y}
                width={activeBboxRect.width}
                height={activeBboxRect.height}
                fill="rgba(59, 130, 246, 0.16)"
                stroke="#3b82f6"
                strokeWidth="1.2"
                strokeDasharray="2,2"
              />
            )}
          </svg>

          {/* HTML Overlay for Interactive Tile Buttons & Handles */}
          <div className="relative w-full h-full">
            {/* Draft polygon vertex handles */}
            {draftPoints.map(([lon, lat], idx) => {
              const { x, y } = projectToPercent(lon, lat);
              return (
                <div
                  key={`draft-pt-${idx}`}
                  style={{ left: `${x}%`, top: `${y}%` }}
                  className="absolute -translate-x-1/2 -translate-y-1/2 h-2.5 w-2.5 rounded-full bg-amber-400 border-2 border-gray-950 z-30 shadow"
                />
              );
            })}

            {/* AOI Polygon vertex handles */}
            {aoiPolygon &&
              aoiPolygon.map(([lon, lat], idx) => {
                const { x, y } = projectToPercent(lon, lat);
                return (
                  <div
                    key={`poly-pt-${idx}`}
                    style={{ left: `${x}%`, top: `${y}%` }}
                    className="absolute -translate-x-1/2 -translate-y-1/2 h-2 w-2 rounded-full bg-amber-400 border border-gray-950 z-20 shadow"
                    title={`Vertex ${idx + 1}: ${formatCoord(lat, 4)}, ${formatCoord(lon, 4)}`}
                  />
                );
              })}

            {/* Empty Results Placeholder */}
            {results.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center text-gray-500 text-xs">
                <Crosshair className="h-8 w-8 mb-2 text-gray-600 animate-pulse" />
                <p className="font-medium text-gray-400">No search results to display.</p>
                <p className="text-[11px] text-gray-600 mt-1 max-w-sm">
                  {hasAoiFilter
                    ? "AOI filter active. Execute search or expand filters to retrieve matching tiles inside the selected region."
                    : "Enter a natural language query or draw an Area of Interest to retrieve matching tiles."}
                </p>
              </div>
            ) : (
              results.map((item) => {
                const topLeft = projectToPercent(item.bbox.west, item.bbox.north);
                const bottomRight = projectToPercent(item.bbox.east, item.bbox.south);
                const widthPct = Math.max(3.5, bottomRight.x - topLeft.x);
                const heightPct = Math.max(3.5, bottomRight.y - topLeft.y);

                const isSelected = selectedTileId === item.tile_id;
                const isHovered = hoveredTile?.tile_id === item.tile_id;
                const color = getScoreColor(item.similarity_score);

                return (
                  <div
                    key={item.tile_id}
                    onClick={(e) => {
                      if (activeTool !== "none") return;
                      e.stopPropagation();
                      onSelectTile(item);
                    }}
                    onMouseEnter={() => setHoveredTile(item)}
                    onMouseLeave={() => setHoveredTile(null)}
                    style={{
                      left: `${topLeft.x}%`,
                      top: `${topLeft.y}%`,
                      width: `${widthPct}%`,
                      height: `${heightPct}%`,
                    }}
                    className={`absolute transition-all duration-150 rounded border ${
                      activeTool !== "none" ? "pointer-events-none" : "cursor-pointer"
                    } ${
                      isSelected
                        ? `${color.bg} border-white ring-2 ring-cyan-400 z-30 shadow-lg shadow-cyan-500/50 scale-[1.05]`
                        : isHovered
                        ? `${color.bg} ${color.border} z-20 scale-[1.02] ring-1 ring-cyan-300`
                        : `${color.bg} ${color.border} opacity-85 hover:opacity-100`
                    }`}
                  >
                    {/* Centered Rank Badge */}
                    <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none p-0.5">
                      <span className="text-[9px] font-bold font-mono text-white drop-shadow">
                        #{item.rank}
                      </span>
                      <span className="text-[8px] font-mono text-cyan-200 opacity-90">
                        {(item.similarity_score * 100).toFixed(0)}%
                      </span>
                    </div>

                    {/* Quality Dot */}
                    <div
                      className={`absolute top-0.5 right-0.5 h-1.5 w-1.5 rounded-full ${
                        item.quality_score != null && item.quality_score >= 0.8
                          ? "bg-emerald-400"
                          : item.quality_score != null && item.quality_score >= 0.5
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

      {/* Bottom Status / Hover Inspector Bar */}
      <div className="px-4 py-2 bg-gray-950/90 border-t border-gray-800 flex items-center justify-between text-xs text-gray-400">
        <div className="flex items-center gap-3">
          {hoveredTile ? (
            <div className="flex items-center gap-2 font-mono text-[11px] text-gray-200">
              <span className="text-cyan-400 font-bold">Rank #{hoveredTile.rank}</span>
              <span>•</span>
              <span className="text-emerald-400 font-semibold">
                {(hoveredTile.similarity_score * 100).toFixed(1)}% match
              </span>
              <span>•</span>
              <span>{hoveredTile.sensor || "Unknown sensor"}</span>
              <span>•</span>
              <span className="text-gray-400">
                [{formatCoord(hoveredTile.center_coordinates.lat, 3)}, {formatCoord(hoveredTile.center_coordinates.lon, 3)}]
              </span>
              {onInspectTile && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onInspectTile(hoveredTile);
                  }}
                  className="ml-2 px-1.5 py-0.5 bg-cyan-900/60 hover:bg-cyan-800 text-cyan-300 text-[10px] rounded flex items-center gap-1 border border-cyan-700/60"
                >
                  <Eye className="h-3 w-3" /> Inspect
                </button>
              )}
            </div>
          ) : cursorGeo ? (
            <div className="flex items-center gap-2 font-mono text-[11px] text-gray-400">
              <MapPin className="h-3 w-3 text-cyan-400" />
              <span>Cursor: [{formatCoord(cursorGeo.lat, 4)}°, {formatCoord(cursorGeo.lon, 4)}°]</span>
              {hasAoiFilter && (
                <span className="text-amber-400 ml-2">
                  (AOI Mode: {spatialFilterMode})
                </span>
              )}
            </div>
          ) : (
            <div className="flex items-center gap-1.5 text-gray-500 text-[11px]">
              <Sparkles className="h-3.5 w-3.5 text-cyan-400" />
              <span>Hover or click any tile to view coordinates, similarity, and provenance.</span>
            </div>
          )}
        </div>

        {/* Similarity Legend */}
        <div className="hidden sm:flex items-center gap-3 text-[10px] font-mono">
          <div className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-emerald-400" />
            <span className="text-gray-400">&gt;80%</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-cyan-400" />
            <span className="text-gray-400">65–80%</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-blue-400" />
            <span className="text-gray-400">50–65%</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-amber-400" />
            <span className="text-gray-400">&lt;50%</span>
          </div>
        </div>
      </div>
    </div>
  );
}
