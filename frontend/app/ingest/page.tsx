"use client";

import { useState, useCallback, useRef } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api, type IngestionStatus } from "@/lib/api";
import {
  formatDate,
  formatBytes,
  statusBadgeClass,
  qualityColor,
  formatCoord,
} from "@/lib/utils";
import {
  Upload,
  Satellite,
  Trash2,
  RefreshCw,
  AlertCircle,
  CloudOff,
  CheckCircle,
  Settings2,
  Layers,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import { ProcessSceneDialog } from "@/components/scenes/ProcessSceneDialog";
import type { SceneSummary } from "@/lib/api";

export default function IngestPage() {
  const qc = useQueryClient();
  const [processSceneTarget, setProcessSceneTarget] = useState<SceneSummary | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [activeJobs, setActiveJobs] = useState<
    Record<string, { jobId: string; filename: string }>
  >({});
  const [jobStatuses, setJobStatuses] = useState<Record<string, IngestionStatus>>({});
  const pollRefs = useRef<Record<string, NodeJS.Timeout>>({});

  // ── Scene list ─────────────────────────────────────────────
  const { data, isLoading, refetch } = useQuery({
    queryKey: ["scenes"],
    queryFn: () => api.listScenes(1, 50),
    refetchInterval: 10_000,
  });

  // ── Delete scene ───────────────────────────────────────────
  const deleteMutation = useMutation({
    mutationFn: (sceneId: string) => api.deleteScene(sceneId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["scenes"] }),
  });

  // ── File upload handler ────────────────────────────────────
  const handleFiles = useCallback(async (files: FileList | File[]) => {
    const arr = Array.from(files);
    const geotiffs = arr.filter((f) =>
      [".tif", ".tiff", ".geotiff"].some((ext) =>
        f.name.toLowerCase().endsWith(ext)
      )
    );

    if (geotiffs.length === 0) {
      alert("Please select GeoTIFF files (.tif, .tiff, .geotiff)");
      return;
    }

    for (const file of geotiffs) {
      try {
        const res = await api.ingestScene(file);
        const tempKey = `${file.name}-${Date.now()}`;
        setActiveJobs((prev) => ({
          ...prev,
          [tempKey]: { jobId: res.job_id, filename: file.name },
        }));
        startPolling(res.job_id, tempKey);
      } catch (err: any) {
        console.error("Ingest failed:", err);
        const msg = err?.response?.data?.detail || err?.message || "Is the backend running?";
        alert(`Failed to start ingestion for ${file.name}: ${msg}`);
      }
    }
  }, []);

  const startPolling = (jobId: string, key: string) => {
    const poll = async () => {
      try {
        const status = await api.getIngestionStatus(jobId);
        setJobStatuses((prev) => ({ ...prev, [key]: status }));

        if (status.status === "COMPLETED" || status.status === "FAILED") {
          clearInterval(pollRefs.current[key]);
          delete pollRefs.current[key];
          qc.invalidateQueries({ queryKey: ["scenes"] });
          // Remove from active jobs after 5s
          setTimeout(() => {
            setActiveJobs((prev) => {
              const next = { ...prev };
              delete next[key];
              return next;
            });
            setJobStatuses((prev) => {
              const next = { ...prev };
              delete next[key];
              return next;
            });
          }, 5000);
        }
      } catch {
        // Silent fail on poll
      }
    };
    poll();
    pollRefs.current[key] = setInterval(poll, 2000);
  };

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      handleFiles(e.dataTransfer.files);
    },
    [handleFiles]
  );

  const scenes = data?.scenes ?? [];
  const hasActiveJobs = Object.keys(activeJobs).length > 0;

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Imagery Ingestion</h1>
        <p className="text-gray-400 text-sm mt-1">
          Upload GeoTIFF files to extract metadata, generate tiles, and index for search.
        </p>
      </div>

      {/* Drop zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        className={`relative border-2 border-dashed rounded-xl p-10 text-center transition-all cursor-pointer ${
          dragOver
            ? "border-blue-400 bg-blue-900/20"
            : "border-gray-700 hover:border-gray-600 bg-gray-900"
        }`}
        onClick={() => document.getElementById("file-input")?.click()}
      >
        <input
          id="file-input"
          type="file"
          multiple
          accept=".tif,.tiff,.geotiff"
          className="hidden"
          onChange={(e) => e.target.files && handleFiles(e.target.files)}
        />
        <Upload className="h-10 w-10 text-gray-600 mx-auto mb-3" />
        <p className="text-gray-300 font-medium">
          Drag & drop GeoTIFF files, or click to browse
        </p>
        <p className="text-gray-500 text-sm mt-1">
          Supports GeoTIFF (.tif, .tiff) and Cloud Optimized GeoTIFF (COG)
        </p>
        <p className="text-gray-600 text-xs mt-1">Maximum file size: 5 GB per file</p>
      </div>

      {/* Active job progress */}
      <AnimatePresence>
        {hasActiveJobs && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="space-y-3"
          >
            <h2 className="text-sm font-semibold text-white">Active Ingestion Jobs</h2>
            {Object.entries(activeJobs).map(([key, { jobId, filename }]) => {
              const status = jobStatuses[key];
              return (
                <JobProgress
                  key={key}
                  filename={filename}
                  status={status}
                />
              );
            })}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Scene list */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-white">
            Ingested Scenes ({data?.total ?? 0})
          </h2>
          <button
            onClick={() => refetch()}
            className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-200 transition-colors"
          >
            <RefreshCw className="h-3 w-3" />
            Refresh
          </button>
        </div>

        {isLoading ? (
          <div className="text-center py-12 text-gray-500">Loading scenes...</div>
        ) : scenes.length === 0 ? (
          <div className="text-center py-12 bg-gray-900 border border-gray-800 rounded-xl">
            <Satellite className="h-10 w-10 text-gray-700 mx-auto mb-3" />
            <p className="text-gray-400">No scenes ingested yet.</p>
            <p className="text-gray-600 text-sm">
              Upload a GeoTIFF above to get started.
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            {scenes.map((scene) => (
              <motion.div
                key={scene.id}
                layout
                className="bg-gray-900 border border-gray-800 rounded-xl p-4 hover:border-gray-700 transition-colors"
              >
                <div className="flex items-start gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <p className="text-sm font-medium text-white truncate">
                        {scene.filename}
                      </p>
                      <span
                        className={`text-xs px-2 py-0.5 rounded border font-mono ${statusBadgeClass(scene.ingestion_status)}`}
                      >
                        {scene.ingestion_status}
                      </span>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-x-6 gap-y-1 mt-2 text-xs text-gray-500">
                      <span>
                        📅 {formatDate(scene.acquisition_date)}
                      </span>
                      <span>
                        🛰️ {scene.sensor ?? "Unknown sensor"}
                      </span>
                      <span>
                        🧩 {scene.tile_count.toLocaleString()} tiles
                      </span>
                      <span>
                        🧠 {scene.embedded_count.toLocaleString()} embedded
                      </span>
                    </div>

                    {scene.bbox_west != null && (
                      <p className="text-xs text-gray-600 mt-1 font-mono">
                        Bounds: [{formatCoord(scene.bbox_west)}, {formatCoord(scene.bbox_south)}] →
                        [{formatCoord(scene.bbox_east)}, {formatCoord(scene.bbox_north)}]
                      </p>
                    )}

                    <div className="flex items-center gap-3 mt-2">
                      {scene.quality_score != null && (
                        <span
                          className={`text-xs font-medium ${qualityColor(scene.quality_score)}`}
                        >
                          Quality: {(scene.quality_score * 100).toFixed(0)}%
                        </span>
                      )}
                      {scene.cloud_cover_pct != null && (
                        <span className="flex items-center gap-1 text-xs text-gray-500">
                          <CloudOff className="h-3 w-3" />
                          Cloud: {scene.cloud_cover_pct.toFixed(1)}%
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    <button
                      onClick={() => setProcessSceneTarget(scene)}
                      className="px-3 py-1.5 bg-blue-600/10 hover:bg-blue-600/20 text-blue-400 border border-blue-500/30 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-colors shadow-sm"
                      title="Configure and run tiling pipeline"
                    >
                      <Settings2 className="h-3.5 w-3.5" />
                      Process Tiles
                    </button>
                    <Link
                      href="/datasets"
                      className="px-2.5 py-1.5 bg-gray-800 hover:bg-gray-700 text-gray-300 border border-gray-700 rounded-lg text-xs font-medium flex items-center gap-1 transition-colors"
                      title="Explore scene and tiles"
                    >
                      <Layers className="h-3.5 w-3.5 text-gray-400" />
                      Explore
                    </Link>
                    <button
                      onClick={() => {
                        if (
                          confirm(
                            `Delete "${scene.filename}" and all its tiles? This cannot be undone.`
                          )
                        ) {
                          deleteMutation.mutate(scene.id);
                        }
                      }}
                      className="p-1.5 text-gray-600 hover:text-red-400 hover:bg-red-900/20 rounded-lg transition-colors"
                      title="Delete scene"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </div>

      {/* Process Scene Modal Dialog */}
      <ProcessSceneDialog
        scene={processSceneTarget}
        isOpen={!!processSceneTarget}
        onClose={() => setProcessSceneTarget(null)}
        onSuccess={() => qc.invalidateQueries({ queryKey: ["scenes"] })}
      />
    </div>
  );
}

function JobProgress({
  filename,
  status,
}: {
  filename: string;
  status?: IngestionStatus;
}) {
  const pct = status?.progress_pct ?? 0;
  const isDone = status?.status === "COMPLETED";
  const isFailed = status?.status === "FAILED";

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          {isDone ? (
            <CheckCircle className="h-4 w-4 text-emerald-400" />
          ) : isFailed ? (
            <AlertCircle className="h-4 w-4 text-red-400" />
          ) : (
            <RefreshCw className="h-4 w-4 text-blue-400 animate-spin" />
          )}
          <span className="text-sm text-gray-200 font-medium">{filename}</span>
        </div>
        <span className="text-xs text-gray-500 font-mono">{pct.toFixed(0)}%</span>
      </div>

      <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <motion.div
          className={`h-full rounded-full ${isDone ? "bg-emerald-500" : isFailed ? "bg-red-500" : "bg-blue-500"}`}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.3 }}
        />
      </div>

      {status?.message && (
        <p className="text-xs text-gray-500 mt-1.5">{status.message}</p>
      )}
      {(status?.tile_count ?? 0) > 0 && (
        <p className="text-xs text-gray-600 mt-0.5">
          {status?.tile_count} tiles generated
        </p>
      )}
      {isFailed && status?.error && (
        <p className="text-xs text-red-400 mt-1">{status?.error}</p>
      )}
    </div>
  );
}
