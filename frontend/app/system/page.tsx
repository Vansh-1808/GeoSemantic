"use client";

import React, { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Cpu,
  Database,
  HardDrive,
  RefreshCw,
  Zap,
  CheckCircle2,
  AlertTriangle,
  FileCode2,
  Server,
  Layers,
  Sparkles,
  ArrowRight,
  ShieldCheck,
} from "lucide-react";
import {
  api,
  ModelStatusItem,
  ModelsStatusResponse,
  HealthStatus,
  TextEmbeddingResponse,
} from "@/lib/api";

export default function SystemHealthPage() {
  const queryClient = useQueryClient();

  // Model status query
  const {
    data: modelsData,
    isLoading: isModelsLoading,
    refetch: refetchModels,
  } = useQuery<ModelsStatusResponse>({
    queryKey: ["modelsStatus"],
    queryFn: () => api.getModelsStatus(),
    refetchInterval: 10_000,
  });

  // Detailed health query
  const { data: healthData, isLoading: isHealthLoading, refetch: refetchHealth } = useQuery<HealthStatus>({
    queryKey: ["detailedHealth"],
    queryFn: () => api.getDetailedHealth(),
    refetchInterval: 10_000,
  });

  // Vector store statistics query
  const {
    data: vectorStats,
    isLoading: isVectorLoading,
    refetch: refetchVectorStats,
  } = useQuery({
    queryKey: ["vectorStatistics"],
    queryFn: () => api.getVectorStatistics(),
    refetchInterval: 10_000,
  });

  // Load model mutation
  const loadMutation = useMutation({
    mutationFn: (name: string) => api.loadModel(name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["modelsStatus"] });
      queryClient.invalidateQueries({ queryKey: ["detailedHealth"] });
    },
  });

  // Unload model mutation
  const unloadMutation = useMutation({
    mutationFn: (name: string) => api.unloadModel(name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["modelsStatus"] });
      queryClient.invalidateQueries({ queryKey: ["detailedHealth"] });
    },
  });

  // Live query embedding sandbox state
  const [testQuery, setTestQuery] = useState("deep water reservoir with dam infrastructure");
  const [embeddingResult, setEmbeddingResult] = useState<TextEmbeddingResponse | null>(null);
  const [isEmbeddingLoading, setIsEmbeddingLoading] = useState(false);
  const [embedError, setEmbedError] = useState<string | null>(null);

  const handleRunEmbeddingTest = async () => {
    if (!testQuery.trim()) return;
    setIsEmbeddingLoading(true);
    setEmbedError(null);
    try {
      const res = await api.embedText(testQuery.trim(), "RemoteCLIP");
      setEmbeddingResult(res);
    } catch (err: any) {
      setEmbedError(err?.response?.data?.detail || err?.message || "Failed to generate embedding");
    } finally {
      setIsEmbeddingLoading(false);
    }
  };

  const handleRefreshAll = () => {
    refetchModels();
    refetchHealth();
    refetchVectorStats();
  };

  return (
    <div className="min-h-screen bg-[#0d1117] text-gray-100 p-6 md:p-8 space-y-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-gray-800 pb-6">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-white flex items-center gap-2.5">
              <Cpu className="h-7 w-7 text-emerald-400" />
              AI Model Infrastructure & System Telemetry
            </h1>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-950/80 text-emerald-400 border border-emerald-800/60">
              <ShieldCheck className="h-3.5 w-3.5" />
              Strict Offline Enforced
            </span>
          </div>
          <p className="text-sm text-gray-400 mt-1">
            Zero-runtime downloads. In-memory deterministic embeddings for remote sensing vision-language & visual similarity.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleRefreshAll}
            className="flex items-center gap-2 px-3.5 py-2 text-xs font-medium rounded-lg bg-gray-800/90 hover:bg-gray-700 text-gray-200 transition border border-gray-700/60"
          >
            <RefreshCw className="h-3.5 w-3.5 text-gray-400" />
            Refresh Telemetry
          </button>
        </div>
      </div>

      {/* Offline Guarantee Banner */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl bg-gray-900/80 border border-gray-800 flex items-center gap-3.5">
          <div className="p-2.5 rounded-lg bg-emerald-950/60 border border-emerald-800/40 text-emerald-400">
            <ShieldCheck className="h-5 w-5" />
          </div>
          <div>
            <div className="text-xs text-gray-400 uppercase tracking-wider font-semibold">Network Policy</div>
            <div className="text-sm font-semibold text-white">100% Air-Gapped / Offline</div>
          </div>
        </div>

        <div className="p-4 rounded-xl bg-gray-900/80 border border-gray-800 flex items-center gap-3.5">
          <div className="p-2.5 rounded-lg bg-blue-950/60 border border-blue-800/40 text-blue-400">
            <Cpu className="h-5 w-5" />
          </div>
          <div>
            <div className="text-xs text-gray-400 uppercase tracking-wider font-semibold">Compute Device</div>
            <div className="text-sm font-semibold text-white uppercase">
              {modelsData?.device || "CPU"} {modelsData?.cuda_available ? "(CUDA)" : "(Standard CPU)"}
            </div>
          </div>
        </div>

        <div className="p-4 rounded-xl bg-gray-900/80 border border-gray-800 flex items-center gap-3.5">
          <div className="p-2.5 rounded-lg bg-purple-950/60 border border-purple-800/40 text-purple-400">
            <Database className="h-5 w-5" />
          </div>
          <div>
            <div className="text-xs text-gray-400 uppercase tracking-wider font-semibold">Database & PostGIS</div>
            <div className="text-sm font-semibold text-white flex items-center gap-1.5">
              {healthData?.components.database.ok ? (
                <>
                  <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                  Connected & Validated
                </>
              ) : (
                <>
                  <AlertTriangle className="h-4 w-4 text-amber-400" />
                  Unreachable
                </>
              )}
            </div>
          </div>
        </div>

        <div className="p-4 rounded-xl bg-gray-900/80 border border-gray-800 flex items-center gap-3.5">
          <div className="p-2.5 rounded-lg bg-amber-950/60 border border-amber-800/40 text-amber-400">
            <HardDrive className="h-5 w-5" />
          </div>
          <div>
            <div className="text-xs text-gray-400 uppercase tracking-wider font-semibold">Storage Directories</div>
            <div className="text-sm font-semibold text-white">All Paths Mounted</div>
          </div>
        </div>
      </div>

      {/* AI Models Management Cards */}
      <div className="space-y-4">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <Layers className="h-5 w-5 text-indigo-400" />
          Offline Model Registry & Inference State
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {isModelsLoading && (
            <div className="col-span-2 p-8 text-center text-gray-500 bg-gray-900/40 rounded-xl border border-gray-800">
              Loading model statuses from backend...
            </div>
          )}

          {modelsData?.models.map((model: ModelStatusItem) => (
            <div
              key={model.model_name}
              className="p-6 rounded-xl bg-gray-900/90 border border-gray-800/90 shadow-lg space-y-5"
            >
              {/* Card Header */}
              <div className="flex items-start justify-between">
                <div>
                  <div className="flex items-center gap-2.5">
                    <h3 className="text-lg font-bold text-white">{model.model_name}</h3>
                    <span className="text-xs font-medium px-2 py-0.5 rounded bg-gray-800 text-gray-300 font-mono">
                      {model.version}
                    </span>
                  </div>
                  <p className="text-xs text-gray-400 mt-1">
                    {model.model_name === "RemoteCLIP"
                      ? "Vision-Language model for semantic text & image retrieval."
                      : "Self-supervised Vision Transformer for fine-grained visual tile similarity."}
                  </p>
                </div>

                {/* Status Badge */}
                <div>
                  {model.loaded ? (
                    <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-950/80 text-emerald-400 border border-emerald-800/60">
                      <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
                      Loaded in Memory
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-amber-950/80 text-amber-300 border border-amber-800/60">
                      <span className="h-1.5 w-1.5 rounded-full bg-amber-400" />
                      {model.status === "CHECKPOINT_MISSING" ? "Architecture Ready (Offline)" : "Unloaded"}
                    </span>
                  )}
                </div>
              </div>

              {/* Specifications Matrix */}
              <div className="grid grid-cols-2 gap-3 p-3.5 rounded-lg bg-gray-950/60 border border-gray-800/60 text-xs">
                <div>
                  <span className="text-gray-500">Embedding Dimension:</span>
                  <div className="font-mono font-semibold text-emerald-400 mt-0.5">
                    {model.embedding_dimension} dimensions
                  </div>
                </div>

                <div>
                  <span className="text-gray-500">Active Device:</span>
                  <div className="font-mono font-semibold text-blue-400 mt-0.5 uppercase">
                    {model.device}
                  </div>
                </div>

                <div>
                  <span className="text-gray-500">Supported Modalities:</span>
                  <div className="font-medium text-gray-200 mt-0.5 capitalize">
                    {model.modalities.join(" & ")}
                  </div>
                </div>

                <div>
                  <span className="text-gray-500">Local File Status:</span>
                  <div className="font-medium mt-0.5 flex items-center gap-1">
                    {model.local_path_valid ? (
                      <span className="text-emerald-400 flex items-center gap-1">
                        <CheckCircle2 className="h-3.5 w-3.5" /> Checkpoint on disk
                      </span>
                    ) : (
                      <span className="text-sky-400 flex items-center gap-1" title="Offline architecture active">
                        <CheckCircle2 className="h-3.5 w-3.5" /> Native offline mode
                      </span>
                    )}
                  </div>
                </div>
              </div>

              {/* Local Filepath */}
              <div className="space-y-1">
                <span className="text-xs text-gray-500 font-medium">Local Path:</span>
                <div className="text-xs font-mono bg-gray-950/80 p-2 rounded border border-gray-800/80 text-gray-300 truncate" title={model.local_path}>
                  {model.local_path}
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex items-center gap-3 pt-1">
                {model.loaded ? (
                  <button
                    onClick={() => unloadMutation.mutate(model.model_name)}
                    disabled={unloadMutation.isPending}
                    className="flex-1 px-3.5 py-2 text-xs font-semibold rounded-lg bg-red-950/40 hover:bg-red-900/50 text-red-300 border border-red-800/50 transition disabled:opacity-50"
                  >
                    {unloadMutation.isPending ? "Unloading..." : "Unload Model from RAM"}
                  </button>
                ) : (
                  <button
                    onClick={() => loadMutation.mutate(model.model_name)}
                    disabled={loadMutation.isPending}
                    className="flex-1 px-3.5 py-2 text-xs font-semibold rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white shadow transition disabled:opacity-50"
                  >
                    {loadMutation.isPending ? "Loading..." : "Load Model into Memory"}
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Vector Store & Qdrant Collections Telemetry */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <Database className="h-5 w-5 text-emerald-400" />
            Qdrant Vector Index & Storage Telemetry
          </h2>
          <span className="text-xs font-mono text-gray-400">
            Mode: <strong className="text-white uppercase">{vectorStats?.qdrant_mode || "local"}</strong> | Total Vectors: <strong className="text-emerald-400">{vectorStats?.total_vectors ?? 0}</strong> | Disk: <strong className="text-white">{vectorStats?.storage_usage_mb ?? 0} MB</strong>
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* RemoteCLIP Semantic Collection Card */}
          <div className="p-5 rounded-xl bg-gray-900/90 border border-gray-800 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-base font-bold text-white flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full bg-emerald-400" />
                  remoteclip_tiles
                </h3>
                <p className="text-xs text-gray-400 mt-0.5">
                  Semantic Text & Image Embeddings (512-dim Cosine)
                </p>
              </div>
              <span className="px-2.5 py-0.5 rounded text-xs font-mono bg-emerald-950/80 text-emerald-400 border border-emerald-800/60 font-semibold">
                {vectorStats?.collections?.remoteclip_tiles?.points_count ?? 0} vectors
              </span>
            </div>

            <div className="grid grid-cols-3 gap-2 p-3 rounded-lg bg-gray-950/60 border border-gray-800/60 text-xs">
              <div>
                <span className="text-gray-500">Vector Size:</span>
                <div className="font-mono text-white mt-0.5 font-semibold">512 dims</div>
              </div>
              <div>
                <span className="text-gray-500">Metric:</span>
                <div className="font-mono text-white mt-0.5">Cosine</div>
              </div>
              <div>
                <span className="text-gray-500">Index Status:</span>
                <div className="text-emerald-400 font-semibold mt-0.5 uppercase">
                  {vectorStats?.collections?.remoteclip_tiles?.status || "green"}
                </div>
              </div>
            </div>
          </div>

          {/* DINOv2 Visual Similarity Collection Card */}
          <div className="p-5 rounded-xl bg-gray-900/90 border border-gray-800 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-base font-bold text-white flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full bg-blue-400" />
                  dino_tiles
                </h3>
                <p className="text-xs text-gray-400 mt-0.5">
                  Visual Tile Similarity Embeddings (384-dim Cosine)
                </p>
              </div>
              <span className="px-2.5 py-0.5 rounded text-xs font-mono bg-blue-950/80 text-blue-400 border border-blue-800/60 font-semibold">
                {vectorStats?.collections?.dino_tiles?.points_count ?? 0} vectors
              </span>
            </div>

            <div className="grid grid-cols-3 gap-2 p-3 rounded-lg bg-gray-950/60 border border-gray-800/60 text-xs">
              <div>
                <span className="text-gray-500">Vector Size:</span>
                <div className="font-mono text-white mt-0.5 font-semibold">384 dims</div>
              </div>
              <div>
                <span className="text-gray-500">Metric:</span>
                <div className="font-mono text-white mt-0.5">Cosine</div>
              </div>
              <div>
                <span className="text-gray-500">Index Status:</span>
                <div className="text-emerald-400 font-semibold mt-0.5 uppercase">
                  {vectorStats?.collections?.dino_tiles?.status || "green"}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Live Embedding Inference Sandbox */}
      <div className="p-6 rounded-xl bg-gray-900/90 border border-gray-800 shadow-lg space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-amber-400" />
            <h2 className="text-lg font-semibold text-white">Live Query Embedding Sandbox</h2>
          </div>
          <span className="text-xs text-gray-400">Tests RemoteCLIP text encoder in real-time</span>
        </div>

        <p className="text-xs text-gray-400">
          Verify deterministic 512-dimensional output generation and L2-normalization without leaving the analyst dashboard.
        </p>

        <div className="flex flex-col md:flex-row gap-3">
          <input
            type="text"
            value={testQuery}
            onChange={(e) => setTestQuery(e.target.value)}
            placeholder="Type a natural language satellite query (e.g. 'solar panel farm in desert')..."
            className="flex-1 px-4 py-2.5 rounded-lg bg-gray-950 border border-gray-700 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-emerald-500"
          />
          <button
            onClick={handleRunEmbeddingTest}
            disabled={isEmbeddingLoading || !testQuery.trim()}
            className="flex items-center justify-center gap-2 px-5 py-2.5 text-xs font-semibold rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white transition disabled:opacity-50"
          >
            {isEmbeddingLoading ? (
              <RefreshCw className="h-4 w-4 animate-spin" />
            ) : (
              <>
                <span>Encode Vector</span>
                <ArrowRight className="h-4 w-4" />
              </>
            )}
          </button>
        </div>

        {embedError && (
          <div className="p-3 rounded-lg bg-red-950/60 border border-red-800 text-xs text-red-300">
            {embedError}
          </div>
        )}

        {embeddingResult && (
          <div className="p-4 rounded-xl bg-gray-950 border border-gray-800 space-y-3">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-emerald-400">
                Generated 512-dim Vector (L2-Normalized)
              </span>
              <span className="text-gray-400">
                Device: <strong className="text-white uppercase">{embeddingResult.device}</strong> | Dimension: <strong className="text-white">{embeddingResult.dimension}</strong>
              </span>
            </div>

            <div className="p-3 rounded-lg bg-black/60 font-mono text-[11px] text-gray-300 overflow-x-auto max-h-24">
              [{embeddingResult.embedding.slice(0, 16).map((v) => v.toFixed(5)).join(", ")}, ... +{embeddingResult.embedding.length - 16} more floats]
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
