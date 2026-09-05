"use client";

import { useState } from "react";
import { api, type SceneSummary, type ProcessSceneResponse } from "@/lib/api";
import {
  Layers,
  Settings2,
  Clock,
  CheckCircle2,
  AlertCircle,
  Loader2,
  X,
  Sparkles,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface ProcessSceneDialogProps {
  scene: SceneSummary | null;
  isOpen: boolean;
  onClose: () => void;
  onSuccess?: () => void;
}

export function ProcessSceneDialog({
  scene,
  isOpen,
  onClose,
  onSuccess,
}: ProcessSceneDialogProps) {
  const [tileSize, setTileSize] = useState<number>(512);
  const [overlapPx, setOverlapPx] = useState<number>(0);
  const [normalize, setNormalize] = useState<boolean>(true);
  const [minValidRatio, setMinValidRatio] = useState<number>(0.3);
  const [generatePreview, setGeneratePreview] = useState<boolean>(true);

  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [result, setResult] = useState<ProcessSceneResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen || !scene) return null;

  const handleStartProcessing = async () => {
    setIsProcessing(true);
    setError(null);
    setResult(null);

    try {
      // Run synchronously to get immediate processing time and tile count
      const res = await api.processScene(
        scene.id,
        {
          tile_size: tileSize,
          overlap_px: overlapPx,
          normalize,
          min_valid_pixel_ratio: minValidRatio,
          generate_preview: generatePreview,
        },
        true // synchronous
      );

      setResult(res);
      if (onSuccess) {
        onSuccess();
      }
    } catch (err: any) {
      console.error("Processing failed:", err);
      setError(
        err.response?.data?.detail ||
          err.message ||
          "Failed to process scene into tiles."
      );
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 10 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 10 }}
          className="relative w-full max-w-lg bg-gray-900 border border-gray-800 rounded-2xl shadow-2xl overflow-hidden"
        >
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-gray-800 bg-gray-950/60">
            <div className="flex items-center gap-2.5">
              <div className="p-2 bg-blue-500/10 rounded-lg border border-blue-500/20 text-blue-400">
                <Layers className="h-5 w-5" />
              </div>
              <div>
                <h3 className="text-base font-semibold text-white">
                  Scene Tiling & Preprocessing
                </h3>
                <p className="text-xs text-gray-400 truncate max-w-xs">
                  {scene.filename}
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              disabled={isProcessing}
              className="p-1.5 text-gray-400 hover:text-white rounded-lg hover:bg-gray-800 transition-colors"
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          {/* Body */}
          <div className="p-6 space-y-5">
            {/* Tile Size Selector */}
            <div className="space-y-2">
              <label className="flex items-center justify-between text-xs font-medium text-gray-300">
                <span className="flex items-center gap-1.5">
                  <Settings2 className="h-3.5 w-3.5 text-blue-400" />
                  Tile Dimensions (px)
                </span>
                <span className="text-[11px] text-blue-400 font-mono">
                  {tileSize} × {tileSize} px
                </span>
              </label>
              <div className="grid grid-cols-3 gap-2">
                {[256, 512, 1024].map((size) => (
                  <button
                    key={size}
                    type="button"
                    disabled={isProcessing}
                    onClick={() => setTileSize(size)}
                    className={`py-2 px-3 text-xs font-medium rounded-xl border transition-all ${
                      tileSize === size
                        ? "bg-blue-600/20 border-blue-500 text-blue-300 shadow-sm"
                        : "bg-gray-950 border-gray-800 text-gray-400 hover:border-gray-700 hover:text-gray-200"
                    }`}
                  >
                    {size} × {size}
                    {size === 512 && (
                      <span className="block text-[10px] text-blue-400/80 font-normal">
                        Default
                      </span>
                    )}
                  </button>
                ))}
              </div>
            </div>

            {/* Overlap Configuration */}
            <div className="space-y-2">
              <label className="flex items-center justify-between text-xs font-medium text-gray-300">
                <span>Tile Overlap (px)</span>
                <span className="text-[11px] text-blue-400 font-mono">
                  {overlapPx} px ({((overlapPx / tileSize) * 100).toFixed(0)}%)
                </span>
              </label>
              <div className="grid grid-cols-3 gap-2">
                {[0, 32, 64].map((ov) => (
                  <button
                    key={ov}
                    type="button"
                    disabled={isProcessing}
                    onClick={() => setOverlapPx(ov)}
                    className={`py-2 px-3 text-xs font-medium rounded-xl border transition-all ${
                      overlapPx === ov
                        ? "bg-blue-600/20 border-blue-500 text-blue-300"
                        : "bg-gray-950 border-gray-800 text-gray-400 hover:border-gray-700 hover:text-gray-200"
                    }`}
                  >
                    {ov === 0 ? "None (0 px)" : `${ov} px overlap`}
                  </button>
                ))}
              </div>
            </div>

            {/* Preprocessing Toggles */}
            <div className="p-4 bg-gray-950/60 rounded-xl border border-gray-800/80 space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs font-medium text-gray-200">
                    Band Normalization
                  </p>
                  <p className="text-[11px] text-gray-500">
                    Robust 2%–98% percentile radiometric contrast stretch
                  </p>
                </div>
                <input
                  type="checkbox"
                  checked={normalize}
                  disabled={isProcessing}
                  onChange={(e) => setNormalize(e.target.checked)}
                  className="h-4 w-4 rounded border-gray-700 text-blue-500 focus:ring-blue-600 focus:ring-offset-gray-900 bg-gray-900"
                />
              </div>

              <div className="flex items-center justify-between pt-2 border-t border-gray-800/60">
                <div>
                  <p className="text-xs font-medium text-gray-200">
                    RGB Preview & Thumbnail
                  </p>
                  <p className="text-[11px] text-gray-500">
                    Generate 512px RGB visualization & 128px thumbnail
                  </p>
                </div>
                <input
                  type="checkbox"
                  checked={generatePreview}
                  disabled={isProcessing}
                  onChange={(e) => setGeneratePreview(e.target.checked)}
                  className="h-4 w-4 rounded border-gray-700 text-blue-500 focus:ring-blue-600 focus:ring-offset-gray-900 bg-gray-900"
                />
              </div>
            </div>

            {/* Status Feedback */}
            {isProcessing && (
              <div className="p-4 bg-blue-950/30 border border-blue-800/50 rounded-xl flex items-center gap-3">
                <Loader2 className="h-5 w-5 text-blue-400 animate-spin shrink-0" />
                <div className="text-xs">
                  <p className="font-semibold text-blue-200">
                    Processing Scene...
                  </p>
                  <p className="text-blue-400/80">
                    Reading raster windows, normalizing bands, generating
                    thumbnails & PostGIS footprints.
                  </p>
                </div>
              </div>
            )}

            {result && (
              <motion.div
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                className="p-4 bg-emerald-950/40 border border-emerald-800/60 rounded-xl space-y-1.5"
              >
                <div className="flex items-center gap-2 text-emerald-400 font-semibold text-xs">
                  <CheckCircle2 className="h-4 w-4" />
                  <span>Tiling Completed Successfully!</span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs text-gray-300 mt-2">
                  <div className="p-2 bg-gray-900/60 rounded-lg border border-gray-800">
                    <span className="text-gray-500 block text-[10px]">
                      Tiles Generated
                    </span>
                    <span className="font-bold text-emerald-400 font-mono text-sm">
                      {result.tile_count}
                    </span>
                  </div>
                  <div className="p-2 bg-gray-900/60 rounded-lg border border-gray-800">
                    <span className="text-gray-500 block text-[10px]">
                      Processing Time
                    </span>
                    <span className="font-bold text-emerald-400 font-mono text-sm flex items-center gap-1">
                      <Clock className="h-3 w-3 inline" />
                      {result.processing_time_seconds}s
                    </span>
                  </div>
                </div>
              </motion.div>
            )}

            {error && (
              <div className="p-4 bg-red-950/40 border border-red-800/60 rounded-xl flex items-start gap-2.5 text-xs text-red-300">
                <AlertCircle className="h-4 w-4 text-red-400 shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-red-200">Error</p>
                  <p>{error}</p>
                </div>
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="flex items-center justify-end gap-2.5 px-6 py-4 border-t border-gray-800 bg-gray-950/60">
            <button
              type="button"
              onClick={onClose}
              disabled={isProcessing}
              className="px-4 py-2 text-xs font-medium text-gray-400 hover:text-white rounded-xl border border-gray-800 hover:bg-gray-800 transition-colors"
            >
              {result ? "Close" : "Cancel"}
            </button>
            {!result ? (
              <button
                type="button"
                id="btn-start-processing"
                disabled={isProcessing}
                onClick={handleStartProcessing}
                className="flex items-center gap-2 px-5 py-2 text-xs font-semibold text-white bg-blue-600 hover:bg-blue-500 rounded-xl transition-all shadow-lg shadow-blue-600/20 disabled:opacity-50"
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Sparkles className="h-3.5 w-3.5" />
                    Start Processing
                  </>
                )}
              </button>
            ) : (
              <button
                type="button"
                onClick={onClose}
                className="px-5 py-2 text-xs font-semibold text-white bg-emerald-600 hover:bg-emerald-500 rounded-xl transition-all"
              >
                Done
              </button>
            )}
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
