"use client";

import React, { useState } from "react";
import { ConfidenceBreakdown, FactorDetail, api } from "@/lib/api";
import { ConfidenceGauge } from "./ConfidenceGauge";
import { 
  ShieldAlert, ShieldCheck, AlertTriangle, Sparkles, Sliders, 
  RotateCcw, CheckCircle2, Loader2,
  Layers, Compass, Cloud, Sun, Eye, Activity, Calendar, SplitSquareVertical
} from "lucide-react";

interface FactorBreakdownPanelProps {
  confidenceBreakdown?: ConfidenceBreakdown | null;
  finalConfidence?: number | null;
  confidenceTier?: string | null;
  isSuppressed?: boolean;
  suppressionReasons?: string[] | null;
  positiveFactors?: string[] | null;
}

export function FactorBreakdownPanel({
  confidenceBreakdown,
  finalConfidence = 0,
  confidenceTier,
  isSuppressed = false,
  suppressionReasons,
  positiveFactors,
}: FactorBreakdownPanelProps) {
  const [showSandbox, setShowSandbox] = useState(false);
  const [sandboxWeights, setSandboxWeights] = useState({
    weight_visual: 0.25,
    weight_semantic: 0.25,
    weight_spectral: 0.15,
    weight_spatial: 0.15,
    weight_quality: 0.08,
    weight_registration: 0.08,
  });
  const [simulatedResult, setSimulatedResult] = useState<ConfidenceBreakdown | null>(null);
  const [evaluating, setEvaluating] = useState(false);

  // Active breakdown to display (either simulated from sandbox or the recorded one)
  const activeBreakdown = simulatedResult || confidenceBreakdown;
  const activeTier = activeBreakdown?.confidence_tier || confidenceTier || (isSuppressed ? "Suppressed" : "Medium Confidence");
  const activeConfidence = activeBreakdown?.final_confidence ?? finalConfidence ?? 0;
  const activeIsSuppressed = activeBreakdown?.is_suppressed ?? isSuppressed;
  const activeReasons = activeBreakdown?.reasons ?? suppressionReasons ?? [];
  const activePositive = activeBreakdown?.positive_factors ?? positiveFactors ?? [];
  const factors = activeBreakdown?.factor_breakdown || {};

  // Icons mapped to the 10 signals
  const factorIcons: Record<string, React.ElementType> = {
    visual_change_score: Eye,
    semantic_change_score: Sparkles,
    image_quality_score: Activity,
    cloud_score: Cloud,
    shadow_score: Sun,
    registration_confidence: Compass,
    sensor_compatibility: SplitSquareVertical,
    seasonal_compatibility: Calendar,
    spectral_evidence: Layers,
    spatial_consistency: CheckCircle2,
  };

  // Status badge style helper
  const getStatusBadge = (status: string) => {
    switch (status) {
      case "optimal":
        return "bg-emerald-500/15 border-emerald-500/40 text-emerald-400";
      case "moderate":
        return "bg-cyan-500/15 border-cyan-500/40 text-cyan-400";
      case "caution":
        return "bg-amber-500/15 border-amber-500/40 text-amber-400";
      case "concerning":
      case "suppressed":
        return "bg-rose-500/15 border-rose-500/40 text-rose-400";
      default:
        return "bg-gray-800 border-gray-700 text-gray-300";
    }
  };

  // Run live simulation with altered weights
  const handleWeightChange = async (key: string, value: number) => {
    const updated = { ...sandboxWeights, [key]: value };
    setSandboxWeights(updated);

    if (confidenceBreakdown?.raw_signals) {
      setEvaluating(true);
      try {
        const raw = confidenceBreakdown.raw_signals;
        const res = await api.evaluateConfidence({
          visual_change: raw.visual_change_score,
          semantic_change: raw.semantic_change_score,
          image_quality: raw.image_quality_score,
          cloud_score: raw.cloud_score,
          shadow_score: raw.shadow_score,
          registration_confidence: raw.registration_confidence,
          sensor_compatibility: raw.sensor_compatibility,
          seasonal_compatibility: raw.seasonal_compatibility,
          spectral_evidence: raw.spectral_evidence,
          spatial_consistency: raw.spatial_consistency,
          weights_override: updated,
        });
        setSimulatedResult(res);
      } catch (err) {
        console.error("Simulation error:", err);
      } finally {
        setEvaluating(false);
      }
    }
  };

  const handleResetSandbox = () => {
    setSandboxWeights({
      weight_visual: 0.25,
      weight_semantic: 0.25,
      weight_spectral: 0.15,
      weight_spatial: 0.15,
      weight_quality: 0.08,
      weight_registration: 0.08,
    });
    setSimulatedResult(null);
  };

  return (
    <div className="space-y-6">
      {/* ── Top Row: Gauge & High-Level Decision Summary ───────────────── */}
      <div className="bg-gray-950/80 rounded-2xl border border-gray-800 p-6 flex flex-col md:flex-row items-center justify-between gap-6 shadow-xl relative overflow-hidden">
        {/* Gauge */}
        <div className="flex flex-col items-center justify-center shrink-0">
          <ConfidenceGauge
            score={activeConfidence}
            tier={activeTier}
            isSuppressed={activeIsSuppressed}
            size="lg"
            showLabel={true}
          />
        </div>

        {/* Core Decision Summary & Evidence Callout */}
        <div className="flex-1 space-y-3.5">
          <div className="flex items-center justify-between">
            <div>
              <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider block">
                Multi-Factor Confidence Verdict
              </span>
              <h3 className="text-xl font-bold text-white flex items-center gap-2">
                <span>{activeTier}</span>
                {activeIsSuppressed ? (
                  <span className="text-xs px-2.5 py-0.5 rounded-full bg-rose-500/20 text-rose-300 border border-rose-500/40">
                    False Alarm Suppressed
                  </span>
                ) : (
                  <span className="text-xs px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
                    Persistent Transformation
                  </span>
                )}
              </h3>
            </div>

            <button
              onClick={() => setShowSandbox(!showSandbox)}
              className={`px-3 py-1.5 rounded-xl border text-xs font-medium flex items-center gap-1.5 transition-all ${
                showSandbox
                  ? "bg-blue-600 border-blue-500 text-white shadow-lg"
                  : "bg-gray-800/80 hover:bg-gray-700 border-gray-700 text-gray-300"
              }`}
            >
              <Sliders className="w-3.5 h-3.5" />
              <span>{showSandbox ? "Close Sandbox" : "Weight Sandbox"}</span>
            </button>
          </div>

          {/* Suppression Reasons Banner (if any) */}
          {activeReasons.length > 0 && (
            <div className="p-3.5 bg-rose-950/25 border border-rose-800/50 rounded-xl space-y-1.5">
              <div className="flex items-center gap-2 text-rose-400 text-xs font-bold uppercase tracking-wider">
                <ShieldAlert className="w-4 h-4" />
                <span>Suppression Reasons & Confounder Warnings:</span>
              </div>
              <ul className="list-disc list-inside text-xs text-rose-200/90 space-y-1">
                {activeReasons.map((reason, idx) => (
                  <li key={idx} className="capitalize">
                    {reason}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Positive Corroborating Factors */}
          {activePositive.length > 0 && (
            <div className="p-3 bg-emerald-950/20 border border-emerald-800/40 rounded-xl space-y-1">
              <div className="flex items-center gap-2 text-emerald-400 text-xs font-bold uppercase tracking-wider">
                <ShieldCheck className="w-4 h-4" />
                <span>Corroborating Evidence:</span>
              </div>
              <ul className="list-disc list-inside text-xs text-emerald-200/90 space-y-0.5">
                {activePositive.map((fact, idx) => (
                  <li key={idx}>{fact}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>

      {/* ── Interactive Weights Sandbox Drawer ─────────────────────────── */}
      {showSandbox && (
        <div className="bg-blue-950/20 border border-blue-800/50 rounded-2xl p-5 space-y-4 animate-in fade-in duration-200">
          <div className="flex items-center justify-between border-b border-blue-900/40 pb-3">
            <div className="flex items-center gap-2">
              <Sliders className="w-4 h-4 text-blue-400" />
              <span className="text-sm font-bold text-white">Analyst Weight Sandbox</span>
              {evaluating && <Loader2 className="w-3.5 h-3.5 animate-spin text-blue-400" />}
              <span className="text-xs text-gray-400">(Test transparent re-weighting with zero black box)</span>
            </div>
            {simulatedResult && (
              <button
                onClick={handleResetSandbox}
                className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1 font-medium"
              >
                <RotateCcw className="w-3 h-3" />
                Reset Defaults
              </button>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
            <div>
              <div className="flex justify-between mb-1 text-gray-300">
                <span>Visual Change Weight:</span>
                <span className="font-mono text-blue-400">{sandboxWeights.weight_visual.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="0.8"
                step="0.05"
                value={sandboxWeights.weight_visual}
                onChange={(e) => handleWeightChange("weight_visual", parseFloat(e.target.value))}
                className="w-full accent-blue-500 cursor-pointer"
              />
            </div>

            <div>
              <div className="flex justify-between mb-1 text-gray-300">
                <span>Semantic Change Weight:</span>
                <span className="font-mono text-blue-400">{sandboxWeights.weight_semantic.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="0.8"
                step="0.05"
                value={sandboxWeights.weight_semantic}
                onChange={(e) => handleWeightChange("weight_semantic", parseFloat(e.target.value))}
                className="w-full accent-blue-500 cursor-pointer"
              />
            </div>

            <div>
              <div className="flex justify-between mb-1 text-gray-300">
                <span>Spectral Evidence Weight:</span>
                <span className="font-mono text-blue-400">{sandboxWeights.weight_spectral.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="0.8"
                step="0.05"
                value={sandboxWeights.weight_spectral}
                onChange={(e) => handleWeightChange("weight_spectral", parseFloat(e.target.value))}
                className="w-full accent-blue-500 cursor-pointer"
              />
            </div>

            <div>
              <div className="flex justify-between mb-1 text-gray-300">
                <span>Spatial Coherence Weight:</span>
                <span className="font-mono text-blue-400">{sandboxWeights.weight_spatial.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="0.8"
                step="0.05"
                value={sandboxWeights.weight_spatial}
                onChange={(e) => handleWeightChange("weight_spatial", parseFloat(e.target.value))}
                className="w-full accent-blue-500 cursor-pointer"
              />
            </div>

            <div>
              <div className="flex justify-between mb-1 text-gray-300">
                <span>Registration Quality Weight:</span>
                <span className="font-mono text-blue-400">{sandboxWeights.weight_registration.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="0.5"
                step="0.02"
                value={sandboxWeights.weight_registration}
                onChange={(e) => handleWeightChange("weight_registration", parseFloat(e.target.value))}
                className="w-full accent-blue-500 cursor-pointer"
              />
            </div>

            <div>
              <div className="flex justify-between mb-1 text-gray-300">
                <span>Image Quality Weight:</span>
                <span className="font-mono text-blue-400">{sandboxWeights.weight_quality.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="0.5"
                step="0.02"
                value={sandboxWeights.weight_quality}
                onChange={(e) => handleWeightChange("weight_quality", parseFloat(e.target.value))}
                className="w-full accent-blue-500 cursor-pointer"
              />
            </div>
          </div>
        </div>
      )}

      {/* ── 10-Factor Comprehensive Signal Breakdown Grid ──────────────── */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-purple-400" />
            <h4 className="text-sm font-bold text-white">10-Factor Scientific Breakdown</h4>
          </div>
          <span className="text-xs text-gray-400">100% Mathematically Explainable</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
          {Object.entries(factors).map(([key, factor]) => {
            const IconComponent = factorIcons[key] || Activity;
            const pct = Math.round(factor.raw_value * 100);

            // Is this factor a penalty?
            const isPenalty = key === "cloud_score" || key === "shadow_score";

            return (
              <div
                key={key}
                className="bg-gray-950/70 p-3.5 rounded-xl border border-gray-800 hover:border-gray-700 transition-colors space-y-2.5"
              >
                {/* Header: Name + Status Badge */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="p-1.5 rounded-lg bg-gray-800/80 text-gray-300">
                      <IconComponent className="w-3.5 h-3.5" />
                    </div>
                    <div>
                      <span className="text-xs font-semibold text-gray-200 block">{factor.name}</span>
                      <span className="text-[10px] text-gray-400 font-mono">
                        Weight: {factor.weight} | Contr: {factor.contribution}
                      </span>
                    </div>
                  </div>

                  <span
                    className={`text-[10px] uppercase font-bold px-2 py-0.5 rounded-full border ${getStatusBadge(
                      factor.status
                    )}`}
                  >
                    {factor.status}
                  </span>
                </div>

                {/* Meter Bar */}
                <div className="space-y-1">
                  <div className="flex justify-between items-center text-[11px] font-mono">
                    <span className="text-gray-400">{isPenalty ? "Interference Level:" : "Score:"}</span>
                    <span className="font-bold text-white">{pct}%</span>
                  </div>
                  <div className="w-full bg-gray-800 rounded-full h-1.5 overflow-hidden">
                    <div
                      className={`h-1.5 rounded-full transition-all duration-500 ${
                        isPenalty
                          ? pct > 25
                            ? "bg-rose-500"
                            : pct > 10
                            ? "bg-amber-500"
                            : "bg-emerald-500"
                          : pct >= 70
                          ? "bg-emerald-500"
                          : pct >= 40
                          ? "bg-blue-500"
                          : "bg-gray-500"
                      }`}
                      style={{ width: `${Math.max(4, pct)}%` }}
                    />
                  </div>
                </div>

                {/* Description */}
                <p className="text-[11px] text-gray-400 leading-snug">{factor.description}</p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
