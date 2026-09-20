"use client";

import React from "react";
import { ShieldCheck, ShieldAlert, AlertTriangle, ShieldX, Activity } from "lucide-react";

interface ConfidenceGaugeProps {
  score: number; // 0.0 to 1.0 or 0 to 100
  tier?: "High Confidence" | "Medium Confidence" | "Low Confidence" | "Suppressed" | string | null;
  isSuppressed?: boolean;
  size?: "sm" | "md" | "lg";
  showLabel?: boolean;
}

export function ConfidenceGauge({
  score,
  tier,
  isSuppressed = false,
  size = "md",
  showLabel = true,
}: ConfidenceGaugeProps) {
  // Normalize score to percentage 0-100
  const normalizedScore = score <= 1.0 ? Math.round(score * 100) : Math.round(score);
  const clampedScore = Math.max(0, Math.min(100, normalizedScore));

  // Determine effective tier if not explicitly provided
  let effectiveTier = tier;
  if (!effectiveTier) {
    if (isSuppressed || clampedScore < 25) {
      effectiveTier = "Suppressed";
    } else if (clampedScore >= 70) {
      effectiveTier = "High Confidence";
    } else if (clampedScore >= 45) {
      effectiveTier = "Medium Confidence";
    } else {
      effectiveTier = "Low Confidence";
    }
  }

  // Tier configuration: colors, gradients, icons
  const tierConfig = {
    "High Confidence": {
      label: "High Confidence",
      color: "#10b981", // emerald-500
      glowColor: "rgba(16, 185, 129, 0.4)",
      gradientId: "gauge-emerald",
      gradientStops: [
        { offset: "0%", color: "#059669" },
        { offset: "100%", color: "#34d399" },
      ],
      badgeBg: "bg-emerald-500/15 border-emerald-500/40 text-emerald-400",
      badgeText: "High Confidence",
      Icon: ShieldCheck,
    },
    "Medium Confidence": {
      label: "Medium Confidence",
      color: "#06b6d4", // cyan-500
      glowColor: "rgba(6, 182, 212, 0.4)",
      gradientId: "gauge-cyan",
      gradientStops: [
        { offset: "0%", color: "#0284c7" },
        { offset: "100%", color: "#38bdf8" },
      ],
      badgeBg: "bg-cyan-500/15 border-cyan-500/40 text-cyan-400",
      badgeText: "Medium Confidence",
      Icon: Activity,
    },
    "Low Confidence": {
      label: "Low Confidence",
      color: "#f59e0b", // amber-500
      glowColor: "rgba(245, 158, 11, 0.4)",
      gradientId: "gauge-amber",
      gradientStops: [
        { offset: "0%", color: "#d97706" },
        { offset: "100%", color: "#fbbf24" },
      ],
      badgeBg: "bg-amber-500/15 border-amber-500/40 text-amber-400",
      badgeText: "Low Confidence",
      Icon: AlertTriangle,
    },
    "Suppressed": {
      label: "Suppressed",
      color: "#ef4444", // red-500
      glowColor: "rgba(239, 68, 68, 0.4)",
      gradientId: "gauge-red",
      gradientStops: [
        { offset: "0%", color: "#b91c1c" },
        { offset: "100%", color: "#f87171" },
      ],
      badgeBg: "bg-rose-500/15 border-rose-500/40 text-rose-400",
      badgeText: "Suppressed (False Alarm)",
      Icon: ShieldX,
    },
  };

  const currentConfig = tierConfig[effectiveTier as keyof typeof tierConfig] || tierConfig["Low Confidence"];
  const TierIcon = currentConfig.Icon;

  // Gauge Dimensions based on size
  const dimMap = {
    sm: { width: 90, height: 58, stroke: 7, radius: 34, fontSize: "text-base", subSize: "text-[9px]" },
    md: { width: 140, height: 90, stroke: 10, radius: 52, fontSize: "text-2xl", subSize: "text-[11px]" },
    lg: { width: 220, height: 135, stroke: 15, radius: 82, fontSize: "text-4xl", subSize: "text-xs" },
  };
  const { width, height, stroke, radius, fontSize, subSize } = dimMap[size];

  // Semicircular Arc calculations (180 degrees, from -180 to 0)
  const cx = width / 2;
  const cy = height - stroke / 2 - 4;
  const arcLength = Math.PI * radius;
  const strokeDashoffset = arcLength - (clampedScore / 100) * arcLength;

  return (
    <div className="flex flex-col items-center justify-center select-none">
      <div className="relative" style={{ width, height }}>
        <svg width={width} height={height} className="overflow-visible">
          <defs>
            <linearGradient id={currentConfig.gradientId} x1="0%" y1="0%" x2="100%" y2="0%">
              {currentConfig.gradientStops.map((stop, i) => (
                <stop key={i} offset={stop.offset} stopColor={stop.color} />
              ))}
            </linearGradient>
            {/* Ambient drop shadow filter */}
            <filter id={`glow-${size}`} x="-20%" y="-20%" width="140%" height="140%">
              <feDropShadow dx="0" dy="0" stdDeviation="4" floodColor={currentConfig.glowColor} />
            </filter>
          </defs>

          {/* Background Track Arc */}
          <path
            d={`M ${cx - radius} ${cy} A ${radius} ${radius} 0 0 1 ${cx + radius} ${cy}`}
            fill="none"
            stroke="#1f2937" // gray-800
            strokeWidth={stroke}
            strokeLinecap="round"
          />

          {/* Active Colored Value Arc */}
          <path
            d={`M ${cx - radius} ${cy} A ${radius} ${radius} 0 0 1 ${cx + radius} ${cy}`}
            fill="none"
            stroke={`url(#${currentConfig.gradientId})`}
            strokeWidth={stroke}
            strokeDasharray={arcLength}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            filter={`url(#glow-${size})`}
            className="transition-all duration-700 ease-out"
          />

          {/* Ticks at 25%, 50%, 75% */}
          {size !== "sm" && [25, 50, 75].map((tick) => {
            const angle = Math.PI * (1 - tick / 100);
            const tickR1 = radius - stroke / 2 - 2;
            const tickR2 = radius - stroke / 2 - 6;
            const x1 = cx + tickR1 * Math.cos(angle);
            const y1 = cy - tickR1 * Math.sin(angle);
            const x2 = cx + tickR2 * Math.cos(angle);
            const y2 = cy - tickR2 * Math.sin(angle);
            return (
              <line
                key={tick}
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke="#4b5563"
                strokeWidth="1.5"
              />
            );
          })}
        </svg>

        {/* Center Percentage Display */}
        <div 
          className="absolute inset-x-0 bottom-1 flex flex-col items-center justify-center pointer-events-none"
          style={{ bottom: size === "lg" ? "12px" : size === "md" ? "8px" : "4px" }}
        >
          <span className={`font-mono font-extrabold text-white tracking-tight ${fontSize} drop-shadow-md`}>
            {clampedScore}%
          </span>
          {size !== "sm" && (
            <span className={`text-gray-400 font-medium uppercase tracking-wider ${subSize}`}>
              Confidence
            </span>
          )}
        </div>
      </div>

      {/* Tier Badge */}
      {showLabel && (
        <div className="mt-2">
          <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full border text-xs font-semibold shadow-sm backdrop-blur-sm ${currentConfig.badgeBg}`}>
            <TierIcon className="w-3.5 h-3.5" />
            <span>{currentConfig.badgeText}</span>
          </span>
        </div>
      )}
    </div>
  );
}
