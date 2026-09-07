"use client";

import React from "react";
import { CheckCircle2, AlertTriangle, XCircle, ShieldCheck, ShieldAlert, ShieldX } from "lucide-react";

interface QualityBadgeProps {
  suitability?: "Suitable" | "Caution" | "Poor Quality" | string | null;
  qualityScore?: number | null;
  showScore?: boolean;
  size?: "xs" | "sm" | "md" | "lg";
  className?: string;
  variant?: "pill" | "subtle" | "tag";
}

export function QualityBadge({
  suitability,
  qualityScore,
  showScore = false,
  size = "sm",
  className = "",
  variant = "pill",
}: QualityBadgeProps) {
  // Infer suitability from score if not provided
  let status = suitability;
  if (!status && qualityScore != null) {
    if (qualityScore >= 0.75) status = "Suitable";
    else if (qualityScore >= 0.40) status = "Caution";
    else status = "Poor Quality";
  }

  const isSuitable = status === "Suitable";
  const isCaution = status === "Caution";
  const isPoor = status === "Poor Quality";

  const sizeClasses = {
    xs: "text-[10px] px-1.5 py-0.5 gap-1",
    sm: "text-xs px-2 py-0.5 gap-1.5",
    md: "text-xs px-2.5 py-1 gap-1.5",
    lg: "text-sm px-3 py-1.5 gap-2",
  }[size];

  const iconSizes = {
    xs: "h-2.5 w-2.5",
    sm: "h-3.5 w-3.5",
    md: "h-4 w-4",
    lg: "h-4.5 w-4.5",
  }[size];

  let colorClasses = "bg-gray-800/80 text-gray-300 border-gray-700";
  let Icon = AlertTriangle;

  if (isSuitable) {
    colorClasses = "bg-emerald-950/80 text-emerald-300 border-emerald-500/40 shadow-emerald-950/20";
    Icon = CheckCircle2;
  } else if (isCaution) {
    colorClasses = "bg-amber-950/80 text-amber-300 border-amber-500/40 shadow-amber-950/20";
    Icon = AlertTriangle;
  } else if (isPoor) {
    colorClasses = "bg-rose-950/80 text-rose-300 border-rose-500/40 shadow-rose-950/20";
    Icon = XCircle;
  }

  const scoreText = qualityScore != null ? `${Math.round(qualityScore * 100)}%` : null;

  return (
    <span
      className={`inline-flex items-center font-medium rounded-full border backdrop-blur-sm shadow-sm select-none transition-colors ${sizeClasses} ${colorClasses} ${className}`}
      title={
        isSuitable
          ? "Suitable: High optical quality, verified for change analysis"
          : isCaution
          ? "Caution: Marginal cloud, shadow, or noise present"
          : "Poor Quality: Excessive occlusion or nodata, unsuitable for change analysis"
      }
    >
      <Icon className={`${iconSizes} flex-shrink-0`} />
      <span>{status || "Assessing"}</span>
      {showScore && scoreText && (
        <span className="opacity-75 font-mono text-[0.9em] border-l border-current/30 pl-1">
          {scoreText}
        </span>
      )}
    </span>
  );
}
