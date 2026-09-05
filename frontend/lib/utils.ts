import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatDate(dateStr: string | null): string {
  if (!dateStr) return "Unknown date";
  return new Date(dateStr).toLocaleDateString("en-IN", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
}

export function formatBytes(bytes: number | null): string {
  if (!bytes) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

export function formatCoord(val: number | null, decimals = 4): string {
  return val != null ? val.toFixed(decimals) : "—";
}

export function qualityColor(score: number | null): string {
  if (score == null) return "text-gray-500";
  if (score >= 0.8) return "text-emerald-400";
  if (score >= 0.5) return "text-amber-400";
  return "text-red-400";
}

export function statusBadgeClass(status: string): string {
  switch (status) {
    case "COMPLETED":
      return "bg-emerald-900/50 text-emerald-400 border-emerald-700";
    case "PROCESSING":
    case "RUNNING":
      return "bg-blue-900/50 text-blue-400 border-blue-700 animate-pulse";
    case "PENDING":
      return "bg-gray-800 text-gray-400 border-gray-700";
    case "FAILED":
      return "bg-red-900/50 text-red-400 border-red-700";
    default:
      return "bg-gray-800 text-gray-400 border-gray-700";
  }
}
