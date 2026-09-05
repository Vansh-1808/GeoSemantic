"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect } from "react";
import {
  LayoutDashboard,
  Search,
  Image,
  GitCompare,
  Clock,
  Layers,
  ClipboardCheck,
  Upload,
  Database,
  Settings,
  Satellite,
  Wifi,
  WifiOff,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/lib/store";
import { api } from "@/lib/api";

const NAV_ITEMS = [
  { href: "/", icon: LayoutDashboard, label: "Dashboard" },
  { href: "/search", icon: Search, label: "Search" },
  { href: "/image-search", icon: Image, label: "Image Search" },
  { href: "/change", icon: GitCompare, label: "Change Analysis" },
  { href: "/timeline", icon: Clock, label: "Timeline" },
  { href: "/clusters", icon: Layers, label: "Clusters" },
  { href: "/review", icon: ClipboardCheck, label: "Review Queue" },
  { href: "/ingest", icon: Upload, label: "Ingestion" },
  { href: "/datasets", icon: Database, label: "Datasets" },
  { href: "/system", icon: Settings, label: "System" },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { isOffline, dbConnected, setSystemStatus } = useAppStore();

  // Poll system health every 30s
  useEffect(() => {
    const check = async () => {
      try {
        const health = await api.getDetailedHealth();
        setSystemStatus({
          isOffline: health.offline_mode,
          dbConnected: health.components.database.ok,
          modelsReady:
            health.components.models.remoteclip.present &&
            health.components.models.dino.present,
        });
      } catch {
        setSystemStatus({ dbConnected: false });
      }
    };
    check();
    const interval = setInterval(check, 30_000);
    return () => clearInterval(interval);
  }, [setSystemStatus]);

  return (
    <div className="flex h-screen overflow-hidden bg-gray-950">
      {/* ── Sidebar ─────────────────────────────────────────── */}
      <aside className="flex flex-col w-56 shrink-0 bg-gray-900 border-r border-gray-800">
        {/* Logo */}
        <div className="flex items-center gap-2 px-4 py-4 border-b border-gray-800">
          <Satellite className="h-6 w-6 text-blue-400" />
          <div>
            <p className="text-sm font-bold text-white leading-tight">GeoSemantic</p>
            <p className="text-[10px] text-gray-500 leading-tight">SIH 2026</p>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto py-2 px-2 space-y-0.5">
          {NAV_ITEMS.map(({ href, icon: Icon, label }) => {
            const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className={cn(
                  "flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors",
                  active
                    ? "bg-blue-600/20 text-blue-400 font-medium"
                    : "text-gray-400 hover:text-gray-100 hover:bg-gray-800"
                )}
              >
                <Icon className="h-4 w-4 shrink-0" />
                {label}
              </Link>
            );
          })}
        </nav>

        {/* Status indicators */}
        <div className="px-3 py-3 border-t border-gray-800 space-y-1.5">
          {/* Offline indicator */}
          <div
            className={cn(
              "flex items-center gap-2 px-2 py-1.5 rounded text-xs font-medium",
              isOffline
                ? "bg-amber-900/40 text-amber-400"
                : "bg-emerald-900/30 text-emerald-400"
            )}
          >
            {isOffline ? (
              <WifiOff className="h-3 w-3" />
            ) : (
              <Wifi className="h-3 w-3" />
            )}
            {isOffline ? "OFFLINE MODE" : "Online"}
          </div>

          {/* DB status */}
          <div
            className={cn(
              "flex items-center gap-2 px-2 py-1 rounded text-xs",
              dbConnected ? "text-gray-400" : "text-red-400"
            )}
          >
            <span
              className={cn(
                "inline-block h-1.5 w-1.5 rounded-full",
                dbConnected ? "bg-emerald-400" : "bg-red-400"
              )}
            />
            {dbConnected ? "Database OK" : "DB unavailable"}
          </div>
        </div>
      </aside>

      {/* ── Main Content ─────────────────────────────────────── */}
      <main className="flex-1 overflow-y-auto">
        {children}
      </main>
    </div>
  );
}
