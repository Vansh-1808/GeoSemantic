"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { formatDate, qualityColor, statusBadgeClass } from "@/lib/utils";
import {
  Satellite,
  Layers,
  GitCompare,
  ClipboardCheck,
  AlertCircle,
  CheckCircle,
  TrendingUp,
  Calendar,
} from "lucide-react";
import Link from "next/link";

export default function DashboardPage() {
  const { data: health } = useQuery({
    queryKey: ["health-detailed"],
    queryFn: () => api.getDetailedHealth(),
    refetchInterval: 30_000,
  });

  const { data: scenesData } = useQuery({
    queryKey: ["scenes"],
    queryFn: () => api.listScenes(1, 100),
    enabled: !!health?.components.database.ok,
  });

  const scenes = scenesData?.scenes ?? [];
  const completedScenes = scenes.filter((s) => s.ingestion_status === "COMPLETED");
  const totalTiles = completedScenes.reduce((sum, s) => sum + s.tile_count, 0);
  const totalEmbedded = completedScenes.reduce((sum, s) => sum + s.embedded_count, 0);
  const sensors = [...new Set(completedScenes.map((s) => s.sensor).filter(Boolean))];

  const recentScenes = [...scenes]
    .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
    .slice(0, 5);

  const dbOk = health?.components.database.ok ?? false;
  const modelsReady =
    (health?.components.models.remoteclip.present &&
      health?.components.models.dino.present) ??
    false;

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white">Mission Dashboard</h1>
        <p className="text-gray-400 text-sm mt-1">
          Satellite imagery intelligence — real-time archive status
        </p>
      </div>

      {/* System alerts */}
      {!dbOk && (
        <div className="flex items-center gap-3 p-3 bg-red-900/30 border border-red-700 rounded-lg text-red-300 text-sm">
          <AlertCircle className="h-4 w-4 shrink-0" />
          <span>
            Database unavailable. Start PostgreSQL and run{" "}
            <code className="font-mono bg-red-900/40 px-1 rounded">python scripts/setup_db.py</code>
          </span>
        </div>
      )}
      {dbOk && !modelsReady && (
        <div className="flex items-center gap-3 p-3 bg-amber-900/30 border border-amber-700 rounded-lg text-amber-300 text-sm">
          <AlertCircle className="h-4 w-4 shrink-0" />
          <span>
            AI models not found. Run{" "}
            <code className="font-mono bg-amber-900/40 px-1 rounded">python scripts/download_models.py</code>{" "}
            to enable semantic search.
          </span>
        </div>
      )}
      {dbOk && modelsReady && (
        <div className="flex items-center gap-3 p-3 bg-emerald-900/30 border border-emerald-700 rounded-lg text-emerald-300 text-sm">
          <CheckCircle className="h-4 w-4 shrink-0" />
          <span>System operational. All components ready.</span>
        </div>
      )}

      {/* Stats grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          icon={Satellite}
          label="Scenes Indexed"
          value={completedScenes.length}
          sub={`${scenes.length} total ingested`}
          color="blue"
        />
        <StatCard
          icon={Layers}
          label="Tiles Indexed"
          value={totalTiles.toLocaleString()}
          sub={`${totalEmbedded.toLocaleString()} embedded`}
          color="purple"
        />
        <StatCard
          icon={GitCompare}
          label="Sensors"
          value={sensors.length}
          sub={sensors.slice(0, 2).join(", ") || "None yet"}
          color="amber"
        />
        <StatCard
          icon={ClipboardCheck}
          label="Reviews Pending"
          value="—"
          sub="Change review queue"
          color="emerald"
        />
      </div>

      {/* Two columns */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Recent scenes */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-white">Recent Scenes</h2>
            <Link href="/ingest" className="text-xs text-blue-400 hover:text-blue-300">
              Ingest new →
            </Link>
          </div>
          {recentScenes.length === 0 ? (
            <div className="text-center py-8">
              <Satellite className="h-8 w-8 text-gray-700 mx-auto mb-2" />
              <p className="text-gray-500 text-sm">No scenes ingested yet.</p>
              <Link
                href="/ingest"
                className="text-blue-400 text-sm hover:underline mt-1 block"
              >
                Upload your first GeoTIFF
              </Link>
            </div>
          ) : (
            <div className="space-y-2">
              {recentScenes.map((scene) => (
                <div
                  key={scene.id}
                  className="flex items-center gap-3 p-2 rounded-lg hover:bg-gray-800 transition-colors"
                >
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-gray-200 truncate">{scene.filename}</p>
                    <div className="flex items-center gap-2 text-xs text-gray-500 mt-0.5">
                      <Calendar className="h-3 w-3" />
                      {formatDate(scene.acquisition_date)}
                      {scene.sensor && (
                        <span className="text-gray-600">· {scene.sensor}</span>
                      )}
                    </div>
                  </div>
                  <div className="flex flex-col items-end gap-1">
                    <span
                      className={`text-xs px-2 py-0.5 rounded border ${statusBadgeClass(scene.ingestion_status)}`}
                    >
                      {scene.ingestion_status}
                    </span>
                    {scene.quality_score != null && (
                      <span
                        className={`text-xs font-mono ${qualityColor(scene.quality_score)}`}
                      >
                        Q:{(scene.quality_score * 100).toFixed(0)}%
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Quick actions */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
          <h2 className="text-sm font-semibold text-white mb-4">Quick Actions</h2>
          <div className="grid grid-cols-2 gap-3">
            {[
              { href: "/ingest", icon: "📥", label: "Ingest Imagery", desc: "Upload GeoTIFF" },
              { href: "/search", icon: "🔍", label: "Semantic Search", desc: "Natural language" },
              { href: "/image-search", icon: "🖼️", label: "Image Search", desc: "Find similar tiles" },
              { href: "/change", icon: "📊", label: "Change Analysis", desc: "Compare dates" },
              { href: "/clusters", icon: "🗺️", label: "Clusters", desc: "Explore similar sites" },
              { href: "/review", icon: "✅", label: "Review Queue", desc: "Analyst decisions" },
            ].map(({ href, icon, label, desc }) => (
              <Link
                key={href}
                href={href}
                className="flex flex-col p-3 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg transition-colors"
              >
                <span className="text-xl mb-1">{icon}</span>
                <span className="text-sm font-medium text-white">{label}</span>
                <span className="text-xs text-gray-400">{desc}</span>
              </Link>
            ))}
          </div>
        </div>
      </div>

      {/* System status */}
      {health && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
          <h2 className="text-sm font-semibold text-white mb-3">System Status</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
            <StatusItem
              label="Database"
              ok={health.components.database.ok}
              detail={health.components.database.message}
            />
            <StatusItem
              label="RemoteCLIP"
              ok={health.components.models.remoteclip.present}
              detail={
                health.components.models.remoteclip.size_mb
                  ? `${health.components.models.remoteclip.size_mb} MB (Ready)`
                  : health.components.models.remoteclip.present
                  ? "577.2 MB (Ready)"
                  : "Not downloaded"
              }
            />
            <StatusItem
              label="DINOv2"
              ok={health.components.models.dino.present}
              detail={
                health.components.models.dino.size_mb
                  ? `${health.components.models.dino.size_mb} MB (Ready)`
                  : health.components.models.dino.note ||
                    (health.components.models.dino.present ? "84.2 MB (Ready)" : "Not downloaded")
              }
            />
            <StatusItem
              label="Network"
              ok={!health.offline_mode}
              detail={health.offline_mode ? "Offline — local only" : "Online"}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function StatCard({
  icon: Icon,
  label,
  value,
  sub,
  color,
}: {
  icon: React.ElementType;
  label: string;
  value: string | number;
  sub: string;
  color: "blue" | "purple" | "amber" | "emerald";
}) {
  const colors = {
    blue: "text-blue-400 bg-blue-900/20",
    purple: "text-purple-400 bg-purple-900/20",
    amber: "text-amber-400 bg-amber-900/20",
    emerald: "text-emerald-400 bg-emerald-900/20",
  };
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
      <div className={`inline-flex p-2 rounded-lg ${colors[color]} mb-3`}>
        <Icon className={`h-4 w-4 ${colors[color].split(" ")[0]}`} />
      </div>
      <p className="text-2xl font-bold text-white">{value}</p>
      <p className="text-sm text-gray-400 font-medium mt-0.5">{label}</p>
      <p className="text-xs text-gray-600 mt-0.5">{sub}</p>
    </div>
  );
}

function StatusItem({
  label,
  ok,
  detail,
}: {
  label: string;
  ok: boolean;
  detail: string;
}) {
  return (
    <div className="flex items-start gap-2">
      <span
        className={`mt-0.5 inline-block h-2 w-2 rounded-full shrink-0 ${ok ? "bg-emerald-400" : "bg-red-400"}`}
      />
      <div>
        <p className="text-gray-300 font-medium">{label}</p>
        <p className="text-gray-500 text-[11px]">{detail}</p>
      </div>
    </div>
  );
}
