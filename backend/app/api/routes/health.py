"""
Health check and system status endpoints.
These must work even when the database is unavailable.
"""
from __future__ import annotations

import platform
import socket
from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.db.database import check_database_connection

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic liveness check — always returns 200."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get("/health/detailed")
async def detailed_health():
    """Full system status — database, models directory, offline mode detection."""
    db_ok = await check_database_connection()

    # Check if internet is reachable (offline mode detection)
    online = _check_internet()

    # Check model files
    models_status = _check_models()

    # Check data directories
    dirs_status = {
        "imagery": settings.imagery_dir.exists(),
        "tiles": settings.tiles_dir.exists(),
        "thumbnails": settings.thumbnails_dir.exists(),
        "qdrant_store": settings.qdrant_store_dir.exists(),
        "models": settings.models_dir.exists(),
    }

    all_ok = db_ok and all(dirs_status.values())

    return JSONResponse(
        status_code=200 if all_ok else 207,
        content={
            "status": "healthy" if all_ok else "degraded",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "offline_mode": not online,
            "components": {
                "database": {"ok": db_ok, "message": "Connected" if db_ok else "Unreachable"},
                "models": models_status,
                "directories": dirs_status,
            },
            "system": {
                "platform": platform.system(),
                "python": platform.python_version(),
                "hostname": socket.gethostname(),
            },
        },
    )


def _check_internet() -> bool:
    """Non-blocking internet reachability check via DNS lookup."""
    try:
        socket.setdefaulttimeout(2)
        socket.getaddrinfo("dns.google", 80)
        return True
    except Exception:
        return False


def _check_models() -> dict:
    """Check which model files are present on disk and query model manager."""
    try:
        from app.services.model_manager import model_manager
        health_info = model_manager.check_health()
        remoteclip_meta = health_info.get("RemoteCLIP", {})
        dino_meta = health_info.get("DINOv2", {})
        rc_present = bool(
            remoteclip_meta.get("local_path_valid", False)
            or remoteclip_meta.get("loaded", False)
            or settings.remoteclip_checkpoint_path.exists()
        )
        dino_present = bool(
            dino_meta.get("local_path_valid", False)
            or dino_meta.get("loaded", False)
            or settings.dino_weights_path.exists()
            or settings.dino_checkpoint_path.exists()
        )
        from pathlib import Path
        rc_path_str = remoteclip_meta.get("local_path", str(settings.remoteclip_checkpoint_path))
        rc_p = Path(rc_path_str) if rc_path_str else settings.remoteclip_checkpoint_path
        rc_size_mb = round(rc_p.stat().st_size / (1024 * 1024), 1) if rc_p.exists() and rc_p.is_file() else None

        dino_path_str = dino_meta.get("local_path", str(settings.dino_weights_path))
        dino_p = Path(dino_path_str) if dino_path_str else settings.dino_weights_path
        dino_size_mb = round(dino_p.stat().st_size / (1024 * 1024), 1) if dino_p.exists() and dino_p.is_file() else None

        return {
            "remoteclip": {
                "present": rc_present,
                "path": str(rc_p),
                "loaded": remoteclip_meta.get("loaded", False),
                "embedding_dim": remoteclip_meta.get("embedding_dimension", 512),
                "size_mb": rc_size_mb,
                "note": f"{rc_size_mb} MB" if rc_size_mb else ("Loaded in memory" if remoteclip_meta.get("loaded") else "Not found"),
            },
            "dino": {
                "present": dino_present,
                "path": str(dino_p),
                "loaded": dino_meta.get("loaded", False),
                "embedding_dim": dino_meta.get("embedding_dimension", 384),
                "size_mb": dino_size_mb,
                "note": f"{dino_size_mb} MB" if dino_size_mb else ("Loaded in memory" if dino_meta.get("loaded") else "Not found"),
            },
            "registry": health_info,
        }
    except Exception as exc:
        remoteclip_path = settings.remoteclip_checkpoint_path
        return {
            "remoteclip": {
                "present": remoteclip_path.exists(),
                "path": str(remoteclip_path),
            },
            "dino": {
                "present": (settings.models_dir / "dino_cache").exists(),
            },
            "error": str(exc),
        }
