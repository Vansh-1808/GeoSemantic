"""
Application configuration using pydantic-settings.
Reads from environment variables and .env file.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[
            str(Path(__file__).resolve().parents[2] / ".env"),
            ".env",
        ],
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    # ── Application ───────────────────────────────────────────
    app_env: Literal["development", "production", "test"] = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    secret_key: str = "change-me"

    # ── Database ──────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/geosemantic"
    )
    database_sync_url: str = Field(
        default="postgresql+psycopg2://postgres:postgres@localhost:5432/geosemantic"
    )

    # ── Data Directories ──────────────────────────────────────
    data_dir: Path = Path("../data")
    imagery_dir: Path = Path("../data/imagery")
    tiles_dir: Path = Path("../data/tiles")
    thumbnails_dir: Path = Path("../data/thumbnails")
    previews_dir: Path = Path("../data/previews")
    models_dir: Path = Path("../data/models")
    exports_dir: Path = Path("../data/exports")
    maps_dir: Path = Path("../data/maps")
    qdrant_store_dir: Path = Path("../data/qdrant_store")

    # ── Tile Configuration ────────────────────────────────────
    default_tile_size: int = 512
    default_tile_overlap: int = 0
    min_valid_pixel_ratio: float = 0.3

    # ── Model Configuration ───────────────────────────────────
    remoteclip_model_name: str = "ViT-B-32"
    remoteclip_checkpoint_path: Path = Path("../data/models/RemoteCLIP-ViT-B-32.pt")
    dino_model_name: str = "vit_small_patch14_dinov2.lvd142m"
    dino_checkpoint_path: Path = Path("../data/models/dino_cache")
    dino_weights_path: Path = Path("../data/models/dinov2_vits14.pt")
    model_manifest_path: Path = Path("../data/models/MODEL_MANIFEST.json")
    inference_device: str = "cpu"
    embedding_batch_size: int = 4
    offline_mode: bool = True

    # ── Qdrant ────────────────────────────────────────────────
    qdrant_mode: Literal["local", "server"] = "local"
    qdrant_local_path: Path = Path("../data/qdrant_store")
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_remoteclip_collection: str = "remoteclip_tiles"
    qdrant_dino_collection: str = "dino_tiles"

    # ── Change Detection ──────────────────────────────────────
    change_confidence_threshold: float = 0.45
    change_suppression_threshold: float = 0.30

    # ── CORS ──────────────────────────────────────────────────
    frontend_url: str = "http://localhost:3000"

    @model_validator(mode="after")
    def resolve_paths(self) -> "Settings":
        """Resolve all Path fields to absolute paths relative to the backend dir."""
        base = Path(__file__).parent.parent
        path_fields = [
            "data_dir", "imagery_dir", "tiles_dir", "thumbnails_dir", "previews_dir",
            "models_dir", "exports_dir", "maps_dir", "qdrant_store_dir",
            "remoteclip_checkpoint_path", "dino_checkpoint_path",
            "dino_weights_path", "model_manifest_path", "qdrant_local_path",
        ]
        for field in path_fields:
            value: Path = getattr(self, field)
            if not value.is_absolute():
                object.__setattr__(self, field, (base / value).resolve())

        # Enforce offline mode by default
        if self.offline_mode:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        return self

    def find_local_checkpoint(self, filename: str) -> Path | None:
        """
        Search candidate locations for a local model checkpoint file.
        Checks:
        1. settings.models_dir / filename
        2. Workspace root / data / models / filename
        3. settings.data_dir / filename
        """
        candidates = [
            self.models_dir / filename,
            Path(__file__).resolve().parents[3] / "data" / "models" / filename,
            self.data_dir / filename,
        ]
        for c in candidates:
            if c.exists():
                return c.resolve()
        return None

    def ensure_directories(self) -> None:
        """Create all required data directories if they don't exist."""
        dirs = [
            self.data_dir,
            self.imagery_dir,
            self.tiles_dir,
            self.thumbnails_dir,
            self.previews_dir,
            self.models_dir,
            self.exports_dir,
            self.maps_dir,
            self.qdrant_store_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# Module-level singleton
settings = Settings()
