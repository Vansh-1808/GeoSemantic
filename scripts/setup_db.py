"""
Database setup script.
Creates the PostgreSQL database, enables PostGIS, and runs all Alembic migrations.
Run this ONCE before starting the application.

Usage:
    python scripts/setup_db.py

Requires:
    - PostgreSQL running locally
    - .env file with DATABASE_URL and DATABASE_SYNC_URL
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make backend/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from app.core.config import settings


def create_database() -> None:
    """Create the database if it doesn't exist."""
    # Parse connection details from sync URL
    url = str(settings.database_sync_url)
    # postgresql+psycopg2://user:pass@host:port/dbname
    url = url.replace("postgresql+psycopg2://", "")
    if "@" in url:
        credentials, host_db = url.split("@", 1)
        user, password = credentials.split(":", 1) if ":" in credentials else (credentials, "")
    else:
        host_db = url
        user, password = "postgres", ""

    if "/" in host_db:
        host_port, dbname = host_db.rsplit("/", 1)
    else:
        host_port, dbname = host_db, "geosemantic"

    host = host_port.split(":")[0]
    port = int(host_port.split(":")[1]) if ":" in host_port else 5432

    print(f"Connecting to PostgreSQL at {host}:{port} as '{user}'...")

    # Connect to the default 'postgres' database to create our DB
    conn = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database="postgres",
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    # Check if database exists
    cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (dbname,))
    exists = cur.fetchone()

    if not exists:
        print(f"Creating database '{dbname}'...")
        cur.execute(f'CREATE DATABASE "{dbname}"')
        print(f"  [OK] Database '{dbname}' created")
    else:
        print(f"  [OK] Database '{dbname}' already exists")

    cur.close()
    conn.close()

    # Now connect to the target database and enable extensions
    print("Enabling PostGIS extensions...")
    conn2 = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=dbname,
    )
    conn2.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur2 = conn2.cursor()

    extensions = ["postgis", "postgis_topology", "uuid-ossp"]
    for ext in extensions:
        try:
            cur2.execute(f'CREATE EXTENSION IF NOT EXISTS "{ext}"')
            print(f"  [OK] Extension '{ext}' enabled")
        except Exception as e:
            print(f"  [WARN] Could not enable '{ext}': {e}")

    cur2.close()
    conn2.close()


def run_migrations() -> None:
    """Run Alembic migrations."""
    import subprocess

    print("\nRunning Alembic migrations...")
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=str(Path(__file__).parent.parent / "backend"),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Migration failed:\n{result.stderr}")
        sys.exit(1)
    else:
        print("  [OK] Migrations applied successfully")
        if result.stdout:
            print(result.stdout)


def ensure_data_directories() -> None:
    """Create all required data directories."""
    print("\nCreating data directories...")
    settings.ensure_directories()
    dirs = [
        settings.imagery_dir,
        settings.tiles_dir,
        settings.thumbnails_dir,
        settings.models_dir,
        settings.exports_dir,
        settings.maps_dir,
        settings.qdrant_store_dir,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] {d}")


if __name__ == "__main__":
    print("=" * 60)
    print("GeoSemantic Platform — Database Setup")
    print("=" * 60)

    # Check .env exists
    env_path = Path(__file__).parent.parent / "backend" / ".env"
    if not env_path.exists():
        example = env_path.parent / ".env.example"
        if example.exists():
            import shutil
            print("[WARN] No .env found. Copying from .env.example...")
            shutil.copy(example, env_path)
            print("  Please edit backend/.env and set your PostgreSQL credentials,")
            print("  then re-run this script.")
            sys.exit(1)
        else:
            print("[FAIL] No .env or .env.example found. Cannot proceed.")
            sys.exit(1)

    ensure_data_directories()
    create_database()
    run_migrations()

    print("\n" + "=" * 60)
    print("[OK] Database setup complete!")
    print("  Next step: python scripts/download_models.py")
    print("=" * 60)
