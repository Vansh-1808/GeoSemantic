"""
Generate synthetic demo GeoTIFFs for development/testing.
Creates realistic georeferenced imagery with varied land-cover patterns.

IMPORTANT: These are synthetic files for pipeline testing ONLY.
They must never be presented as real satellite data in the SIH evaluation.
Real imagery must be used for actual demonstrations.

Usage:
    python scripts/generate_demo_data.py
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import os
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
try:
    import pyproj
    os.environ["PROJ_LIB"] = pyproj.datadir.get_data_dir()
    os.environ["PROJ_DATA"] = pyproj.datadir.get_data_dir()
except Exception:
    pass

from app.core.config import settings

try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds
except ImportError:
    print("✗ rasterio not installed. Run: pip install rasterio")
    sys.exit(1)


# ── Scene definitions ─────────────────────────────────────────
SCENES = [
    {
        "name": "demo_urban_2022",
        "description": "Urban area — 2022 baseline",
        "bbox": (77.10, 28.55, 77.25, 28.65),  # Delhi region (lon_min, lat_min, lon_max, lat_max)
        "date": "2022-06-15",
        "sensor": "Demo-MSI",
        "pattern": "urban",
        "size": (512, 512),
    },
    {
        "name": "demo_urban_2024",
        "description": "Urban area — 2024 with new construction",
        "bbox": (77.10, 28.55, 77.25, 28.65),
        "date": "2024-03-20",
        "sensor": "Demo-MSI",
        "pattern": "urban_construction",
        "size": (512, 512),
    },
    {
        "name": "demo_agricultural_2023",
        "description": "Agricultural area — seasonal variation",
        "bbox": (75.80, 30.10, 75.95, 30.20),
        "date": "2023-11-10",
        "sensor": "Demo-MSI",
        "pattern": "agricultural",
        "size": (512, 512),
    },
    {
        "name": "demo_water_body_2023",
        "description": "Reservoir with surrounding land",
        "bbox": (76.40, 29.20, 76.55, 29.30),
        "date": "2023-07-05",
        "sensor": "Demo-MSI",
        "pattern": "water",
        "size": (512, 512),
    },
    {
        "name": "demo_industrial_2024",
        "description": "Industrial zone near roads",
        "bbox": (77.30, 28.80, 77.45, 28.90),
        "date": "2024-01-12",
        "sensor": "Demo-MSI",
        "pattern": "industrial",
        "size": (512, 512),
    },
]


def generate_scene(scene_def: dict, out_dir: Path) -> Path:
    """Generate a synthetic 3-band GeoTIFF for a scene definition."""
    name = scene_def["name"]
    w, s, e, n = scene_def["bbox"]
    width, height = scene_def["size"]
    pattern = scene_def["pattern"]
    date_str = scene_def["date"]
    sensor = scene_def["sensor"]

    out_path = out_dir / f"{name}_{date_str}.tif"
    if out_path.exists():
        print(f"  [OK] Already exists: {out_path.name}")
        return out_path

    data = _generate_pattern(width, height, pattern)
    transform = from_bounds(w, s, e, n, width, height)

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs=CRS.from_epsg(4326),
        transform=transform,
        compress="lzw",
    ) as ds:
        ds.write(data)
        ds.update_tags(
            SCENE_ID=name,
            ACQUISITION_DATE=date_str,
            ACQUISITIONDATETIME=f"{date_str}T10:30:00+05:30",
            SENSOR_ID=sensor,
            SPACECRAFT_ID=sensor,
            DESCRIPTION=scene_def["description"],
            SYNTHETIC="TRUE",
            GENERATED_BY="GeoSemantic Demo Generator",
        )

    size_kb = out_path.stat().st_size // 1024
    print(f"  [OK] {out_path.name} ({size_kb} KB, {pattern} pattern)")
    return out_path


def _generate_pattern(width: int, height: int, pattern: str) -> np.ndarray:
    """Generate a 3-band (RGB) array for a given land-cover pattern."""
    rng = np.random.default_rng(hash(pattern) % (2**32))
    r = np.zeros((height, width), dtype=np.uint8)
    g = np.zeros((height, width), dtype=np.uint8)
    b = np.zeros((height, width), dtype=np.uint8)

    if pattern == "urban":
        # Grey tones with structured grid-like features
        base = rng.integers(80, 150, (height, width), dtype=np.uint8)
        r[:] = base
        g[:] = (base * 0.95).astype(np.uint8)
        b[:] = (base * 0.9).astype(np.uint8)
        # Road grid
        for i in range(0, height, 40):
            r[i:i+4, :] = 160
            g[i:i+4, :] = 160
            b[i:i+4, :] = 160
        for j in range(0, width, 40):
            r[:, j:j+4] = 160
            g[:, j:j+4] = 160
            b[:, j:j+4] = 160

    elif pattern == "urban_construction":
        # Same as urban but with bright construction patches
        base = rng.integers(80, 150, (height, width), dtype=np.uint8)
        r[:] = base
        g[:] = (base * 0.95).astype(np.uint8)
        b[:] = (base * 0.9).astype(np.uint8)
        for i in range(0, height, 40):
            r[i:i+4, :] = 160
            g[i:i+4, :] = 160
            b[i:i+4, :] = 160
        for j in range(0, width, 40):
            r[:, j:j+4] = 160
            g[:, j:j+4] = 160
            b[:, j:j+4] = 160
        # New construction patches: bright bare soil colour
        for _ in range(5):
            py = rng.integers(50, height - 80)
            px = rng.integers(50, width - 80)
            ph, pw = rng.integers(30, 80), rng.integers(30, 80)
            r[py:py+ph, px:px+pw] = rng.integers(190, 220)
            g[py:py+ph, px:px+pw] = rng.integers(170, 200)
            b[py:py+ph, px:px+pw] = rng.integers(140, 170)

    elif pattern == "agricultural":
        # Green tones with field parcels
        r[:] = rng.integers(50, 100, (height, width), dtype=np.uint8)
        g[:] = rng.integers(120, 180, (height, width), dtype=np.uint8)
        b[:] = rng.integers(40, 80, (height, width), dtype=np.uint8)
        # Field boundaries
        for i in range(0, height, 60):
            r[i:i+3, :] = 100
            g[i:i+3, :] = 80
            b[i:i+3, :] = 60
        for j in range(0, width, 60):
            r[:, j:j+3] = 100
            g[:, j:j+3] = 80
            b[:, j:j+3] = 60

    elif pattern == "water":
        # Blue water body in center, green/brown surroundings
        r[:] = rng.integers(80, 130, (height, width), dtype=np.uint8)
        g[:] = rng.integers(100, 150, (height, width), dtype=np.uint8)
        b[:] = rng.integers(60, 100, (height, width), dtype=np.uint8)
        # Central water body (dark blue)
        cy, cx = height // 2, width // 2
        ry, rx = height // 4, width // 4
        for y in range(height):
            for x in range(width):
                if ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 < 1:
                    r[y, x] = rng.integers(20, 50)
                    g[y, x] = rng.integers(50, 90)
                    b[y, x] = rng.integers(120, 180)

    elif pattern == "industrial":
        # Grey/tan with large rectangular structures
        r[:] = rng.integers(130, 170, (height, width), dtype=np.uint8)
        g[:] = rng.integers(125, 165, (height, width), dtype=np.uint8)
        b[:] = rng.integers(110, 150, (height, width), dtype=np.uint8)
        # Industrial buildings
        for _ in range(8):
            py = rng.integers(20, height - 80)
            px = rng.integers(20, width - 80)
            ph, pw = rng.integers(40, 100), rng.integers(60, 150)
            shade = rng.integers(100, 140)
            r[py:py+ph, px:px+pw] = shade
            g[py:py+ph, px:px+pw] = shade - 5
            b[py:py+ph, px:px+pw] = shade - 10

    else:
        # Generic noise
        r[:] = rng.integers(50, 200, (height, width), dtype=np.uint8)
        g[:] = rng.integers(50, 200, (height, width), dtype=np.uint8)
        b[:] = rng.integers(50, 200, (height, width), dtype=np.uint8)

    # Add realistic noise
    noise = rng.integers(-10, 10, (3, height, width))
    data = np.stack([r, g, b], axis=0).astype(np.int16) + noise
    return np.clip(data, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    print("=" * 60)
    print("GeoSemantic Platform - Demo Data Generator")
    print("[WARNING] SYNTHETIC DATA - Development/testing only.")
    print("   Do NOT use in SIH evaluation demonstrations.")
    print("=" * 60)

    settings.ensure_directories()

    print(f"\nGenerating {len(SCENES)} synthetic scenes in {settings.imagery_dir}...\n")
    generated = []
    for scene_def in SCENES:
        path = generate_scene(scene_def, settings.imagery_dir)
        generated.append({"name": scene_def["name"], "path": str(path)})

    manifest_path = settings.imagery_dir / "DEMO_MANIFEST.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "note": "SYNTHETIC DATA — pipeline testing only",
                "scenes": generated,
            },
            f,
            indent=2,
        )

    print(f"\n[OK] Generated {len(generated)} scenes")
    print(f"[OK] Manifest: {manifest_path}")
    print("\nNext: python scripts/setup_db.py (if not done)")
    print("Then: start the backend and ingest these files via the UI")
