"""
fetch_my_area.py - Download real Sentinel-2 true-color satellite GeoTIFFs for ANY location on Earth.
Uses Microsoft Planetary Computer's free, anonymous STAC API (100% open, zero login / no API key required).

Usage:
  python scripts/fetch_my_area.py --city "Delhi"
  python scripts/fetch_my_area.py --city "Mumbai"
  python scripts/fetch_my_area.py --city "Bengaluru"
  python scripts/fetch_my_area.py --city "Jaipur"
  python scripts/fetch_my_area.py --lat 28.5562 --lon 77.1000 --name "delhi_airport" --size 2048
"""
import argparse
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

# Safeguard: Ensure rasterio and GDAL use modern pyproj database
try:
    import pyproj
    proj_dir = pyproj.datadir.get_data_dir()
    os.environ["PROJ_LIB"] = proj_dir
    os.environ["PROJ_DATA"] = proj_dir
except Exception:
    pass

import httpx
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEST_DIR = Path(__file__).resolve().parent.parent / "backend" / "data" / "raw_scenes"
DEST_DIR.mkdir(parents=True, exist_ok=True)

# Preset coordinates for major cities (with fallback to geocoding)
POPULAR_CITIES = {
    "delhi": {"lat": 28.6139, "lon": 77.2090, "name": "delhi"},
    "delhi airport": {"lat": 28.5562, "lon": 77.1000, "name": "delhi_airport"},
    "mumbai": {"lat": 18.9500, "lon": 72.8500, "name": "mumbai"},
    "mumbai port": {"lat": 18.9400, "lon": 72.9500, "name": "mumbai_port"},
    "bengaluru": {"lat": 12.9716, "lon": 77.5946, "name": "bengaluru"},
    "jaipur": {"lat": 26.9124, "lon": 75.7873, "name": "jaipur"},
    "kolkata": {"lat": 22.5726, "lon": 88.3639, "name": "kolkata"},
    "hyderabad": {"lat": 17.3850, "lon": 78.4867, "name": "hyderabad"},
    "chennai": {"lat": 13.0827, "lon": 80.2707, "name": "chennai"},
    "chandigarh": {"lat": 30.7333, "lon": 76.7794, "name": "chandigarh"},
    "pune": {"lat": 18.5204, "lon": 73.8567, "name": "pune"},
    "ahmedabad": {"lat": 23.0225, "lon": 72.5714, "name": "ahmedabad"},
}

def geocode_city(city_name: str):
    key = city_name.strip().lower()
    if key in POPULAR_CITIES:
        return POPULAR_CITIES[key]["lat"], POPULAR_CITIES[key]["lon"], POPULAR_CITIES[key]["name"]
    
    # Try OpenStreetMap Nominatim for any arbitrary city/town worldwide
    print(f"[*] Geocoding '{city_name}' via OpenStreetMap Nominatim...")
    url = f"https://nominatim.openstreetmap.org/search?q={city_name}&format=json&limit=1"
    headers = {"User-Agent": "GeoSemantic-Downloader/1.0"}
    try:
        r = httpx.get(url, headers=headers, timeout=10)
        if r.status_code == 200 and len(r.json()) > 0:
            item = r.json()[0]
            lat = float(item["lat"])
            lon = float(item["lon"])
            clean_name = city_name.strip().lower().replace(" ", "_").replace(",", "")
            return lat, lon, clean_name
    except Exception as e:
        print(f"    [!] Geocoding error: {e}")
        
    return None, None, None

def fetch_area_scene(lat: float, lon: float, name: str, size_px: int = 2048):
    print("\n" + "=" * 70)
    print(f" Fetching Real Sentinel-2 Satellite GeoTIFF for: {name.upper()}")
    print(f" Center: Latitude {lat:.4f}°, Longitude {lon:.4f}° | Size: {size_px}x{size_px} px (~20 km)")
    print("=" * 70)

    # 1. Get anonymous SAS token from Microsoft Planetary Computer
    print("[1/4] Generating anonymous Planetary Computer token (zero-login)...")
    token_resp = httpx.get("https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a", timeout=15)
    if token_resp.status_code != 200:
        print("[!] Failed to obtain anonymous token from Planetary Computer.")
        return None
    token = token_resp.json().get("token", "")

    # 2. Query STAC catalog for the freshest, cloud-free Sentinel-2 L2A scene
    print("[2/4] Searching Sentinel-2 L2A satellite catalog for cloud-free imagery...")
    delta = 0.05
    stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": [lon - delta, lat - delta, lon + delta, lat + delta],
        "query": {"eo:cloud_cover": {"lt": 20}},
        "limit": 5,
        "sortby": [{"field": "datetime", "direction": "desc"}]
    }

    stac_resp = httpx.post(stac_url, json=payload, timeout=20)
    if stac_resp.status_code != 200:
        print("[!] STAC catalog search failed.")
        return None

    features = stac_resp.json().get("features", [])
    if not features:
        print("[!] No cloud-free Sentinel-2 scenes found for this area. Retrying with higher cloud allowance...")
        payload["query"]["eo:cloud_cover"]["lt"] = 50
        stac_resp = httpx.post(stac_url, json=payload, timeout=20)
        features = stac_resp.json().get("features", [])
        if not features:
            print("[!] Still no scenes found.")
            return None

    scene_item = features[0]
    scene_id = scene_item["id"]
    acq_date = scene_item["properties"].get("datetime", "")[:10]
    cloud_pct = scene_item["properties"].get("eo:cloud_cover", 0)
    visual_asset = scene_item["assets"].get("visual", {})
    if not visual_asset:
        print("[!] Visual true-color asset not found in scene.")
        return None

    visual_href = visual_asset["href"] + "?" + token
    print(f"    Selected Scene: {scene_id}")
    print(f"    Acquisition:    {acq_date} (Cloud cover: {cloud_pct:.1f}%)")

    # 3. Stream windowed region from the remote Cloud-Optimized GeoTIFF
    print(f"[3/4] Streaming {size_px}x{size_px} window directly from remote COG...")
    out_file = DEST_DIR / f"{name}_sentinel2_{acq_date}.tif"

    with rasterio.open(visual_href) as src:
        # Transform WGS84 lat/lon to raster native CRS coordinates
        xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
        cx, cy = xs[0], ys[0]

        # Convert native coordinates to pixel row/col
        center_row, center_col = src.index(cx, cy)
        half_size = size_px // 2
        col_off = max(0, min(src.width - size_px, center_col - half_size))
        row_off = max(0, min(src.height - size_px, center_row - half_size))

        window = Window(col_off=col_off, row_off=row_off, width=size_px, height=size_px)
        data = src.read(window=window)
        win_transform = rasterio.windows.transform(window, src.transform)

        profile = src.profile.copy()
        profile.update({
            "driver": "GTiff",
            "height": size_px,
            "width": size_px,
            "transform": win_transform,
            "compress": "lzw",
        })

        # Write georeferenced output GeoTIFF locally
        with rasterio.open(out_file, "w", **profile) as dst:
            dst.write(data)
            # Add metadata tags
            dst.update_tags(
                SENSOR="Sentinel-2B",
                ACQUISITION_DATE=acq_date,
                SCENE_ID=scene_id,
                AREA_NAME=name,
            )

    file_size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"[4/4] Successfully saved real satellite scene ({file_size_mb:.2f} MB):")
    print(f"    File: {out_file.name}")
    print(f"    Path: {out_file}")
    print("=" * 70)
    print(" [READY] You can now ingest this file directly on http://localhost:3000/ingest")
    print("=" * 70)
    return out_file

def main():
    parser = argparse.ArgumentParser(description="Fetch real Sentinel-2 satellite imagery for any area.")
    parser.add_argument("--city", type=str, default=None, help="Name of city or landmark (e.g. Delhi, Mumbai, Jaipur, Bengaluru)")
    parser.add_argument("--lat", type=float, default=None, help="Latitude of area center")
    parser.add_argument("--lon", type=float, default=None, help="Longitude of area center")
    parser.add_argument("--name", type=str, default=None, help="Output filename tag")
    parser.add_argument("--size", type=int, default=2048, help="Pixel size of square crop (default 2048 for ~20 km)")

    args = parser.parse_args()

    if args.city:
        lat, lon, clean_name = geocode_city(args.city)
        if lat is None:
            print(f"[!] Could not find coordinates for '{args.city}'. Please provide --lat and --lon.")
            sys.exit(1)
        name = args.name or clean_name
    elif args.lat is not None and args.lon is not None:
        lat, lon = args.lat, args.lon
        name = args.name or f"loc_{lat:.3f}_{lon:.3f}"
    else:
        # Default to Delhi if no arguments given
        print("[i] No city specified. Defaulting to 'Delhi'...")
        lat, lon, name = 28.6139, 77.2090, "delhi"

    fetch_area_scene(lat=lat, lon=lon, name=name, size_px=args.size)

if __name__ == "__main__":
    main()
