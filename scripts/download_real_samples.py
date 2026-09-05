"""
Download real-world georeferenced satellite GeoTIFF samples into backend/data/raw_scenes.
All URLs are public, direct, and require zero authentication.
"""
import sys
import os
from pathlib import Path
import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEST_DIR = Path(__file__).resolve().parent.parent / "backend" / "data" / "raw_scenes"
DEST_DIR.mkdir(parents=True, exist_ok=True)

# Direct public satellite GeoTIFF samples
SAMPLES = [
    {
        "filename": "sentinel2_real_scene.tif",
        "url": "https://opengeos.github.io/data/raster/sentinel2.tif",
        "desc": "Real Sentinel-2 True-Color (RGB) Multispectral GeoTIFF",
    },
    {
        "filename": "landsat7_highlands_scene.tif",
        "url": "https://github.com/rasterio/rasterio/raw/main/tests/data/RGB.byte.tif",
        "desc": "Real Landsat 7 Satellite Scene (Highlands & Water, EPSG:32613)",
    },
    {
        "filename": "cog_agricultural_scene.tif",
        "url": "https://github.com/opengeos/datasets/releases/download/raster/cog.tif",
        "desc": "Real Cloud-Optimized GeoTIFF (COG) Agricultural & River Corridor",
    },
]

def download_file(url: str, dest: Path, desc: str):
    print(f"\n[*] Downloading: {desc}...")
    print(f"    URL: {url}")
    print(f"    Destination: {dest.name}")
    
    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        with client.stream("GET", url) as response:
            if response.status_code != 200:
                print(f"    [!] Failed to download: HTTP {response.status_code}")
                return False
            
            total = int(response.headers.get("content-length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=65536):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = (downloaded / total) * 100
                        print(f"\r    Progress: {downloaded / (1024*1024):.2f} MB / {total / (1024*1024):.2f} MB ({pct:.1f}%)", end="")
                    else:
                        print(f"\r    Downloaded: {downloaded / (1024*1024):.2f} MB", end="")
    print(f"\n    [+] Saved successfully ({dest.stat().st_size / (1024*1024):.2f} MB)")
    return True

def main():
    print("=" * 70)
    print(" Downloading Real-World Satellite GeoTIFF Datasets")
    print("=" * 70)
    
    success_count = 0
    for s in SAMPLES:
        dest = DEST_DIR / s["filename"]
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"\n[i] File already exists: {dest.name} ({dest.stat().st_size / (1024*1024):.2f} MB), skipping.")
            success_count += 1
            continue
            
        if download_file(s["url"], dest, s["desc"]):
            success_count += 1
            
    print("\n" + "=" * 70)
    print(f" [SUCCESS] {success_count}/{len(SAMPLES)} real satellite scenes ready in:")
    print(f" {DEST_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
