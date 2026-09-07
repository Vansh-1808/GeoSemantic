import planetary_computer
from pystac_client import Client
import requests
import os
import sys
if "PROJ_LIB" in os.environ: del os.environ["PROJ_LIB"]
if "PROJ_DATA" in os.environ: del os.environ["PROJ_DATA"]
import rasterio

# Search parameters for a changing area
# Let's pick a rapidly changing area: Dubai construction or an airport.
# Dubai Al Maktoum Airport expansion: lon, lat: 55.17, 24.89
bbox = [55.15, 24.87, 55.19, 24.91]

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

print("Searching for Before image (2020)...")
search_before = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime="2020-01-01/2020-03-31",
    query={"eo:cloud_cover": {"lt": 5}}
)
items_before = list(search_before.items())

print("Searching for After image (2026)...")
search_after = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime="2025-01-01/2026-09-01",
    query={"eo:cloud_cover": {"lt": 5}}
)
items_after = list(search_after.items())

if not items_before or not items_after:
    print("Could not find suitable images.")
    exit(1)

item_b = items_before[0]
item_a = items_after[0]

print(f"Before: {item_b.id} ({item_b.datetime})")
print(f"After: {item_a.id} ({item_a.datetime})")

output_dir = os.path.abspath("../data/imagery")
os.makedirs(output_dir, exist_ok=True)

for prefix, item in [("before", item_b), ("after", item_a)]:
    asset = item.assets["visual"] # True color image
    href = asset.href
    print(f"Downloading {prefix} image from {href}...")
    
    # We can use rasterio to read a window if the image is huge, but let's just download the visual TIF.
    # Sentinel-2 visual is a COG. We can read just the window we want.
    
    with rasterio.open(href) as src:
        import pyproj
        from rasterio.windows import from_bounds
        from rasterio.warp import transform_bounds
        
        # Transform bbox to CRS of the image
        dst_bbox = transform_bounds("EPSG:4326", src.crs, *bbox)
        window = from_bounds(*dst_bbox, src.transform)
        
        # Read data
        data = src.read(window=window)
        
        # Save locally
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": window.height,
            "width": window.width,
            "transform": src.window_transform(window)
        })
        
        out_path = os.path.join(output_dir, f"dubai_airport_{prefix}_{item.datetime.strftime('%Y%m%d')}.tif")
        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(data)
            
        print(f"Saved to {out_path}")

print("Done downloading sample datasets!")
