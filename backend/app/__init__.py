"""
GeoSemantic backend application package.
Initializes PROJ environment to ensure rasterio uses Python's internal PROJ data.
"""
from __future__ import annotations

import os

try:
    import pyproj
    proj_dir = pyproj.datadir.get_data_dir()
    os.environ["PROJ_LIB"] = proj_dir
    os.environ["PROJ_DATA"] = proj_dir
except Exception:
    pass
