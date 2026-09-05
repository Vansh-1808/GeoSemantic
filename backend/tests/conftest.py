"""
Pytest configuration and environment fixtures.
Configures PROJ data path before rasterio is imported.
"""
import os
import sys
from pathlib import Path

# Ensure backend root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["APP_ENV"] = "test"

try:
    import pyproj
    proj_dir = pyproj.datadir.get_data_dir()
    os.environ["PROJ_LIB"] = proj_dir
    os.environ["PROJ_DATA"] = proj_dir
except Exception:
    pass
