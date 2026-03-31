"""
Fix PROJ_LIB to use a proj.db with DATABASE.LAYOUT.VERSION.MINOR >= 4.

The ArcGIS conda environment has multiple PROJ installations that can
conflict. This module finds a compatible proj.db and sets PROJ_LIB
before rasterio or geopandas are imported.
"""

import os
import sqlite3
from pathlib import Path


def _proj_db_is_compatible(proj_dir: str) -> bool:
    """Check if a proj.db has VERSION.MINOR >= 4."""
    db_path = Path(proj_dir) / "proj.db"
    if not db_path.exists():
        return False
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.execute(
            "SELECT value FROM metadata "
            "WHERE key = 'DATABASE.LAYOUT.VERSION.MINOR'"
        )
        row = cur.fetchone()
        conn.close()
        return row is not None and int(row[0]) >= 4
    except Exception:
        return False


def fix_proj_lib():
    # Candidate directories, in priority order
    candidates = []

    # 1. rasterio's bundled proj_data (most likely to match rasterio's PROJ)
    try:
        import rasterio
        rasterio_proj = Path(rasterio.__file__).parent / "proj_data"
        candidates.append(str(rasterio_proj))
    except Exception:
        pass

    # 2. ArcGIS Pro's PROJ data
    candidates.append(
        r"C:\Program Files\ArcGIS\Pro\Resources\pedata\gdaldata"
    )

    # 3. pyproj's bundled data
    try:
        import pyproj
        candidates.append(pyproj.datadir.get_data_dir())
    except Exception:
        pass

    for candidate in candidates:
        if candidate and _proj_db_is_compatible(candidate):
            os.environ['PROJ_LIB'] = candidate
            os.environ['PROJ_DATA'] = candidate  # newer PROJ uses this
            return

    # If nothing worked, at least try pyproj's dir
    try:
        import pyproj
        d = pyproj.datadir.get_data_dir()
        if d:
            os.environ['PROJ_LIB'] = d
    except Exception:
        pass


fix_proj_lib()
