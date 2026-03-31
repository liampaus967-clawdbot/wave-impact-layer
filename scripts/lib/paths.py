"""
Canonical path conventions for lake data and outputs.
"""

from pathlib import Path
from typing import Optional


class LakePaths:
    """
    Provides canonical paths for all lake-related files.

    Layout:
        data/lakes/{lake_id}/config.json
        data/lakes/{lake_id}/polygon.geojson
        data/lakes/{lake_id}/raster.tif
        data/lakes/{lake_id}/bays.json          (optional)
        data/fetch_rasters/{lake_id}/fetch_*.tif
        data/output/{lake_id}/*.geojson
    """

    def __init__(self, lake_id: str, data_root: Optional[Path] = None):
        if data_root is None:
            # Walk up from this file to find the project root.
            # Look for data/lakes/ to avoid false matches.
            current = Path(__file__).resolve()
            for parent in current.parents:
                candidate = parent / "data"
                if (candidate / "lakes").is_dir():
                    data_root = candidate
                    break
            if data_root is None:
                for parent in current.parents:
                    candidate = parent / "data"
                    if candidate.is_dir():
                        data_root = candidate
                        break
            if data_root is None:
                data_root = Path("data")

        self.lake_id = lake_id
        self.data_root = Path(data_root)
        self.lake_dir = self.data_root / "lakes" / lake_id
        self.fetch_dir = self.data_root / "fetch_rasters" / lake_id
        self.output_dir = self.data_root / "output" / lake_id

    @property
    def config(self) -> Path:
        return self.lake_dir / "config.json"

    @property
    def polygon(self) -> Path:
        return self.lake_dir / "polygon.geojson"

    @property
    def raster(self) -> Path:
        return self.lake_dir / "raster.tif"

    @property
    def bays(self) -> Path:
        return self.lake_dir / "bays.json"

    def ensure_dirs(self):
        """Create all directories for this lake."""
        self.lake_dir.mkdir(parents=True, exist_ok=True)
        self.fetch_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
