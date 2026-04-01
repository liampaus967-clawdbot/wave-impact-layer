"""
Lake configuration loading and management.

Primary source: local data/lakes/{lake_id}/config.json files
Fallback: PostGIS database (hydrology.lakes)
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .geo_utils import utm_epsg_from_lonlat

logger = logging.getLogger(__name__)


@dataclass
class LakeConfig:
    """Configuration for a single lake."""
    lake_id: str
    name: str
    bbox: list  # [min_lon, min_lat, max_lon, max_lat]
    center: list  # [lon, lat]
    avg_depth_m: float
    utm_epsg: int = 0
    state: str = ""
    area_km2: float = 0.0
    gnis_id: str = ""
    uuid: str = ""

    def __post_init__(self):
        if self.utm_epsg == 0:
            self.utm_epsg = utm_epsg_from_lonlat(self.center[0], self.center[1])

    @property
    def utm_crs(self) -> str:
        return f"EPSG:{self.utm_epsg}"

    @property
    def lat(self) -> float:
        return self.center[1]

    @property
    def lon(self) -> float:
        return self.center[0]


def _find_data_root() -> Path:
    """Find the data/ directory relative to the project root."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "data"
        if (candidate / "lakes").is_dir():
            return candidate
    for parent in current.parents:
        candidate = parent / "data"
        if candidate.is_dir():
            return candidate
    return Path("data")


def load_lake_config_from_db(lake_name: str) -> LakeConfig:
    """
    Load lake config by querying PostGIS.

    Args:
        lake_name: Lake name (partial match, case-insensitive)

    Returns:
        LakeConfig instance
    """
    from .db import get_lake_by_name, lake_row_to_config

    gdf = get_lake_by_name(lake_name)
    if len(gdf) == 0:
        raise FileNotFoundError(f"Lake '{lake_name}' not found in database")

    row = gdf.iloc[0]
    cfg = lake_row_to_config(row)

    return LakeConfig(
        lake_id=_slugify(cfg['name']),
        name=cfg['name'],
        bbox=cfg['bbox'],
        center=cfg['center'],
        avg_depth_m=cfg['avg_depth_m'],
        utm_epsg=cfg['utm_epsg'],
        area_km2=cfg['area_km2'],
        uuid=cfg['uuid'],
    )


def load_lake_config(lake_id: str, data_root: Optional[Path] = None) -> LakeConfig:
    """
    Load lake configuration. Uses local JSON if available, falls back to database.

    Args:
        lake_id: Lake identifier (e.g. 'champlain', 'Lake Champlain')
        data_root: Path to data/ directory (auto-detected if None)

    Returns:
        LakeConfig instance
    """
    if data_root is None:
        data_root = _find_data_root()

    # Try local JSON config first (avoids DB query when data already exists)
    # Check multiple slug variants: full slug, raw id, and without "lake-" prefix
    slug = _slugify(lake_id)
    candidates = [slug, lake_id]
    # "Lake Champlain" slugifies to "lake-champlain" but dir might be "champlain"
    if slug.startswith('lake-'):
        candidates.append(slug[5:])

    config_path = None
    for candidate in candidates:
        path = data_root / "lakes" / candidate / "config.json"
        if path.exists():
            config_path = path
            break

    if config_path is not None:
        logger.debug(f"Loading '{lake_id}' from local config: {config_path}")
        with open(config_path) as f:
            raw = json.load(f)

        return LakeConfig(
            lake_id=config_path.parent.name,
            name=raw["name"],
            gnis_id=raw.get("gnis_id", ""),
            bbox=raw["bbox"],
            center=raw["center"],
            avg_depth_m=raw.get("avg_depth_m", 10.0),
            utm_epsg=raw.get("utm_epsg", 0),
            state=raw.get("state", ""),
            area_km2=raw.get("area_km2", 0.0),
        )

    # Fall back to database
    try:
        return load_lake_config_from_db(lake_id)
    except Exception as e:
        raise FileNotFoundError(
            f"Lake '{lake_id}' not found locally (tried {candidates}) or in database: {e}"
        )


def list_lakes_local(min_area_km2: float = 5.0,
                     states: list = None,
                     data_root: Optional[Path] = None) -> list:
    """
    List lakes from local config files, filtering by area and state.

    Returns:
        List of LakeConfig instances matching the filters
    """
    if data_root is None:
        data_root = _find_data_root()

    lakes_dir = data_root / "lakes"
    if not lakes_dir.exists():
        return []

    configs = []
    for d in sorted(lakes_dir.iterdir()):
        config_path = d / "config.json"
        if not d.is_dir() or not config_path.exists():
            continue
        with open(config_path) as f:
            raw = json.load(f)
        area = raw.get("area_km2", 0.0)
        state = raw.get("state", "").upper()
        if area < min_area_km2:
            continue
        if states and state not in states:
            continue
        configs.append(LakeConfig(
            lake_id=d.name,
            name=raw["name"],
            gnis_id=raw.get("gnis_id", ""),
            bbox=raw["bbox"],
            center=raw["center"],
            avg_depth_m=raw.get("avg_depth_m", 10.0),
            utm_epsg=raw.get("utm_epsg", 0),
            state=state,
            area_km2=area,
        ))

    return configs


def list_lakes_from_db(min_area_km2: float = 5.0,
                       states: list = None) -> list:
    """
    List lakes from PostGIS, returning LakeConfig objects.

    Args:
        min_area_km2: Minimum lake area filter
        states: Optional state abbreviation filter

    Returns:
        List of LakeConfig instances
    """
    from .db import query_lakes, lake_row_to_config

    gdf = query_lakes(min_area_km2=min_area_km2, states=states)
    configs = []

    for _, row in gdf.iterrows():
        cfg = lake_row_to_config(row)
        configs.append(LakeConfig(
            lake_id=_slugify(cfg['name']),
            name=cfg['name'],
            bbox=cfg['bbox'],
            center=cfg['center'],
            avg_depth_m=cfg['avg_depth_m'],
            utm_epsg=cfg['utm_epsg'],
            area_km2=cfg['area_km2'],
            uuid=cfg['uuid'],
        ))

    return configs


def list_lakes(data_root: Optional[Path] = None) -> list:
    """Return list of lake_id strings that have a local config.json."""
    if data_root is None:
        data_root = _find_data_root()

    lakes_dir = data_root / "lakes"
    if not lakes_dir.exists():
        return []

    return sorted([
        d.name for d in lakes_dir.iterdir()
        if d.is_dir() and (d / "config.json").exists()
    ])


def _slugify(name: str) -> str:
    """Convert lake name to a filesystem-safe ID."""
    import re
    slug = name.lower().strip()
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    return slug.strip('-')
