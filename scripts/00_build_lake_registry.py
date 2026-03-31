#!/usr/bin/env python3
"""
Build National Lake Registry

Queries NHD HR (National Hydrography Dataset High-Resolution) for all US lakes
above a configurable area threshold and generates per-lake config files.

Usage:
    python scripts/00_build_lake_registry.py --min-area-km2 5.0 --all-states
    python scripts/00_build_lake_registry.py --min-area-km2 1.0 --states VT,NY,NH
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path
import sys

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.geo_utils import utm_epsg_from_lonlat
from lib.depth_estimation import estimate_depth

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# NHD HR WFS endpoint for waterbodies (NHDWaterbody layer)
NHD_URL = "https://hydro.nationalmap.gov/arcgis/rest/services/NHDPlus_HR/MapServer/9/query"

# HUC2 regions covering CONUS + Alaska/Hawaii
# Each region covers a major drainage basin
HUC2_REGIONS = {
    '01': 'New England',
    '02': 'Mid-Atlantic',
    '03': 'South Atlantic-Gulf',
    '04': 'Great Lakes',
    '05': 'Ohio',
    '06': 'Tennessee',
    '07': 'Upper Mississippi',
    '08': 'Lower Mississippi',
    '09': 'Souris-Red-Rainy',
    '10': 'Missouri',
    '11': 'Arkansas-White-Red',
    '12': 'Texas-Gulf',
    '13': 'Rio Grande',
    '14': 'Upper Colorado',
    '15': 'Lower Colorado',
    '16': 'Great Basin',
    '17': 'Pacific Northwest',
    '18': 'California',
    '19': 'Alaska',
    '20': 'Hawaii',
    '21': 'Caribbean',
}

# US state FIPS codes for state filtering via centroid
# We use a simple bounding box approach per state
STATE_ABBREVS = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
]

# Max records per API request (NHD typically allows up to 2000)
PAGE_SIZE = 1000


def slugify(name: str) -> str:
    """Convert lake name to a filesystem-safe ID."""
    slug = name.lower().strip()
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    slug = slug.strip('-')
    return slug


def query_nhd_lakes(min_area_km2: float = 5.0, states: list = None,
                    max_results: int = 10000) -> list:
    """
    Query NHD HR for lakes/ponds above a minimum area threshold.

    Args:
        min_area_km2: Minimum lake area in km^2
        states: Optional list of state abbreviations to filter by
        max_results: Maximum total results to return

    Returns:
        List of dicts with lake attributes
    """
    # Convert area to square meters for NHD query (AreaSqKm field is in km2)
    where_clause = (
        f"FTYPE = 390 AND AreaSqKm >= {min_area_km2}"
    )

    if states:
        # NHD doesn't have a state field directly, so we query broadly
        # and filter by centroid later
        logger.info(f"Will filter by states: {', '.join(states)}")

    all_lakes = []
    offset = 0

    while len(all_lakes) < max_results:
        params = {
            'where': where_clause,
            'outFields': 'GNIS_ID,GNIS_NAME,AreaSqKm,OBJECTID',
            'returnGeometry': 'true',
            'outSR': '4326',
            'f': 'geojson',
            'resultRecordCount': PAGE_SIZE,
            'resultOffset': offset,
            'orderByFields': 'AreaSqKm DESC',
        }

        try:
            logger.info(f"Querying NHD HR (offset={offset})...")
            response = requests.get(NHD_URL, params=params, timeout=120)
            response.raise_for_status()
            data = response.json()

            features = data.get('features', [])
            if not features:
                logger.info("No more features returned.")
                break

            for feat in features:
                props = feat.get('properties', {})
                geom = feat.get('geometry', {})

                name = props.get('GNIS_NAME', '')
                if not name:
                    continue  # Skip unnamed water bodies

                gnis_id = props.get('GNIS_ID', '')
                area_km2 = props.get('AreaSqKm', 0)

                # Calculate centroid from geometry
                coords = geom.get('coordinates', [])
                if not coords:
                    continue

                centroid = _geometry_centroid(geom)
                if centroid is None:
                    continue

                lon, lat = centroid

                # Calculate bounding box from geometry
                bbox = _geometry_bbox(geom)

                all_lakes.append({
                    'name': name,
                    'gnis_id': str(gnis_id),
                    'area_km2': round(area_km2, 2),
                    'center': [round(lon, 4), round(lat, 4)],
                    'bbox': [round(v, 4) for v in bbox],
                    'lat': lat,
                    'lon': lon,
                })

            logger.info(f"  Retrieved {len(features)} features, total: {len(all_lakes)}")

            if len(features) < PAGE_SIZE:
                break  # Last page

            offset += PAGE_SIZE
            time.sleep(0.5)  # Be polite to the API

        except requests.exceptions.RequestException as e:
            logger.error(f"NHD query failed at offset {offset}: {e}")
            break

    logger.info(f"Total lakes retrieved: {len(all_lakes)}")

    # Deduplicate by GNIS_ID
    seen = set()
    unique_lakes = []
    for lake in all_lakes:
        key = lake['gnis_id'] or lake['name']
        if key not in seen:
            seen.add(key)
            unique_lakes.append(lake)

    logger.info(f"Unique lakes after dedup: {len(unique_lakes)}")
    return unique_lakes


def _geometry_centroid(geom: dict) -> tuple:
    """Calculate approximate centroid from GeoJSON geometry."""
    coords = _flatten_coords(geom)
    if not coords:
        return None
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return (sum(lons) / len(lons), sum(lats) / len(lats))


def _geometry_bbox(geom: dict) -> list:
    """Calculate bounding box from GeoJSON geometry."""
    coords = _flatten_coords(geom)
    if not coords:
        return [0, 0, 0, 0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return [min(lons), min(lats), max(lons), max(lats)]


def _flatten_coords(geom: dict) -> list:
    """Flatten nested GeoJSON coordinate arrays into list of [lon, lat]."""
    gtype = geom.get('type', '')
    coords = geom.get('coordinates', [])

    if gtype == 'Point':
        return [coords]
    elif gtype == 'MultiPoint' or gtype == 'LineString':
        return coords
    elif gtype == 'Polygon':
        # First ring is exterior
        return coords[0] if coords else []
    elif gtype == 'MultiPolygon':
        result = []
        for polygon in coords:
            if polygon:
                result.extend(polygon[0])
        return result
    return []


def build_registry(lakes: list, data_root: Path, states_filter: list = None):
    """
    Build per-lake config files and master registry.

    Args:
        lakes: List of lake dicts from query_nhd_lakes
        data_root: Path to data/ directory
        states_filter: Optional list of state abbreviations to keep
    """
    registry = []

    for lake in lakes:
        lake_id = slugify(lake['name'])

        # Skip duplicates (same slug)
        lake_dir = data_root / "lakes" / lake_id
        config_path = lake_dir / "config.json"

        # Estimate depth
        depth = estimate_depth(lake['area_km2'], lake['name'])

        # Compute UTM EPSG
        utm_epsg = utm_epsg_from_lonlat(lake['center'][0], lake['center'][1])

        config = {
            'name': lake['name'],
            'gnis_id': lake['gnis_id'],
            'bbox': lake['bbox'],
            'center': lake['center'],
            'avg_depth_m': round(depth, 1),
            'utm_epsg': utm_epsg,
            'state': '',  # Could be enriched with reverse geocoding
            'area_km2': lake['area_km2'],
        }

        # Write per-lake config
        lake_dir.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        registry.append({
            'lake_id': lake_id,
            'name': lake['name'],
            'gnis_id': lake['gnis_id'],
            'area_km2': lake['area_km2'],
            'center': lake['center'],
            'status': 'discovered',
        })

    # Write master registry
    registry_path = data_root / "lake_registry.json"
    with open(registry_path, 'w') as f:
        json.dump({
            'total_lakes': len(registry),
            'lakes': registry,
        }, f, indent=2)

    logger.info(f"Registry saved to {registry_path} ({len(registry)} lakes)")
    logger.info(f"Per-lake configs written to {data_root / 'lakes'}/")


def main():
    parser = argparse.ArgumentParser(description='Build national lake registry from NHD HR')
    parser.add_argument('--min-area-km2', type=float, default=5.0,
                        help='Minimum lake area in km^2 (default: 5.0)')
    parser.add_argument('--states', type=str, default=None,
                        help='Comma-separated state abbreviations (e.g., VT,NY,NH)')
    parser.add_argument('--all-states', action='store_true',
                        help='Query all US states')
    parser.add_argument('--max-results', type=int, default=10000,
                        help='Maximum number of lakes to retrieve')
    parser.add_argument('--data-dir', type=Path, default=Path('data'),
                        help='Data root directory')

    args = parser.parse_args()

    states = None
    if args.states:
        states = [s.strip().upper() for s in args.states.split(',')]
    elif args.all_states:
        states = STATE_ABBREVS

    logger.info(f"Building lake registry (min area: {args.min_area_km2} km²)")

    # Query NHD
    lakes = query_nhd_lakes(
        min_area_km2=args.min_area_km2,
        states=states,
        max_results=args.max_results,
    )

    if not lakes:
        logger.error("No lakes found!")
        return

    # Build registry and per-lake configs
    build_registry(lakes, args.data_dir, states)

    logger.info("Done!")


if __name__ == '__main__':
    main()
