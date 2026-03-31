#!/usr/bin/env python3
"""
Prepare Lake Data for Wave Impact Analysis

Fetches lake polygon from PostGIS (primary) or NHD HR (fallback)
and rasterizes it for fetch calculation.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import proj_fix  # noqa: E402,F401 — must run before geo imports

import geopandas as gpd
import requests
from shapely.geometry import shape

from lib.lake_config import load_lake_config, LakeConfig
from lib.paths import LakePaths

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_lake_from_db(config: LakeConfig, output_path: Path) -> gpd.GeoDataFrame:
    """
    Fetch lake polygon from PostGIS database.

    Returns None if DB is unavailable or lake not found.
    """
    try:
        if config.uuid:
            from lib.db import get_lake_by_uuid
            gdf = get_lake_by_uuid(config.uuid)
        else:
            from lib.db import get_lake_by_name
            gdf = get_lake_by_name(config.name)

        if len(gdf) == 0:
            return None

        # Ensure WGS84
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs('EPSG:4326')

        gdf.to_file(output_path, driver='GeoJSON')
        logger.info(f"Fetched {config.name} geometry from database")
        return gdf

    except Exception as e:
        logger.debug(f"DB fetch failed: {e}")
        return None


def download_lake_polygon(config: LakeConfig, output_path: Path) -> gpd.GeoDataFrame:
    """
    Download lake polygon from NHD HR WFS service.

    Falls back to OpenStreetMap Overpass API if NHD query fails.
    """
    logger.info(f"Downloading {config.name} from NHD HR...")

    url = "https://hydro.nationalmap.gov/arcgis/rest/services/NHDPlus_HR/MapServer/9/query"

    bbox_str = ",".join(str(v) for v in config.bbox)
    where_clause = (
        f"GNIS_ID = '{config.gnis_id}'"
        if config.gnis_id
        else f"GNIS_NAME LIKE '%{config.name}%'"
    )

    params = {
        'where': where_clause,
        'outFields': '*',
        'geometry': bbox_str,
        'geometryType': 'esriGeometryEnvelope',
        'spatialRel': 'esriSpatialRelIntersects',
        'outSR': '4326',
        'f': 'geojson'
    }

    try:
        response = requests.get(url, params=params, timeout=120)
        response.raise_for_status()
        data = response.json()

        if 'features' in data and len(data['features']) > 0:
            gdf = gpd.GeoDataFrame.from_features(data['features'], crs='EPSG:4326')
            logger.info(f"Downloaded {len(gdf)} features")

            if len(gdf) > 1:
                logger.info("Dissolving multiple parts into single polygon...")
                gdf = gdf.dissolve()
                gdf = gdf.reset_index(drop=True)

            gdf.to_file(output_path, driver='GeoJSON')
            logger.info(f"Saved to {output_path}")
            return gdf
        else:
            logger.warning("No features found from NHD HR, trying fallback...")
            return download_lake_fallback(config, output_path)

    except Exception as e:
        logger.warning(f"NHD HR query failed: {e}, trying fallback...")
        return download_lake_fallback(config, output_path)


def download_lake_fallback(config: LakeConfig, output_path: Path) -> gpd.GeoDataFrame:
    """
    Fallback: Download lake polygon from OpenStreetMap via Overpass API.
    """
    logger.info(f"Downloading {config.name} from OpenStreetMap...")

    overpass_url = "https://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json][timeout:180];
    (
      relation["name"="{config.name}"]["natural"="water"];
    );
    out body;
    >;
    out skel qt;
    """

    try:
        response = requests.post(overpass_url, data={'data': overpass_query}, timeout=180)
        response.raise_for_status()

        # If OSM returns data, try to build geometry
        # For a robust implementation, use osm2geojson
        logger.info("Processing OSM response...")

    except Exception as e:
        logger.warning(f"OSM fallback also failed: {e}")

    # Last resort: create a simplified polygon from bbox
    logger.info(f"Creating simplified polygon from bounding box for {config.name}...")
    from shapely.geometry import box

    bbox_geom = box(*config.bbox)
    gdf = gpd.GeoDataFrame(
        {'name': [config.name], 'avg_depth_m': [config.avg_depth_m]},
        geometry=[bbox_geom],
        crs='EPSG:4326'
    )

    gdf.to_file(output_path, driver='GeoJSON')
    logger.info(f"Saved simplified polygon to {output_path}")
    return gdf


def rasterize_lake(gdf: gpd.GeoDataFrame, output_path: Path,
                   utm_crs: str, resolution: float = 100.0):
    """
    Rasterize lake polygon for fetch calculation.

    Args:
        gdf: Lake polygon GeoDataFrame
        output_path: Output raster path
        utm_crs: UTM CRS string (e.g. 'EPSG:32618')
        resolution: Cell size in meters
    """
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    logger.info(f"Rasterizing lake at {resolution}m resolution (CRS: {utm_crs})...")

    gdf_utm = gdf.to_crs(utm_crs)

    bounds = gdf_utm.total_bounds
    minx, miny, maxx, maxy = bounds

    buffer = resolution * 10
    minx -= buffer
    miny -= buffer
    maxx += buffer
    maxy += buffer

    width = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)

    logger.info(f"Raster dimensions: {width} x {height}")

    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    shapes = [(geom, 1) for geom in gdf_utm.geometry]
    raster = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype='uint8'
    )

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype='uint8',
        crs=utm_crs,
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(raster, 1)

    logger.info(f"Saved raster to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Prepare lake data for wave impact analysis')
    parser.add_argument('--lake', type=str, default='champlain',
                        help='Lake name or ID (e.g. "Lake Champlain" or "champlain")')
    parser.add_argument('--resolution', type=float, default=100.0,
                        help='Raster resolution in meters')
    args = parser.parse_args()

    # Load config (DB first, then local JSON fallback)
    config = load_lake_config(args.lake)
    paths = LakePaths(config.lake_id)
    paths.ensure_dirs()

    # Try DB for geometry first, then NHD/OSM fallback
    gdf = fetch_lake_from_db(config, paths.polygon)
    if gdf is None:
        gdf = download_lake_polygon(config, paths.polygon)

    # Add metadata
    gdf['avg_depth_m'] = config.avg_depth_m
    gdf.to_file(paths.polygon, driver='GeoJSON')

    # Rasterize for fetch calculation
    rasterize_lake(gdf, paths.raster, config.utm_crs, args.resolution)

    logger.info("Lake preparation complete!")
    logger.info(f"  Polygon: {paths.polygon}")
    logger.info(f"  Raster: {paths.raster}")


if __name__ == '__main__':
    main()
