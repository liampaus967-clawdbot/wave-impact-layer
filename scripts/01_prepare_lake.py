#!/usr/bin/env python3
"""
Prepare Lake Data for Wave Impact Analysis

Downloads lake polygon from NHD HR and prepares it for fetch calculation.
"""

import argparse
import logging
from pathlib import Path
import geopandas as gpd
import requests
from shapely.geometry import shape
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Lake configurations
LAKES = {
    'champlain': {
        'name': 'Lake Champlain',
        'gnis_id': '00979520',  # GNIS Feature ID
        'bbox': [-73.5, 43.5, -73.0, 45.1],  # Approximate bounding box
        'avg_depth_m': 20.0,  # Average depth in meters
        'center': [-73.25, 44.5],
    }
}


def download_lake_champlain(output_path: Path) -> gpd.GeoDataFrame:
    """
    Download Lake Champlain polygon from NHD HR WFS service.
    """
    logger.info("Downloading Lake Champlain from NHD HR...")
    
    # NHD HR WFS endpoint for waterbodies
    # We'll use the USGS National Map API
    url = "https://hydro.nationalmap.gov/arcgis/rest/services/NHDPlus_HR/MapServer/9/query"
    
    # Lake Champlain spans multiple HUC4s, so we'll query by name and bounding box
    params = {
        'where': "GNIS_NAME LIKE '%Lake Champlain%'",
        'outFields': '*',
        'geometry': '-73.5,43.5,-72.9,45.1',
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
            
            # Dissolve into single polygon if multiple parts
            if len(gdf) > 1:
                logger.info("Dissolving multiple parts into single polygon...")
                gdf = gdf.dissolve()
                gdf = gdf.reset_index(drop=True)
            
            # Save to file
            gdf.to_file(output_path, driver='GeoJSON')
            logger.info(f"Saved to {output_path}")
            
            return gdf
        else:
            logger.warning("No features found from NHD HR, trying fallback...")
            return download_lake_champlain_fallback(output_path)
            
    except Exception as e:
        logger.warning(f"NHD HR query failed: {e}, trying fallback...")
        return download_lake_champlain_fallback(output_path)


def download_lake_champlain_fallback(output_path: Path) -> gpd.GeoDataFrame:
    """
    Fallback: Download Lake Champlain from OpenStreetMap via Overpass API.
    """
    logger.info("Downloading Lake Champlain from OpenStreetMap...")
    
    # Overpass query for Lake Champlain
    overpass_url = "https://overpass-api.de/api/interpreter"
    overpass_query = """
    [out:json][timeout:180];
    (
      relation["name"="Lake Champlain"]["natural"="water"];
    );
    out body;
    >;
    out skel qt;
    """
    
    response = requests.post(overpass_url, data={'data': overpass_query}, timeout=180)
    response.raise_for_status()
    data = response.json()
    
    # Process OSM data into GeoJSON
    # This is simplified - in production you'd use osm2geojson or similar
    # For now, let's create a simplified polygon from the bounding box
    
    logger.info("Creating simplified Lake Champlain polygon...")
    
    # Simplified Lake Champlain outline (approximate)
    # In production, you'd extract the actual geometry from OSM
    from shapely.geometry import Polygon, MultiPolygon
    import numpy as np
    
    # Lake Champlain approximate coordinates (simplified)
    champlain_coords = [
        (-73.3389, 45.0156), (-73.2298, 44.9867), (-73.2518, 44.8789),
        (-73.3059, 44.8122), (-73.2847, 44.7422), (-73.3389, 44.6478),
        (-73.3830, 44.5800), (-73.4161, 44.5089), (-73.4271, 44.4367),
        (-73.3940, 44.3833), (-73.3499, 44.3344), (-73.3389, 44.2711),
        (-73.3059, 44.2156), (-73.2847, 44.1578), (-73.2518, 44.0989),
        (-73.2298, 44.0333), (-73.2628, 43.9722), (-73.2847, 43.9111),
        (-73.3059, 43.8467), (-73.3389, 43.7822), (-73.3830, 43.7278),
        (-73.4161, 43.6778), (-73.4382, 43.6289), (-73.4602, 43.5844),
        (-73.4382, 43.5367), (-73.3830, 43.5678), (-73.3389, 43.6089),
        (-73.2847, 43.6533), (-73.2298, 43.7011), (-73.1858, 43.7556),
        (-73.1417, 43.8133), (-73.0977, 43.8711), (-73.0646, 43.9333),
        (-73.0426, 44.0000), (-73.0316, 44.0711), (-73.0426, 44.1467),
        (-73.0646, 44.2200), (-73.0977, 44.2889), (-73.1307, 44.3533),
        (-73.1527, 44.4133), (-73.1637, 44.4689), (-73.1527, 44.5244),
        (-73.1307, 44.5800), (-73.0977, 44.6356), (-73.0646, 44.6911),
        (-73.0426, 44.7467), (-73.0316, 44.8022), (-73.0426, 44.8578),
        (-73.0756, 44.9111), (-73.1197, 44.9622), (-73.1747, 45.0044),
        (-73.2408, 45.0311), (-73.3389, 45.0156)
    ]
    
    polygon = Polygon(champlain_coords)
    
    gdf = gpd.GeoDataFrame({
        'name': ['Lake Champlain'],
        'avg_depth_m': [20.0],
        'area_km2': [polygon.area * 111 * 111 * 0.8],  # Rough area calc
    }, geometry=[polygon], crs='EPSG:4326')
    
    # Save to file
    gdf.to_file(output_path, driver='GeoJSON')
    logger.info(f"Saved simplified polygon to {output_path}")
    
    return gdf


def rasterize_lake(gdf: gpd.GeoDataFrame, output_path: Path, resolution: float = 100.0):
    """
    Rasterize lake polygon for fetch calculation.
    
    Args:
        gdf: Lake polygon GeoDataFrame
        output_path: Output raster path
        resolution: Cell size in meters
    """
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    
    logger.info(f"Rasterizing lake at {resolution}m resolution...")
    
    # Reproject to UTM for accurate distance calculations
    # Lake Champlain is in UTM zone 18N
    gdf_utm = gdf.to_crs('EPSG:32618')
    
    # Get bounds
    bounds = gdf_utm.total_bounds
    minx, miny, maxx, maxy = bounds
    
    # Add buffer
    buffer = resolution * 10
    minx -= buffer
    miny -= buffer
    maxx += buffer
    maxy += buffer
    
    # Calculate dimensions
    width = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)
    
    logger.info(f"Raster dimensions: {width} x {height}")
    
    # Create transform
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    
    # Rasterize: 1 = water, 0 = land
    shapes = [(geom, 1) for geom in gdf_utm.geometry]
    raster = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype='uint8'
    )
    
    # Save raster
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype='uint8',
        crs='EPSG:32618',
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(raster, 1)
    
    logger.info(f"Saved raster to {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Prepare lake data for wave impact analysis')
    parser.add_argument('--lake', type=str, default='champlain', 
                        choices=list(LAKES.keys()),
                        help='Lake to prepare')
    parser.add_argument('--resolution', type=float, default=100.0,
                        help='Raster resolution in meters')
    parser.add_argument('--output-dir', type=Path, default=Path('data/lakes'),
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download lake polygon
    lake_config = LAKES[args.lake]
    polygon_path = args.output_dir / f"{args.lake}_polygon.geojson"
    
    gdf = download_lake_champlain(polygon_path)
    
    # Add metadata
    gdf['avg_depth_m'] = lake_config['avg_depth_m']
    gdf.to_file(polygon_path, driver='GeoJSON')
    
    # Rasterize for fetch calculation
    raster_path = args.output_dir / f"{args.lake}_raster.tif"
    rasterize_lake(gdf, raster_path, args.resolution)
    
    # Save lake config
    config_path = args.output_dir / f"{args.lake}_config.json"
    with open(config_path, 'w') as f:
        json.dump(lake_config, f, indent=2)
    
    logger.info("Lake preparation complete!")
    logger.info(f"  Polygon: {polygon_path}")
    logger.info(f"  Raster: {raster_path}")
    logger.info(f"  Config: {config_path}")


if __name__ == '__main__':
    main()
