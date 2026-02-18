#!/usr/bin/env python3
"""
Micro-Shelter Detection

Identifies sheltered coves and bays based on current wind direction.
A micro-shelter is an area with low fetch (< threshold) in the wind direction.

Output: micro_shelters.geojson with labeled shelter polygons
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import json
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import Point, Polygon, MultiPolygon, shape, mapping
from shapely.ops import unary_union
from scipy import ndimage
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Known bay/cove names for Lake Champlain (can expand)
KNOWN_BAYS = {
    'champlain': [
        {'name': 'Malletts Bay', 'center': (-73.17, 44.53), 'radius': 3000},
        {'name': 'Shelburne Bay', 'center': (-73.22, 44.40), 'radius': 2000},
        {'name': 'Burlington Bay', 'center': (-73.23, 44.48), 'radius': 1500},
        {'name': 'St. Albans Bay', 'center': (-73.15, 44.80), 'radius': 2500},
        {'name': 'Missisquoi Bay', 'center': (-73.10, 44.97), 'radius': 4000},
        {'name': 'Cumberland Bay', 'center': (-73.43, 44.68), 'radius': 2000},
        {'name': 'Willsboro Bay', 'center': (-73.38, 44.38), 'radius': 2000},
        {'name': 'Port Henry Bay', 'center': (-73.44, 44.05), 'radius': 1500},
        {'name': 'Bulwagga Bay', 'center': (-73.42, 44.02), 'radius': 1500},
        {'name': 'Crown Point Bay', 'center': (-73.43, 43.94), 'radius': 1500},
        {'name': 'Ticonderoga Bay', 'center': (-73.42, 43.85), 'radius': 1500},
        {'name': 'South Bay', 'center': (-73.38, 43.60), 'radius': 2000},
        {'name': 'East Bay', 'center': (-73.32, 43.58), 'radius': 1500},
        {'name': 'Northwest Bay', 'center': (-73.40, 43.65), 'radius': 2000},
        {'name': 'Button Bay', 'center': (-73.37, 44.18), 'radius': 1000},
        {'name': 'Converse Bay', 'center': (-73.28, 44.33), 'radius': 1500},
        {'name': 'Thompsons Point', 'center': (-73.30, 44.26), 'radius': 1000},
        {'name': 'McNeil Cove', 'center': (-73.24, 44.46), 'radius': 800},
        {'name': 'Appletree Bay', 'center': (-73.21, 44.47), 'radius': 800},
        {'name': 'Perkins Pier', 'center': (-73.22, 44.48), 'radius': 500},
        {'name': 'Oakledge Cove', 'center': (-73.23, 44.45), 'radius': 600},
        {'name': 'Red Rocks', 'center': (-73.24, 44.43), 'radius': 800},
        {'name': 'Sunset Cliff', 'center': (-73.25, 44.42), 'radius': 600},
        {'name': 'Kingsland Bay', 'center': (-73.26, 44.22), 'radius': 1200},
        {'name': 'Basin Harbor', 'center': (-73.35, 44.17), 'radius': 800},
        {'name': 'Dead Creek', 'center': (-73.34, 44.10), 'radius': 1000},
        {'name': 'Otter Creek', 'center': (-73.30, 44.07), 'radius': 1000},
    ]
}


def detect_micro_shelters(lake_polygon_path: Path, fetch_dir: Path,
                          wind_direction: float, output_path: Path,
                          lake_name: str = 'champlain',
                          fetch_threshold: float = 2000.0,
                          min_shelter_area: float = 50000.0):
    """
    Detect micro-shelters based on low fetch in wind direction.
    
    Args:
        lake_polygon_path: Path to lake polygon
        fetch_dir: Directory with fetch rasters
        wind_direction: Wind direction in degrees (FROM)
        output_path: Output GeoJSON path
        lake_name: Lake name for known bay lookup
        fetch_threshold: Max fetch (m) to be considered sheltered
        min_shelter_area: Minimum shelter area in m²
    """
    logger.info(f"Detecting micro-shelters for wind from {wind_direction}°...")
    
    # Load lake polygon
    lake = gpd.read_file(lake_polygon_path)
    lake_utm = lake.to_crs('EPSG:32618')
    lake_geom = lake_utm.geometry.iloc[0]
    
    # Load fetch raster for wind direction
    index_path = fetch_dir / "fetch_index.json"
    with open(index_path) as f:
        fetch_index = json.load(f)
    
    # Find closest direction raster
    directions = list(fetch_index['files'].keys())
    directions_float = [float(d) for d in directions]
    closest_idx = min(range(len(directions_float)), 
                      key=lambda i: min(abs(directions_float[i] - wind_direction),
                                        360 - abs(directions_float[i] - wind_direction)))
    closest_dir_key = directions[closest_idx]
    
    fetch_path = fetch_dir / fetch_index['files'][closest_dir_key]
    logger.info(f"Using fetch raster for {closest_dir_key}°")
    
    with rasterio.open(fetch_path) as src:
        fetch_data = src.read(1)
        fetch_transform = src.transform
        fetch_crs = src.crs
    
    # Create shelter mask: areas with fetch < threshold
    shelter_mask = (fetch_data > 0) & (fetch_data < fetch_threshold)
    
    # Clean up mask with morphological operations
    # Remove tiny isolated pixels
    shelter_mask = ndimage.binary_opening(shelter_mask, iterations=2)
    # Fill small holes
    shelter_mask = ndimage.binary_closing(shelter_mask, iterations=2)
    
    # Label connected regions
    labeled, num_features = ndimage.label(shelter_mask)
    logger.info(f"Found {num_features} potential shelter regions")
    
    # Convert to polygons
    shelter_polygons = []
    
    for geom, value in shapes(labeled.astype('int32'), transform=fetch_transform):
        if value == 0:  # Background
            continue
        
        poly = shape(geom)
        
        # Filter by minimum area
        if poly.area < min_shelter_area:
            continue
        
        # Calculate centroid for naming
        centroid = poly.centroid
        
        # Get average fetch in this shelter
        # Sample a few points
        avg_fetch = 0
        sample_count = 0
        for pt in [poly.centroid, poly.representative_point()]:
            try:
                row, col = rasterio.transform.rowcol(fetch_transform, pt.x, pt.y)
                if 0 <= row < fetch_data.shape[0] and 0 <= col < fetch_data.shape[1]:
                    avg_fetch += fetch_data[row, col]
                    sample_count += 1
            except:
                pass
        
        if sample_count > 0:
            avg_fetch /= sample_count
        
        shelter_polygons.append({
            'geometry': poly,
            'area_m2': poly.area,
            'area_acres': poly.area / 4047,
            'avg_fetch_m': float(avg_fetch),
            'centroid_x': centroid.x,
            'centroid_y': centroid.y
        })
    
    logger.info(f"Found {len(shelter_polygons)} shelters above minimum area")
    
    # Create GeoDataFrame
    if shelter_polygons:
        gdf = gpd.GeoDataFrame(shelter_polygons, crs=fetch_crs)
    else:
        gdf = gpd.GeoDataFrame({'geometry': []}, crs=fetch_crs)
    
    # Name shelters based on known bays
    if lake_name in KNOWN_BAYS and len(gdf) > 0:
        gdf['name'] = None
        gdf['is_named'] = False
        
        known_bays = KNOWN_BAYS[lake_name]
        
        for idx, row in gdf.iterrows():
            centroid_wgs84 = gpd.GeoSeries([Point(row['centroid_x'], row['centroid_y'])], 
                                           crs=fetch_crs).to_crs('EPSG:4326').iloc[0]
            
            # Find closest known bay
            best_match = None
            best_dist = float('inf')
            
            for bay in known_bays:
                dist = np.sqrt((centroid_wgs84.x - bay['center'][0])**2 + 
                               (centroid_wgs84.y - bay['center'][1])**2)
                # Convert to approximate meters
                dist_m = dist * 111000
                
                if dist_m < bay['radius'] and dist_m < best_dist:
                    best_match = bay['name']
                    best_dist = dist_m
            
            if best_match:
                gdf.at[idx, 'name'] = best_match
                gdf.at[idx, 'is_named'] = True
            else:
                # Generate generic name based on location
                gdf.at[idx, 'name'] = f"Shelter Zone {idx + 1}"
                gdf.at[idx, 'is_named'] = False
    
    # Add protection level based on fetch
    if len(gdf) > 0:
        gdf['protection'] = gdf['avg_fetch_m'].apply(
            lambda f: 'excellent' if f < 500 else ('good' if f < 1000 else 'moderate')
        )
        
        # Add wind direction info
        gdf['sheltered_from'] = get_wind_direction_name(wind_direction)
        gdf['wind_direction_deg'] = wind_direction
    
    # Reproject to WGS84
    gdf = gdf.to_crs('EPSG:4326')
    
    # Save
    gdf.to_file(output_path, driver='GeoJSON')
    logger.info(f"Saved {len(gdf)} micro-shelters to {output_path}")
    
    # Log named shelters
    named = gdf[gdf['is_named'] == True] if 'is_named' in gdf.columns else gdf
    for _, row in named.iterrows():
        logger.info(f"  {row.get('name', 'Unknown')}: {row.get('protection', '?')} protection, "
                   f"{row.get('area_acres', 0):.0f} acres")
    
    return gdf


def generate_shelter_labels(shelters_gdf: gpd.GeoDataFrame, output_path: Path):
    """
    Generate point labels for shelter zones (for map labeling).
    """
    if len(shelters_gdf) == 0:
        gpd.GeoDataFrame({'geometry': []}).to_file(output_path, driver='GeoJSON')
        return
    
    labels = []
    
    for idx, row in shelters_gdf.iterrows():
        centroid = row.geometry.centroid
        
        labels.append({
            'geometry': centroid,
            'name': row.get('name', f'Shelter {idx}'),
            'protection': row.get('protection', 'moderate'),
            'area_acres': row.get('area_acres', 0),
            'avg_fetch_m': row.get('avg_fetch_m', 0),
            'sheltered_from': row.get('sheltered_from', ''),
        })
    
    gdf = gpd.GeoDataFrame(labels, crs=shelters_gdf.crs)
    gdf.to_file(output_path, driver='GeoJSON')
    logger.info(f"Saved {len(gdf)} shelter labels to {output_path}")
    
    return gdf


def get_wind_direction_name(direction_deg: float) -> str:
    """Convert degrees to compass direction name."""
    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    idx = int((direction_deg + 11.25) / 22.5) % 16
    return directions[idx]


def main():
    parser = argparse.ArgumentParser(description='Detect micro-shelters')
    parser.add_argument('--lake', type=str, default='champlain',
                        help='Lake name')
    parser.add_argument('--wind-dir', type=float, required=True,
                        help='Wind direction in degrees (FROM)')
    parser.add_argument('--lakes-dir', type=Path, default=Path('data/lakes'),
                        help='Directory with lake data')
    parser.add_argument('--fetch-dir', type=Path, default=Path('data/fetch_rasters'),
                        help='Directory with fetch rasters')
    parser.add_argument('--output-dir', type=Path, default=Path('data/output'),
                        help='Output directory')
    parser.add_argument('--fetch-threshold', type=float, default=2000.0,
                        help='Max fetch (m) to be considered sheltered')
    parser.add_argument('--min-area', type=float, default=50000.0,
                        help='Minimum shelter area in m²')
    
    args = parser.parse_args()
    
    # Paths
    lake_polygon_path = args.lakes_dir / f"{args.lake}_polygon.geojson"
    fetch_dir = args.fetch_dir / args.lake
    
    if not lake_polygon_path.exists():
        logger.error(f"Lake polygon not found: {lake_polygon_path}")
        return
    
    if not fetch_dir.exists():
        logger.error(f"Fetch rasters not found: {fetch_dir}")
        return
    
    # Detect shelters
    shelters_path = args.output_dir / "micro_shelters.geojson"
    shelters = detect_micro_shelters(
        lake_polygon_path, fetch_dir,
        args.wind_dir, shelters_path,
        lake_name=args.lake,
        fetch_threshold=args.fetch_threshold,
        min_shelter_area=args.min_area
    )
    
    # Generate labels
    labels_path = args.output_dir / "shelter_labels.geojson"
    generate_shelter_labels(shelters, labels_path)
    
    logger.info("Micro-shelter detection complete!")


if __name__ == '__main__':
    main()
