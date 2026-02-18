#!/usr/bin/env python3
"""
Generate Additional Styled Layers for Wave Impact Visualization

Creates:
1. wave_polylines.geojson - Horizontal wavy lines across the lake surface
2. bank_impact_points.geojson - Points along shoreline with impact values
3. wind_indicator.geojson - Wind direction arrow(s)

These are ADDITIONAL outputs - original files are not modified.
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import json
import geopandas as gpd
import rasterio
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import unary_union
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_wave_polylines(lake_polygon_path: Path, fetch_dir: Path, 
                            wind_speed_ms: float, wind_direction: float,
                            output_path: Path, line_spacing: float = 800.0,
                            wave_amplitude: float = 150.0, wave_frequency: float = 0.002,
                            segment_length: float = 300.0):
    """
    Generate horizontal wavy polylines across the lake surface.
    
    Lines are SPLIT INTO SEGMENTS, each colored by local fetch/intensity.
    
    Args:
        lake_polygon_path: Path to lake polygon GeoJSON
        fetch_dir: Directory with fetch rasters
        wind_speed_ms: Wind speed in m/s
        wind_direction: Wind direction in degrees
        output_path: Output GeoJSON path
        line_spacing: Vertical spacing between lines in meters
        wave_amplitude: Height of wave oscillation in meters
        wave_frequency: Frequency of wave oscillation (radians per meter)
        segment_length: Length of each line segment in meters (for varying intensity)
    """
    logger.info("Generating wave polylines (segmented)...")
    
    # Load lake polygon
    lake = gpd.read_file(lake_polygon_path)
    lake_utm = lake.to_crs('EPSG:32618')
    lake_geom = lake_utm.geometry.iloc[0]
    
    # Get bounds
    minx, miny, maxx, maxy = lake_geom.bounds
    
    # Load fetch index to get wave intensity
    index_path = fetch_dir / "fetch_index.json"
    with open(index_path) as f:
        fetch_index = json.load(f)
    
    # Find closest direction raster
    directions = list(fetch_index['files'].keys())
    directions_float = [float(d) for d in directions]
    closest_idx = min(range(len(directions_float)), key=lambda i: abs(directions_float[i] - wind_direction) if abs(directions_float[i] - wind_direction) <= 180 else 360 - abs(directions_float[i] - wind_direction))
    closest_dir_key = directions[closest_idx]
    
    fetch_path = fetch_dir / fetch_index['files'][closest_dir_key]
    
    with rasterio.open(fetch_path) as src:
        fetch_data = src.read(1)
        fetch_transform = src.transform
    
    def get_fetch_at_point(x, y):
        """Get fetch value at a point."""
        try:
            row, col = rasterio.transform.rowcol(fetch_transform, x, y)
            if 0 <= row < fetch_data.shape[0] and 0 <= col < fetch_data.shape[1]:
                return fetch_data[row, col]
        except Exception:
            pass
        return 0
    
    def calc_wave_height(fetch_m):
        """Calculate wave height from fetch."""
        if fetch_m > 0 and wind_speed_ms > 0:
            fetch_km = fetch_m / 1000.0
            return 0.0016 * (wind_speed_ms ** 2) * np.sqrt(fetch_km)
        return 0
    
    def classify_intensity(wave_height):
        """Classify wave height into intensity category."""
        if wave_height < 0.05:
            return 'calm'
        elif wave_height < 0.15:
            return 'light'
        elif wave_height < 0.30:
            return 'moderate'
        elif wave_height < 0.50:
            return 'rough'
        else:
            return 'very_rough'
    
    segments = []
    line_id = 0
    
    # Generate horizontal wavy lines from south to north
    y = miny + line_spacing
    
    while y < maxy:
        # Create base horizontal line with fine resolution
        x_points = np.arange(minx - wave_amplitude, maxx + wave_amplitude, 30)  # 30m resolution
        
        # Add wave oscillation
        y_points = y + wave_amplitude * np.sin(wave_frequency * x_points)
        
        # Create line coordinates
        coords = list(zip(x_points, y_points))
        
        if len(coords) < 2:
            y += line_spacing
            continue
        
        line = LineString(coords)
        
        # Clip to lake polygon
        try:
            clipped = line.intersection(lake_geom)
        except Exception:
            y += line_spacing
            continue
        
        if clipped.is_empty:
            y += line_spacing
            continue
        
        # Handle MultiLineString (line may be split by islands)
        if clipped.geom_type == 'MultiLineString':
            line_parts = list(clipped.geoms)
        elif clipped.geom_type == 'LineString':
            line_parts = [clipped]
        else:
            y += line_spacing
            continue
        
        for part in line_parts:
            if part.length < 50:  # Skip tiny segments
                continue
            
            # Split this line part into segments
            total_length = part.length
            num_segments = max(1, int(total_length / segment_length))
            
            for i in range(num_segments):
                start_frac = i / num_segments
                end_frac = (i + 1) / num_segments
                
                # Get segment endpoints
                start_pt = part.interpolate(start_frac, normalized=True)
                end_pt = part.interpolate(end_frac, normalized=True)
                mid_pt = part.interpolate((start_frac + end_frac) / 2, normalized=True)
                
                # Extract the actual segment geometry
                # For accuracy, we need to cut the line at these distances
                start_dist = start_frac * total_length
                end_dist = end_frac * total_length
                
                # Get coordinates along segment
                seg_coords = []
                coords_list = list(part.coords)
                cumulative_dist = 0
                
                for j in range(len(coords_list) - 1):
                    p1 = coords_list[j]
                    p2 = coords_list[j + 1]
                    seg_dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                    
                    # Check if this segment overlaps with our target range
                    seg_start = cumulative_dist
                    seg_end = cumulative_dist + seg_dist
                    
                    if seg_end >= start_dist and seg_start <= end_dist:
                        # This segment is within our range
                        if seg_start >= start_dist and seg_end <= end_dist:
                            # Entire segment is within range
                            if not seg_coords:
                                seg_coords.append(p1)
                            seg_coords.append(p2)
                        elif seg_start < start_dist and seg_end > start_dist:
                            # Segment starts before range, crosses start
                            t = (start_dist - seg_start) / seg_dist
                            interp_pt = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
                            seg_coords.append(interp_pt)
                            if seg_end <= end_dist:
                                seg_coords.append(p2)
                        elif seg_start < end_dist and seg_end > end_dist:
                            # Segment crosses end
                            if not seg_coords:
                                seg_coords.append(p1)
                            t = (end_dist - seg_start) / seg_dist
                            interp_pt = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
                            seg_coords.append(interp_pt)
                    
                    cumulative_dist = seg_end
                
                if len(seg_coords) >= 2:
                    segment_geom = LineString(seg_coords)
                else:
                    # Fallback: create simple line from start to end
                    segment_geom = LineString([(start_pt.x, start_pt.y), (end_pt.x, end_pt.y)])
                
                # Get fetch at segment midpoint
                fetch_m = get_fetch_at_point(mid_pt.x, mid_pt.y)
                wave_height = calc_wave_height(fetch_m)
                intensity = classify_intensity(wave_height)
                
                segments.append({
                    'geometry': segment_geom,
                    'line_id': line_id,
                    'segment_id': i,
                    'wave_height_m': float(wave_height),
                    'intensity': intensity,
                    'fetch_m': float(fetch_m)
                })
            
            line_id += 1
        
        y += line_spacing
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(segments, crs='EPSG:32618')
    
    # Reproject to WGS84
    gdf = gdf.to_crs('EPSG:4326')
    
    # Save
    gdf.to_file(output_path, driver='GeoJSON')
    logger.info(f"Saved {len(gdf)} wave polyline segments to {output_path}")
    
    # Stats
    for intensity in ['calm', 'light', 'moderate', 'rough', 'very_rough']:
        count = len(gdf[gdf['intensity'] == intensity])
        logger.info(f"  {intensity}: {count} segments")
    
    return gdf


def generate_bank_impact_points(bank_impact_path: Path, output_path: Path,
                                 point_spacing: float = 100.0):
    """
    Convert bank impact line segments to points along shoreline.
    
    Args:
        bank_impact_path: Path to bank_impact.geojson
        output_path: Output GeoJSON path  
        point_spacing: Spacing between points in meters
    """
    logger.info("Generating bank impact points...")
    
    # Load bank impact segments
    gdf = gpd.read_file(bank_impact_path)
    
    # Reproject to UTM for distance calculations
    gdf_utm = gdf.to_crs('EPSG:32618')
    
    points = []
    
    for idx, row in gdf_utm.iterrows():
        geom = row.geometry
        
        if geom is None or geom.is_empty:
            continue
        
        # Get line length
        length = geom.length
        
        if length < point_spacing / 2:
            # Short segment - just use midpoint
            pt = geom.interpolate(0.5, normalized=True)
            points.append({
                'geometry': pt,
                'impact': row.get('impact', 0),
                'intensity': row.get('intensity', 'calm'),
                'angle_diff': row.get('angle_diff', 0),
                'shore_normal': row.get('shore_normal', 0)
            })
        else:
            # Generate points along segment
            num_points = max(1, int(length / point_spacing))
            
            for i in range(num_points + 1):
                fraction = i / num_points if num_points > 0 else 0.5
                pt = geom.interpolate(fraction, normalized=True)
                
                points.append({
                    'geometry': pt,
                    'impact': row.get('impact', 0),
                    'intensity': row.get('intensity', 'calm'),
                    'angle_diff': row.get('angle_diff', 0),
                    'shore_normal': row.get('shore_normal', 0)
                })
    
    # Create GeoDataFrame
    gdf_points = gpd.GeoDataFrame(points, crs='EPSG:32618')
    
    # Reproject to WGS84
    gdf_points = gdf_points.to_crs('EPSG:4326')
    
    # Save
    gdf_points.to_file(output_path, driver='GeoJSON')
    logger.info(f"Saved {len(gdf_points)} bank impact points to {output_path}")
    
    # Stats
    for intensity in ['calm', 'light', 'moderate', 'rough', 'very_rough']:
        count = len(gdf_points[gdf_points['intensity'] == intensity])
        logger.info(f"  {intensity}: {count} points")
    
    return gdf_points


def generate_wind_indicator(lake_polygon_path: Path, wind_speed_ms: float,
                            wind_direction: float, output_path: Path):
    """
    Generate wind direction indicator arrow(s).
    
    Args:
        lake_polygon_path: Path to lake polygon
        wind_speed_ms: Wind speed in m/s
        wind_direction: Wind direction in degrees (FROM direction)
        output_path: Output GeoJSON path
    """
    logger.info("Generating wind indicator...")
    
    # Load lake to get center
    lake = gpd.read_file(lake_polygon_path)
    lake_utm = lake.to_crs('EPSG:32618')
    
    centroid = lake_utm.geometry.iloc[0].centroid
    
    # Create arrow line pointing in wind direction
    # Arrow shows direction wind is GOING (opposite of FROM)
    going_direction = (wind_direction + 180) % 360
    direction_rad = np.radians(going_direction)
    
    # Arrow length proportional to wind speed (min 5km, max 20km)
    arrow_length = min(20000, max(5000, wind_speed_ms * 1000))
    
    # Calculate arrow endpoints
    dx = arrow_length * np.sin(direction_rad)
    dy = arrow_length * np.cos(direction_rad)
    
    # Start point (upwind of center)
    start_x = centroid.x - dx * 0.3
    start_y = centroid.y - dy * 0.3
    
    # End point (downwind)
    end_x = centroid.x + dx * 0.7
    end_y = centroid.y + dy * 0.7
    
    arrow_line = LineString([(start_x, start_y), (end_x, end_y)])
    
    # Create arrowhead
    head_length = arrow_length * 0.15
    head_angle = 25  # degrees
    
    # Arrowhead points
    angle1 = direction_rad + np.radians(180 - head_angle)
    angle2 = direction_rad + np.radians(180 + head_angle)
    
    head_x1 = end_x + head_length * np.sin(angle1)
    head_y1 = end_y + head_length * np.cos(angle1)
    head_x2 = end_x + head_length * np.sin(angle2)
    head_y2 = end_y + head_length * np.cos(angle2)
    
    arrowhead1 = LineString([(end_x, end_y), (head_x1, head_y1)])
    arrowhead2 = LineString([(end_x, end_y), (head_x2, head_y2)])
    
    # Wind direction name
    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    dir_idx = int((wind_direction + 11.25) / 22.5) % 16
    dir_name = directions[dir_idx]
    
    wind_mph = wind_speed_ms * 2.237
    
    features = [
        {
            'geometry': arrow_line,
            'type': 'arrow_shaft',
            'wind_speed_ms': wind_speed_ms,
            'wind_speed_mph': wind_mph,
            'wind_direction_deg': wind_direction,
            'wind_direction_name': dir_name
        },
        {
            'geometry': arrowhead1,
            'type': 'arrow_head',
            'wind_speed_ms': wind_speed_ms,
            'wind_direction_deg': wind_direction
        },
        {
            'geometry': arrowhead2,
            'type': 'arrow_head',
            'wind_speed_ms': wind_speed_ms,
            'wind_direction_deg': wind_direction
        }
    ]
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(features, crs='EPSG:32618')
    gdf = gdf.to_crs('EPSG:4326')
    
    # Save
    gdf.to_file(output_path, driver='GeoJSON')
    logger.info(f"Saved wind indicator to {output_path}")
    logger.info(f"  Wind: {wind_mph:.0f} mph from {dir_name}")
    
    return gdf


def main():
    parser = argparse.ArgumentParser(description='Generate styled wave layers')
    parser.add_argument('--lake', type=str, default='champlain',
                        help='Lake name')
    parser.add_argument('--wind-speed', type=float, required=True,
                        help='Wind speed in m/s')
    parser.add_argument('--wind-dir', type=float, required=True,
                        help='Wind direction in degrees')
    parser.add_argument('--lakes-dir', type=Path, default=Path('data/lakes'),
                        help='Directory with lake data')
    parser.add_argument('--fetch-dir', type=Path, default=Path('data/fetch_rasters'),
                        help='Directory with fetch rasters')
    parser.add_argument('--output-dir', type=Path, default=Path('data/output'),
                        help='Output directory')
    parser.add_argument('--line-spacing', type=float, default=800.0,
                        help='Spacing between wave lines in meters')
    parser.add_argument('--point-spacing', type=float, default=100.0,
                        help='Spacing between bank impact points in meters')
    parser.add_argument('--segment-length', type=float, default=300.0,
                        help='Length of wave line segments in meters (for varying intensity)')
    
    args = parser.parse_args()
    
    # Paths
    lake_polygon_path = args.lakes_dir / f"{args.lake}_polygon.geojson"
    fetch_dir = args.fetch_dir / args.lake
    bank_impact_path = args.output_dir / "bank_impact.geojson"
    
    # Check inputs
    if not lake_polygon_path.exists():
        logger.error(f"Lake polygon not found: {lake_polygon_path}")
        return
    
    if not fetch_dir.exists():
        logger.error(f"Fetch rasters not found: {fetch_dir}")
        return
    
    if not bank_impact_path.exists():
        logger.error(f"Bank impact not found: {bank_impact_path}")
        logger.error("Run 03_generate_wave_layer.py first")
        return
    
    logger.info(f"Generating styled layers for {args.lake}")
    logger.info(f"Wind: {args.wind_speed} m/s from {args.wind_dir}°")
    
    # Generate wave polylines
    wave_lines_path = args.output_dir / "wave_polylines.geojson"
    generate_wave_polylines(
        lake_polygon_path, fetch_dir,
        args.wind_speed, args.wind_dir,
        wave_lines_path, args.line_spacing,
        segment_length=args.segment_length
    )
    
    # Generate bank impact points
    bank_points_path = args.output_dir / "bank_impact_points.geojson"
    generate_bank_impact_points(bank_impact_path, bank_points_path, args.point_spacing)
    
    # Generate wind indicator
    wind_path = args.output_dir / "wind_indicator.geojson"
    generate_wind_indicator(lake_polygon_path, args.wind_speed, args.wind_dir, wind_path)
    
    logger.info("Styled layer generation complete!")
    logger.info(f"New outputs in {args.output_dir}:")
    logger.info(f"  - wave_polylines.geojson")
    logger.info(f"  - bank_impact_points.geojson")
    logger.info(f"  - wind_indicator.geojson")


if __name__ == '__main__':
    main()
