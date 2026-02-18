#!/usr/bin/env python3
"""
Generate Wave Impact Layer from Live HRRR Wind Data

Fetches current or forecast HRRR wind data and generates wave impact layers.
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import json
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Lake center coordinates for HRRR extraction
LAKE_CENTERS = {
    'champlain': {'lat': 44.5, 'lon': -73.25}
}


def get_latest_hrrr_time() -> datetime:
    """Get the latest available HRRR forecast time (~3 hours ago)."""
    now = datetime.utcnow()
    latest = now - timedelta(hours=3)
    latest = latest.replace(minute=0, second=0, microsecond=0)
    return latest


def fetch_hrrr_wind(date: datetime, lat: float, lon: float, fxx: int = 0):
    """
    Fetch HRRR wind components for a specific location.
    
    Args:
        date: Forecast initialization datetime
        lat: Latitude
        lon: Longitude
        fxx: Forecast hour (0 = analysis)
        
    Returns:
        Tuple of (u_wind, v_wind) in m/s
    """
    from herbie import Herbie
    
    logger.info(f"Fetching HRRR wind for {date.strftime('%Y-%m-%d %H:00')} F{fxx:02d}")
    logger.info(f"Location: {lat}, {lon}")
    
    try:
        # Create Herbie object
        H = Herbie(date, model='hrrr', product='sfc', fxx=fxx)
        
        # Download U and V wind components at 10m
        ds_u = H.xarray("UGRD:10 m")
        ds_v = H.xarray("VGRD:10 m")
        
        # Extract point values using nearest neighbor
        # Find nearest grid point
        u_data = ds_u['u10'].values
        v_data = ds_v['v10'].values
        lats = ds_u['latitude'].values
        lons = ds_u['longitude'].values
        
        # Handle 2D lat/lon arrays
        if lats.ndim == 2:
            # Find nearest point
            dist = np.sqrt((lats - lat)**2 + (lons - lon)**2)
            idx = np.unravel_index(np.argmin(dist), dist.shape)
            u_wind = float(u_data[idx])
            v_wind = float(v_data[idx])
        else:
            # 1D coordinate arrays
            lat_idx = np.argmin(np.abs(lats - lat))
            lon_idx = np.argmin(np.abs(lons - lon))
            u_wind = float(u_data[lat_idx, lon_idx])
            v_wind = float(v_data[lat_idx, lon_idx])
        
        logger.info(f"U wind: {u_wind:.2f} m/s, V wind: {v_wind:.2f} m/s")
        
        return u_wind, v_wind
        
    except Exception as e:
        logger.error(f"Failed to fetch HRRR data: {e}")
        raise


def calculate_wind_speed_direction(u: float, v: float) -> tuple:
    """
    Calculate wind speed and direction from U/V components.
    
    Args:
        u: U (eastward) wind component in m/s
        v: V (northward) wind component in m/s
        
    Returns:
        Tuple of (speed_ms, direction_deg)
        Direction is where wind comes FROM (meteorological convention)
    """
    # Speed
    speed = np.sqrt(u**2 + v**2)
    
    # Direction (meteorological: where wind comes FROM)
    # atan2 gives direction wind is going TO, so add 180
    direction = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
    
    return speed, direction


def generate_wave_layer_from_hrrr(lake: str, date: datetime = None, fxx: int = 0,
                                  output_dir: Path = Path('data/output')):
    """
    Generate wave impact layer using live HRRR data.
    """
    if date is None:
        date = get_latest_hrrr_time()
    
    # Get lake center
    center = LAKE_CENTERS[lake]
    
    # Fetch wind data
    u_wind, v_wind = fetch_hrrr_wind(date, center['lat'], center['lon'], fxx)
    
    # Calculate speed and direction
    wind_speed, wind_direction = calculate_wind_speed_direction(u_wind, v_wind)
    
    logger.info(f"Wind: {wind_speed:.1f} m/s from {wind_direction:.0f}°")
    
    # Convert to common wind descriptions
    wind_mph = wind_speed * 2.237
    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    dir_idx = int((wind_direction + 11.25) / 22.5) % 16
    dir_name = directions[dir_idx]
    
    logger.info(f"Wind: {wind_mph:.0f} mph from {dir_name}")
    
    # Call the wave layer generator
    cmd = [
        'python', 'scripts/03_generate_wave_layer.py',
        '--lake', lake,
        '--wind-speed', str(wind_speed),
        '--wind-dir', str(wind_direction),
        '--output-dir', str(output_dir)
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Wave layer generation failed:\n{result.stderr}")
        return None
    
    logger.info(result.stdout)
    
    # Update metadata with HRRR info
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        metadata['hrrr'] = {
            'initialization_time': date.strftime('%Y-%m-%d %H:00 UTC'),
            'forecast_hour': fxx,
            'valid_time': (date + timedelta(hours=fxx)).strftime('%Y-%m-%d %H:00 UTC'),
            'u_wind_ms': u_wind,
            'v_wind_ms': v_wind,
            'wind_speed_ms': wind_speed,
            'wind_speed_mph': wind_mph,
            'wind_direction_deg': wind_direction,
            'wind_direction_name': dir_name
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    return output_dir


def generate_forecast_sequence(lake: str, date: datetime = None, 
                               forecast_hours: list = None,
                               output_dir: Path = Path('data/output')):
    """
    Generate wave layers for a sequence of forecast hours.
    
    Useful for timeline scrubbing visualization.
    """
    if date is None:
        date = get_latest_hrrr_time()
    
    if forecast_hours is None:
        forecast_hours = list(range(0, 25, 3))  # 0, 3, 6, ... 24
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for fxx in forecast_hours:
        fxx_dir = output_dir / f"f{fxx:02d}"
        fxx_dir.mkdir(exist_ok=True)
        
        try:
            result = generate_wave_layer_from_hrrr(lake, date, fxx, fxx_dir)
            
            valid_time = date + timedelta(hours=fxx)
            results.append({
                'forecast_hour': fxx,
                'valid_time': valid_time.strftime('%Y-%m-%d %H:00 UTC'),
                'output_dir': str(fxx_dir),
                'status': 'success'
            })
        except Exception as e:
            logger.error(f"Failed to generate F{fxx:02d}: {e}")
            results.append({
                'forecast_hour': fxx,
                'status': 'failed',
                'error': str(e)
            })
    
    # Save forecast index
    index = {
        'lake': lake,
        'initialization_time': date.strftime('%Y-%m-%d %H:00 UTC'),
        'forecasts': results
    }
    
    index_path = output_dir / "forecast_index.json"
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
    
    logger.info(f"Generated {len([r for r in results if r['status'] == 'success'])} forecast layers")
    logger.info(f"Index saved to {index_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Generate wave layer from HRRR')
    parser.add_argument('--lake', type=str, default='champlain',
                        choices=list(LAKE_CENTERS.keys()),
                        help='Lake name')
    parser.add_argument('--latest', action='store_true',
                        help='Use latest HRRR forecast')
    parser.add_argument('--date', type=str,
                        help='Forecast date (YYYY-MM-DD)')
    parser.add_argument('--cycle', type=int,
                        help='Forecast cycle hour (0-23)')
    parser.add_argument('--fxx', type=int, default=0,
                        help='Forecast hour (0-48)')
    parser.add_argument('--forecast-sequence', action='store_true',
                        help='Generate sequence of forecast hours')
    parser.add_argument('--output-dir', type=Path, default=Path('data/output'),
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Determine forecast time
    if args.latest:
        date = get_latest_hrrr_time()
    elif args.date:
        date = datetime.strptime(f"{args.date} {args.cycle or 0}", "%Y-%m-%d %H")
    else:
        date = get_latest_hrrr_time()
    
    logger.info(f"Using HRRR forecast: {date.strftime('%Y-%m-%d %H:00 UTC')}")
    
    if args.forecast_sequence:
        generate_forecast_sequence(args.lake, date, output_dir=args.output_dir)
    else:
        generate_wave_layer_from_hrrr(args.lake, date, args.fxx, args.output_dir)


if __name__ == '__main__':
    main()
