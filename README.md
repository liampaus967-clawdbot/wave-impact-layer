# Wave Impact Layer

Generate wind-driven wave impact visualizations for lakes, similar to Deep Dive's "Wave Impacts" feature.

## Features

- **Fetch Calculation**: Compute wind fetch distances for any lake polygon
- **Wave Height Estimation**: SMB method wave height from wind speed + fetch
- **Bank Impact**: Shoreline wave exposure based on wind direction
- **Calm Zone Detection**: Identify sheltered areas
- **Mapbox-Ready Output**: GeoJSON for styling in Mapbox GL

## Data Sources

- **Wind**: HRRR (High-Resolution Rapid Refresh) via Herbie
- **Lakes**: NHD HR waterbody polygons
- **Depth**: Constant average depth (Phase 1)

## Prototype

Lake Champlain, Vermont/New York

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Prepare Lake Data
```bash
python scripts/01_prepare_lake.py --lake "Lake Champlain"
```

### 2. Calculate Fetch Rasters
```bash
python scripts/02_calculate_fetch.py --lake champlain
```

### 3. Generate Wave Impact Layer
```bash
python scripts/03_generate_wave_layer.py --lake champlain --wind-speed 10 --wind-dir 225
```

### 4. Generate from Live HRRR
```bash
python scripts/04_hrrr_wave_layer.py --lake champlain --latest
```

## Output

- `data/output/wave_intensity.geojson` - Wave height grid for water surface
- `data/output/bank_impact.geojson` - Shoreline impact segments
- `data/output/calm_zones.geojson` - Sheltered areas

## Algorithm

### Fetch Calculation
For each cell, trace rays in 16 directions until hitting land. Store fetch distance per direction.

### Wave Height (SMB Method)
```
Hs = 0.0016 × U² × √(X)
```
Where:
- Hs = significant wave height (m)
- U = wind speed (m/s)
- X = fetch distance (m)

### Bank Impact
```
Impact = wave_height × cos(angle)
```
Where angle = difference between wind direction and shoreline normal

## License

MIT
