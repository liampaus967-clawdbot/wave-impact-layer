"""
Geospatial utilities for wave impact layer processing.
"""

import math


def utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    """
    Return the EPSG code for the UTM zone containing the given lon/lat.

    Args:
        lon: Longitude in degrees (WGS84)
        lat: Latitude in degrees (WGS84)

    Returns:
        EPSG code (e.g. 32618 for UTM zone 18N)
    """
    zone = math.floor((lon + 180) / 6) + 1
    if lat >= 0:
        return 32600 + zone
    else:
        return 32700 + zone


def utm_crs_from_lonlat(lon: float, lat: float) -> str:
    """Return CRS string like 'EPSG:32618' for the UTM zone at lon/lat."""
    return f"EPSG:{utm_epsg_from_lonlat(lon, lat)}"
