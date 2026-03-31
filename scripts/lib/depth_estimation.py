"""
Depth estimation for lakes without published bathymetry data.
"""

import math

# Known average depths for major US lakes (meters)
# Sources: USGS, state agencies, published bathymetric surveys
KNOWN_DEPTHS = {
    # Great Lakes
    "Lake Superior": 149.0,
    "Lake Michigan": 85.0,
    "Lake Huron": 59.0,
    "Lake Erie": 19.0,
    "Lake Ontario": 86.0,
    # Large western lakes
    "Lake Tahoe": 301.0,
    "Crater Lake": 350.0,
    "Flathead Lake": 50.0,
    "Lake Chelan": 143.0,
    "Lake Pend Oreille": 164.0,
    "Lake Coeur d'Alene": 22.0,
    "Priest Lake": 30.0,
    # Northeast
    "Lake Champlain": 20.0,
    "Lake George": 18.0,
    "Moosehead Lake": 16.0,
    "Sebago Lake": 31.0,
    "Lake Winnipesaukee": 13.0,
    "Rangeley Lake": 20.0,
    # Southeast
    "Lake Okeechobee": 2.7,
    "Lake Marion": 4.0,
    "Lake Moultrie": 5.5,
    # Midwest
    "Lake of the Woods": 8.0,
    "Mille Lacs Lake": 8.5,
    "Lake Sakakawea": 18.0,
    "Fort Peck Lake": 40.0,
    "Lake Oahe": 27.0,
    "Lake Francis Case": 24.0,
    "Lewis and Clark Lake": 6.0,
    "Lake Red Rock": 5.0,
    "Lake McConaughy": 43.0,
    "Table Rock Lake": 24.0,
    "Bull Shoals Lake": 27.0,
    "Lake of the Ozarks": 18.0,
    "Truman Reservoir": 9.0,
    # South Central
    "Toledo Bend Reservoir": 9.0,
    "Sam Rayburn Reservoir": 11.0,
    "Lake Texoma": 10.0,
    "Robert S. Kerr Reservoir": 7.0,
    "Eufaula Lake": 8.0,
    "Grand Lake O' the Cherokees": 12.0,
    "Lake Pontchartrain": 3.7,
    # West
    "Lake Powell": 40.0,
    "Lake Mead": 55.0,
    "Lake Havasu": 10.0,
    "Shasta Lake": 48.0,
    "Lake Berryessa": 30.0,
    "Clear Lake": 8.0,
    "Mono Lake": 17.0,
    "Pyramid Lake": 60.0,
    "Walker Lake": 21.0,
    "Utah Lake": 2.7,
    "Bear Lake": 28.0,
    "Yellowstone Lake": 42.0,
    "Jackson Lake": 43.0,
    "American Falls Reservoir": 12.0,
}


def estimate_depth(area_km2: float, name: str = None) -> float:
    """
    Estimate average lake depth in meters.

    Uses published depth for known lakes, otherwise falls back to an
    empirical area-depth relationship.

    Args:
        area_km2: Lake surface area in km^2
        name: Lake name for lookup (optional)

    Returns:
        Estimated average depth in meters
    """
    # Check known depths first
    if name:
        for known_name, depth in KNOWN_DEPTHS.items():
            if known_name.lower() in name.lower() or name.lower() in known_name.lower():
                return depth

    # Empirical fallback: depth tends to scale with sqrt of area
    # Capped at 50m to avoid unrealistic estimates
    return min(50.0, 3.0 * math.sqrt(area_km2))
