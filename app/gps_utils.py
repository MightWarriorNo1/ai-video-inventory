"""
GPS Coordinate Conversion Utilities

Converts local world coordinates (meters) to GPS coordinates (latitude/longitude).
"""

import numpy as np
from typing import Optional, Tuple


def meters_to_gps(
    x_meters: float,
    y_meters: float,
    ref_lat: float,
    ref_lon: float
) -> Tuple[float, float]:
    """
    Convert local coordinates (meters) to GPS coordinates (lat/lon).
    
    This uses a simple approximation suitable for small areas (< 10km).
    For larger areas or higher accuracy, consider using UTM or other projections.
    
    Args:
        x_meters: Eastward distance in meters (positive = east)
        y_meters: Northward distance in meters (positive = north)
        ref_lat: Reference latitude (degrees) - corresponds to origin (0,0)
        ref_lon: Reference longitude (degrees) - corresponds to origin (0,0)
    
    Returns:
        Tuple of (latitude, longitude) in degrees
    
    Note:
        - 1 degree of latitude ≈ 111,320 meters (constant)
        - 1 degree of longitude ≈ 111,320 * cos(latitude) meters (varies with latitude)
        - y_meters maps to latitude (north/south)
        - x_meters maps to longitude (east/west)
    """
    # Convert reference to radians
    ref_lat_rad = np.radians(ref_lat)
    
    # Meters per degree
    meters_per_degree_lat = 111320.0  # Approximately constant
    meters_per_degree_lon = 111320.0 * np.cos(ref_lat_rad)  # Varies with latitude
    
    # Convert meters to degrees
    lat = ref_lat + (y_meters / meters_per_degree_lat)
    lon = ref_lon + (x_meters / meters_per_degree_lon)
    
    return (lat, lon)


def gps_to_meters(
    lat: float,
    lon: float,
    ref_lat: float,
    ref_lon: float
) -> Tuple[float, float]:
    """
    Convert GPS coordinates (lat/lon) to local coordinates (meters).
    
    Inverse of meters_to_gps().
    
    Args:
        lat: Latitude (degrees)
        lon: Longitude (degrees)
        ref_lat: Reference latitude (degrees) - corresponds to origin (0,0)
        ref_lon: Reference longitude (degrees) - corresponds to origin (0,0)
    
    Returns:
        Tuple of (x_meters, y_meters) where:
        - x_meters: Eastward distance in meters (positive = east)
        - y_meters: Northward distance in meters (positive = north)
    """
    # Convert reference to radians
    ref_lat_rad = np.radians(ref_lat)
    
    # Meters per degree
    meters_per_degree_lat = 111320.0
    meters_per_degree_lon = 111320.0 * np.cos(ref_lat_rad)
    
    # Convert degrees to meters
    y_meters = (lat - ref_lat) * meters_per_degree_lat
    x_meters = (lon - ref_lon) * meters_per_degree_lon
    
    return (x_meters, y_meters)


def validate_gps_coordinate(lat: float, lon: float) -> bool:
    """
    Validate GPS coordinates are within valid ranges.
    
    Args:
        lat: Latitude (degrees)
        lon: Longitude (degrees)
    
    Returns:
        True if valid, False otherwise
    """
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0




