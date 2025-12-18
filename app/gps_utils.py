"""
GPS Coordinate Conversion Utilities

Converts local world coordinates (meters) to GPS coordinates (latitude/longitude).
Uses pyproj for accurate geodesic calculations based on WGS84 ellipsoid.
Implements ENU (East-North-Up) local coordinate system for precise conversions.
"""

import numpy as np
from typing import Optional, Tuple
from pyproj import Geod, Transformer
from pyproj.crs import CRS


# Initialize Geod object with WGS84 ellipsoid for accurate geodesic calculations
_geod = Geod(ellps='WGS84')


def meters_to_gps(
    x_meters: float,
    y_meters: float,
    ref_lat: float,
    ref_lon: float
) -> Tuple[float, float]:
    """
    Convert local ENU coordinates (meters) to GPS coordinates (lat/lon).
    
    Uses pyproj for accurate geodesic calculations based on WGS84 ellipsoid.
    Implements proper ENU (East-North-Up) local coordinate system.
    This provides sub-meter accuracy for distances up to hundreds of kilometers.
    
    Args:
        x_meters: Eastward distance in meters (positive = east, maps to longitude)
        y_meters: Northward distance in meters (positive = north, maps to latitude)
        ref_lat: Reference latitude (degrees) - corresponds to origin (0,0)
        ref_lon: Reference longitude (degrees) - corresponds to origin (0,0)
    
    Returns:
        Tuple of (latitude, longitude) in degrees
    
    Note:
        - Uses geodesic forward calculation for accurate results
        - Implements ENU coordinate system: X=east, Y=north
        - Azimuth is measured clockwise from north (0° = north, 90° = east)
        - For small distances, this is equivalent to simple approximation
        - For larger distances, accounts for Earth's curvature
    """
    # Calculate distance and azimuth from (x_meters, y_meters)
    # Distance is the hypotenuse
    distance = np.sqrt(x_meters**2 + y_meters**2)
    
    # If distance is zero, return reference point
    if distance < 1e-9:  # Essentially zero
        return (ref_lat, ref_lon)
    
    # Azimuth: atan2(x, y) gives angle from north (0° = north, 90° = east)
    # In ENU: x is east, y is north, so atan2(x, y) gives azimuth from north
    # Convert to degrees and ensure it's in [0, 360) range
    azimuth = np.degrees(np.arctan2(x_meters, y_meters))
    if azimuth < 0:
        azimuth += 360.0
    
    # Use geodesic forward calculation to get destination point
    # geod.fwd returns (lon, lat, back_azimuth)
    lon, lat, _ = _geod.fwd(ref_lon, ref_lat, azimuth, distance)
    
    return (lat, lon)


def gps_to_meters(
    lat: float,
    lon: float,
    ref_lat: float,
    ref_lon: float
) -> Tuple[float, float]:
    """
    Convert GPS coordinates (lat/lon) to local ENU coordinates (meters).
    
    Uses pyproj for accurate geodesic calculations based on WGS84 ellipsoid.
    Implements proper ENU (East-North-Up) local coordinate system.
    Inverse of meters_to_gps().
    
    Args:
        lat: Latitude (degrees)
        lon: Longitude (degrees)
        ref_lat: Reference latitude (degrees) - corresponds to origin (0,0)
        ref_lon: Reference longitude (degrees) - corresponds to origin (0,0)
    
    Returns:
        Tuple of (x_meters, y_meters) where:
        - x_meters: Eastward distance in meters (positive = east, maps to longitude)
        - y_meters: Northward distance in meters (positive = north, maps to latitude)
    
    Note:
        - Uses geodesic inverse calculation for accurate distance and azimuth
        - Implements ENU coordinate system: X=east, Y=north
        - Azimuth is measured clockwise from north (0° = north, 90° = east)
        - For small distances, this is equivalent to simple approximation
        - For larger distances, accounts for Earth's curvature
    """
    # Use geodesic inverse calculation to get distance and azimuth
    # geod.inv returns (forward_azimuth, back_azimuth, distance)
    azimuth, back_azimuth, distance = _geod.inv(ref_lon, ref_lat, lon, lat)
    
    # Convert azimuth and distance to ENU components
    # In ENU: x is eastward, y is northward
    # x = distance * sin(azimuth)  [east component]
    # y = distance * cos(azimuth)  [north component]
    azimuth_rad = np.radians(azimuth)
    x_meters = distance * np.sin(azimuth_rad)
    y_meters = distance * np.cos(azimuth_rad)
    
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


def verify_coordinate_system(
    ref_lat: float,
    ref_lon: float,
    test_points: list,
    expected_gps_points: list
) -> dict:
    """
    Verify that the coordinate system conversion is working correctly.
    
    This function helps debug coordinate system issues by testing
    round-trip conversions and comparing with expected GPS coordinates.
    
    Args:
        ref_lat: Reference latitude (degrees)
        ref_lon: Reference longitude (degrees)
        test_points: List of (x_meters, y_meters) tuples to test
        expected_gps_points: List of (lat, lon) tuples expected for test_points
    
    Returns:
        Dictionary with verification results including:
        - errors: List of error distances in meters for each point
        - max_error: Maximum error in meters
        - avg_error: Average error in meters
        - round_trip_errors: Errors from round-trip conversion (GPS->meters->GPS)
    """
    errors = []
    round_trip_errors = []
    
    for (x_m, y_m), (exp_lat, exp_lon) in zip(test_points, expected_gps_points):
        # Forward conversion: meters -> GPS
        calc_lat, calc_lon = meters_to_gps(x_m, y_m, ref_lat, ref_lon)
        
        # Calculate error distance
        _, _, error_dist = _geod.inv(exp_lon, exp_lat, calc_lon, calc_lat)
        errors.append(abs(error_dist))
        
        # Round-trip test: GPS -> meters -> GPS
        x_back, y_back = gps_to_meters(calc_lat, calc_lon, ref_lat, ref_lon)
        lat_back, lon_back = meters_to_gps(x_back, y_back, ref_lat, ref_lon)
        _, _, rt_error = _geod.inv(calc_lon, calc_lat, lon_back, lat_back)
        round_trip_errors.append(abs(rt_error))
    
    return {
        'errors': errors,
        'max_error': max(errors) if errors else 0.0,
        'avg_error': np.mean(errors) if errors else 0.0,
        'round_trip_errors': round_trip_errors,
        'max_round_trip_error': max(round_trip_errors) if round_trip_errors else 0.0
    }




