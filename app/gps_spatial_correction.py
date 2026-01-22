"""
Spatial GPS Correction Module

Corrects GPS coordinates based on location to account for homography extrapolation errors.
The correction varies spatially because homography errors increase with distance from 
calibration region and vary with perspective distortion.
"""
import numpy as np
from typing import Tuple, Optional
from app.gps_utils import gps_to_meters, meters_to_gps


# Door GPS coordinates (ground truth)
DOOR_COORDINATES = {
    'DD042': {
        'center': (41.911639525, -89.0447665625),  # Average of 4 corners
        'corners': [
            (41.91163919, -89.04478826),
            (41.91182147, -89.04479633),
            (41.91182219, -89.0447451),
            (41.91163986, -89.04473684)
        ]
    },
    'DD045': {
        'center': (41.9116427625, -89.0446126625),
        'corners': [
            (41.91164119, -89.04463401),
            (41.91182363, -89.04464264),
            (41.91182435, -89.04459141),
            (41.91164186, -89.04458259)
        ]
    },
    'DD046': {
        'center': (41.911642575, -89.0445615625),
        'corners': [
            (41.91164186, -89.04458259),
            (41.91182435, -89.04459141),
            (41.91182507, -89.04454018),
            (41.91164253, -89.04453117)
        ]
    },
    'DD048': {
        'center': (41.91164379, -89.04445894),
        'corners': [
            (41.9116432, -89.04447975),
            (41.91182579, -89.04448895),
            (41.9118265, -89.04443772),
            (41.91164387, -89.04442834)
        ]
    }
}

# GPS reference from calibration
GPS_REFERENCE = {
    'lat': 41.9125436,
    'lon': -89.04487889
}


def get_door_correction(world_x_m: float, world_y_m: float, 
                        gps_ref_lat: float, gps_ref_lon: float) -> Optional[Tuple[float, float]]:
    """
    Get GPS correction offset based on location (world coordinates in meters).
    
    This function identifies which door/region the point belongs to and returns
    a location-specific correction offset.
    
    Args:
        world_x_m: World X coordinate in meters (eastward from reference)
        world_y_m: World Y coordinate in meters (northward from reference)
        gps_ref_lat: GPS reference latitude
        gps_ref_lon: GPS reference longitude
    
    Returns:
        Tuple of (lat_offset_deg, lon_offset_deg) correction, or None if no correction available
    """
    # Convert world coordinates to GPS
    calc_lat, calc_lon = meters_to_gps(world_x_m, world_y_m, gps_ref_lat, gps_ref_lon)
    
    # Find closest door center
    min_dist = float('inf')
    closest_door = None
    closest_door_center = None
    
    for door_name, door_data in DOOR_COORDINATES.items():
        door_lat, door_lon = door_data['center']
        
        # Calculate distance in meters
        x_m, y_m = gps_to_meters(door_lat, door_lon, calc_lat, calc_lon)
        dist = np.sqrt(x_m**2 + y_m**2)
        
        if dist < min_dist:
            min_dist = dist
            closest_door = door_name
            closest_door_center = (door_lat, door_lon)
    
    # If we're close enough to a door center (within ~50m), apply correction
    if closest_door and min_dist < 50.0:
        # Calculate correction: difference between calculated and expected
        # For now, we'll need to learn these corrections from actual data
        # This is a placeholder - actual corrections should be calculated based on
        # comparing calculated GPS vs expected door centers from real measurements
        
        # TODO: Calculate actual corrections by comparing homography output
        # vs ground truth door centers. For now, return None to disable correction
        # until proper calibration data is available.
        
        return None
    
    return None


def apply_spatial_gps_correction(lat: float, lon: float, 
                                 world_x_m: float, world_y_m: float,
                                 gps_ref_lat: float, gps_ref_lon: float) -> Tuple[float, float]:
    """
    Apply spatial correction to GPS coordinates based on location.
    
    This function corrects GPS coordinates to account for homography extrapolation
    errors that vary by location. For now, this is a placeholder that returns
    uncorrected coordinates until proper correction data is available.
    
    Args:
        lat: Calculated latitude
        lon: Calculated longitude  
        world_x_m: World X coordinate in meters
        world_y_m: World Y coordinate in meters
        gps_ref_lat: GPS reference latitude
        gps_ref_lon: GPS reference longitude
    
    Returns:
        Corrected (lat, lon) tuple
    """
    correction = get_door_correction(world_x_m, world_y_m, gps_ref_lat, gps_ref_lon)
    
    if correction is not None:
        lat_offset, lon_offset = correction
        return (lat + lat_offset, lon + lon_offset)
    
    # No correction available - return original coordinates
    # NOTE: The homography may have errors for points far from calibration region
    return (lat, lon)
