"""
Resolve world coordinates to exact spot image coordinates.

Simple direct lookup using exact coordinates from CSV.
"""

import json
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import cv2


# Cache
_SPOT_IMAGE_COORDS = None
_SPOT_WORLD_COORDS_CACHE = {}  # (homography_hash, spot) -> world_coords


def _load_spot_coords():
    """Load spot image coordinates."""
    global _SPOT_IMAGE_COORDS
    if _SPOT_IMAGE_COORDS is not None:
        return _SPOT_IMAGE_COORDS
    
    coords_path = Path(__file__).parent.parent / "config" / "spot_image_coords.json"
    if coords_path.exists():
        with open(coords_path, 'r') as f:
            _SPOT_IMAGE_COORDS = json.load(f)
    else:
        _SPOT_IMAGE_COORDS = {}
    
    return _SPOT_IMAGE_COORDS


def get_exact_image_coords(spot: str) -> Optional[Tuple[float, float]]:
    """Get exact image coordinates for spot."""
    coords = _load_spot_coords()
    if spot in coords:
        return (float(coords[spot][0]), float(coords[spot][1]))
    return None


def find_closest_spot(
    x_world: float,
    y_world: float,
    homography_inv: np.ndarray,
    max_distance: float = 10.0
) -> Optional[Tuple[str, float]]:
    """
    Find closest spot by converting spot image coords to world coords.
    
    Returns: (spot_name, distance) or None
    """
    coords = _load_spot_coords()
    if not coords or homography_inv is None:
        return None
    
    min_dist = float('inf')
    closest_spot = None
    
    for spot, img_coords in coords.items():
        # Convert image coords to world coords using inverse homography
        point = np.array([[img_coords[0], img_coords[1]]], dtype=np.float32)
        point = np.array([point])
        world_point = cv2.perspectiveTransform(point, homography_inv)
        spot_x, spot_y = world_point[0][0]
        
        # Calculate distance
        dist = np.sqrt((x_world - spot_x)**2 + (y_world - spot_y)**2)
        
        if dist < min_dist:
            min_dist = dist
            closest_spot = spot
    
    if min_dist < max_distance:
        return (closest_spot, min_dist)
    
    return None
