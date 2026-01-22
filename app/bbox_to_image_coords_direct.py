"""
Direct bbox-to-image-coords calculation using exact coordinates from GPS.

This is a simpler, more direct approach that uses the exact image coordinates
from the CSV data when available.
"""

from typing import Tuple, List, Optional
import json
from pathlib import Path

# Load exact coordinates once
_SPOT_COORDS = None

def _load_exact_coords():
    """Load exact spot coordinates from config."""
    global _SPOT_COORDS
    if _SPOT_COORDS is not None:
        return _SPOT_COORDS
    
    coords_path = Path(__file__).parent.parent / "config" / "spot_image_coords.json"
    if coords_path.exists():
        with open(coords_path, 'r') as f:
            _SPOT_COORDS = json.load(f)
    else:
        _SPOT_COORDS = {}
    
    return _SPOT_COORDS


def calculate_image_coords_direct(
    bbox: List[float],
    spot: Optional[str] = None
) -> Tuple[float, float]:
    """
    Calculate image coordinates directly.
    
    If spot is known and in lookup table, use exact coordinates.
    Otherwise, use simple bottom-center method (most reliable fallback).
    
    Args:
        bbox: [x1, y1, x2, y2]
        spot: Spot name (DD042, YARD182, etc.)
        
    Returns:
        (x_img, y_img)
    """
    # Try exact coordinates first
    if spot and spot != "unknown":
        coords = _load_exact_coords()
        if spot in coords:
            return (float(coords[spot][0]), float(coords[spot][1]))
    
    # Fallback: bottom-center (simple and reliable)
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2.0
    bottom_y = float(y2)
    return (center_x, bottom_y)
