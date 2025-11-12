"""
Parking Spot Resolver

Resolves world coordinates (x, y) to named parking spots using GeoJSON polygons.
Uses point-in-polygon testing with nearest-centroid fallback.
"""

import json
import os
from typing import Dict, List, Tuple, Optional
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import LinearRing
import numpy as np


class SpotResolver:
    """
    Resolves world coordinates to parking spot names using GeoJSON polygons.
    """
    
    def __init__(self, geojson_path: str):
        """
        Initialize spot resolver with GeoJSON polygons.
        
        Args:
            geojson_path: Path to spots.geojson file
        """
        self.geojson_path = geojson_path
        self.spots = []  # List of {name, polygon, centroid}
        
        self._load_polygons()
    
    def _load_polygons(self):
        """Load parking spot polygons from GeoJSON file."""
        if not os.path.exists(self.geojson_path):
            print(f"Warning: GeoJSON file not found: {self.geojson_path}")
            return
        
        with open(self.geojson_path, 'r') as f:
            geojson = json.load(f)
        
        for feature in geojson.get('features', []):
            if feature['geometry']['type'] != 'Polygon':
                continue
            
            name = feature['properties'].get('name', 'unknown')
            coords = feature['geometry']['coordinates'][0]  # Exterior ring
            
            # Create Shapely polygon
            polygon = Polygon(coords)
            
            # Compute centroid
            centroid = polygon.centroid
            
            self.spots.append({
                'name': name,
                'polygon': polygon,
                'centroid': centroid
            })
        
        print(f"Loaded {len(self.spots)} parking spots from {self.geojson_path}")
    
    def resolve(self, x_world: float, y_world: float) -> Dict[str, str]:
        """
        Resolve world coordinates to a parking spot.
        
        Args:
            x_world: World X coordinate (meters)
            y_world: World Y coordinate (meters)
            
        Returns:
            Dict with keys: spot (spot name), method ('point-in-polygon' or 'nearest-centroid')
        """
        point = Point(x_world, y_world)
        
        # Try point-in-polygon first
        for spot in self.spots:
            if spot['polygon'].contains(point):
                return {
                    'spot': spot['name'],
                    'method': 'point-in-polygon'
                }
        
        # Fallback to nearest centroid
        if len(self.spots) == 0:
            return {
                'spot': 'unknown',
                'method': 'no-spots'
            }
        
        min_dist = float('inf')
        nearest_spot = None
        
        for spot in self.spots:
            centroid = spot['centroid']
            dist = point.distance(centroid)
            if dist < min_dist:
                min_dist = dist
                nearest_spot = spot
        
        return {
            'spot': nearest_spot['name'],
            'method': 'nearest-centroid'
        }
    
    def get_all_spots(self) -> List[Dict[str, any]]:
        """
        Get all parking spots with their polygons and centroids.
        
        Returns:
            List of spot dicts with keys: name, polygon, centroid
        """
        return self.spots.copy()

