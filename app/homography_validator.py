"""
Homography Validation and Correction

Validates homography results based on distance from calibration points
and applies corrections for points outside the trusted region.
"""

import numpy as np
from typing import Tuple, Optional, List

# Try to import scipy for efficient distance calculation, fallback to numpy
try:
    from scipy.spatial.distance import cdist
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class HomographyValidator:
    """
    Validates and corrects homography results based on calibration point distribution.
    
    Homography accuracy degrades with distance from calibration points.
    This class helps identify trusted regions and apply corrections.
    """
    
    def __init__(self, calib_data: dict):
        """
        Initialize validator with calibration data.
        
        Args:
            calib_data: Dictionary with calibration data including:
                - image_points: List of [x, y] calibration points
                - world_points: List of [x, y] world coordinates
                - gps_points: List of [lat, lon] GPS coordinates
        """
        self.calib_data = calib_data
        self.image_points = np.array(calib_data['image_points'])
        self.world_points = np.array(calib_data['world_points'])
        
        # Calculate calibration region bounds
        self._calculate_trusted_region()
    
    def _calculate_trusted_region(self):
        """Calculate the trusted region based on calibration points."""
        # Image space bounds
        self.img_x_min = np.min(self.image_points[:, 0])
        self.img_x_max = np.max(self.image_points[:, 0])
        self.img_y_min = np.min(self.image_points[:, 1])
        self.img_y_max = np.max(self.image_points[:, 1])
        
        # World space bounds
        self.world_x_min = np.min(self.world_points[:, 0])
        self.world_x_max = np.max(self.world_points[:, 0])
        self.world_y_min = np.min(self.world_points[:, 1])
        self.world_y_max = np.max(self.world_points[:, 1])
        
        # Calculate typical distance between calibration points
        if len(self.world_points) > 1:
            distances = []
            for i in range(len(self.world_points)):
                for j in range(i + 1, len(self.world_points)):
                    dist = np.sqrt(
                        (self.world_points[i, 0] - self.world_points[j, 0])**2 +
                        (self.world_points[i, 1] - self.world_points[j, 1])**2
                    )
                    distances.append(dist)
            self.avg_calib_distance = np.mean(distances)
            self.max_calib_distance = np.max(distances)
        else:
            self.avg_calib_distance = 10.0  # Default
            self.max_calib_distance = 20.0  # Default
        
        # Trusted region: within 1.5x the max calibration distance
        self.trusted_radius = self.max_calib_distance * 1.5
    
    def is_in_trusted_region(self, x_world: float, y_world: float) -> bool:
        """
        Check if a world point is within the trusted calibration region.
        
        Args:
            x_world: World X coordinate (meters)
            y_world: World Y coordinate (meters)
        
        Returns:
            True if point is within trusted region
        """
        # Check if point is within bounds of calibration region
        in_x_bounds = self.world_x_min - self.trusted_radius <= x_world <= self.world_x_max + self.trusted_radius
        in_y_bounds = self.world_y_min - self.trusted_radius <= y_world <= self.world_y_max + self.trusted_radius
        
        # Also check distance from nearest calibration point
        if len(self.world_points) > 0:
            if HAS_SCIPY:
                point = np.array([[x_world, y_world]])
                distances = cdist(point, self.world_points)
                min_distance = np.min(distances)
            else:
                # Fallback to numpy calculation
                point = np.array([x_world, y_world])
                distances = np.sqrt(np.sum((self.world_points - point)**2, axis=1))
                min_distance = np.min(distances)
            within_radius = min_distance <= self.trusted_radius
        else:
            within_radius = True
        
        return in_x_bounds and in_y_bounds and within_radius
    
    def get_distance_from_calibration(self, x_world: float, y_world: float) -> float:
        """
        Get minimum distance from point to nearest calibration point.
        
        Args:
            x_world: World X coordinate (meters)
            y_world: World Y coordinate (meters)
        
        Returns:
            Distance in meters to nearest calibration point
        """
        if len(self.world_points) == 0:
            return 0.0
        
        if HAS_SCIPY:
            point = np.array([[x_world, y_world]])
            distances = cdist(point, self.world_points)
        else:
            # Fallback to numpy calculation
            point = np.array([x_world, y_world])
            distances = np.sqrt(np.sum((self.world_points - point)**2, axis=1))
        return float(np.min(distances))
    
    def get_confidence_score(self, x_world: float, y_world: float) -> float:
        """
        Get confidence score (0.0 to 1.0) based on distance from calibration.
        
        Args:
            x_world: World X coordinate (meters)
            y_world: World Y coordinate (meters)
        
        Returns:
            Confidence score: 1.0 = very close, 0.0 = very far
        """
        distance = self.get_distance_from_calibration(x_world, y_world)
        
        # Confidence decreases with distance
        # Within trusted radius: high confidence (0.8-1.0)
        # 1.5x trusted radius: medium confidence (0.5-0.8)
        # 2x trusted radius: low confidence (0.2-0.5)
        # Beyond 2x: very low confidence (<0.2)
        
        if distance <= self.trusted_radius:
            # Within trusted region: 0.8 to 1.0
            confidence = 1.0 - (distance / self.trusted_radius) * 0.2
        elif distance <= self.trusted_radius * 1.5:
            # Slightly outside: 0.5 to 0.8
            excess = distance - self.trusted_radius
            max_excess = self.trusted_radius * 0.5
            confidence = 0.8 - (excess / max_excess) * 0.3
        elif distance <= self.trusted_radius * 2.0:
            # Further outside: 0.2 to 0.5
            excess = distance - self.trusted_radius * 1.5
            max_excess = self.trusted_radius * 0.5
            confidence = 0.5 - (excess / max_excess) * 0.3
        else:
            # Very far: <0.2
            confidence = max(0.0, 0.2 - (distance - self.trusted_radius * 2.0) / self.trusted_radius * 0.2)
        
        return max(0.0, min(1.0, confidence))
    
    def should_use_result(self, x_world: float, y_world: float, 
                         min_confidence: float = 0.5) -> Tuple[bool, float]:
        """
        Determine if homography result should be used based on confidence.
        
        Args:
            x_world: World X coordinate (meters)
            y_world: World Y coordinate (meters)
            min_confidence: Minimum confidence threshold (default 0.5)
        
        Returns:
            Tuple of (should_use, confidence_score)
        """
        confidence = self.get_confidence_score(x_world, y_world)
        should_use = confidence >= min_confidence
        return (should_use, confidence)
    
    def get_trusted_region_info(self) -> dict:
        """
        Get information about the trusted calibration region.
        
        Returns:
            Dictionary with region bounds and statistics
        """
        return {
            'world_bounds': {
                'x_min': float(self.world_x_min),
                'x_max': float(self.world_x_max),
                'y_min': float(self.world_y_min),
                'y_max': float(self.world_y_max)
            },
            'image_bounds': {
                'x_min': float(self.img_x_min),
                'x_max': float(self.img_x_max),
                'y_min': float(self.img_y_min),
                'y_max': float(self.img_y_max)
            },
            'trusted_radius': float(self.trusted_radius),
            'avg_calib_distance': float(self.avg_calib_distance),
            'max_calib_distance': float(self.max_calib_distance),
            'num_calibration_points': len(self.image_points)
        }


def validate_homography_result(
    calib_data: dict,
    x_world: float,
    y_world: float,
    min_confidence: float = 0.5
) -> dict:
    """
    Quick validation function for homography results.
    
    Args:
        calib_data: Calibration data dictionary
        x_world: World X coordinate (meters)
        y_world: World Y coordinate (meters)
        min_confidence: Minimum confidence threshold
    
    Returns:
        Dictionary with validation results:
        - is_trusted: bool
        - confidence: float (0.0-1.0)
        - distance_from_calib: float (meters)
        - should_use: bool
    """
    validator = HomographyValidator(calib_data)
    distance = validator.get_distance_from_calibration(x_world, y_world)
    confidence = validator.get_confidence_score(x_world, y_world)
    should_use, _ = validator.should_use_result(x_world, y_world, min_confidence)
    is_trusted = validator.is_in_trusted_region(x_world, y_world)
    
    return {
        'is_trusted': is_trusted,
        'confidence': confidence,
        'distance_from_calib': distance,
        'should_use': should_use,
        'trusted_radius': validator.trusted_radius
    }

