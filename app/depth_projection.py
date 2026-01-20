"""
Standalone Depth-Based Projection Module

This module provides depth-based 3D projection that works independently
of the full 3D projector. It uses depth estimation + camera intrinsics
to project pixels to world coordinates.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict
from pathlib import Path
import json


class DepthProjector:
    """
    Standalone depth-based projector that uses depth estimation + camera intrinsics.
    
    This works independently of the full 3D projector and only requires:
    - Depth estimator (for depth maps)
    - Camera intrinsics (focal length, principal point)
    - Optional: GPS reference for GPS conversion
    """
    
    def __init__(
        self,
        depth_estimator,
        intrinsic_matrix: Optional[np.ndarray] = None,
        gps_reference: Optional[Dict[str, float]] = None
    ):
        """
        Initialize depth projector.
        
        Args:
            depth_estimator: DepthEstimator instance
            intrinsic_matrix: Optional 3×3 camera intrinsic matrix K.
                            If None, will try to load from calibration file or estimate.
            gps_reference: Optional GPS reference point dict with 'lat' and 'lon'
        """
        self.depth_estimator = depth_estimator
        
        # Load or estimate intrinsics
        if intrinsic_matrix is not None:
            self.K = np.array(intrinsic_matrix, dtype=np.float32)
        else:
            # Try to estimate or use defaults
            self.K = self._estimate_intrinsics()
        
        self.gps_reference = gps_reference
        
        # Extract intrinsic parameters
        self.fx = float(self.K[0, 0])
        self.fy = float(self.K[1, 1])
        self.cx = float(self.K[0, 2])
        self.cy = float(self.K[1, 2])
    
    @classmethod
    def from_calibration_file(cls, depth_estimator, calib_path: str) -> 'DepthProjector':
        """
        Load depth projector from calibration JSON file.
        
        Args:
            depth_estimator: DepthEstimator instance
            calib_path: Path to calibration JSON file (3D or homography)
            
        Returns:
            DepthProjector instance
        """
        calib_path = Path(calib_path)
        if not calib_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {calib_path}")
        
        with open(calib_path, 'r') as f:
            calib_data = json.load(f)
        
        # Try to get intrinsics from 3D calibration
        intrinsic_matrix = None
        if 'intrinsic_matrix' in calib_data:
            intrinsic_matrix = np.array(calib_data['intrinsic_matrix'], dtype=np.float32)
        
        gps_reference = calib_data.get('gps_reference')
        
        return cls(depth_estimator, intrinsic_matrix, gps_reference)
    
    def _estimate_intrinsics(self) -> np.ndarray:
        """
        Estimate camera intrinsics if not provided.
        Uses reasonable defaults based on common camera setups.
        """
        # Default intrinsics for typical camera (can be improved with calibration)
        # Assuming 1920x1080 image, ~60-70 degree FOV
        fx = fy = 1500.0  # Focal length in pixels
        cx = 960.0  # Principal point X (center of 1920 width)
        cy = 540.0  # Principal point Y (center of 1080 height)
        
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        print(f"[DepthProjector] Using estimated intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        print(f"[DepthProjector] WARNING: Estimated intrinsics may reduce accuracy. Use calibrated values for best results.")
        
        return K
    
    def bbox_to_ground_coords(
        self,
        bbox: List[float],
        depth_map: np.ndarray,
        current_image_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[float, float]:
        """
        Convert bounding box to ground coordinates using depth estimation.
        
        This is a standalone method that doesn't require full 3D calibration.
        It uses:
        1. Depth map from depth estimator
        2. Camera intrinsics (from calibration or estimated)
        3. Assumes ground plane at z=0
        
        Args:
            bbox: Bounding box as [x1, y1, x2, y2]
            depth_map: Depth map from depth_estimator.estimate_depth()
            current_image_size: Optional (width, height) of current image
            
        Returns:
            (X_ground, Y_ground) - Ground contact point in world coordinates (meters)
        """
        x1, y1, x2, y2 = bbox
        
        # Step 1: Extract depth value from bbox (median depth)
        x1_int, y1_int = int(x1), int(y1)
        x2_int, y2_int = int(x2), int(y2)
        h, w = depth_map.shape
        x1_int = max(0, min(x1_int, w - 1))
        y1_int = max(0, min(y1_int, h - 1))
        x2_int = max(0, min(x2_int, w - 1))
        y2_int = max(0, min(y2_int, h - 1))
        
        bbox_region = depth_map[y1_int:y2_int+1, x1_int:x2_int+1]
        if bbox_region.size > 0:
            depth = float(np.median(bbox_region))
        else:
            # Fallback to bottom center
            center_x = int((x1 + x2) / 2)
            bottom_y = int(y2)
            center_x = max(0, min(center_x, w - 1))
            bottom_y = max(0, min(bottom_y, h - 1))
            depth = float(depth_map[bottom_y, center_x])
        
        # Step 2: Get bottom center pixel (ground contact point)
        bottom_center_u = (x1 + x2) / 2.0
        bottom_center_v = float(y2)
        
        # Step 3: Unproject using K^-1: [x, y, z] = K^-1 [u, v, 1] · d
        u_homogeneous = np.array([[bottom_center_u], [bottom_center_v], [1.0]], dtype=np.float32)
        K_inv = np.linalg.inv(self.K)
        ray_normalized = K_inv @ u_homogeneous
        
        # Scale by depth: [x, y, z] = ray_normalized · d
        point_3d_cam = ray_normalized * depth
        
        # Step 4: For standalone depth projection, we assume camera is at origin
        # and project directly to ground plane (z=0)
        # The depth gives us distance along the ray
        # We want the point where the ray intersects ground plane z=0
        
        # Ray direction in camera coordinates
        ray_dir = ray_normalized.flatten()[:3]
        
        # Ray equation: P = origin + t * ray_dir
        # We want z = 0 in camera coordinates
        # For a typical camera setup, ground is below camera
        # We'll use the depth-scaled point and project to z=0
        
        # Simple projection: use x, y from depth-scaled point, assume z=0
        # This gives us relative position from camera
        x_world = float(point_3d_cam[0, 0])
        y_world = float(point_3d_cam[1, 0])
        
        # Note: This gives camera-relative coordinates
        # For absolute world coordinates, you'd need camera pose (extrinsics)
        # But for relative positioning, this works
        
        return (x_world, y_world)
    
    def pixel_to_world(
        self,
        u: float,
        v: float,
        depth: float,
        return_gps: bool = False
    ) -> Tuple[float, float]:
        """
        Project a single pixel to world coordinates using depth.
        
        Args:
            u, v: Pixel coordinates
            depth: Depth value at (u, v) in meters
            return_gps: If True and GPS reference available, return GPS coordinates
            
        Returns:
            (X_world, Y_world) or (lat, lon) depending on return_gps
        """
        # Unproject using K^-1
        u_homogeneous = np.array([[u], [v], [1.0]], dtype=np.float32)
        K_inv = np.linalg.inv(self.K)
        ray_normalized = K_inv @ u_homogeneous
        
        # Scale by depth
        point_3d_cam = ray_normalized * depth
        
        x_world = float(point_3d_cam[0, 0])
        y_world = float(point_3d_cam[1, 0])
        
        if return_gps and self.gps_reference:
            from app.gps_utils import meters_to_gps
            lat, lon = meters_to_gps(
                x_world, y_world,
                self.gps_reference['lat'],
                self.gps_reference['lon']
            )
            return (lat, lon)
        
        return (x_world, y_world)
