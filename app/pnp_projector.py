"""
PnP (Perspective-n-Point) Projection Module

Projects 2D image coordinates to 3D world coordinates using camera pose
estimated from PnP calibration. Replaces homography-based projection.

Usage:
    projector = PnPProjector.load_from_file('config/calib/camera-id_pnp.json')
    x_world, y_world = projector.project_point(x_img, y_img)
    lat, lon = projector.project_to_gps(x_img, y_img)
"""

import numpy as np
import cv2
import json
from typing import Optional, Tuple, Dict
from pathlib import Path


class PnPProjector:
    """
    Projects 2D image points to 3D world coordinates using PnP camera pose.
    """
    
    def __init__(self, rvec: np.ndarray, tvec: np.ndarray, 
                 camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                 gps_reference: Optional[Dict] = None):
        """
        Initialize PnP projector.
        
        Args:
            rvec: Rotation vector (3x1) from PnP calibration
            tvec: Translation vector (3x1) from PnP calibration
            camera_matrix: Camera intrinsic matrix (3x3)
            dist_coeffs: Distortion coefficients (4x1 or 5x1)
            gps_reference: Optional dict with 'lat' and 'lon' for GPS conversion
        """
        self.rvec = np.array(rvec, dtype=np.float32)
        self.tvec = np.array(tvec, dtype=np.float32)
        self.camera_matrix = np.array(camera_matrix, dtype=np.float32)
        self.dist_coeffs = np.array(dist_coeffs, dtype=np.float32)
        self.gps_reference = gps_reference
        
        # Convert rotation vector to rotation matrix for easier use
        self.rotation_matrix, _ = cv2.Rodrigues(self.rvec)
        
        # Pre-compute inverse rotation and translation for efficiency
        self.inv_rotation = self.rotation_matrix.T
        self.inv_translation = -self.inv_rotation @ self.tvec
    
    @classmethod
    def load_from_file(cls, calib_path: str) -> 'PnPProjector':
        """
        Load PnP calibration from JSON file.
        
        Args:
            calib_path: Path to calibration JSON file
            
        Returns:
            PnPProjector instance
        """
        with open(calib_path, 'r') as f:
            calib_data = json.load(f)
        
        if calib_data.get('method') != 'pnp':
            raise ValueError(f"Calibration file is not PnP format: {calib_path}")
        
        rvec = np.array(calib_data['rvec'], dtype=np.float32)
        tvec = np.array(calib_data['tvec'], dtype=np.float32)
        camera_matrix = np.array(calib_data['camera_matrix'], dtype=np.float32)
        dist_coeffs = np.array(calib_data['dist_coeffs'], dtype=np.float32)
        
        gps_reference = None
        if 'gps_reference' in calib_data:
            gps_ref = calib_data['gps_reference']
            gps_reference = {
                'lat': gps_ref['lat'],
                'lon': gps_ref['lon']
            }
        
        return cls(rvec, tvec, camera_matrix, dist_coeffs, gps_reference)
    
    def project_point(self, x_img: float, y_img: float, z_world: float = 0.0) -> Tuple[float, float]:
        """
        Project a 2D image point to 3D world coordinates.
        
        This uses ray casting: we cast a ray from the camera through the image point
        and find where it intersects the ground plane (Z=z_world).
        
        Args:
            x_img: Image X coordinate (pixels)
            y_img: Image Y coordinate (pixels)
            z_world: World Z coordinate (meters, default 0.0 for ground plane)
        
        Returns:
            Tuple of (x_world, y_world) in meters
        """
        # Convert image point to normalized camera coordinates
        # First, undistort the point
        img_point = np.array([[x_img, y_img]], dtype=np.float32)
        undistorted = cv2.undistortPoints(
            img_point.reshape(1, 1, 2),
            self.camera_matrix,
            self.dist_coeffs,
            P=self.camera_matrix
        )
        x_norm, y_norm = undistorted[0, 0]
        
        # Get camera center in world coordinates
        camera_center_world = -self.inv_rotation @ self.tvec
        
        # Direction vector in camera coordinates (normalized)
        # In camera coordinates: Z is forward, X is right, Y is down
        direction_cam = np.array([x_norm, y_norm, 1.0], dtype=np.float32)
        direction_cam = direction_cam / np.linalg.norm(direction_cam)
        
        # Transform direction to world coordinates
        direction_world = self.inv_rotation @ direction_cam
        
        # Find intersection with ground plane (Z = z_world)
        # Ray equation: P = camera_center + t * direction
        # We want P_z = z_world
        # camera_center_z + t * direction_z = z_world
        # t = (z_world - camera_center_z) / direction_z
        
        camera_z = camera_center_world[2]
        direction_z = direction_world[2]
        
        if abs(direction_z) < 1e-6:
            # Ray is parallel to ground plane, cannot intersect
            raise ValueError("Image point projects to ray parallel to ground plane")
        
        t = (z_world - camera_z) / direction_z
        
        if t < 0:
            # Ray points away from ground plane
            raise ValueError("Image point projects behind camera")
        
        # Calculate world point
        world_point = camera_center_world + t * direction_world
        
        return float(world_point[0]), float(world_point[1])
    
    def project_to_gps(self, x_img: float, y_img: float, z_world: float = 0.0) -> Optional[Tuple[float, float]]:
        """
        Project a 2D image point to GPS coordinates.
        
        Args:
            x_img: Image X coordinate (pixels)
            y_img: Image Y coordinate (pixels)
            z_world: World Z coordinate (meters, default 0.0 for ground plane)
        
        Returns:
            Tuple of (latitude, longitude) in degrees, or None if GPS reference not available
        """
        x_world, y_world = self.project_point(x_img, y_img, z_world)
        
        if self.gps_reference is None:
            return None
        
        from app.gps_utils import meters_to_gps
        lat, lon = meters_to_gps(
            x_world, y_world,
            self.gps_reference['lat'],
            self.gps_reference['lon']
        )
        
        return (float(lat), float(lon))
    
    def project_multiple_points(self, image_points: np.ndarray, z_world: float = 0.0) -> np.ndarray:
        """
        Project multiple 2D image points to 3D world coordinates.
        
        Args:
            image_points: Array of shape (N, 2) with [x_img, y_img] pairs
            z_world: World Z coordinate (meters, default 0.0 for ground plane)
        
        Returns:
            Array of shape (N, 2) with [x_world, y_world] pairs
        """
        world_points = []
        for point in image_points:
            try:
                x_w, y_w = self.project_point(point[0], point[1], z_world)
                world_points.append([x_w, y_w])
            except (ValueError, ZeroDivisionError):
                # Point cannot be projected (e.g., parallel to ground)
                world_points.append([np.nan, np.nan])
        
        return np.array(world_points, dtype=np.float32)
    
    def get_camera_pose(self) -> Dict:
        """
        Get camera pose information.
        
        Returns:
            Dictionary with rotation matrix, translation vector, and camera center
        """
        camera_center_world = -self.inv_rotation @ self.tvec
        
        return {
            'rotation_matrix': self.rotation_matrix.tolist(),
            'rotation_vector': self.rvec.tolist(),
            'translation_vector': self.tvec.tolist(),
            'camera_center_world': camera_center_world.tolist()
        }
    
    def verify_calibration(self, image_points: np.ndarray, 
                          expected_world_points: np.ndarray) -> Dict:
        """
        Verify calibration by reprojecting world points and comparing with image points.
        
        Args:
            image_points: Array of shape (N, 2) with original image points
            expected_world_points: Array of shape (N, 3) with expected world points [X, Y, Z]
        
        Returns:
            Dictionary with reprojection errors and statistics
        """
        # Reproject world points to image
        projected_2d, _ = cv2.projectPoints(
            expected_world_points,
            self.rvec,
            self.tvec,
            self.camera_matrix,
            self.dist_coeffs
        )
        projected_2d = projected_2d.reshape(-1, 2)
        
        # Calculate errors
        errors = np.sqrt(np.sum((projected_2d - image_points) ** 2, axis=1))
        
        return {
            'reprojection_errors': errors.tolist(),
            'rmse': float(np.sqrt(np.mean(errors ** 2))),
            'max_error': float(np.max(errors)),
            'min_error': float(np.min(errors)),
            'mean_error': float(np.mean(errors))
        }


def load_pnp_calibration(calib_path: str) -> Optional[PnPProjector]:
    """
    Convenience function to load PnP calibration.
    
    Args:
        calib_path: Path to calibration JSON file
        
    Returns:
        PnPProjector instance, or None if file doesn't exist
    """
    if not Path(calib_path).exists():
        return None
    
    try:
        return PnPProjector.load_from_file(calib_path)
    except Exception as e:
        print(f"Warning: Failed to load PnP calibration from {calib_path}: {e}")
        return None
