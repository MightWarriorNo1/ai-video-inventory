"""
Alternative GPS Coordinate Calculation Methods

Provides multiple approaches to get GPS coordinates from image points
when homography is not working correctly.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from app.gps_utils import meters_to_gps, gps_to_meters


# ============================================================================
# 1. Camera Pose Estimation (PnP)
# ============================================================================

class CameraPoseEstimator:
    """
    Estimate camera pose using PnP and project image points to GPS coordinates.
    """
    
    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None,
        gps_reference: Optional[dict] = None
    ):
        """
        Initialize camera pose estimator.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix (focal length, principal point)
            dist_coeffs: Distortion coefficients (optional)
            gps_reference: GPS reference point {'lat': float, 'lon': float}
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((4, 1))
        self.gps_reference = gps_reference
        
        # Camera pose (will be set during calibration)
        self.rvec = None  # Rotation vector
        self.tvec = None  # Translation vector
    
    def calibrate(
        self,
        image_points: List[Tuple[float, float]],
        world_points_3d: List[Tuple[float, float, float]],
        gps_points: Optional[List[Tuple[float, float]]] = None
    ) -> dict:
        """
        Calibrate camera pose using known 3D points and their image projections.
        
        Args:
            image_points: List of (x, y) pixel coordinates
            world_points_3d: List of (x, y, z) world coordinates in meters
            gps_points: Optional list of (lat, lon) GPS coordinates
        
        Returns:
            Dictionary with calibration results
        """
        if len(image_points) < 4:
            raise ValueError("Need at least 4 points for PnP calibration")
        
        # Convert to numpy arrays
        img_pts = np.array(image_points, dtype=np.float32)
        obj_pts = np.array(world_points_3d, dtype=np.float32)
        
        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            raise ValueError("Failed to solve PnP")
        
        self.rvec = rvec
        self.tvec = tvec
        
        # Calculate reprojection error
        reprojected, _ = cv2.projectPoints(obj_pts, rvec, tvec, 
                                          self.camera_matrix, self.dist_coeffs)
        reprojected = reprojected.reshape(-1, 2)
        errors = np.sqrt(np.sum((img_pts - reprojected)**2, axis=1))
        rmse = np.sqrt(np.mean(errors**2))
        
        # Store GPS reference if provided
        if gps_points and len(gps_points) > 0:
            # Use first GPS point as reference
            self.gps_reference = {
                'lat': gps_points[0][0],
                'lon': gps_points[0][1]
            }
        
        return {
            'rvec': rvec.tolist(),
            'tvec': tvec.tolist(),
            'rmse': float(rmse),
            'reprojection_errors': errors.tolist(),
            'gps_reference': self.gps_reference
        }
    
    def project_to_world(
        self,
        image_point: Tuple[float, float],
        ground_plane_z: float = 0.0
    ) -> Tuple[float, float, float]:
        """
        Project image point to 3D world coordinates.
        
        Args:
            image_point: (x, y) pixel coordinates
            ground_plane_z: Z coordinate of ground plane (default 0.0)
        
        Returns:
            (x, y, z) world coordinates in meters
        """
        if self.rvec is None or self.tvec is None:
            raise ValueError("Camera pose not calibrated. Call calibrate() first.")
        
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(self.rvec)
        
        # Camera center in world coordinates
        camera_center = -R.T @ self.tvec
        
        # Image point in normalized camera coordinates
        img_pt = np.array([[image_point[0], image_point[1]]], dtype=np.float32)
        
        # Undistort if needed
        if np.any(self.dist_coeffs != 0):
            img_pt = cv2.undistortPoints(img_pt, self.camera_matrix, 
                                        self.dist_coeffs, P=self.camera_matrix)
        
        # Convert to normalized coordinates
        x_norm = (img_pt[0, 0] - self.camera_matrix[0, 2]) / self.camera_matrix[0, 0]
        y_norm = (img_pt[0, 1] - self.camera_matrix[1, 2]) / self.camera_matrix[1, 1]
        
        # Direction vector from camera center
        direction = np.array([x_norm, y_norm, 1.0])
        direction = R.T @ direction
        direction = direction / np.linalg.norm(direction)
        
        # Intersect with ground plane
        # Ray: camera_center + t * direction
        # Ground plane: z = ground_plane_z
        if abs(direction[2]) < 1e-6:
            raise ValueError("Ray is parallel to ground plane")
        
        t = (ground_plane_z - camera_center[2]) / direction[2]
        world_point = camera_center + t * direction
        
        return tuple(world_point.flatten())
    
    def project_to_gps(
        self,
        image_point: Tuple[float, float],
        ground_plane_z: float = 0.0
    ) -> Optional[Tuple[float, float]]:
        """
        Project image point directly to GPS coordinates.
        
        Args:
            image_point: (x, y) pixel coordinates
            ground_plane_z: Z coordinate of ground plane
        
        Returns:
            (lat, lon) GPS coordinates or None if no GPS reference
        """
        if self.gps_reference is None:
            return None
        
        x_world, y_world, z_world = self.project_to_world(image_point, ground_plane_z)
        
        lat, lon = meters_to_gps(
            x_world, y_world,
            self.gps_reference['lat'],
            self.gps_reference['lon']
        )
        
        return (lat, lon)


def estimate_camera_intrinsics(image_width: int, image_height: int, 
                               fov_degrees: float = 60.0) -> np.ndarray:
    """
    Estimate camera intrinsic matrix from image dimensions and field of view.
    
    Args:
        image_width: Image width in pixels
        image_height: Image height in pixels
        fov_degrees: Field of view in degrees (horizontal)
    
    Returns:
        3x3 camera intrinsic matrix
    """
    # Calculate focal length from FOV
    fov_rad = np.radians(fov_degrees)
    focal_length = (image_width / 2.0) / np.tan(fov_rad / 2.0)
    
    # Principal point (image center)
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    camera_matrix = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return camera_matrix


# ============================================================================
# 2. Direct GPS Tagging with Known Landmarks
# ============================================================================

class LandmarkGPSMapper:
    """
    Map image coordinates to GPS using known landmarks with interpolation.
    """
    
    def __init__(self, gps_reference: Optional[dict] = None):
        """
        Initialize landmark mapper.
        
        Args:
            gps_reference: Optional GPS reference point
        """
        self.landmarks = []  # List of (image_x, image_y, gps_lat, gps_lon)
        self.gps_reference = gps_reference
        self.interpolation_method = 'cubic'  # 'linear', 'cubic', 'nearest'
    
    def add_landmark(
        self,
        image_x: float,
        image_y: float,
        gps_lat: float,
        gps_lon: float
    ):
        """
        Add a known landmark point.
        
        Args:
            image_x: X pixel coordinate
            image_y: Y pixel coordinate
            gps_lat: GPS latitude
            gps_lon: GPS longitude
        """
        self.landmarks.append((image_x, image_y, gps_lat, gps_lon))
    
    def add_landmarks_batch(self, landmarks: List[Tuple[float, float, float, float]]):
        """
        Add multiple landmarks at once.
        
        Args:
            landmarks: List of (image_x, image_y, gps_lat, gps_lon) tuples
        """
        self.landmarks.extend(landmarks)
    
    def project_to_gps(
        self,
        image_point: Tuple[float, float],
        method: str = 'interpolate'
    ) -> Optional[Tuple[float, float]]:
        """
        Project image point to GPS coordinates.
        
        Args:
            image_point: (x, y) pixel coordinates
            method: 'interpolate' (grid interpolation) or 'nearest' (nearest neighbor)
        
        Returns:
            (lat, lon) GPS coordinates or None if insufficient landmarks
        """
        if len(self.landmarks) < 3:
            return None
        
        img_x, img_y = image_point
        
        if method == 'nearest':
            return self._nearest_neighbor(img_x, img_y)
        else:
            return self._interpolate(img_x, img_y)
    
    def _nearest_neighbor(self, img_x: float, img_y: float) -> Tuple[float, float]:
        """Use nearest landmark."""
        if not self.landmarks:
            return None
        
        # Build KD-tree for fast nearest neighbor search
        image_points = np.array([(x, y) for x, y, _, _ in self.landmarks])
        tree = cKDTree(image_points)
        
        # Find nearest landmark
        distance, idx = tree.query([img_x, img_y])
        _, _, lat, lon = self.landmarks[idx]
        
        return (lat, lon)
    
    def _interpolate(self, img_x: float, img_y: float) -> Tuple[float, float]:
        """Interpolate GPS coordinates from landmarks."""
        if len(self.landmarks) < 3:
            return self._nearest_neighbor(img_x, img_y)
        
        # Extract image and GPS points
        image_points = np.array([(x, y) for x, y, _, _ in self.landmarks])
        gps_points = np.array([(lat, lon) for _, _, lat, lon in self.landmarks])
        
        # Interpolate
        query_point = np.array([[img_x, img_y]])
        
        # Use griddata for interpolation
        lat = griddata(image_points, gps_points[:, 0], query_point, 
                      method=self.interpolation_method)[0]
        lon = griddata(image_points, gps_points[:, 1], query_point,
                      method=self.interpolation_method)[0]
        
        # Check if interpolation succeeded (not NaN)
        if np.isnan(lat) or np.isnan(lon):
            # Fall back to nearest neighbor
            return self._nearest_neighbor(img_x, img_y)
        
        return (float(lat), float(lon))
    
    def get_accuracy_estimate(self, test_points: List[Tuple[float, float, float, float]]) -> Dict:
        """
        Estimate accuracy by testing known points.
        
        Args:
            test_points: List of (img_x, img_y, expected_lat, expected_lon)
        
        Returns:
            Dictionary with accuracy metrics
        """
        errors = []
        
        for img_x, img_y, exp_lat, exp_lon in test_points:
            pred_lat, pred_lon = self.project_to_gps((img_x, img_y))
            
            if pred_lat is None or pred_lon is None:
                continue
            
            # Calculate distance error (simplified)
            lat_error = abs(pred_lat - exp_lat) * 111320  # meters per degree
            lon_error = abs(pred_lon - exp_lon) * 111320 * np.cos(np.radians(exp_lat))
            error = np.sqrt(lat_error**2 + lon_error**2)
            errors.append(error)
        
        if not errors:
            return {
                'mean_error': float('inf'),
                'max_error': float('inf'),
                'min_error': float('inf'),
                'std_error': float('inf'),
                'errors': []
            }
        
        return {
            'mean_error': float(np.mean(errors)),
            'max_error': float(np.max(errors)),
            'min_error': float(np.min(errors)),
            'std_error': float(np.std(errors)),
            'errors': [float(e) for e in errors]
        }


# ============================================================================
# 3. Multi-Camera Triangulation
# ============================================================================

class MultiCameraTriangulator:
    """
    Triangulate 3D position from multiple camera views.
    """
    
    def __init__(self):
        """Initialize triangulator."""
        self.cameras = {}  # camera_id -> camera_info
    
    def add_camera(
        self,
        camera_id: str,
        camera_matrix: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        gps_reference: Optional[dict] = None
    ):
        """
        Add a camera to the system.
        
        Args:
            camera_id: Unique camera identifier
            camera_matrix: 3x3 camera intrinsic matrix
            rvec: Rotation vector (3x1)
            tvec: Translation vector (3x1)
            gps_reference: GPS reference point
        """
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Camera center in world coordinates
        camera_center = -R.T @ tvec
        
        self.cameras[camera_id] = {
            'camera_matrix': camera_matrix,
            'R': R,
            'tvec': tvec,
            'camera_center': camera_center,
            'gps_reference': gps_reference
        }
    
    def triangulate(
        self,
        image_points: Dict[str, Tuple[float, float]]
    ) -> Optional[Tuple[float, float, float]]:
        """
        Triangulate 3D point from image points in multiple cameras.
        
        Args:
            image_points: Dict mapping camera_id -> (x, y) pixel coordinates
        
        Returns:
            (x, y, z) 3D world coordinates or None if insufficient views
        """
        if len(image_points) < 2:
            return None
        
        # Prepare projection matrices and image points
        projection_matrices = []
        image_pts = []
        
        for camera_id, img_pt in image_points.items():
            if camera_id not in self.cameras:
                continue
            
            cam_info = self.cameras[camera_id]
            K = cam_info['camera_matrix']
            R = cam_info['R']
            t = cam_info['tvec']
            
            # Projection matrix: P = K [R | t]
            P = K @ np.hstack([R, t.reshape(3, 1)])
            projection_matrices.append(P)
            image_pts.append(img_pt)
        
        if len(projection_matrices) < 2:
            return None
        
        # Convert to numpy arrays
        image_pts = np.array(image_pts, dtype=np.float32)
        
        # Triangulate using DLT (Direct Linear Transform)
        point_3d = cv2.triangulatePoints(
            projection_matrices[0],
            projection_matrices[1],
            image_pts[0],
            image_pts[1]
        )
        
        # Convert from homogeneous coordinates
        point_3d = point_3d[:3] / point_3d[3]
        
        return tuple(point_3d.flatten())
    
    def triangulate_to_gps(
        self,
        image_points: Dict[str, Tuple[float, float]]
    ) -> Optional[Tuple[float, float]]:
        """
        Triangulate and convert directly to GPS.
        
        Args:
            image_points: Dict mapping camera_id -> (x, y) pixel coordinates
        
        Returns:
            (lat, lon) GPS coordinates or None
        """
        point_3d = self.triangulate(image_points)
        if point_3d is None:
            return None
        
        # Find first camera with GPS reference
        gps_ref = None
        for camera_id in image_points.keys():
            if camera_id in self.cameras:
                gps_ref = self.cameras[camera_id].get('gps_reference')
                if gps_ref:
                    break
        
        if gps_ref is None:
            return None
        
        # Convert to GPS (using x, y from 3D point, assuming z=0 for ground)
        x_world, y_world, z_world = point_3d
        lat, lon = meters_to_gps(
            x_world, y_world,
            gps_ref['lat'],
            gps_ref['lon']
        )
        
        return (lat, lon)


# ============================================================================
# 4. Hybrid Approach
# ============================================================================

class HybridGPSEstimator:
    """
    Combines multiple GPS estimation methods for robustness.
    """
    
    def __init__(self):
        """Initialize hybrid estimator."""
        self.methods = {}
        self.method_weights = {}
    
    def add_method(self, method_name: str, method, weight: float = 1.0):
        """
        Add a GPS estimation method.
        
        Args:
            method_name: Name of the method
            method: Method object with project_to_gps() method
            weight: Weight for this method (higher = more trusted)
        """
        self.methods[method_name] = method
        self.method_weights[method_name] = weight
    
    def estimate_gps(
        self,
        image_point: Tuple[float, float],
        **kwargs
    ) -> Optional[Tuple[float, float]]:
        """
        Estimate GPS using all available methods and combine results.
        
        Args:
            image_point: (x, y) pixel coordinates
            **kwargs: Additional arguments for methods
        
        Returns:
            (lat, lon) GPS coordinates or None
        """
        results = []
        weights = []
        
        for method_name, method in self.methods.items():
            try:
                gps = method.project_to_gps(image_point, **kwargs)
                if gps:
                    results.append(gps)
                    weights.append(self.method_weights[method_name])
            except Exception as e:
                print(f"Method {method_name} failed: {e}")
                continue
        
        if not results:
            return None
        
        # Weighted average
        results = np.array(results)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        lat = np.average(results[:, 0], weights=weights)
        lon = np.average(results[:, 1], weights=weights)
        
        return (float(lat), float(lon))
    
    def estimate_with_confidence(
        self,
        image_point: Tuple[float, float],
        **kwargs
    ) -> Dict:
        """
        Estimate GPS with confidence score.
        
        Returns:
            Dictionary with 'gps', 'confidence', and 'method_results'
        """
        results = []
        weights = []
        method_results = {}
        
        for method_name, method in self.methods.items():
            try:
                gps = method.project_to_gps(image_point, **kwargs)
                if gps:
                    results.append(gps)
                    weights.append(self.method_weights[method_name])
                    method_results[method_name] = gps
                else:
                    method_results[method_name] = None
            except Exception as e:
                method_results[method_name] = None
                continue
        
        if not results:
            return {
                'gps': None,
                'confidence': 0.0,
                'method_results': method_results
            }
        
        # Calculate weighted average
        results = np.array(results)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        lat = np.average(results[:, 0], weights=weights)
        lon = np.average(results[:, 1], weights=weights)
        
        # Calculate confidence based on agreement between methods
        if len(results) > 1:
            # Standard deviation of results
            lat_std = np.std(results[:, 0])
            lon_std = np.std(results[:, 1])
            # Lower std = higher confidence
            confidence = 1.0 / (1.0 + (lat_std + lon_std) * 1000)
        else:
            confidence = 0.5  # Single method, moderate confidence
        
        return {
            'gps': (float(lat), float(lon)),
            'confidence': float(confidence),
            'method_results': method_results
        }
