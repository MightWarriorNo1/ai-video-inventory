"""
3D Camera Projection Module

This module provides 3D projection capabilities for converting image pixel coordinates
to 3D world coordinates using camera intrinsic and extrinsic parameters.

Unlike homography (which assumes all points lie on a plane), 3D projection can handle
points at different elevations, making it ideal for projecting elevated objects like
trailer backsides to their ground contact points.
"""

import numpy as np
import cv2
import json
from typing import Tuple, Optional, Dict, List
from pathlib import Path


class Camera3DProjector:
    """
    3D Camera Projector for accurate world coordinate projection.
    
    Handles projection from image pixels to 3D world coordinates using:
    - Camera intrinsic parameters (focal length, principal point, distortion)
    - Camera extrinsic parameters (rotation, translation)
    """
    
    def __init__(
        self,
        intrinsic_matrix: np.ndarray,
        rotation_matrix: np.ndarray,
        translation_vector: np.ndarray,
        distortion_coeffs: Optional[np.ndarray] = None,
        gps_reference: Optional[Dict[str, float]] = None
    ):
        """
        Initialize 3D projector with camera parameters.
        
        Args:
            intrinsic_matrix: 3×3 camera intrinsic matrix K
            rotation_matrix: 3×3 rotation matrix R (world to camera)
            translation_vector: 3×1 translation vector t (camera position in world)
            distortion_coeffs: Optional distortion coefficients (for undistortion)
            gps_reference: Optional GPS reference point dict with 'lat' and 'lon'
        """
        self.K = np.array(intrinsic_matrix, dtype=np.float32)
        self.R = np.array(rotation_matrix, dtype=np.float32)
        self.t = np.array(translation_vector, dtype=np.float32).reshape(3, 1)
        self.dist_coeffs = np.array(distortion_coeffs, dtype=np.float32) if distortion_coeffs is not None else None
        self.gps_reference = gps_reference
        
        # Extract intrinsic parameters for convenience
        self.fx = float(self.K[0, 0])
        self.fy = float(self.K[1, 1])
        self.cx = float(self.K[0, 2])
        self.cy = float(self.K[1, 2])
        
        # Camera position in world coordinates
        self.camera_pos_world = -self.R.T @ self.t
        
        # Store calibration image size for validation
        self.calibration_image_size = None
    
    @classmethod
    def from_calibration_file(cls, calib_path: str) -> 'Camera3DProjector':
        """
        Load camera projector from calibration JSON file.
        
        Args:
            calib_path: Path to calibration JSON file
            
        Returns:
            Camera3DProjector instance
        """
        calib_path = Path(calib_path)
        if not calib_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {calib_path}")
        
        with open(calib_path, 'r') as f:
            calib_data = json.load(f)
        
        # Extract matrices
        K = np.array(calib_data['intrinsic_matrix'], dtype=np.float32)
        R = np.array(calib_data['rotation_matrix'], dtype=np.float32)
        t = np.array(calib_data['translation_vector'], dtype=np.float32)
        dist_coeffs = None
        if 'distortion_coefficients' in calib_data:
            dist_coeffs = np.array(calib_data['distortion_coefficients'], dtype=np.float32)
        
        gps_reference = calib_data.get('gps_reference')
        
        projector = cls(K, R, t, dist_coeffs, gps_reference)
        
        # Store calibration image size if available
        if 'image_size' in calib_data:
            img_size = calib_data['image_size']
            projector.calibration_image_size = (img_size.get('width'), img_size.get('height'))
        
        return projector
    
    def undistort_point(self, u: float, v: float) -> Tuple[float, float]:
        """
        Undistort a pixel coordinate if distortion coefficients are available.
        
        Args:
            u, v: Distorted pixel coordinates
            
        Returns:
            Undistorted pixel coordinates (u_undist, v_undist)
        """
        if self.dist_coeffs is None:
            return (u, v)
        
        point = np.array([[[u, v]]], dtype=np.float32)
        undistorted = cv2.undistortPoints(point, self.K, self.dist_coeffs, P=self.K)
        return (float(undistorted[0, 0, 0]), float(undistorted[0, 0, 1]))
    
    def pixel_to_world_plane(
        self,
        u: float,
        v: float,
        plane_z: float = 0.0
    ) -> Tuple[float, float, float]:
        """
        Project pixel to world coordinates assuming it lies on a plane at Z=plane_z.
        
        This is useful for projecting points on known planes (ground, trailer backs, etc.).
        
        Args:
            u, v: Pixel coordinates
            plane_z: Z coordinate of the plane in world space (default: 0.0 for ground)
            
        Returns:
            (X_world, Y_world, Z_world) - 3D world coordinates
        """
        # Undistort if needed
        u_undist, v_undist = self.undistort_point(u, v)
        
        # Step 1: Normalize pixel coordinates (convert to camera coordinates)
        x_norm = (u_undist - self.cx) / self.fx
        y_norm = (v_undist - self.cy) / self.fy
        
        # Step 2: Create ray direction in camera coordinates
        ray_dir_cam = np.array([x_norm, y_norm, 1.0])
        
        # Step 3: Transform ray direction to world coordinates
        ray_dir_world = self.R.T @ ray_dir_cam
        
        # Step 4: Find intersection with plane Z = plane_z
        # Ray equation: P = camera_pos_world + lambda * ray_dir_world
        # We want Z = plane_z
        # camera_pos_world[2] + lambda * ray_dir_world[2] = plane_z
        
        if abs(ray_dir_world[2]) < 1e-6:
            # Ray is parallel to the plane, cannot intersect
            raise ValueError(f"Ray is parallel to plane Z={plane_z}")
        
        lambda_val = (plane_z - self.camera_pos_world[2, 0]) / ray_dir_world[2]
        
        # Calculate 3D point
        point_3d = self.camera_pos_world.flatten() + lambda_val * ray_dir_world
        
        return (float(point_3d[0]), float(point_3d[1]), float(point_3d[2]))
    
    def bbox_to_ground_coords(
        self,
        bbox: List[float],
        method: str = "bottom_center",
        trailer_height: Optional[float] = None,
        current_image_size: Optional[Tuple[int, int]] = None,
        depth_map: Optional[np.ndarray] = None,
        depth_estimator: Optional[object] = None
    ) -> Tuple[float, float]:
        """
        Convert trailer bounding box to ground contact point in world coordinates.
        
        Args:
            bbox: Bounding box as [x1, y1, x2, y2]
            method: Method to extract point from bbox:
                - "bottom_center": Use center X, bottom Y (assumes ground contact)
                - "backside_projection": Project elevated backside point down to ground
                - "depth_based": Use monocular depth estimation (requires depth_map or depth_estimator)
            trailer_height: Height of trailer back in meters (required for backside_projection)
            current_image_size: Optional (width, height) of current image. If provided and differs
                              from calibration size, coordinates will be scaled.
            depth_map: Pre-computed depth map (optional, for depth_based method)
            depth_estimator: DepthEstimator instance (optional, for depth_based method if depth_map not provided)
            
        Returns:
            (X_ground, Y_ground) - Ground contact point in world coordinates
        """
        x1, y1, x2, y2 = bbox
        
        # Scale coordinates if image size differs from calibration
        if current_image_size is not None and self.calibration_image_size is not None:
            curr_w, curr_h = current_image_size
            calib_w, calib_h = self.calibration_image_size
            
            if curr_w != calib_w or curr_h != calib_h:
                scale_x = calib_w / curr_w
                scale_y = calib_h / curr_h
                x1 = x1 * scale_x
                y1 = y1 * scale_y
                x2 = x2 * scale_x
                y2 = y2 * scale_y
        
        if method == "bottom_center":
            # Simple method: assume bottom-center is on ground
            center_x = (x1 + x2) / 2.0
            bottom_y = float(y2)
            
            ground_3d = self.pixel_to_world_plane(center_x, bottom_y, plane_z=0.0)
            return (ground_3d[0], ground_3d[1])
        
        elif method == "backside_projection":
            # Advanced method: account for trailer backside elevation
            # This correctly projects the elevated backside center down to ground
            if trailer_height is None:
                trailer_height = 2.6  # Default trailer back height in meters
            
            # Use center of bbox (which is on the backside at elevation)
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            
            # Step 1: Project the center pixel to world coordinates at trailer_height
            # This gives us the 3D point on the trailer backside
            back_3d = self.pixel_to_world_plane(center_x, center_y, plane_z=trailer_height)
            
            # Step 2: The ground contact point is directly below this elevated point
            # In world coordinates, this is the same (X, Y) but at Z=0
            # This is correct because we want the point on the ground directly beneath
            # the center of the trailer backside
            ground_x = back_3d[0]
            ground_y = back_3d[1]
            ground_z = 0.0
            
            return (ground_x, ground_y)
        
        elif method == "depth_based":
            # Monocular depth estimation method (Option A from guide)
            # This uses depth estimation to get distance d, then unprojects using K^-1
            return self._bbox_to_ground_coords_depth_based(
                bbox, depth_map, depth_estimator, current_image_size
            )
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _bbox_to_ground_coords_depth_based(
        self,
        bbox: List[float],
        depth_map: Optional[np.ndarray],
        depth_estimator: Optional[object],
        current_image_size: Optional[Tuple[int, int]]
    ) -> Tuple[float, float]:
        """
        Convert bbox to ground coords using monocular depth estimation.
        
        Implements: [x, y, z] = K^-1 [u, v, 1] · d
        Then refines by intersecting with ground plane if needed.
        """
        from app.depth_estimation import DepthEstimator
        
        # Get depth map if not provided
        if depth_map is None:
            if depth_estimator is None:
                raise ValueError("depth_based method requires either depth_map or depth_estimator with frame")
            # Note: frame should be passed separately, this is a limitation
            # In practice, depth_map should be pre-computed per frame
            raise ValueError("depth_map must be provided for depth_based method")
        
        x1, y1, x2, y2 = bbox
        
        # Scale coordinates if image size differs from calibration
        if current_image_size is not None and self.calibration_image_size is not None:
            curr_w, curr_h = current_image_size
            calib_w, calib_h = self.calibration_image_size
            
            if curr_w != calib_w or curr_h != calib_h:
                scale_x = calib_w / curr_w
                scale_y = calib_h / curr_h
                x1 = x1 * scale_x
                y1 = y1 * scale_y
                x2 = x2 * scale_x
                y2 = y2 * scale_y
        
        # Step 1: Extract depth value from bbox
        # Use median depth within bbox, or bottom center for ground contact approximation
        depth = None
        if depth_estimator is not None:
            try:
                # Use depth estimator's method
                depth = depth_estimator.get_depth_in_bbox(depth_map, [x1, y1, x2, y2], method="median")
            except Exception:
                # Fallback to manual extraction if estimator method fails
                pass
        
        if depth is None:
            # Manual extraction
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
        
        # Step 2: Get bounding box center pixel (u, v)
        center_u = (x1 + x2) / 2.0
        center_v = (y1 + y2) / 2.0
        
        # For ground contact, use bottom center instead
        # This approximates where the trailer touches the ground
        bottom_center_u = (x1 + x2) / 2.0
        bottom_center_v = float(y2)
        
        # Step 3: Unproject using K^-1: [x, y, z] = K^-1 [u, v, 1] · d
        # Undistort if needed
        u_undist, v_undist = self.undistort_point(bottom_center_u, bottom_center_v)
        
        # Convert to normalized camera coordinates
        # [x_norm, y_norm, 1] = K^-1 [u, v, 1]
        u_homogeneous = np.array([[u_undist], [v_undist], [1.0]], dtype=np.float32)
        K_inv = np.linalg.inv(self.K)
        ray_normalized = K_inv @ u_homogeneous
        
        # Scale by depth: [x, y, z] = ray_normalized · d
        point_3d_cam = ray_normalized * depth
        
        # Step 4: Transform from camera coordinates to world coordinates
        # P_world = R^T @ P_cam + camera_pos_world
        point_3d_cam_vec = point_3d_cam.flatten()[:3]
        point_3d_world = self.R.T @ point_3d_cam_vec + self.camera_pos_world.flatten()
        
        # Step 5: Refine by intersecting with ground plane (z=0)
        # If depth model doesn't align perfectly, intersect ray with ground plane
        # The trailer is assumed to be on the ground plane (z=0 in world coordinates)
        # Ray: P = camera_pos_world + lambda * ray_dir_world
        # We want Z = 0
        
        # Get ray direction in world coordinates (normalized direction)
        ray_dir_cam = ray_normalized.flatten()[:3]
        ray_dir_world = self.R.T @ ray_dir_cam
        
        # Intersect with ground plane z=0
        if abs(ray_dir_world[2]) > 1e-6:
            # Ray equation: P = camera_pos_world + lambda * ray_dir_world
            # We want Z = 0
            lambda_val = (0.0 - self.camera_pos_world[2, 0]) / ray_dir_world[2]
            point_ground = self.camera_pos_world.flatten() + lambda_val * ray_dir_world
            return (float(point_ground[0]), float(point_ground[1]))
        else:
            # Ray parallel to ground, use original projection
            return (float(point_3d_world[0]), float(point_3d_world[1]))
    
    def pixel_to_world(
        self,
        u: float,
        v: float,
        plane_z: float = 0.0,
        return_gps: bool = False
    ) -> Tuple[float, float]:
        """
        Project pixel to world coordinates (2D output, ground plane projection).
        
        Convenience method that projects to ground plane and optionally converts to GPS.
        
        Args:
            u, v: Pixel coordinates
            plane_z: Z coordinate of projection plane (default: 0.0 for ground)
            return_gps: If True, return GPS coordinates instead of world meters
            
        Returns:
            (X_world, Y_world) or (lat, lon) depending on return_gps
        """
        world_3d = self.pixel_to_world_plane(u, v, plane_z=plane_z)
        x_world, y_world = world_3d[0], world_3d[1]
        
        if return_gps and self.gps_reference:
            from app.gps_utils import meters_to_gps
            lat, lon = meters_to_gps(
                x_world, y_world,
                self.gps_reference['lat'],
                self.gps_reference['lon']
            )
            return (lat, lon)
        
        return (x_world, y_world)
    
    def world_to_pixel(
        self,
        X_world: float,
        Y_world: float,
        Z_world: float = 0.0
    ) -> Tuple[float, float]:
        """
        Project 3D world point to image pixel coordinates (inverse projection).
        
        Args:
            X_world, Y_world, Z_world: World coordinates
            
        Returns:
            (u, v) - Pixel coordinates
        """
        # Transform world point to camera coordinates
        P_world = np.array([[X_world], [Y_world], [Z_world], [1.0]])
        P_cam_homogeneous = np.vstack([self.R, np.zeros((1, 3))]) @ P_world[:3] + np.vstack([self.t, np.array([[1.0]])])
        P_cam = P_cam_homogeneous[:3]
        
        # Project to image plane
        if P_cam[2] <= 0:
            # Point is behind camera
            return (-1, -1)
        
        x_cam_norm = P_cam[0] / P_cam[2]
        y_cam_norm = P_cam[1] / P_cam[2]
        
        u = self.fx * x_cam_norm + self.cx
        v = self.fy * y_cam_norm + self.cy
        
        return (float(u), float(v))
    
    def save_calibration(self, output_path: str, gps_reference: Optional[Dict[str, float]] = None):
        """
        Save calibration parameters to JSON file.
        
        Args:
            output_path: Path to save calibration file
            gps_reference: Optional GPS reference to include
        """
        calib_data = {
            'intrinsic_matrix': self.K.tolist(),
            'rotation_matrix': self.R.tolist(),
            'translation_vector': self.t.flatten().tolist(),
        }
        
        if self.dist_coeffs is not None:
            calib_data['distortion_coefficients'] = self.dist_coeffs.flatten().tolist()
        
        if gps_reference:
            calib_data['gps_reference'] = gps_reference
        elif self.gps_reference:
            calib_data['gps_reference'] = self.gps_reference
        
        with open(output_path, 'w') as f:
            json.dump(calib_data, f, indent=2)
        
        print(f"Calibration saved to: {output_path}")
