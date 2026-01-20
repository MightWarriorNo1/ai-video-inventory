"""
BEV Utilities - Integration with existing system

This module provides utilities for integrating deep learning-based BEV transformation
with the existing homography-based system. It offers a drop-in replacement interface
that's compatible with the current API.
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Dict
from pathlib import Path
import json

# Try to import BEV transformer (requires PyTorch)
try:
    from app.ai.bev_transformer import BEVTransformer
    BEV_AVAILABLE = True
except ImportError:
    BEV_AVAILABLE = False
    BEVTransformer = None


class BEVProjector:
    """
    BEV-based coordinate projector (drop-in replacement for homography).
    
    This class provides the same interface as the homography-based projection
    but uses deep learning-based BEV transformation instead.
    """
    
    def __init__(
        self,
        model_path: str,
        gps_reference: Optional[Dict[str, float]] = None,
        input_size: Tuple[int, int] = (720, 1280),
        device: Optional[str] = None
    ):
        """
        Initialize BEV projector.
        
        Args:
            model_path: Path to trained BEV model checkpoint
            gps_reference: GPS reference point dict with 'lat' and 'lon' keys
            input_size: Input image size (height, width)
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        if not BEV_AVAILABLE:
            raise ImportError(
                "BEV transformer requires PyTorch. Install with: "
                "pip install torch torchvision"
            )
        
        self.gps_reference = gps_reference
        self.transformer = BEVTransformer(
            model_path=model_path,
            device=device,
            input_size=input_size
        )
    
    def project_to_world(
        self,
        image: np.ndarray,
        x_img: float,
        y_img: float,
        return_gps: bool = False
    ) -> Tuple[float, float]:
        """
        Project image point to world coordinates.
        
        Args:
            image: Input image (BGR format)
            x_img: Image X coordinate
            y_img: Image Y coordinate
            return_gps: If True, return GPS coordinates; if False, return world meters
        
        Returns:
            Tuple of (lat, lon) if return_gps=True, else (x_world, y_world)
        """
        # Transform using BEV network
        x_world, y_world = self.transformer.transform_point(image, x_img, y_img)
        
        if return_gps and self.gps_reference is not None:
            from app.gps_utils import meters_to_gps
            lat, lon = meters_to_gps(
                x_world, y_world,
                self.gps_reference['lat'],
                self.gps_reference['lon']
            )
            return (lat, lon)
        
        return (x_world, y_world)
    
    def project_bbox(
        self,
        image: np.ndarray,
        bbox: list,
        bbox_to_image_fn=None
    ) -> Tuple[float, float]:
        """
        Project bounding box to world/GPS coordinates.
        
        Args:
            image: Input image (BGR format)
            bbox: Bounding box as [x1, y1, x2, y2]
            bbox_to_image_fn: Function to convert bbox to image coordinates
                            (default: uses bottom-center)
        
        Returns:
            Tuple of (x_world, y_world) or (lat, lon) depending on configuration
        """
        # Convert bbox to image coordinates
        if bbox_to_image_fn is not None:
            x_img, y_img = bbox_to_image_fn(bbox)
        else:
            # Default: bottom-center
            x1, y1, x2, y2 = bbox
            x_img = (x1 + x2) / 2.0
            y_img = float(y2)
        
        # Project to world
        return_gps = self.gps_reference is not None
        return self.project_to_world(image, x_img, y_img, return_gps=return_gps)


class HybridProjector:
    """
    Hybrid projector that can use either BEV or homography.
    
    Useful for gradual migration or fallback scenarios.
    """
    
    def __init__(
        self,
        method: str = 'bev',  # 'bev', 'homography', or 'hybrid'
        bev_model_path: Optional[str] = None,
        homography_matrix: Optional[np.ndarray] = None,
        gps_reference: Optional[Dict[str, float]] = None,
        input_size: Tuple[int, int] = (720, 1280)
    ):
        """
        Initialize hybrid projector.
        
        Args:
            method: Projection method ('bev', 'homography', or 'hybrid')
            bev_model_path: Path to BEV model (required if method includes 'bev')
            homography_matrix: Homography matrix (required if method includes 'homography')
            gps_reference: GPS reference point
            input_size: Input image size (height, width)
        """
        self.method = method
        self.gps_reference = gps_reference
        
        # Initialize BEV projector if needed
        if method in ('bev', 'hybrid'):
            if bev_model_path is None:
                raise ValueError("bev_model_path required when method includes 'bev'")
            self.bev_projector = BEVProjector(
                model_path=bev_model_path,
                gps_reference=gps_reference,
                input_size=input_size
            )
        else:
            self.bev_projector = None
        
        # Initialize homography if needed
        if method in ('homography', 'hybrid'):
            if homography_matrix is None:
                raise ValueError("homography_matrix required when method includes 'homography'")
            self.H = np.array(homography_matrix)
        else:
            self.H = None
    
    def project_to_world(
        self,
        image: np.ndarray,
        x_img: float,
        y_img: float,
        return_gps: bool = False
    ) -> Tuple[float, float]:
        """
        Project image point to world coordinates.
        
        Uses BEV, homography, or both depending on configuration.
        """
        if self.method == 'bev':
            return self.bev_projector.project_to_world(
                image, x_img, y_img, return_gps=return_gps
            )
        
        elif self.method == 'homography':
            # Traditional homography projection
            point_img = np.array([[x_img, y_img]], dtype=np.float32)
            point_img = np.array([point_img])
            point_world = cv2.perspectiveTransform(point_img, self.H)
            x_world, y_world = point_world[0][0]
            
            if return_gps and self.gps_reference:
                from app.gps_utils import meters_to_gps
                lat, lon = meters_to_gps(
                    x_world, y_world,
                    self.gps_reference['lat'],
                    self.gps_reference['lon']
                )
                return (lat, lon)
            
            return (x_world, y_world)
        
        elif self.method == 'hybrid':
            # Use both and average (or use BEV with homography fallback)
            try:
                # Try BEV first
                bev_result = self.bev_projector.project_to_world(
                    image, x_img, y_img, return_gps=return_gps
                )
                
                # Also get homography result
                point_img = np.array([[x_img, y_img]], dtype=np.float32)
                point_img = np.array([point_img])
                point_world = cv2.perspectiveTransform(point_img, self.H)
                h_x_world, h_y_world = point_world[0][0]
                
                if return_gps and self.gps_reference:
                    from app.gps_utils import meters_to_gps
                    h_lat, h_lon = meters_to_gps(
                        h_x_world, h_y_world,
                        self.gps_reference['lat'],
                        self.gps_reference['lon']
                    )
                    h_result = (h_lat, h_lon)
                else:
                    h_result = (h_x_world, h_y_world)
                
                # Average the results (weighted: 0.7 BEV, 0.3 homography)
                if return_gps:
                    lat = bev_result[0] * 0.7 + h_result[0] * 0.3
                    lon = bev_result[1] * 0.7 + h_result[1] * 0.3
                    return (lat, lon)
                else:
                    x = bev_result[0] * 0.7 + h_result[0] * 0.3
                    y = bev_result[1] * 0.7 + h_result[1] * 0.3
                    return (x, y)
            
            except Exception as e:
                # Fallback to homography if BEV fails
                print(f"BEV projection failed, using homography: {e}")
                point_img = np.array([[x_img, y_img]], dtype=np.float32)
                point_img = np.array([point_img])
                point_world = cv2.perspectiveTransform(point_img, self.H)
                x_world, y_world = point_world[0][0]
                
                if return_gps and self.gps_reference:
                    from app.gps_utils import meters_to_gps
                    lat, lon = meters_to_gps(
                        x_world, y_world,
                        self.gps_reference['lat'],
                        self.gps_reference['lon']
                    )
                    return (lat, lon)
                
                return (x_world, y_world)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")


def load_bev_projector_from_config(
    config_path: str,
    camera_id: Optional[str] = None
) -> Optional[BEVProjector]:
    """
    Load BEV projector from configuration file.
    
    Args:
        config_path: Path to camera configuration JSON
        camera_id: Camera ID (if config contains multiple cameras)
    
    Returns:
        BEVProjector instance or None if not configured
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        return None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Handle camera-specific config
    if camera_id and 'cameras' in config:
        cameras = config['cameras']
        if camera_id in cameras:
            cam_config = cameras[camera_id]
        else:
            return None
    else:
        cam_config = config
    
    # Check if BEV is configured
    if 'bev_model' not in cam_config:
        return None
    
    bev_model_path = Path(cam_config['bev_model'])
    if not bev_model_path.is_absolute():
        bev_model_path = config_path.parent / bev_model_path
    
    if not bev_model_path.exists():
        print(f"Warning: BEV model not found: {bev_model_path}")
        return None
    
    gps_reference = cam_config.get('gps_reference')
    input_size = cam_config.get('input_size', [720, 1280])
    
    return BEVProjector(
        model_path=str(bev_model_path),
        gps_reference=gps_reference,
        input_size=tuple(input_size)
    )


def visualize_bev_transform(
    image: np.ndarray,
    transformer: BEVTransformer,
    points: Optional[list] = None,
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualize BEV transformation by creating a BEV map overlay.
    
    Args:
        image: Input image
        transformer: BEVTransformer instance
        points: Optional list of (x_img, y_img) points to highlight
        output_path: Optional path to save visualization
    
    Returns:
        Visualization image
    """
    # Get BEV map
    bev_map = transformer.transform_image_to_bev(image)
    
    # Normalize BEV map for visualization
    # bev_map is (2, H, W) with (x, y) coordinates
    bev_x = bev_map[0]
    bev_y = bev_map[1]
    
    # Create visualization
    vis_image = image.copy()
    
    # Overlay points if provided
    if points:
        for x_img, y_img in points:
            x_world, y_world = transformer.transform_point(image, x_img, y_img)
            cv2.circle(vis_image, (int(x_img), int(y_img)), 5, (0, 255, 0), -1)
            cv2.putText(
                vis_image,
                f"({x_world:.1f}, {y_world:.1f})",
                (int(x_img) + 10, int(y_img) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
    
    if output_path:
        cv2.imwrite(output_path, vis_image)
    
    return vis_image







