"""
Deep Learning-based Bird's-Eye View (BEV) Detector

This module provides a neural network-based approach to convert image coordinates
to world/GPS coordinates, replacing or complementing traditional homography.

The BEV detector learns the transformation from camera image space to bird's-eye
view world coordinates using a deep learning model, which can handle:
- Non-linear distortions better than homography
- Occlusion and perspective variations
- Adapting to different camera angles and heights
- Learning from existing calibration data

Usage:
    # Initialize BEV detector
    detector = BEVDetector(model_path="models/bev_model.pth")
    
    # Project image point to world coordinates
    x_world, y_world = detector.predict(image, x_img, y_img)
    
    # Or use with bbox
    x_world, y_world = detector.predict_from_bbox(image, bbox)
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict
import os
import json
from pathlib import Path


class BEVDetector:
    """
    Deep learning-based Bird's-Eye View detector.
    
    Uses a CNN to predict world coordinates (x, y in meters) from image coordinates.
    Can be trained using existing homography calibration data as ground truth.
    """
    
    def __init__(self, model_path: Optional[str] = None, use_cuda: bool = True,
                 image_size: Tuple[int, int] = (1920, 1080),
                 bev_size: Tuple[float, float] = (100.0, 100.0)):  # meters
        """
        Initialize BEV detector.
        
        Args:
            model_path: Path to trained model weights (.pth file). If None, uses default model.
            use_cuda: Whether to use GPU if available
            image_size: Input image size (width, height) in pixels
            bev_size: BEV coverage size (width, height) in meters
        """
        self.model_path = model_path
        self.use_cuda = use_cuda
        self.image_size = image_size
        self.bev_size = bev_size
        self.model = None
        self.device = None
        self.initialized = False
        
        # GPS reference for converting world coords to GPS (optional)
        self.gps_reference = None  # {'lat': float, 'lon': float}
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self._load_model()
    
    def _load_model(self):
        """Load the BEV model."""
        try:
            import torch
            import torch.nn as nn
            
            self.device = torch.device('cuda' if (self.use_cuda and torch.cuda.is_available()) else 'cpu')
            
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch not available")
            
            # Define model architecture
            self.model = BEVCoordinateNet(
                image_size=self.image_size,
                bev_size=self.bev_size
            ).to(self.device)
            
            # Load weights if available
            if self.model_path and os.path.exists(self.model_path):
                print(f"[BEVDetector] Loading model from {self.model_path}")
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                    
                    # Load GPS reference if available in checkpoint
                    if 'gps_reference' in checkpoint:
                        self.gps_reference = checkpoint['gps_reference']
                else:
                    self.model.load_state_dict(checkpoint)
                
                self.model.eval()
                print(f"[BEVDetector] Model loaded successfully on {self.device}")
            else:
                print(f"[BEVDetector] Warning: Model file not found at {self.model_path}, using untrained model")
            
            self.initialized = True
            
        except ImportError:
            raise ImportError(
                "PyTorch not installed. Install with: pip install torch torchvision"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load BEV model: {e}")
    
    def initialize(self):
        """Initialize the model (load if not already loaded)."""
        if not self.initialized:
            self._load_model()
    
    def set_gps_reference(self, lat: float, lon: float):
        """
        Set GPS reference point for converting world coordinates to GPS.
        
        Args:
            lat: Reference latitude (degrees)
            lon: Reference longitude (degrees)
        """
        self.gps_reference = {'lat': lat, 'lon': lon}
        print(f"[BEVDetector] GPS reference set: ({lat:.6f}, {lon:.6f})")
    
    def predict(self, image: np.ndarray, x_img: float, y_img: float,
                return_gps: bool = False) -> Tuple[Optional[float], Optional[float]]:
        """
        Predict world coordinates from image coordinates.
        
        Args:
            image: Input image (BGR format, numpy array)
            x_img: Image X coordinate (pixels)
            y_img: Image Y coordinate (pixels)
            return_gps: If True and GPS reference set, return (lat, lon) instead of (x, y) meters
        
        Returns:
            Tuple of (x_world, y_world) in meters, or (lat, lon) if return_gps=True
            Returns (None, None) if prediction fails
        """
        if not self.initialized:
            self.initialize()
            if not self.initialized:
                return (None, None)
        
        try:
            import torch
            
            # Normalize image coordinates to [0, 1]
            x_norm = x_img / self.image_size[0]
            y_norm = y_img / self.image_size[1]
            
            # Prepare input tensor: [batch, channels, height, width]
            # Use image patch around the point for context
            patch_size = 128
            h, w = image.shape[:2]
            
            # Extract patch centered at (x_img, y_img)
            x1 = max(0, int(x_img - patch_size // 2))
            y1 = max(0, int(y_img - patch_size // 2))
            x2 = min(w, x1 + patch_size)
            y2 = min(h, y1 + patch_size)
            
            # Pad if necessary
            patch = image[y1:y2, x1:x2].copy()
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                pad_h = patch_size - patch.shape[0]
                pad_w = patch_size - patch.shape[1]
                patch = cv2.copyMakeBorder(patch, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
            
            # Resize to standard size
            patch = cv2.resize(patch, (patch_size, patch_size))
            
            # Convert BGR to RGB and normalize
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            patch_tensor = torch.from_numpy(patch_rgb).float().permute(2, 0, 1) / 255.0
            patch_tensor = patch_tensor.unsqueeze(0).to(self.device)
            
            # Normalized coordinates as additional input
            coord_tensor = torch.tensor([[x_norm, y_norm]], dtype=torch.float32).to(self.device)
            
            # Predict
            with torch.no_grad():
                output = self.model(patch_tensor, coord_tensor)
                x_world = output[0, 0].item()
                y_world = output[0, 1].item()
            
            # Convert to GPS if requested
            if return_gps and self.gps_reference:
                from app.gps_utils import meters_to_gps
                lat, lon = meters_to_gps(
                    x_world, y_world,
                    self.gps_reference['lat'],
                    self.gps_reference['lon']
                )
                return (lat, lon)
            
            return (x_world, y_world)
            
        except Exception as e:
            print(f"[BEVDetector] Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return (None, None)
    
    def predict_from_bbox(self, image: np.ndarray, bbox: List[float],
                         return_gps: bool = False) -> Tuple[Optional[float], Optional[float]]:
        """
        Predict world coordinates from bounding box.
        
        Args:
            image: Input image (BGR format)
            bbox: Bounding box [x1, y1, x2, y2]
            return_gps: If True and GPS reference set, return (lat, lon) instead of (x, y) meters
        
        Returns:
            Tuple of (x_world, y_world) in meters, or (lat, lon) if return_gps=True
        """
        # Calculate bottom-center point of bbox (ground contact point)
        x1, y1, x2, y2 = bbox
        x_center = (x1 + x2) / 2.0
        y_bottom = y2  # Use bottom of bbox for ground plane
        
        return self.predict(image, x_center, y_bottom, return_gps=return_gps)
    
    def predict_batch(self, image: np.ndarray, points: List[Tuple[float, float]],
                     return_gps: bool = False) -> List[Tuple[Optional[float], Optional[float]]]:
        """
        Predict world coordinates for multiple image points.
        
        Args:
            image: Input image (BGR format)
            points: List of (x_img, y_img) tuples
            return_gps: If True and GPS reference set, return GPS coordinates
        
        Returns:
            List of (x_world, y_world) or (lat, lon) tuples
        """
        results = []
        for x_img, y_img in points:
            result = self.predict(image, x_img, y_img, return_gps=return_gps)
            results.append(result)
        return results


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


if TORCH_AVAILABLE:
    class BEVCoordinateNet(nn.Module):
        """
        Neural network for predicting world coordinates from image coordinates.
        
        Architecture:
        - CNN encoder extracts features from image patch
        - Coordinate encoder processes normalized image coordinates
        - Fusion layer combines image and coordinate features
        - Regression head predicts (x_world, y_world) in meters
        """
        
        def __init__(self, image_size: Tuple[int, int] = (1920, 1080),
                     bev_size: Tuple[float, float] = (100.0, 100.0)):
            """
            Initialize BEV coordinate network.
            
            Args:
                image_size: Input image size (width, height)
                bev_size: BEV coverage size (width, height) in meters
            """
            super().__init__()
            
            self.image_size = image_size
            self.bev_size = bev_size
        
        # Image encoder (CNN backbone)
        self.image_encoder = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64x64
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))  # 4x4 feature map
        )
        
            # Coordinate encoder (MLP)
            self.coord_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128)
        )
        
            # Fusion and regression
            image_feat_dim = 256 * 4 * 4  # 4096
            coord_feat_dim = 128
            fusion_dim = 512
            
            self.fusion = nn.Sequential(
            nn.Linear(image_feat_dim + coord_feat_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Regression head: predict (x_world, y_world) in meters
        self.regression_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)  # Output: (x_world, y_world)
        )
        
        def forward(self, image_patch, normalized_coords):
            """
            Forward pass.
            
            Args:
                image_patch: Image patch tensor [B, 3, H, W]
                normalized_coords: Normalized image coordinates [B, 2] in range [0, 1]
            
            Returns:
                World coordinates [B, 2] in meters (x_world, y_world)
            """
            # Extract image features
            img_features = self.image_encoder(image_patch)
            img_features = img_features.view(img_features.size(0), -1)  # Flatten
            
            # Encode coordinates
            coord_features = self.coord_encoder(normalized_coords)
            
            # Fuse features
            fused = torch.cat([img_features, coord_features], dim=1)
            fused = self.fusion(fused)
            
            # Predict world coordinates
            world_coords = self.regression_head(fused)
            
            return world_coords
else:
    # Dummy class if PyTorch not available
    class BEVCoordinateNet:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch not installed. Install with: pip install torch torchvision")


def create_bev_detector(config: Dict) -> Optional[BEVDetector]:
    """
    Factory function to create BEV detector from config.
    
    Args:
        config: Configuration dict with keys:
            - 'enabled': bool - Whether to enable BEV
            - 'model_path': str - Path to model weights (optional)
            - 'use_cuda': bool - Whether to use GPU (optional)
            - 'image_size': Tuple[int, int] - Image size (optional)
            - 'bev_size': Tuple[float, float] - BEV size in meters (optional)
            - 'gps_reference': Dict with 'lat' and 'lon' (optional)
    
    Returns:
        BEVDetector instance or None if disabled
    """
    if not config.get('enabled', False):
        return None
    
    model_path = config.get('model_path', None)
    use_cuda = config.get('use_cuda', True)
    image_size = tuple(config.get('image_size', [1920, 1080]))
    bev_size = tuple(config.get('bev_size', [100.0, 100.0]))
    
    detector = BEVDetector(
        model_path=model_path,
        use_cuda=use_cuda,
        image_size=image_size,
        bev_size=bev_size
    )
    
    # Set GPS reference if provided
    gps_ref = config.get('gps_reference', None)
    if gps_ref:
        detector.set_gps_reference(gps_ref['lat'], gps_ref['lon'])
    
    return detector
