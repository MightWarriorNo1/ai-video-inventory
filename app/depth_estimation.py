"""
Monocular Depth Estimation Module

This module provides depth estimation using pre-trained models:
- MiDaS (via torch.hub)
- Depth Anything (via Hugging Face)

The depth maps are used to improve 3D position accuracy for trailer detection.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List
import warnings

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Depth estimation will be disabled.")


class DepthEstimator:
    """
    Depth estimation using pre-trained models.
    """
    
    def __init__(self, model_type: str = "midas", device: Optional[str] = None):
        """
        Initialize depth estimator.
        
        Args:
            model_type: "midas" or "depth_anything"
            device: "cuda" or "cpu" (auto-detect if None)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for depth estimation. Install with: pip install torch torchvision")
        
        self.model_type = model_type.lower()
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the depth estimation model."""
        if self.model_type == "midas":
            self._initialize_midas()
        elif self.model_type == "depth_anything":
            self._initialize_depth_anything()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}. Use 'midas' or 'depth_anything'")
    
    def _initialize_midas(self):
        """Initialize MiDaS model from torch.hub."""
        print(f"[DepthEstimator] Loading MiDaS model on {self.device}...")
        try:
            # Load MiDaS model (DPT_Large for best accuracy)
            self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
            self.model.to(self.device)
            self.model.eval()
            
            # Load MiDaS transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.transform = midas_transforms.dpt_transform
            print(f"[DepthEstimator] MiDaS model loaded successfully")
        except Exception as e:
            print(f"[DepthEstimator] Error loading MiDaS: {e}")
            raise
    
    def _initialize_depth_anything(self):
        """Initialize Depth Anything model from Hugging Face."""
        print(f"[DepthEstimator] Loading Depth Anything model on {self.device}...")
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            
            model_name = "depth-anything/Depth-Anything-V2-Small-hf"  # Smaller, faster
            # Alternative: "depth-anything/Depth-Anything-V2-Base-hf" for better accuracy
            
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"[DepthEstimator] Depth Anything model loaded successfully")
        except Exception as e:
            print(f"[DepthEstimator] Error loading Depth Anything: {e}")
            raise
    
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth map for entire frame.
        
        Args:
            frame: Input frame (BGR format, numpy array)
            
        Returns:
            Depth map (same size as input, float32, in meters or relative units)
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        if self.model_type == "midas":
            return self._estimate_midas(frame)
        elif self.model_type == "depth_anything":
            return self._estimate_depth_anything(frame)
    
    def _estimate_midas(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth using MiDaS."""
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transform
        input_batch = self.transform(img_rgb).to(self.device)
        
        # Predict depth
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # Convert to numpy and normalize
        depth = prediction.cpu().numpy()
        
        # MiDaS outputs inverse depth (larger = closer), convert to depth
        # Normalize to reasonable range (0-100 meters)
        depth_normalized = depth / depth.max() * 100.0
        
        return depth_normalized.astype(np.float32)
    
    def _estimate_depth_anything(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth using Depth Anything."""
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Prepare inputs
        inputs = self.processor(images=img_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict depth
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        # Convert to numpy
        depth = prediction.cpu().numpy()
        
        # Depth Anything outputs depth in meters (approximately)
        # Normalize if needed
        depth_normalized = depth * 0.1  # Scale factor may need adjustment
        
        return depth_normalized.astype(np.float32)
    
    def get_depth_at_point(self, depth_map: np.ndarray, u: float, v: float) -> float:
        """
        Get depth value at a specific pixel coordinate.
        
        Args:
            depth_map: Depth map from estimate_depth()
            u, v: Pixel coordinates
            
        Returns:
            Depth value at (u, v)
        """
        h, w = depth_map.shape
        u_int = int(np.clip(u, 0, w - 1))
        v_int = int(np.clip(v, 0, h - 1))
        return float(depth_map[v_int, u_int])
    
    def get_depth_in_bbox(
        self, 
        depth_map: np.ndarray, 
        bbox: List[float],
        method: str = "median"
    ) -> float:
        """
        Get depth value for a bounding box.
        
        Args:
            depth_map: Depth map from estimate_depth()
            bbox: Bounding box as [x1, y1, x2, y2]
            method: "median", "mean", or "bottom_center"
            
        Returns:
            Depth value (distance in meters)
        """
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        h, w = depth_map.shape
        
        # Clip to image bounds
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        
        if method == "bottom_center":
            # Extract depth at bottom center (approximates ground contact)
            center_x = (x1 + x2) // 2
            bottom_y = y2
            return self.get_depth_at_point(depth_map, center_x, bottom_y)
        
        elif method == "median":
            # Extract median depth within bbox
            bbox_region = depth_map[y1:y2+1, x1:x2+1]
            if bbox_region.size == 0:
                # Fallback to center point
                return self.get_depth_at_point(depth_map, (x1 + x2) / 2, (y1 + y2) / 2)
            return float(np.median(bbox_region))
        
        elif method == "mean":
            # Extract mean depth within bbox
            bbox_region = depth_map[y1:y2+1, x1:x2+1]
            if bbox_region.size == 0:
                return self.get_depth_at_point(depth_map, (x1 + x2) / 2, (y1 + y2) / 2)
            return float(np.mean(bbox_region))
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'median', 'mean', or 'bottom_center'")
