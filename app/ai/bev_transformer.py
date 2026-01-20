"""
Deep Learning-Based BEV (Bird's-Eye View) Transformer

This module provides a CNN-based BEV transformation network that learns
image-to-world coordinate mapping using deep learning instead of traditional homography.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict
from pathlib import Path
from torch.utils.data import Dataset
import json


class BEVNet(nn.Module):
    """
    CNN-based BEV transformation network.
    
    Uses an encoder-decoder architecture to predict world coordinates
    for each pixel location in the input image.
    """
    
    def __init__(
        self,
        input_size: Tuple[int, int] = (720, 1280),
        bev_size: Tuple[int, int] = (256, 256),
        hidden_dim: int = 256,
        use_coord_regression: bool = True
    ):
        """
        Initialize BEV network.
        
        Args:
            input_size: Input image size (height, width)
            bev_size: BEV output size (height, width)
            hidden_dim: Hidden dimension for features
            use_coord_regression: If True, output (x, y) coordinates; else output features
        """
        super(BEVNet, self).__init__()
        self.input_size = input_size
        self.bev_size = bev_size
        self.hidden_dim = hidden_dim
        self.use_coord_regression = use_coord_regression
        
        # Encoder: Simple Sequential structure (matching original checkpoint)
        # Based on checkpoint keys: encoder.4.0, encoder.4.1, encoder.4.3, encoder.4.4
        self.encoder = nn.Sequential(
            # Initial conv (encoder.0)
            nn.Conv2d(3, 64, 7, 2, 3),
            # encoder.1 - BatchNorm
            nn.BatchNorm2d(64),
            # encoder.2 - ReLU
            nn.ReLU(inplace=True),
            # encoder.3 - MaxPool
            nn.MaxPool2d(3, 2, 1),
            
            # encoder.4 - First layer (64 -> 128, based on checkpoint shape [128, 64, 3, 3])
            nn.Sequential(
                nn.Conv2d(64, 128, 3, 1, 1),  # encoder.4.0 (shape [128, 64, 3, 3])
                nn.BatchNorm2d(128),          # encoder.4.1 (has weight/bias in checkpoint)
                nn.ReLU(inplace=True),        # encoder.4.2 (not in checkpoint, but needed)
                nn.Conv2d(128, 128, 3, 1, 1), # encoder.4.3 (shape [128, 128, 3, 3])
                nn.BatchNorm2d(128),          # encoder.4.4 (shape [128])
            ),
            
            # encoder.5 - Second layer (128 -> 256, based on checkpoint)
            nn.Sequential(
                nn.Conv2d(128, 256, 3, 2, 1), # encoder.5.0 (stride=2)
                nn.BatchNorm2d(256),          # encoder.5.1 (has weight in checkpoint)
                nn.ReLU(inplace=True),        # encoder.5.2 (not in checkpoint)
                nn.Conv2d(256, 256, 3, 1, 1), # encoder.5.3
                nn.BatchNorm2d(256),          # encoder.5.4
            ),
            
            # encoder.6 - Third layer (256 -> 512, based on checkpoint)
            nn.Sequential(
                nn.Conv2d(256, 512, 3, 2, 1), # encoder.6.0 (stride=2)
                nn.BatchNorm2d(512),          # encoder.6.1 (has weight in checkpoint)
                nn.ReLU(inplace=True),        # encoder.6.2 (not in checkpoint)
                nn.Conv2d(512, 512, 3, 1, 1), # encoder.6.3
                nn.BatchNorm2d(512),          # encoder.6.4
            ),
            
            # encoder.7 - Fourth layer (512 -> 512, final encoder layer)
            nn.Sequential(
                nn.Conv2d(512, 512, 3, 2, 1), # encoder.7.0 (stride=2)
                nn.BatchNorm2d(512),          # encoder.7.1 (has weight in checkpoint)
                nn.ReLU(inplace=True),        # encoder.7.2 (not in checkpoint)
                nn.Conv2d(512, 512, 3, 1, 1), # encoder.7.3
                nn.BatchNorm2d(512),          # encoder.7.4
            ),
        )
        
        # Bridge: 3x3 conv (matching checkpoint shape [256, 512, 3, 3])
        self.bridge = nn.Sequential(
            nn.Conv2d(512, hidden_dim, 3, 1, 1),  # 3x3 conv, not 1x1
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Decoder: Matching checkpoint structure based on error messages
        # decoder.1.weight: exists in checkpoint -> Conv2d (has weights!)
        # decoder.2.weight: [128] -> BatchNorm2d(128), NOT Conv2d!
        # decoder.3.weight: [128] -> BatchNorm2d(128) or next layer
        # decoder.5.weight: [64, 128, 3, 3] -> Conv2d(128, 64, 3, 1, 1)
        # decoder.6.running_mean: exists -> BatchNorm2d(64)
        # decoder.9.weight: exists in checkpoint -> Conv2d (has weights!)
        # decoder.10.weight: [32] -> BatchNorm2d(32)
        # output_head.0.weight: [64, 32, 3, 3] -> Conv2d(32, 64, 3, 1, 1), NOT Conv2d(32, 16)!
        self.decoder = nn.Sequential(
            # decoder.0 - ConvTranspose (256 -> 256)
            nn.ConvTranspose2d(hidden_dim, 256, 4, 2, 1),
            # decoder.1 - Conv2d (256 -> 128, checkpoint has decoder.1.weight!)
            nn.Conv2d(256, 128, 3, 1, 1),
            # decoder.2 - BatchNorm (checkpoint has decoder.2.weight as [128])
            nn.BatchNorm2d(128),
            # decoder.3 - ReLU
            nn.ReLU(inplace=True),
            # decoder.4 - ConvTranspose (128 -> 64) or keep 128?
            # Actually, decoder.5 is Conv2d(128, 64), so decoder.4 should keep 128
            # Let me check: decoder.5.weight [64, 128, 3, 3] means input is 128
            # So decoder.4 should not change channels, or it's ReLU
            nn.ReLU(inplace=True),
            # decoder.5 - Conv2d (128 -> 64, shape [64, 128, 3, 3])
            nn.Conv2d(128, 64, 3, 1, 1),
            # decoder.6 - BatchNorm (checkpoint has decoder.6.running_mean, shape [64])
            nn.BatchNorm2d(64),
            # decoder.7 - ReLU
            nn.ReLU(inplace=True),
            # decoder.8 - ConvTranspose (64 -> 64, to keep 64 for decoder.9)
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            # decoder.9 - Conv2d (64 -> 32, checkpoint has decoder.9.weight shape [32, 64, 3, 3])
            nn.Conv2d(64, 32, 3, 1, 1),
            # decoder.10 - BatchNorm (checkpoint has decoder.10.weight as [32])
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # Output head: Sequential with multiple layers (matching checkpoint)
        # output_head.0.weight: [64, 32, 3, 3] -> Conv2d(32, 64, 3, 1, 1)
        # output_head.2.weight: [2, 64, 1, 1] -> Conv2d(64, 2, 1, 1, 0), NOT Conv2d(64, 2, 3, 1, 1)!
        if use_coord_regression:
            self.output_head = nn.Sequential(
                nn.Conv2d(32, 64, 3, 1, 1),  # output_head.0 (shape [64, 32, 3, 3])
                nn.ReLU(inplace=True),       # output_head.1
                nn.Conv2d(64, 2, 1, 1, 0),  # output_head.2 (x, y) coordinates, shape [2, 64, 1, 1]
            )
        else:
            self.output_head = nn.Sequential(
                nn.Conv2d(32, hidden_dim, 3, 1, 1),
                nn.ReLU(inplace=True),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image tensor (B, 3, H, W)
        
        Returns:
            BEV coordinate map (B, 2, H_bev, W_bev) if use_coord_regression,
            else feature map (B, hidden_dim, H_bev, W_bev)
        """
        # Encode
        features = self.encoder(x)
        
        # Bridge
        features = self.bridge(features)
        
        # Decode
        features = self.decoder(features)
        
        # Resize to exact BEV size if needed
        if features.shape[2:] != self.bev_size:
            features = F.interpolate(
                features,
                size=self.bev_size,
                mode='bilinear',
                align_corners=False
            )
        
        # Output
        output = self.output_head(features)
        return output


class BEVDataset(Dataset):
    """
    Dataset for BEV training.
    
    Creates dense target maps from calibration points using homography
    or inverse distance weighting.
    """
    
    def __init__(
        self,
        image_paths: List[str],
        image_points: List[Tuple[float, float]],
        world_points: List[Tuple[float, float]],
        image_size: Tuple[int, int] = (720, 1280),
        bev_size: Tuple[int, int] = (256, 256),
        normalize_coords: bool = True,
        homography_matrix: Optional[np.ndarray] = None
    ):
        """
        Initialize dataset.
        
        Args:
            image_paths: List of image file paths
            image_points: List of (x, y) image coordinates
            world_points: List of (x, y) world coordinates (meters)
            image_size: Input image size (height, width)
            bev_size: BEV output size (height, width)
            normalize_coords: Whether to normalize world coordinates to [0, 1]
            homography_matrix: Optional homography matrix for dense target generation
        """
        self.image_paths = image_paths
        self.image_points = image_points
        self.world_points = world_points
        self.image_size = image_size
        self.bev_size = bev_size
        self.normalize_coords = normalize_coords
        self.homography_matrix = homography_matrix
        
        # Calculate normalization parameters
        if normalize_coords and len(world_points) > 0:
            world_x = [float(p[0]) for p in world_points]
            world_y = [float(p[1]) for p in world_points]
            
            self.world_x_min = min(world_x)
            self.world_x_max = max(world_x)
            self.world_y_min = min(world_y)
            self.world_y_max = max(world_y)
            
            # Sample points from homography if available
            if homography_matrix is not None:
                sample_points = []
                h, w = image_size
                # Sample corners and center
                for y in [0, h // 2, h - 1]:
                    for x in [0, w // 2, w - 1]:
                        point = np.array([[x, y]], dtype=np.float32)
                        point = np.array([point])
                        projected = cv2.perspectiveTransform(point, homography_matrix)
                        sample_points.append((float(projected[0][0][0]), float(projected[0][0][1])))
                
                # Include sampled points in range calculation
                all_x = [p[0] for p in world_points] + [p[0] for p in sample_points]
                all_y = [p[1] for p in world_points] + [p[1] for p in sample_points]
                
                self.world_x_min = min(all_x)
                self.world_x_max = max(all_x)
                self.world_y_min = min(all_y)
                self.world_y_max = max(all_y)
            
            # Calculate ranges with margin
            margin_x = max(1.0, (self.world_x_max - self.world_x_min) * 0.1)
            margin_y = max(0.1, (self.world_y_max - self.world_y_min) * 0.1)
            
            self.world_x_min -= margin_x
            self.world_x_max += margin_x
            self.world_y_min -= margin_y
            self.world_y_max += margin_y
            
            self.world_x_range = max(1.0, self.world_x_max - self.world_x_min)
            self.world_y_range = max(0.1, self.world_y_max - self.world_y_min)
        else:
            self.world_x_min = 0.0
            self.world_x_max = 0.0
            self.world_y_min = 0.0
            self.world_y_max = 0.0
            self.world_x_range = 1.0
            self.world_y_range = 1.0
        
        # Cache for dense target maps
        self._dense_target_cache = {}
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Tuple[float, float], Tuple[float, float]]:
        """
        Get a training sample.
        
        Returns:
            Tuple of (image, target_map, image_point, world_point)
        """
        image_path = self.image_paths[idx]
        image_point = self.image_points[idx]
        world_point = self.world_points[idx]
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        image = torch.from_numpy(image)
        
        # Get or create dense target map
        if image_path not in self._dense_target_cache:
            target_map = self._create_dense_target_map(image_path)
            self._dense_target_cache[image_path] = target_map
        else:
            target_map = self._dense_target_cache[image_path]
        
        return image, target_map, image_point, world_point
    
    def _create_dense_target_map(self, image_path: str) -> torch.Tensor:
        """
        Create dense target map for an image.
        
        Uses homography if available, otherwise inverse distance weighting.
        """
        target_map = np.zeros((2, self.bev_size[0], self.bev_size[1]), dtype=np.float32)
        
        if self.homography_matrix is not None:
            # Use homography to project all pixels
            h, w = self.image_size
            for y_bev in range(self.bev_size[0]):
                for x_bev in range(self.bev_size[1]):
                    # Map BEV coordinates back to image coordinates
                    x_img = (x_bev / self.bev_size[1]) * w
                    y_img = (y_bev / self.bev_size[0]) * h
                    
                    # Project using homography
                    point = np.array([[x_img, y_img]], dtype=np.float32)
                    point = np.array([point])
                    projected = cv2.perspectiveTransform(point, self.homography_matrix)
                    x_world, y_world = projected[0][0]
                    
                    # Normalize if needed
                    if self.normalize_coords:
                        x_world = (x_world - self.world_x_min) / self.world_x_range
                        y_world = (y_world - self.world_y_min) / self.world_y_range
                    
                    target_map[0, y_bev, x_bev] = x_world
                    target_map[1, y_bev, x_bev] = y_world
        else:
            # Use inverse distance weighting
            h, w = self.image_size
            for y_bev in range(self.bev_size[0]):
                for x_bev in range(self.bev_size[1]):
                    x_img = (x_bev / self.bev_size[1]) * w
                    y_img = (y_bev / self.bev_size[0]) * h
                    
                    # Calculate weighted average from calibration points
                    weights = []
                    values = []
                    for img_pt, world_pt in zip(self.image_points, self.world_points):
                        dist = np.sqrt((x_img - img_pt[0])**2 + (y_img - img_pt[1])**2)
                        if dist < 1e-6:
                            weight = 1e6
                        else:
                            weight = 1.0 / (dist**2 + 1e-6)
                        weights.append(weight)
                        
                        x_world, y_world = world_pt
                        if self.normalize_coords:
                            x_world = (x_world - self.world_x_min) / self.world_x_range
                            y_world = (y_world - self.world_y_min) / self.world_y_range
                        values.append((x_world, y_world))
                    
                    # Weighted average
                    total_weight = sum(weights)
                    if total_weight > 0:
                        x_world = sum(w * v[0] for w, v in zip(weights, values)) / total_weight
                        y_world = sum(w * v[1] for w, v in zip(weights, values)) / total_weight
                    else:
                        x_world, y_world = 0.0, 0.0
                    
                    target_map[0, y_bev, x_bev] = x_world
                    target_map[1, y_bev, x_bev] = y_world
        
        return torch.from_numpy(target_map)


class BEVTransformer:
    """
    BEV transformer for inference.
    
    Loads a trained BEV model and provides coordinate transformation.
    """
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        input_size: Tuple[int, int] = (720, 1280)
    ):
        """
        Initialize BEV transformer.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to use ('cuda', 'cpu', or None for auto)
            input_size: Input image size (height, width)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.input_size = input_size
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model parameters from checkpoint
        # Default BEV size
        self.bev_size = (256, 256)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # Check for normalization params (try both key names for compatibility)
            norm_params = None
            if 'normalization_params' in checkpoint:
                norm_params = checkpoint['normalization_params']
            elif 'normalization' in checkpoint:
                norm_params = checkpoint['normalization']
            
            if norm_params:
                self.normalize_coords = norm_params.get('normalize_coords', True)
                self.world_x_min = norm_params.get('world_x_min', 0.0)
                self.world_x_max = norm_params.get('world_x_max', 0.0)
                self.world_y_min = norm_params.get('world_y_min', 0.0)
                self.world_y_max = norm_params.get('world_y_max', 0.0)
                self.world_x_range = norm_params.get('world_x_range', 1.0)
                self.world_y_range = norm_params.get('world_y_range', 1.0)
                print(f"Loaded normalization params: X range [{self.world_x_min:.3f}, {self.world_x_max:.3f}], Y range [{self.world_y_min:.3f}, {self.world_y_max:.3f}]")
            else:
                print("Warning: No normalization parameters found in checkpoint, using defaults")
                self.normalize_coords = True
                self.world_x_min = 0.0
                self.world_x_max = 0.0
                self.world_y_min = 0.0
                self.world_y_max = 0.0
                self.world_x_range = 1.0
                self.world_y_range = 1.0
        else:
            state_dict = checkpoint
            self.normalize_coords = True
            self.world_x_min = 0.0
            self.world_x_max = 0.0
            self.world_y_min = 0.0
            self.world_y_max = 0.0
            self.world_x_range = 1.0
            self.world_y_range = 1.0
        
        # Initialize model
        self.model = BEVNet(
            input_size=input_size,
            bev_size=self.bev_size,
            use_coord_regression=True
        ).to(self.device)
        
        # Load weights with flexible matching
        # The checkpoint may have been saved with a different architecture
        model_dict = self.model.state_dict()
        
        # Filter to only matching keys with matching shapes
        pretrained_dict = {}
        skipped_keys = []
        for k, v in state_dict.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    pretrained_dict[k] = v
                else:
                    skipped_keys.append(f"{k} (shape mismatch: {v.shape} vs {model_dict[k].shape})")
            else:
                skipped_keys.append(f"{k} (not in model)")
        
        # Update model dict with matching parameters
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        
        if skipped_keys:
            print(f"Warning: Skipped {len(skipped_keys)} parameters that don't match current architecture")
            if len(skipped_keys) <= 10:
                for key in skipped_keys[:10]:
                    print(f"  - {key}")
            else:
                for key in skipped_keys[:10]:
                    print(f"  - {key}")
                print(f"  ... and {len(skipped_keys) - 10} more")
        
        print(f"Loaded {len(pretrained_dict)}/{len(state_dict)} matching parameters from checkpoint")
        
        self.model.eval()
    
    def transform_point(
        self,
        image: np.ndarray,
        x_img: float,
        y_img: float
    ) -> Tuple[float, float]:
        """
        Transform a single image point to world coordinates.
        
        Args:
            image: Input image (BGR format)
            x_img: Image X coordinate
            y_img: Image Y coordinate
        
        Returns:
            Tuple of (x_world, y_world) in meters
        """
        # Preprocess image
        orig_h, orig_w = image.shape[:2]
        image_resized = cv2.resize(image, (self.input_size[1], self.input_size[0]))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb.astype(np.float32) / 255.0)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            bev_map = self.model(image_tensor)  # (1, 2, H_bev, W_bev)
        
        # Map image coordinates to BEV grid
        scale_x = self.input_size[1] / orig_w
        scale_y = self.input_size[0] / orig_h
        x_img_resized = x_img * scale_x
        y_img_resized = y_img * scale_y
        
        bev_x = int((x_img_resized / self.input_size[1]) * self.bev_size[1])
        bev_y = int((y_img_resized / self.input_size[0]) * self.bev_size[0])
        
        # Clamp to valid range
        bev_x = max(0, min(self.bev_size[1] - 1, bev_x))
        bev_y = max(0, min(self.bev_size[0] - 1, bev_y))
        
        # Extract coordinates
        x_world_norm = bev_map[0, 0, bev_y, bev_x].item()
        y_world_norm = bev_map[0, 1, bev_y, bev_x].item()
        
        # Denormalize
        if self.normalize_coords:
            x_world = x_world_norm * self.world_x_range + self.world_x_min
            y_world = y_world_norm * self.world_y_range + self.world_y_min
        else:
            x_world = x_world_norm
            y_world = y_world_norm
        
        return (float(x_world), float(y_world))
    
    def transform_image_to_bev(self, image: np.ndarray) -> np.ndarray:
        """
        Transform entire image to BEV coordinate map.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            BEV coordinate map (2, H_bev, W_bev) with (x, y) coordinates
        """
        # Preprocess image
        image_resized = cv2.resize(image, (self.input_size[1], self.input_size[0]))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb.astype(np.float32) / 255.0)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            bev_map = self.model(image_tensor)  # (1, 2, H_bev, W_bev)
        
        # Convert to numpy and denormalize
        bev_map = bev_map[0].cpu().numpy()  # (2, H_bev, W_bev)
        
        if self.normalize_coords:
            bev_map[0] = bev_map[0] * self.world_x_range + self.world_x_min
            bev_map[1] = bev_map[1] * self.world_y_range + self.world_y_min
        
        return bev_map


