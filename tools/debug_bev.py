"""
Debug BEV Model Predictions

This script helps debug what the BEV model is actually predicting.
"""

import argparse
import cv2
import numpy as np
import json
import torch
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from app.ai.bev_transformer import BEVTransformer


def debug_bev_output(model_path: str, calib_path: str, image_path: str = None):
    """Debug what the BEV model is predicting."""
    
    # Load calibration data
    with open(calib_path, 'r') as f:
        calib_data = json.load(f)
    
    # Get image path
    if image_path is None:
        image_path = calib_data.get('image_path')
        if image_path is None:
            raise ValueError("image_path not provided and not in calibration data")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Initialize BEV transformer
    print(f"Loading BEV model from {model_path}")
    transformer = BEVTransformer(model_path=model_path)
    
    # Get BEV output map
    print("Getting BEV output map...")
    bev_map = transformer.transform_image_to_bev(image)
    
    print(f"BEV map shape: {bev_map.shape}")  # Should be (2, H, W)
    print(f"BEV map dtype: {bev_map.dtype}")
    
    # Get statistics
    bev_x = bev_map[0]  # X coordinates
    bev_y = bev_map[1]  # Y coordinates
    
    print(f"\nX coordinate statistics:")
    print(f"  Min: {bev_x.min():.6f}")
    print(f"  Max: {bev_x.max():.6f}")
    print(f"  Mean: {bev_x.mean():.6f}")
    print(f"  Std: {bev_x.std():.6f}")
    
    print(f"\nY coordinate statistics:")
    print(f"  Min: {bev_y.min():.6f}")
    print(f"  Max: {bev_y.max():.6f}")
    print(f"  Mean: {bev_y.mean():.6f}")
    print(f"  Std: {bev_y.std():.6f}")
    
    # Check normalization parameters
    print(f"\nNormalization parameters:")
    print(f"  normalize_coords: {transformer.normalize_coords}")
    print(f"  world_x_min: {transformer.world_x_min}")
    print(f"  world_x_max: {transformer.world_x_max}")
    print(f"  world_x_range: {transformer.world_x_range}")
    print(f"  world_y_min: {transformer.world_y_min}")
    print(f"  world_y_max: {transformer.world_y_max}")
    print(f"  world_y_range: {transformer.world_y_range}")
    
    # Test a few points
    image_points = calib_data.get('image_points', [])
    world_points = calib_data.get('world_points', [])
    
    print(f"\nTesting calibration points:")
    print("-" * 80)
    for i, (img_pt, world_pt) in enumerate(zip(image_points[:5], world_points[:5])):
        x_img, y_img = img_pt
        x_world_expected, y_world_expected = world_pt
        
        # Get prediction
        x_world_pred, y_world_pred = transformer.transform_point(image, x_img, y_img)
        
        print(f"Point {i+1}:")
        print(f"  Image: ({x_img:.1f}, {y_img:.1f})")
        print(f"  Expected: ({x_world_expected:.3f}, {y_world_expected:.3f})")
        print(f"  Predicted: ({x_world_pred:.3f}, {y_world_pred:.3f})")
        
        # Show raw BEV output at that location
        orig_h, orig_w = image.shape[:2]
        scale_x = transformer.input_size[1] / orig_w
        scale_y = transformer.input_size[0] / orig_h
        x_img_resized = x_img * scale_x
        y_img_resized = y_img * scale_y
        
        bev_x_idx = int((x_img_resized / transformer.input_size[1]) * transformer.bev_size[1])
        bev_y_idx = int((y_img_resized / transformer.input_size[0]) * transformer.bev_size[0])
        bev_x_idx = max(0, min(bev_x_idx, transformer.bev_size[1] - 1))
        bev_y_idx = max(0, min(bev_y_idx, transformer.bev_size[0] - 1))
        
        raw_x = bev_x[bev_y_idx, bev_x_idx]
        raw_y = bev_y[bev_y_idx, bev_x_idx]
        
        print(f"  Raw BEV output: ({raw_x:.6f}, {raw_y:.6f})")
        print()


def main():
    parser = argparse.ArgumentParser(description='Debug BEV Model')
    parser.add_argument('--model', required=True, help='Path to BEV model')
    parser.add_argument('--calib', required=True, help='Path to calibration JSON')
    parser.add_argument('--image', help='Path to image (if different from calib)')
    
    args = parser.parse_args()
    
    try:
        debug_bev_output(args.model, args.calib, args.image)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())




