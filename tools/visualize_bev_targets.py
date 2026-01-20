"""
Visualize BEV Target Maps

This script helps debug BEV training by visualizing what target maps are being created.
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
from app.ai.bev_transformer import BEVDataset


def visualize_target_map(calib_path: str, output_dir: str = None):
    """Visualize BEV target maps."""
    
    # Load calibration data
    with open(calib_path, 'r') as f:
        calib_data = json.load(f)
    
    image_path = calib_data.get('image_path')
    if image_path is None:
        raise ValueError("image_path not found in calibration data")
    
    image_points = calib_data.get('image_points', [])
    world_points = calib_data.get('world_points', [])
    
    # Get homography if available
    homography_matrix = None
    if 'H' in calib_data:
        homography_matrix = np.array(calib_data['H'], dtype=np.float32)
    
    # Create dataset
    dataset = BEVDataset(
        image_paths=[image_path] * len(image_points),
        image_points=image_points,
        world_points=world_points,
        image_size=(720, 1280),
        bev_size=(256, 256),
        normalize_coords=True,
        homography_matrix=homography_matrix
    )
    
    # Get target map
    target_map = dataset._create_dense_target_map()  # (2, H, W)
    
    # Visualize X and Y maps separately
    x_map = target_map[0].numpy()  # (H, W)
    y_map = target_map[1].numpy()  # (H, W)
    
    print(f"X map range: [{x_map.min():.3f}, {x_map.max():.3f}]")
    print(f"Y map range: [{y_map.min():.3f}, {y_map.max():.3f}]")
    print(f"X map mean: {x_map.mean():.3f}, std: {x_map.std():.3f}")
    print(f"Y map mean: {y_map.mean():.3f}, std: {y_map.std():.3f}")
    
    # Normalize to 0-255 for visualization
    x_map_viz = ((x_map - x_map.min()) / (x_map.max() - x_map.min() + 1e-6) * 255).astype(np.uint8)
    y_map_viz = ((y_map - y_map.min()) / (y_map.max() - y_map.min() + 1e-6) * 255).astype(np.uint8)
    
    # Apply colormap
    x_map_colored = cv2.applyColorMap(x_map_viz, cv2.COLORMAP_JET)
    y_map_colored = cv2.applyColorMap(y_map_viz, cv2.COLORMAP_JET)
    
    # Mark calibration points
    for img_pt in image_points:
        x_img, y_img = img_pt
        bev_x = int((x_img / 1280) * 256)
        bev_y = int((y_img / 720) * 256)
        bev_x = max(0, min(bev_x, 255))
        bev_y = max(0, min(bev_y, 255))
        cv2.circle(x_map_colored, (bev_x, bev_y), 3, (255, 255, 255), -1)
        cv2.circle(y_map_colored, (bev_x, bev_y), 3, (255, 255, 255), -1)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / "bev_target_x.jpg"), x_map_colored)
        cv2.imwrite(str(output_dir / "bev_target_y.jpg"), y_map_colored)
        print(f"Visualizations saved to {output_dir}")
    else:
        # Display
        cv2.imshow("BEV Target X", cv2.resize(x_map_colored, (512, 512)))
        cv2.imshow("BEV Target Y", cv2.resize(y_map_colored, (512, 512)))
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Visualize BEV Target Maps')
    parser.add_argument('--calib', required=True, help='Path to calibration JSON')
    parser.add_argument('--output', help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    try:
        visualize_target_map(args.calib, args.output)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())




