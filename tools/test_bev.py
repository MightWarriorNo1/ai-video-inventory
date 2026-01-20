"""
Test BEV Transformation Model

Test and visualize the trained BEV model on calibration points or test images.
"""

import argparse
import cv2
import numpy as np
import json
from pathlib import Path
import sys
import os
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from app.ai.bev_transformer import BEVTransformer
from app.bev_utils import BEVProjector, visualize_bev_transform
from app.gps_utils import gps_to_meters, meters_to_gps


def test_bev_on_calibration(
    model_path: str,
    calib_path: str,
    image_path: Optional[str] = None,
    output_dir: Optional[str] = None
):
    """
    Test BEV model on calibration points.
    
    Args:
        model_path: Path to trained BEV model
        calib_path: Path to calibration JSON file
        image_path: Path to calibration image (if different from calib file)
        output_dir: Directory to save test results
    """
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
    
    # Get calibration points
    image_points = calib_data.get('image_points', [])
    world_points = calib_data.get('world_points', [])
    gps_reference = calib_data.get('gps_reference')
    
    if len(image_points) != len(world_points):
        raise ValueError("Mismatch between image_points and world_points")
    
    print(f"\nTesting on {len(image_points)} calibration points...")
    print("-" * 80)
    
    errors = []
    
    for i, (img_pt, world_pt) in enumerate(zip(image_points, world_points)):
        x_img, y_img = img_pt
        x_world_expected, y_world_expected = world_pt
        
        # Predict using BEV
        x_world_pred, y_world_pred = transformer.transform_point(image, x_img, y_img)
        
        # Calculate error
        error = np.sqrt(
            (x_world_pred - x_world_expected)**2 + 
            (y_world_pred - y_world_expected)**2
        )
        errors.append(error)
        
        # Print result
        print(f"Point {i+1}:")
        print(f"  Image: ({x_img:.1f}, {y_img:.1f})")
        print(f"  Expected World: ({x_world_expected:.3f}, {y_world_expected:.3f}) m")
        print(f"  Predicted World: ({x_world_pred:.3f}, {y_world_pred:.3f}) m")
        print(f"  Error: {error:.3f} m")
        
        # If GPS reference available, also check GPS
        if gps_reference:
            lat_exp, lon_exp = meters_to_gps(
                x_world_expected, y_world_expected,
                gps_reference['lat'], gps_reference['lon']
            )
            lat_pred, lon_pred = meters_to_gps(
                x_world_pred, y_world_pred,
                gps_reference['lat'], gps_reference['lon']
            )
            
            # Calculate GPS error in meters
            from pyproj import Geod
            geod = Geod(ellps='WGS84')
            _, _, gps_error = geod.inv(lon_exp, lat_exp, lon_pred, lat_pred)
            gps_error = abs(gps_error)
            
            print(f"  Expected GPS: ({lat_exp:.8f}, {lon_exp:.8f})")
            print(f"  Predicted GPS: ({lat_pred:.8f}, {lon_pred:.8f})")
            print(f"  GPS Error: {gps_error:.3f} m")
        
        print()
    
    # Statistics
    errors = np.array(errors)
    print("-" * 80)
    print(f"Statistics:")
    print(f"  Mean Error: {np.mean(errors):.3f} m")
    print(f"  Median Error: {np.median(errors):.3f} m")
    print(f"  Std Error: {np.std(errors):.3f} m")
    print(f"  Max Error: {np.max(errors):.3f} m")
    print(f"  Min Error: {np.min(errors):.3f} m")
    
    # Create visualization
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Visualize
        vis_image = visualize_bev_transform(
            image,
            transformer,
            points=image_points
        )
        
        output_path = output_dir / "bev_test_visualization.jpg"
        cv2.imwrite(str(output_path), vis_image)
        print(f"\nVisualization saved to: {output_path}")


def test_bev_on_image(
    model_path: str,
    image_path: str,
    points: Optional[list] = None,
    output_path: Optional[str] = None
):
    """
    Test BEV model on arbitrary image with optional points.
    
    Args:
        model_path: Path to trained BEV model
        image_path: Path to test image
        points: Optional list of (x_img, y_img) points to test
        output_path: Optional path to save visualization
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Initialize BEV transformer
    print(f"Loading BEV model from {model_path}")
    transformer = BEVTransformer(model_path=model_path)
    
    # Visualize
    vis_image = visualize_bev_transform(
        image,
        transformer,
        points=points
    )
    
    if output_path:
        cv2.imwrite(output_path, vis_image)
        print(f"Visualization saved to: {output_path}")
    
    return vis_image


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Test BEV Transformation Model')
    parser.add_argument('--model', required=True, help='Path to trained BEV model')
    parser.add_argument('--calib', help='Path to calibration JSON file (for testing on calibration points)')
    parser.add_argument('--image', help='Path to test image')
    parser.add_argument('--points', help='Points to test as JSON array: [[x1,y1], [x2,y2], ...]')
    parser.add_argument('--output', help='Output directory or file path for visualization')
    
    args = parser.parse_args()
    
    try:
        if args.calib:
            # Test on calibration data
            test_bev_on_calibration(
                model_path=args.model,
                calib_path=args.calib,
                image_path=args.image,
                output_dir=args.output
            )
        elif args.image:
            # Test on single image
            points = None
            if args.points:
                import json
                points = json.loads(args.points)
            
            test_bev_on_image(
                model_path=args.model,
                image_path=args.image,
                points=points,
                output_path=args.output
            )
        else:
            parser.error("Either --calib or --image must be provided")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())







