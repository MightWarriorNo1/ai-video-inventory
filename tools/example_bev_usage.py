"""
Example: Using Deep Learning-Based BEV Transformation

This script demonstrates how to use the BEV transformer in your code.
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.bev_utils import BEVProjector, HybridProjector
from app.bbox_to_image_coords import calculate_image_coords_from_bbox


def example_basic_projection():
    """Example: Basic point projection using BEV."""
    print("=" * 60)
    print("Example 1: Basic BEV Projection")
    print("=" * 60)
    
    # Initialize BEV projector
    model_path = "models/bev/camera-01_bev.pth"  # Update with your model path
    gps_reference = {
        "lat": 41.91164053,
        "lon": -89.04468542
    }
    
    try:
        bev_projector = BEVProjector(
            model_path=model_path,
            gps_reference=gps_reference
        )
        
        # Load test image
        image_path = "path/to/your/image.jpg"  # Update with your image path
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return
        
        # Project image point to GPS
        x_img, y_img = 1150, 783  # Example image coordinates
        lat, lon = bev_projector.project_to_world(
            image, x_img, y_img, return_gps=True
        )
        
        print(f"Image point: ({x_img}, {y_img})")
        print(f"GPS coordinates: ({lat:.8f}, {lon:.8f})")
        
    except FileNotFoundError:
        print(f"Model not found: {model_path}")
        print("Please train a model first using tools/train_bev.py")
    except Exception as e:
        print(f"Error: {e}")


def example_bbox_projection():
    """Example: Project bounding box to GPS."""
    print("\n" + "=" * 60)
    print("Example 2: Bounding Box Projection")
    print("=" * 60)
    
    model_path = "models/bev/camera-01_bev.pth"
    gps_reference = {
        "lat": 41.91164053,
        "lon": -89.04468542
    }
    
    try:
        bev_projector = BEVProjector(
            model_path=model_path,
            gps_reference=gps_reference
        )
        
        # Load image
        image_path = "path/to/your/image.jpg"
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return
        
        # Example bounding box from YOLO detection
        bbox = [1100, 750, 1200, 850]  # [x1, y1, x2, y2]
        
        # Project bbox to GPS
        lat, lon = bev_projector.project_bbox(
            image=image,
            bbox=bbox,
            bbox_to_image_fn=lambda b: calculate_image_coords_from_bbox(b, "adaptive_percentage")
        )
        
        print(f"Bounding box: {bbox}")
        print(f"GPS coordinates: ({lat:.8f}, {lon:.8f})")
        
    except FileNotFoundError:
        print(f"Model not found: {model_path}")
        print("Please train a model first using tools/train_bev.py")
    except Exception as e:
        print(f"Error: {e}")


def example_hybrid_projection():
    """Example: Hybrid BEV + Homography projection."""
    print("\n" + "=" * 60)
    print("Example 3: Hybrid BEV + Homography")
    print("=" * 60)
    
    model_path = "models/bev/camera-01_bev.pth"
    gps_reference = {
        "lat": 41.91164053,
        "lon": -89.04468542
    }
    
    # Example homography matrix (load from your calibration file)
    H = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])  # Replace with your actual homography matrix
    
    try:
        hybrid_projector = HybridProjector(
            method='hybrid',
            bev_model_path=model_path,
            homography_matrix=H,
            gps_reference=gps_reference
        )
        
        # Load image
        image_path = "path/to/your/image.jpg"
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return
        
        # Project using hybrid method
        x_img, y_img = 1150, 783
        lat, lon = hybrid_projector.project_to_world(
            image, x_img, y_img, return_gps=True
        )
        
        print(f"Image point: ({x_img}, {y_img})")
        print(f"GPS coordinates (hybrid): ({lat:.8f}, {lon:.8f})")
        print("(Uses both BEV and homography, averaged)")
        
    except FileNotFoundError:
        print(f"Model not found: {model_path}")
        print("Please train a model first using tools/train_bev.py")
    except Exception as e:
        print(f"Error: {e}")


def example_integration_with_detection():
    """Example: Integration with YOLO detection pipeline."""
    print("\n" + "=" * 60)
    print("Example 4: Integration with Detection Pipeline")
    print("=" * 60)
    
    model_path = "models/bev/camera-01_bev.pth"
    gps_reference = {
        "lat": 41.91164053,
        "lon": -89.04468542
    }
    
    try:
        bev_projector = BEVProjector(
            model_path=model_path,
            gps_reference=gps_reference
        )
        
        # Simulate YOLO detection results
        detections = [
            {"bbox": [1100, 750, 1200, 850], "conf": 0.95, "class": "trailer"},
            {"bbox": [500, 600, 650, 750], "conf": 0.87, "class": "trailer"},
        ]
        
        image_path = "path/to/your/image.jpg"
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return
        
        # Process each detection
        results = []
        for det in detections:
            bbox = det["bbox"]
            
            # Project to GPS
            lat, lon = bev_projector.project_bbox(
                image=image,
                bbox=bbox,
                bbox_to_image_fn=lambda b: calculate_image_coords_from_bbox(b, "adaptive_percentage")
            )
            
            results.append({
                "bbox": bbox,
                "confidence": det["conf"],
                "class": det["class"],
                "gps": {"lat": lat, "lon": lon}
            })
            
            print(f"Detection: {det['class']} (conf: {det['conf']:.2f})")
            print(f"  BBox: {bbox}")
            print(f"  GPS: ({lat:.8f}, {lon:.8f})")
        
        print(f"\nProcessed {len(results)} detections")
        
    except FileNotFoundError:
        print(f"Model not found: {model_path}")
        print("Please train a model first using tools/train_bev.py")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("\nBEV Transformation Usage Examples")
    print("=" * 60)
    print("\nNOTE: Update file paths before running!")
    print("1. Set model_path to your trained BEV model")
    print("2. Set image_path to your test image")
    print("3. Update gps_reference with your camera's GPS reference point")
    print("\n")
    
    # Run examples
    example_basic_projection()
    example_bbox_projection()
    example_hybrid_projection()
    example_integration_with_detection()
    
    print("\n" + "=" * 60)
    print("For more information, see BEV_GUIDE.md")
    print("=" * 60)







