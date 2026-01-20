"""
Validate 3D Camera Calibration Accuracy

This tool analyzes a 3D calibration file to assess accuracy and identify potential issues.

Usage:
    python tools/validate_3d_calibration.py --calib config/calib/camera-01_3d.json
"""

import json
import numpy as np
import argparse
from pathlib import Path
import cv2


def validate_calibration(calib_path: str):
    """Validate and analyze 3D calibration accuracy."""
    with open(calib_path, 'r') as f:
        calib_data = json.load(f)
    
    print("=" * 80)
    print("3D Camera Calibration Validation")
    print("=" * 80)
    print(f"\nCalibration file: {calib_path}")
    
    # Extract data
    K = np.array(calib_data['intrinsic_matrix'])
    R = np.array(calib_data['rotation_matrix'])
    t = np.array(calib_data['translation_vector']).reshape(3, 1)
    dist_coeffs = np.array(calib_data['distortion_coefficients'])
    image_points = np.array(calib_data['image_points'], dtype=np.float32)
    world_points = np.array(calib_data['world_points'], dtype=np.float32)
    reprojection_error = calib_data.get('reprojection_error', None)
    image_size = calib_data.get('image_size', {})
    
    print(f"\nImage size: {image_size.get('width', 'unknown')}x{image_size.get('height', 'unknown')}")
    print(f"Number of calibration points: {len(image_points)}")
    
    # Analyze intrinsic parameters
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    print(f"\n--- Intrinsic Parameters ---")
    print(f"Focal length: fx={fx:.2f}, fy={fy:.2f}")
    print(f"Principal point: cx={cx:.2f}, cy={cy:.2f}")
    
    # Check if intrinsics look reasonable
    w = image_size.get('width', fx * 2)
    h = image_size.get('height', fy * 2)
    
    print(f"\n--- Intrinsic Parameter Analysis ---")
    if abs(fx - fy) / fx > 0.1:
        print("⚠️  WARNING: Significant difference between fx and fy - may indicate skew or distortion")
    else:
        print("✓ Focal lengths fx and fy are similar (expected)")
    
    # Principal point should be near image center
    center_x = w / 2.0
    center_y = h / 2.0
    cx_offset = abs(cx - center_x) / w
    cy_offset = abs(cy - center_y) / h
    
    if cx_offset > 0.15 or cy_offset > 0.15:
        print(f"⚠️  WARNING: Principal point ({cx:.1f}, {cy:.1f}) is far from image center ({center_x:.1f}, {center_y:.1f})")
        print(f"    Offset: {cx_offset*100:.1f}% (X), {cy_offset*100:.1f}% (Y)")
    else:
        print(f"✓ Principal point is near image center (offset: {cx_offset*100:.1f}% X, {cy_offset*100:.1f}% Y)")
    
    # Focal length should be roughly image width (for standard cameras)
    focal_ratio = fx / w if w > 0 else 1.0
    if focal_ratio < 0.5 or focal_ratio > 2.0:
        print(f"⚠️  WARNING: Focal length (fx={fx:.2f}) is unusual compared to image width ({w:.0f})")
        print(f"    Typical range: 0.5-2.0x image width, got {focal_ratio:.2f}x")
        print(f"    Suggestion: Verify intrinsic parameters or recalibrate with known camera specs")
    else:
        print(f"✓ Focal length ({fx:.2f}) is within typical range ({focal_ratio:.2f}x image width)")
    
    # Analyze extrinsic parameters
    print(f"\n--- Extrinsic Parameters ---")
    camera_pos = -R.T @ t
    print(f"Camera position: ({camera_pos[0,0]:.2f}, {camera_pos[1,0]:.2f}, {camera_pos[2,0]:.2f}) meters")
    print(f"Camera height above ground: {camera_pos[2,0]:.2f} meters")
    
    # Check camera height is reasonable
    if camera_pos[2,0] < 1.0 or camera_pos[2,0] > 100.0:
        print(f"⚠️  WARNING: Camera height ({camera_pos[2,0]:.2f}m) seems unusual")
        print(f"    Typical range: 1-50 meters above ground")
    
    # Reprojection error analysis
    print(f"\n--- Reprojection Error Analysis ---")
    if reprojection_error is not None:
        print(f"Reported reprojection error: {reprojection_error:.4f} pixels")
        
        # Recalculate reprojection error
        projected_points, _ = cv2.projectPoints(
            world_points.reshape(-1, 1, 3),
            cv2.Rodrigues(R)[0],
            t,
            K,
            dist_coeffs
        )
        projected_points = projected_points.reshape(-1, 2)
        
        errors = np.linalg.norm(projected_points - image_points, axis=1)
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        std_error = np.std(errors)
        
        print(f"Recalculated mean error: {mean_error:.4f} pixels")
        print(f"Recalculated max error: {max_error:.4f} pixels")
        print(f"Recalculated std deviation: {std_error:.4f} pixels")
        
        if mean_error > 10.0:
            print("❌ ERROR: Reprojection error is HIGH (>10 pixels)")
            print("    This indicates poor calibration quality")
            print("    Suggestions:")
            print("    - Recalibrate with more points (8+ recommended)")
            print("    - Ensure GPS coordinates are accurate")
            print("    - Verify all points are on the ground plane (Z=0)")
            print("    - Check if camera intrinsics need refinement")
        elif mean_error > 5.0:
            print("⚠️  WARNING: Reprojection error is MODERATE (5-10 pixels)")
            print("    Accuracy may be limited")
            print("    Suggestions:")
            print("    - Add more calibration points")
            print("    - Verify GPS coordinate accuracy")
            print("    - Consider refining intrinsic parameters")
        else:
            print("✓ Reprojection error is LOW (<5 pixels) - good calibration quality")
    else:
        print("⚠️  WARNING: Reprojection error not available in calibration file")
    
    # Point distribution analysis
    print(f"\n--- Calibration Point Distribution ---")
    world_x = world_points[:, 0]
    world_y = world_points[:, 1]
    
    x_range = np.max(world_x) - np.min(world_x)
    y_range = np.max(world_y) - np.min(world_y)
    
    print(f"X range: {np.min(world_x):.2f} to {np.max(world_x):.2f} meters (span: {x_range:.2f}m)")
    print(f"Y range: {np.min(world_y):.2f} to {np.max(world_y):.2f} meters (span: {y_range:.2f}m)")
    
    # Check point distribution
    if len(image_points) < 6:
        print(f"⚠️  WARNING: Only {len(image_points)} calibration points (recommended: 8+)")
        print("    More points improve accuracy, especially spread across the image")
    else:
        print(f"✓ Good number of calibration points ({len(image_points)})")
    
    # Check if points are spread across image
    img_x = image_points[:, 0]
    img_y = image_points[:, 1]
    img_w = image_size.get('width', np.max(img_x))
    img_h = image_size.get('height', np.max(img_y))
    
    x_coverage = (np.max(img_x) - np.min(img_x)) / img_w if img_w > 0 else 0
    y_coverage = (np.max(img_y) - np.min(img_y)) / img_h if img_h > 0 else 0
    
    if x_coverage < 0.5 or y_coverage < 0.5:
        print(f"⚠️  WARNING: Points are clustered (coverage: {x_coverage*100:.0f}% X, {y_coverage*100:.0f}% Y)")
        print("    Spread points across the entire image for better calibration")
    else:
        print(f"✓ Points are well-distributed (coverage: {x_coverage*100:.0f}% X, {y_coverage*100:.0f}% Y)")
    
    # Recommendations
    print(f"\n--- Recommendations ---")
    recommendations = []
    
    if reprojection_error and reprojection_error > 5.0:
        recommendations.append("Recalibrate with more points and verify GPS accuracy")
    
    if len(image_points) < 8:
        recommendations.append("Add more calibration points (8-12 recommended)")
    
    if focal_ratio < 0.5 or focal_ratio > 2.0:
        recommendations.append("Consider calibrating with known camera intrinsics or using camera calibration checkerboard")
    
    if not recommendations:
        recommendations.append("Calibration looks good! Consider testing with real video to verify accuracy.")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Validate 3D camera calibration")
    parser.add_argument("--calib", required=True, help="Path to calibration JSON file")
    args = parser.parse_args()
    
    calib_path = Path(args.calib)
    if not calib_path.exists():
        print(f"Error: Calibration file not found: {calib_path}")
        return
    
    validate_calibration(str(calib_path))


if __name__ == "__main__":
    main()
