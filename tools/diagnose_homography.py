# tools/diagnose_homography.py
"""
Diagnostic tool to identify homography calibration issues.
"""
import json
import numpy as np
import cv2
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from app.gps_utils import meters_to_gps, gps_to_meters
from app.homography_validator import HomographyValidator

def analyze_calibration(calib_path: str):
    """Analyze calibration file for issues."""
    with open(calib_path, 'r') as f:
        calib_data = json.load(f)
    
    print("=" * 80)
    print("HOMOGRAPHY CALIBRATION DIAGNOSTIC")
    print("=" * 80)
    print()
    
    # Load data - ensure correct data types
    H = np.array(calib_data['H'], dtype=np.float32)
    image_points = np.array(calib_data['image_points'], dtype=np.float32)
    world_points = np.array(calib_data['world_points'], dtype=np.float32)
    gps_points = calib_data.get('gps_points', [])
    gps_ref = calib_data.get('gps_reference', {})
    
    print(f"Calibration File: {calib_path}")
    print(f"Number of calibration points: {len(image_points)}")
    print(f"RMSE: {calib_data.get('rmse', 'N/A'):.3f} meters")
    print()
    
    # 1. Check point distribution
    print("1. CALIBRATION POINT DISTRIBUTION")
    print("-" * 80)
    
    img_x_range = np.max(image_points[:, 0]) - np.min(image_points[:, 0])
    img_y_range = np.max(image_points[:, 1]) - np.min(image_points[:, 1])
    img_x_span = img_x_range / 1920.0 if img_x_range > 0 else 0  # Normalize to typical image width
    img_y_span = img_y_range / 1080.0 if img_y_range > 0 else 0  # Normalize to typical image height
    
    world_x_range = np.max(world_points[:, 0]) - np.min(world_points[:, 0])
    world_y_range = np.max(world_points[:, 1]) - np.min(world_points[:, 1])
    world_span = np.sqrt(world_x_range**2 + world_y_range**2)
    
    print(f"Image space coverage:")
    print(f"  X range: {img_x_range:.0f} pixels ({img_x_span*100:.1f}% of image width)")
    print(f"  Y range: {img_y_range:.0f} pixels ({img_y_span*100:.1f}% of image height)")
    print(f"World space coverage: {world_span:.2f} meters")
    print()
    
    if img_x_span < 0.3 or img_y_span < 0.3:
        print("⚠️  WARNING: Calibration points are CLUSTERED!")
        print("   Points cover less than 30% of image dimensions.")
        print("   Accuracy will degrade quickly away from calibration points.")
        print("   SOLUTION: Add more points spread across the entire image.")
        if img_y_span < 0.3:
            print(f"   ⚠️  CRITICAL: Vertical coverage is only {img_y_span*100:.1f}%!")
            print("      Points are clustered in Y direction (top/bottom).")
            print("      Add points at TOP and BOTTOM of image for better coverage.")
    elif img_x_span < 0.5 or img_y_span < 0.5:
        print("⚠️  WARNING: Calibration points have LIMITED coverage.")
        print("   Points cover 30-50% of image dimensions.")
        if img_y_span < 0.5:
            print(f"   ⚠️  Vertical coverage is only {img_y_span*100:.1f}% - add more top/bottom points.")
        print("   Consider adding more points for better coverage.")
    else:
        print("✓ Calibration points have GOOD distribution.")
    print()
    
    # 2. Check world space distribution
    print("2. WORLD SPACE DISTRIBUTION")
    print("-" * 80)
    print(f"World X range: {world_x_range:.2f} meters")
    print(f"World Y range: {world_y_range:.2f} meters")
    print(f"Total span: {world_span:.2f} meters")
    print()
    
    if world_span < 20:
        print("⚠️  WARNING: Calibration points span less than 20 meters.")
        print("   This is a very small area. Homography will only be accurate")
        print("   within this region. Points outside will have large errors.")
    elif world_span < 50:
        print("⚠️  WARNING: Calibration points span less than 50 meters.")
        print("   Consider adding points further from the center for better coverage.")
    else:
        print("✓ World space coverage is adequate.")
    print()
    
    # 3. Test accuracy at calibration points
    print("3. ACCURACY AT CALIBRATION POINTS")
    print("-" * 80)
    
    # Ensure correct shape and dtype for cv2.perspectiveTransform
    img_pts_reshaped = image_points.reshape(-1, 1, 2).astype(np.float32)
    projected = cv2.perspectiveTransform(img_pts_reshaped, H).reshape(-1, 2)
    
    errors = np.sqrt(np.sum((projected - world_points) ** 2, axis=1))
    max_error = np.max(errors)
    mean_error = np.mean(errors)
    
    print(f"Max error at calibration points: {max_error:.3f} meters")
    print(f"Mean error at calibration points: {mean_error:.3f} meters")
    print()
    
    for i, (err, img_pt, world_pt) in enumerate(zip(errors, image_points, world_points), 1):
        print(f"  Point {i}: Error = {err:.3f}m (Image: {img_pt}, World: {world_pt})")
    print()
    
    # 4. Test accuracy at image corners (extrapolation)
    print("4. EXTRAPOLATION ACCURACY (Image Corners)")
    print("-" * 80)
    
    # Get image dimensions from calibration or image file
    img_width = int(np.max(image_points[:, 0]) * 1.1)  # Estimate
    img_height = int(np.max(image_points[:, 1]) * 1.1)  # Estimate
    
    # Try to get actual image dimensions from image_path if available
    image_path = calib_data.get('image_path', '')
    if image_path:
        full_image_path = Path(calib_path).parent.parent / image_path
        if not full_image_path.exists():
            # Try relative to project root
            full_image_path = Path(calib_path).parent.parent.parent / image_path
        if full_image_path.exists():
            try:
                img = cv2.imread(str(full_image_path))
                if img is not None:
                    img_height, img_width = img.shape[:2]
                    print(f"Loaded image dimensions: {img_width}x{img_height}")
            except:
                pass
    
    # Test corners
    corners = [
        (0, 0, "Top-Left"),
        (img_width, 0, "Top-Right"),
        (0, img_height, "Bottom-Left"),
        (img_width, img_height, "Bottom-Right"),
        (img_width//2, img_height//2, "Center")
    ]
    
    validator = HomographyValidator(calib_data)
    
    print("Testing points at image boundaries:")
    for x, y, name in corners:
        # Ensure correct shape: (1, 1, 2) for cv2.perspectiveTransform
        point = np.array([[[x, y]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(point, H)
        x_world, y_world = projected[0][0]
        
        # Check distance from calibration region
        distance = validator.get_distance_from_calibration(x_world, y_world)
        confidence = validator.get_confidence_score(x_world, y_world)
        in_trusted = validator.is_in_trusted_region(x_world, y_world)
        
        status = "✓" if in_trusted else "⚠️"
        print(f"  {status} {name:12} ({x:4.0f}, {y:4.0f}): "
              f"World=({x_world:7.2f}, {y_world:7.2f})m, "
              f"Dist={distance:6.2f}m, Confidence={confidence:.2f}")
    print()
    
    # 5. Recommendations
    print("5. RECOMMENDATIONS")
    print("-" * 80)
    
    recommendations = []
    
    if len(image_points) < 6:
        recommendations.append("Add more calibration points (aim for 6-8 minimum)")
    
    if img_x_span < 0.5 or img_y_span < 0.5:
        recommendations.append("Spread calibration points across entire image area")
        if img_y_span < 0.5:
            recommendations.append(f"CRITICAL: Add points at TOP and BOTTOM of image (current Y coverage: {img_y_span*100:.1f}%)")
        if img_x_span < 0.5:
            recommendations.append(f"Add points at LEFT and RIGHT edges of image (current X coverage: {img_x_span*100:.1f}%)")
        recommendations.append("Add points near image corners and edges")
    
    if world_span < 50:
        recommendations.append("Add calibration points further from center (50+ meters)")
    
    if 'max_error' in locals() and max_error > 1.0:
        recommendations.append("Recalibrate - errors at calibration points are too high")
    
    if len(recommendations) == 0:
        print("✓ Calibration looks good! If you're still seeing issues, check:")
        print("  - Ground contact point calculation accuracy")
        print("  - GPS reference point correctness")
        print("  - Camera stability (no movement between calibration and processing)")
    else:
        print("Issues found. Recommended fixes:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    print()
    print("=" * 80)
    
    return {
        'num_points': len(image_points),
        'rmse': calib_data.get('rmse', 0),
        'img_coverage_x': img_x_span,
        'img_coverage_y': img_y_span,
        'world_span': world_span,
        'max_error': max_error,
        'mean_error': mean_error
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Diagnose homography calibration')
    parser.add_argument('calib_file', help='Path to calibration JSON file')
    args = parser.parse_args()
    
    analyze_calibration(args.calib_file)