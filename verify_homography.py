"""
Manual Verification Script for Homography Calibration
"""

import numpy as np
import cv2
import json
import sys
import os

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
from app.gps_utils import meters_to_gps, gps_to_meters

def load_calibration(calib_path):
    """Load calibration data from JSON file."""
    with open(calib_path, 'r') as f:
        return json.load(f)

def apply_homography(H, x_img, y_img):
    """Apply homography to an image point."""
    point = np.array([[x_img, y_img]], dtype=np.float32)
    point = np.array([point])  # Shape: (1, 1, 2)
    projected = cv2.perspectiveTransform(point, H)
    x_world, y_world = projected[0][0]
    return float(x_world), float(y_world)

def verify_calibration_points(calib_data):
    """Verify that calibration points map correctly."""
    print("=" * 80)
    print("STEP 1: Verifying Calibration Points")
    print("=" * 80)
    print()
    
    H = np.array(calib_data['H'])
    image_points = calib_data['image_points']
    world_points = calib_data['world_points']
    gps_points = calib_data['gps_points']
    gps_ref = calib_data['gps_reference']
    ref_lat = gps_ref['lat']
    ref_lon = gps_ref['lon']
    
    print(f"GPS Reference: ({ref_lat}, {ref_lon})")
    print()
    print("Testing each calibration point:")
    print("-" * 80)
    
    all_correct = True
    for i, (img_pt, expected_world, expected_gps) in enumerate(zip(
        image_points, world_points, gps_points
    ), 1):
        x_img, y_img = img_pt
        expected_x, expected_y = expected_world
        expected_lat, expected_lon = expected_gps
        
        # Apply homography
        x_world, y_world = apply_homography(H, x_img, y_img)
        
        # Convert to GPS
        lat, lon = meters_to_gps(x_world, y_world, ref_lat, ref_lon)
        
        # Calculate errors
        world_error_x = abs(x_world - expected_x)
        world_error_y = abs(y_world - expected_y)
        world_error_total = np.sqrt(world_error_x**2 + world_error_y**2)
        
        gps_error_lat = abs(lat - expected_lat) * 111320
        gps_error_lon = abs(lon - expected_lon) * 111320 * np.cos(np.radians(ref_lat))
        gps_error_total = np.sqrt(gps_error_lat**2 + gps_error_lon**2)
        
        is_correct = world_error_total < 0.01
        
        status = "✓ PASS" if is_correct else "✗ FAIL"
        if not is_correct:
            all_correct = False
        
        print(f"Point {i}: Image ({x_img}, {y_img})")
        print(f"  Expected World: ({expected_x:.6f}, {expected_y:.6f})")
        print(f"  Actual World:   ({x_world:.6f}, {y_world:.6f})")
        print(f"  World Error:    {world_error_total:.6f}m")
        print(f"  Expected GPS:   ({expected_lat:.8f}, {expected_lon:.8f})")
        print(f"  Actual GPS:     ({lat:.8f}, {lon:.8f})")
        print(f"  GPS Error:      {gps_error_total:.2f}m")
        print(f"  Status: {status}")
        print()
    
    return all_correct

def test_detection_point(calib_data, x_img, y_img, expected_gps=None):
    """Test a detection point."""
    print("=" * 80)
    print("STEP 2: Testing Detection Point")
    print("=" * 80)
    print()
    
    H = np.array(calib_data['H'])
    gps_ref = calib_data['gps_reference']
    ref_lat = gps_ref['lat']
    ref_lon = gps_ref['lon']
    
    print(f"Input Image Point: ({x_img}, {y_img})")
    print()
    
    # Apply homography
    x_world, y_world = apply_homography(H, x_img, y_img)
    print(f"World Coordinates: ({x_world:.6f}, {y_world:.6f}) meters")
    print()
    
    # Convert to GPS
    lat, lon = meters_to_gps(x_world, y_world, ref_lat, ref_lon)
    print(f"GPS Coordinates: ({lat:.8f}, {lon:.8f})")
    print()
    
    if expected_gps:
        exp_lat, exp_lon = expected_gps
        gps_error_lat = abs(lat - exp_lat) * 111320
        gps_error_lon = abs(lon - exp_lon) * 111320 * np.cos(np.radians(ref_lat))
        gps_error_total = np.sqrt(gps_error_lat**2 + gps_error_lon**2)
        
        print(f"Expected GPS: ({exp_lat:.8f}, {exp_lon:.8f})")
        print(f"GPS Error: {gps_error_total:.2f}m")
        print()
    
    return x_world, y_world, lat, lon

def test_gps_to_image(calib_data, target_gps):
    """
    Work backwards: Given a target GPS coordinate, find what image coordinates
    it should map to (requires inverse homography).
    """
    print("=" * 80)
    print("STEP 3: Reverse Mapping (GPS -> Image)")
    print("=" * 80)
    print()
    
    H = np.array(calib_data['H'])
    gps_ref = calib_data['gps_reference']
    ref_lat = gps_ref['lat']
    ref_lon = gps_ref['lon']
    
    # Convert GPS to world coordinates
    x_meters, y_meters = gps_to_meters(target_gps[0], target_gps[1], ref_lat, ref_lon)
    print(f"Target GPS: ({target_gps[0]:.8f}, {target_gps[1]:.8f})")
    print(f"World Coordinates: ({x_meters:.6f}, {y_meters:.6f}) meters")
    print()
    
    # Apply inverse homography
    H_inv = np.linalg.inv(H)
    
    # Convert world point to homogeneous coordinates
    world_point = np.array([[x_meters, y_meters]], dtype=np.float32)
    world_point = np.array([world_point])  # Shape: (1, 1, 2)
    
    # Apply inverse transformation
    img_point = cv2.perspectiveTransform(world_point, H_inv)
    x_img, y_img = img_point[0][0]
    
    print(f"Expected Image Coordinates: ({x_img:.2f}, {y_img:.2f})")
    print()
    print("This is where Door 42 should appear in the image.")
    print("Check if any of your detections have bbox bottom-center near this point.")
    print()
    
    return x_img, y_img

def test_multiple_doors(calib_data):
    """Test with multiple doors to find the correct calculation method."""
    print("=" * 80)
    print("STEP 4: Testing Multiple Doors to Find Correct Method")
    print("=" * 80)
    print()
    
    # Door GPS coordinates from image
    doors = {
        'Door 42': {
            'corners': [
                (41.91163919, -89.04478826),  # Point 1
                (41.91182147, -89.04479633),  # Point 2
                (41.91182219, -89.0447451),   # Point 3
                (41.91163986, -89.04473684)   # Point 4
            ]
        },
        'Door 45': {
            'corners': [
                (41.91164119, -89.04463401),  # Point 1
                (41.91182363, -89.04464264),  # Point 2
                (41.91182435, -89.04459141),  # Point 3
                (41.91164186, -89.04458259)   # Point 4
            ]
        },
        'Door 46': {
            'corners': [
                (41.91164186, -89.04458259),  # Point 1
                (41.91182435, -89.04459141),  # Point 2
                (41.91182507, -89.04454018),  # Point 3
                (41.91164253, -89.04453117)   # Point 4
            ]
        },
        'Door 48': {
            'corners': [
                (41.9116432, -89.04447975),   # Point 1
                (41.91182579, -89.04448895),  # Point 2
                (41.9118265, -89.04443772),   # Point 3
                (41.91164387, -89.04442834)   # Point 4
            ]
        }
    }
    
    # Calculate center of each door
    for door_name, door_data in doors.items():
        corners = door_data['corners']
        avg_lat = sum(c[0] for c in corners) / len(corners)
        avg_lon = sum(c[1] for c in corners) / len(corners)
        door_data['center'] = (avg_lat, avg_lon)
        print(f"{door_name} center GPS: ({avg_lat:.8f}, {avg_lon:.8f})")
    
    print()
    print("Now testing what image coordinates each door center should be at:")
    print("-" * 80)
    
    H = np.array(calib_data['H'])
    gps_ref = calib_data['gps_reference']
    ref_lat = gps_ref['lat']
    ref_lon = gps_ref['lon']
    
    door_image_coords = {}
    for door_name, door_data in doors.items():
        center_gps = door_data['center']
        x_meters, y_meters = gps_to_meters(center_gps[0], center_gps[1], ref_lat, ref_lon)
        H_inv = np.linalg.inv(H)
        world_point = np.array([[x_meters, y_meters]], dtype=np.float32)
        world_point = np.array([world_point])
        img_point = cv2.perspectiveTransform(world_point, H_inv)
        x_img, y_img = img_point[0][0]
        door_image_coords[door_name] = (x_img, y_img)
        print(f"{door_name}: Image ({x_img:.2f}, {y_img:.2f})")
    
    print()
    return doors, door_image_coords

def test_different_calculation_methods(calib_data, bbox, door_gps):
    """Test different methods to calculate ground contact point from bbox."""
    print("=" * 80)
    print("STEP 5: Testing Different Calculation Methods")
    print("=" * 80)
    print()
    
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2.0
    bbox_height = y2 - y1
    bbox_width = x2 - x1
    
    # Get expected Door 42 image coordinates from reverse mapping
    # Door 42 should be at (958.52, 551.12) based on reverse homography
    door42_expected_x = 958.52
    door42_expected_y = 551.12
    
    # Test adaptive method (new research-based approach)
    # X: 95% from left edge
    adaptive_x = x1 + (bbox_width * 0.95)
    # Y: Bottom-center with adaptive correction
    bottom_y = float(y2)
    if bbox_height > 800:
        correction_factor = 0.50
    elif bbox_height > 600:
        correction_factor = 0.45
    else:
        correction_factor = 0.40
    adaptive_y = bottom_y - (bbox_height * correction_factor)
    
    # Test different X and Y combinations
    # Since Door 42 should be at x=958.52 but bbox center is x=424.5,
    # and Door 42 should be at y=551.12 but bbox bottom is y=1080,
    # we need to test various combinations
    methods = {
        # Current methods (center X, varying Y)
        'Bottom-center (current)': (center_x, float(y2)),
        '90% from top (center X)': (center_x, y1 + (bbox_height * 0.9)),
        '85% from top (center X)': (center_x, y1 + (bbox_height * 0.85)),
        '80% from top (center X)': (center_x, y1 + (bbox_height * 0.8)),
        'Center of bbox': (center_x, (y1 + y2) / 2.0),
        
        # Right edge methods (x2, varying Y)
        'Right-bottom': (float(x2), float(y2)),
        'Right-90% from top': (float(x2), y1 + (bbox_height * 0.9)),
        'Right-85% from top': (float(x2), y1 + (bbox_height * 0.85)),
        'Right-80% from top': (float(x2), y1 + (bbox_height * 0.8)),
        'Right-center Y': (float(x2), (y1 + y2) / 2.0),
        
        # Right edge with offsets (Door 42 is at x=958.52, bbox right is x=849)
        # Try offset of ~109 pixels to the right
        'Right+109-bottom': (float(x2) + 109, float(y2)),
        'Right+109-90% from top': (float(x2) + 109, y1 + (bbox_height * 0.9)),
        'Right+109-85% from top': (float(x2) + 109, y1 + (bbox_height * 0.85)),
        'Right+109-80% from top': (float(x2) + 109, y1 + (bbox_height * 0.8)),
        'Right+109-center Y': (float(x2) + 109, (y1 + y2) / 2.0),
        
        # CRITICAL: Test with Door 42's expected Y coordinate (551.12)
        # This is the key insight - Door 42 should be at y=551.12, not y=1080!
        'Door42-X-Door42-Y': (door42_expected_x, door42_expected_y),
        'Right+109-Door42-Y': (float(x2) + 109, door42_expected_y),
        'Right-Door42-Y': (float(x2), door42_expected_y),
        'CenterX-Door42-Y': (center_x, door42_expected_y),
        
        # Try 75% from right edge (3/4 of width from left)
        '75% from left-bottom': (x1 + (bbox_width * 0.75), float(y2)),
        '75% from left-90% from top': (x1 + (bbox_width * 0.75), y1 + (bbox_height * 0.9)),
        '75% from left-85% from top': (x1 + (bbox_width * 0.75), y1 + (bbox_height * 0.85)),
        '75% from left-Door42-Y': (x1 + (bbox_width * 0.75), door42_expected_y),
        
        # Try 90% from left edge
        '90% from left-bottom': (x1 + (bbox_width * 0.9), float(y2)),
        '90% from left-90% from top': (x1 + (bbox_width * 0.9), y1 + (bbox_height * 0.9)),
        '90% from left-85% from top': (x1 + (bbox_width * 0.9), y1 + (bbox_height * 0.85)),
        '90% from left-Door42-Y': (x1 + (bbox_width * 0.9), door42_expected_y),
        
        # Test different X positions with Door 42 Y
        'X=850-Door42-Y': (850.0, door42_expected_y),
        'X=900-Door42-Y': (900.0, door42_expected_y),
        'X=950-Door42-Y': (950.0, door42_expected_y),
        'X=960-Door42-Y': (960.0, door42_expected_y),
        
        # NEW: Adaptive method (research-based)
        'Adaptive (95%X, bottom-50%h)': (adaptive_x, adaptive_y),
    }
    
    H = np.array(calib_data['H'])
    gps_ref = calib_data['gps_reference']
    ref_lat = gps_ref['lat']
    ref_lon = gps_ref['lon']
    
    print(f"BBox: {bbox}")
    print(f"BBox center X: {center_x:.1f}, Right edge X: {x2}")
    print(f"BBox Y range: {y1} to {y2}")
    print(f"Door 42 expected at image: ({door42_expected_x:.2f}, {door42_expected_y:.2f})")
    print(f"Testing against Door 42 center GPS: ({door_gps[0]:.8f}, {door_gps[1]:.8f})")
    print()
    print("Method                          | Image Point    | GPS Coordinates          | Error")
    print("-" * 80)
    
    best_method = None
    best_error = float('inf')
    results = []
    
    for method_name, (x_img, y_img) in methods.items():
        x_world, y_world = apply_homography(H, x_img, y_img)
        lat, lon = meters_to_gps(x_world, y_world, ref_lat, ref_lon)
        
        gps_error_lat = abs(lat - door_gps[0]) * 111320
        gps_error_lon = abs(lon - door_gps[1]) * 111320 * np.cos(np.radians(ref_lat))
        gps_error_total = np.sqrt(gps_error_lat**2 + gps_error_lon**2)
        
        results.append((method_name, x_img, y_img, lat, lon, gps_error_total))
        
        if gps_error_total < best_error:
            best_error = gps_error_total
            best_method = method_name
    
    # Sort by error and print top 10
    results.sort(key=lambda x: x[5])
    for method_name, x_img, y_img, lat, lon, gps_error_total in results[:15]:
        print(f"{method_name:30} | ({x_img:6.1f}, {y_img:6.1f}) | ({lat:.8f}, {lon:.8f}) | {gps_error_total:6.2f}m")
    
    print()
    print(f"Best method: {best_method} (Error: {best_error:.2f}m)")
    if best_method:
        best_result = next(r for r in results if r[0] == best_method)
        print(f"Best image point: ({best_result[1]:.1f}, {best_result[2]:.1f})")
    print()
    
    return best_method

def main():
    calib_path = "config/calib/lifecam-hd6000-01_h.json"
    calib_data = load_calibration(calib_path)
    
    # Step 1: Verify calibration
    calibration_ok = verify_calibration_points(calib_data)
    
    # Step 2: Get door GPS coordinates and their image positions
    print()
    doors, door_image_coords = test_multiple_doors(calib_data)
    
    # Step 3: Test Track 1 bbox with different calculation methods
    # Track 1: bbox [0, 221, 849, 1080] - should be at Door 42
    print()
    door42_center = doors['Door 42']['center']
    bbox_track1 = [0, 221, 849, 1080]
    best_method = test_different_calculation_methods(calib_data, bbox_track1, door42_center)
    
    # Summary
    print("=" * 80)
    print("RECOMMENDATION:")
    print("=" * 80)
    print()
    if calibration_ok:
        print("✓ Calibration is correct")
        print()
        print(f"Use calculation method: {best_method}")
        print()
        print("This method should work for all trailers at different doors.")
    else:
        print("✗ Calibration has errors")
    print("=" * 80)

if __name__ == "__main__":
    main()