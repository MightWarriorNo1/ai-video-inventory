"""
Test bbox-to-image-coords calculation with actual data.
"""

import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.bbox_to_image_coords_advanced import calculate_image_coords_from_bbox_with_config

# Expected coordinates from calibration
expected_coords = {
    "DD042": (1026.95977511, 719.29697004),
    "DD045": (1026.95633396, 719.29690812),
    "DD046": (1026.95518566, 719.29688746),
    "DD048": (1026.95288718, 719.2968461),
}

# Test with actual bbox values from the new folder
test_bboxes = {
    "DD042": [0, 221, 849, 1134],  # Track 1
    "DD045": [551, 115, 1394, 1050],  # Track 2 (from calibration)
    "DD046": [3, 134, 879, 1035],  # Track 3
    "DD048": [1129, 191, 1886, 1023],  # Track 4 (from new folder)
}

print("Testing bbox-to-image-coords calculation:")
print("=" * 80)

for spot, bbox in test_bboxes.items():
    if spot not in expected_coords:
        continue
    
    expected = expected_coords[spot]
    calculated = calculate_image_coords_from_bbox_with_config(bbox, debug=True)
    
    error_x = abs(calculated[0] - expected[0])
    error_y = abs(calculated[1] - expected[1])
    error_total = (error_x**2 + error_y**2)**0.5
    
    print(f"\n{spot}:")
    print(f"  BBox: {bbox}")
    print(f"  Expected: ({expected[0]:.2f}, {expected[1]:.2f})")
    print(f"  Calculated: ({calculated[0]:.2f}, {calculated[1]:.2f})")
    print(f"  Error: X={error_x:.2f}px, Y={error_y:.2f}px, Total={error_total:.2f}px")
    print("-" * 80)

# Test with actual metadata from new folder
metadata_path = Path("out/crops/test-video/5e68c303-de88-44da-ad3e-be0fdc8d5abe_IMG_1409/crops_metadata.json")
if metadata_path.exists():
    print("\n\nTesting with actual metadata from new folder:")
    print("=" * 80)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    for crop in metadata:
        bbox_orig = crop.get('bbox_original', crop.get('bbox', []))
        if len(bbox_orig) != 4:
            continue
        
        calculated = calculate_image_coords_from_bbox_with_config(bbox_orig)
        stored_coords = crop.get('image_coords')
        
        print(f"\n{crop['crop_filename']} (Track {crop['track_id']}):")
        print(f"  BBox original: {bbox_orig}")
        print(f"  Calculated now: ({calculated[0]:.2f}, {calculated[1]:.2f})")
        if stored_coords:
            print(f"  Stored in metadata: ({stored_coords[0]:.2f}, {stored_coords[1]:.2f})")
            diff = ((calculated[0] - stored_coords[0])**2 + (calculated[1] - stored_coords[1])**2)**0.5
            print(f"  Difference: {diff:.2f}px")
        else:
            print(f"  Stored in metadata: NOT FOUND (needs reprocessing)")
        print("-" * 80)



