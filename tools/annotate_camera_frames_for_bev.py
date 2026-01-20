"""
Annotate camera frames with GPS coordinates for BEV training.

This tool helps you click points on camera frames and assign GPS coordinates
from your satellite calibration file.
"""

import cv2
import json
import numpy as np
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from app.gps_utils import gps_to_meters, validate_gps_coordinate


class CameraFrameAnnotator:
    def __init__(self, calib_path: str, output_path: str):
        """Initialize annotator with satellite calibration."""
        with open(calib_path, 'r') as f:
            self.calib_data = json.load(f)
        
        self.gps_ref = self.calib_data['gps_reference']
        self.ref_lat = self.gps_ref['lat']
        self.ref_lon = self.gps_ref['lon']
        
        # Store annotations: {image_path: [(x, y, lat, lon), ...]}
        self.annotations = {}
        self.output_path = output_path
        
        # Load existing annotations if file exists
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                existing_data = json.load(f)
                self.annotations = existing_data.get('annotations', {})
                print(f"Loaded existing annotations: {len(self.annotations)} images")
    
    def annotate_image(self, image_path: str):
        """Annotate a single camera frame."""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return
        
        if image_path not in self.annotations:
            self.annotations[image_path] = []
        
        points = []
        window_name = f"Annotate: {Path(image_path).name} - Click points, press 'q' when done"
        
        # Create a copy for drawing
        display_image = image.copy()
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f"\nPoint clicked: ({x}, {y})")
                print("Enter GPS coordinates for this point:")
                print("  - Use same GPS coords from satellite calibration")
                print("  - Or right-click on Google Maps to get coordinates")
                print("  - Format: 'lat, lon' or enter separately")
                
                try:
                    user_input = input("  GPS (lat, lon): ").strip()
                    
                    # Try to parse as comma-separated
                    if ',' in user_input:
                        parts = user_input.split(',')
                        lat = float(parts[0].strip())
                        lon = float(parts[1].strip())
                    else:
                        # Try as single latitude value
                        lat = float(user_input)
                        lon = float(input("  Longitude: "))
                    
                    if validate_gps_coordinate(lat, lon):
                        points.append((x, y, lat, lon))
                        cv2.circle(display_image, (x, y), 5, (0, 255, 0), -1)
                        cv2.putText(display_image, f"{len(points)}", (x+10, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.imshow(window_name, display_image)
                        print(f"  ✓ Added point {len(points)}: ({lat:.6f}, {lon:.6f})")
                    else:
                        print("  ✗ Invalid GPS coordinates! Latitude must be -90 to 90, Longitude -180 to 180")
                except ValueError:
                    print("  ✗ Invalid input! Please enter numbers.")
        
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)
        cv2.imshow(window_name, display_image)
        
        print(f"\n{'='*60}")
        print(f"Annotating: {Path(image_path).name}")
        print(f"{'='*60}")
        print("Instructions:")
        print("1. Click on landmark points visible in this camera frame")
        print("2. Enter GPS coordinates (same points from satellite calibration)")
        print("3. Press 'q' when done")
        print(f"\nGPS Reference: ({self.ref_lat:.6f}, {self.ref_lon:.6f})")
        print(f"Already annotated: {len(self.annotations[image_path])} points")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        
        # Convert GPS to meters and store
        for x, y, lat, lon in points:
            x_meters, y_meters = gps_to_meters(lat, lon, self.ref_lat, self.ref_lon)
            self.annotations[image_path].append({
                'image_point': [float(x), float(y)],
                'gps_point': [float(lat), float(lon)],
                'world_point': [float(x_meters), float(y_meters)]
            })
        
        print(f"\n✓ Added {len(points)} points for {Path(image_path).name}")
        print(f"  Total points for this image: {len(self.annotations[image_path])}")
    
    def save(self):
        """Save annotations to JSON file."""
        output = {
            'gps_reference': self.gps_ref,
            'annotations': self.annotations,
            'total_images': len(self.annotations),
            'total_points': sum(len(points) for points in self.annotations.values())
        }
        
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Annotations saved to: {self.output_path}")
        print(f"  Total: {output['total_images']} images, {output['total_points']} points")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Annotate camera frames for BEV training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Annotate training images
  python tools/annotate_camera_frames_for_bev.py \\
    --calib config/calib/lifecam-hd6000-01_h.json \\
    --images yolo_trailer_dataset/images/train \\
    --output bev_training_annotations.json \\
    --num-images 20
  
  # Continue annotating (adds to existing file)
  python tools/annotate_camera_frames_for_bev.py \\
    --calib config/calib/lifecam-hd6000-01_h.json \\
    --images yolo_trailer_dataset/images/val \\
    --output bev_training_annotations.json \\
    --num-images 5
        """
    )
    parser.add_argument('--calib', required=True, help='Satellite calibration JSON file')
    parser.add_argument('--images', required=True, help='Directory with camera frames')
    parser.add_argument('--output', required=True, help='Output annotations JSON file')
    parser.add_argument('--num-images', type=int, default=10, help='Number of images to annotate')
    
    args = parser.parse_args()
    
    # Validate calibration file exists
    if not os.path.exists(args.calib):
        print(f"Error: Calibration file not found: {args.calib}")
        return 1
    
    # Validate images directory exists
    image_dir = Path(args.images)
    if not image_dir.exists():
        print(f"Error: Images directory not found: {args.images}")
        return 1
    
    annotator = CameraFrameAnnotator(args.calib, args.output)
    
    # Get list of images
    image_files = sorted(list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')))
    
    if len(image_files) == 0:
        print(f"Error: No images found in {args.images}")
        return 1
    
    print(f"Found {len(image_files)} images")
    print(f"Will annotate {min(args.num_images, len(image_files))} images")
    print(f"Output file: {args.output}")
    print()
    
    annotated_count = 0
    skipped_count = 0
    
    for i, img_path in enumerate(image_files[:args.num_images]):
        img_path_str = str(img_path)
        
        # Check if already has enough points (skip if has 4+ points)
        if img_path_str in annotator.annotations:
            existing_points = len(annotator.annotations[img_path_str])
            if existing_points >= 4:
                print(f"\n[{i+1}/{args.num_images}] Skipping {img_path.name} (already has {existing_points} points)")
                skipped_count += 1
                continue
        
        print(f"\n[{i+1}/{args.num_images}] Processing: {img_path.name}")
        annotator.annotate_image(img_path_str)
        annotator.save()  # Save after each image
        annotated_count += 1
    
    print(f"\n{'='*60}")
    print("Annotation session complete!")
    print(f"  Annotated: {annotated_count} images")
    print(f"  Skipped: {skipped_count} images (already annotated)")
    print(f"  Total images in file: {len(annotator.annotations)}")
    print(f"  Total points: {sum(len(points) for points in annotator.annotations.values())}")
    print(f"{'='*60}")
    
    return 0


if __name__ == '__main__':
    exit(main())

