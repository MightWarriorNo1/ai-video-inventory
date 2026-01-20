"""
Convert camera frame annotations to BEV training format.

This script converts the annotation JSON file into a format suitable
for BEV model training.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple
import argparse
import sys
import os


def convert_annotations_to_bev_format(
    annotations_path: str,
    output_path: str
):
    """
    Convert annotations to BEV training data format.
    
    Args:
        annotations_path: Path to annotations JSON file
        output_path: Path to save BEV training data JSON
    """
    if not os.path.exists(annotations_path):
        print(f"Error: Annotations file not found: {annotations_path}")
        return False
    
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    # Extract all image points and world points
    image_paths = []
    image_points = []
    world_points = []
    gps_points = []
    
    for img_path, points in data['annotations'].items():
        for point_data in points:
            image_paths.append(img_path)
            image_points.append(tuple(point_data['image_point']))
            world_points.append(tuple(point_data['world_point']))
            gps_points.append(tuple(point_data['gps_point']))
    
    if len(image_paths) == 0:
        print("Error: No annotations found in file!")
        return False
    
    # Create BEV training data structure
    bev_data = {
        'image_paths': image_paths,
        'image_points': image_points,
        'world_points': world_points,
        'gps_points': gps_points,
        'gps_reference': data.get('gps_reference'),
        'total_samples': len(image_paths),
        'unique_images': len(set(image_paths)),
        'points_per_image': {
            'min': min(len(points) for points in data['annotations'].values()),
            'max': max(len(points) for points in data['annotations'].values()),
            'avg': sum(len(points) for points in data['annotations'].values()) / len(data['annotations'])
        }
    }
    
    # Calculate world coordinate ranges
    world_x = [p[0] for p in world_points]
    world_y = [p[1] for p in world_points]
    
    bev_data['world_coord_ranges'] = {
        'x_min': float(min(world_x)),
        'x_max': float(max(world_x)),
        'y_min': float(min(world_y)),
        'y_max': float(max(world_y)),
        'x_range': float(max(world_x) - min(world_x)),
        'y_range': float(max(world_y) - min(world_y))
    }
    
    # Save to file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(bev_data, f, indent=2)
    
    print(f"✓ Created BEV training data")
    print(f"  Total samples: {len(image_paths)}")
    print(f"  Unique images: {bev_data['unique_images']}")
    print(f"  Points per image: {bev_data['points_per_image']['min']}-{bev_data['points_per_image']['max']} (avg: {bev_data['points_per_image']['avg']:.1f})")
    print(f"  World X range: [{bev_data['world_coord_ranges']['x_min']:.2f}, {bev_data['world_coord_ranges']['x_max']:.2f}] ({bev_data['world_coord_ranges']['x_range']:.2f}m)")
    print(f"  World Y range: [{bev_data['world_coord_ranges']['y_min']:.2f}, {bev_data['world_coord_ranges']['y_max']:.2f}] ({bev_data['world_coord_ranges']['y_range']:.2f}m)")
    print(f"  Saved to: {output_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert camera frame annotations to BEV training format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python tools/prepare_bev_training_data.py \\
    --annotations bev_training_annotations.json \\
    --output bev_training_data.json
        """
    )
    parser.add_argument('--annotations', required=True, help='Input annotations JSON file')
    parser.add_argument('--output', required=True, help='Output BEV training data JSON file')
    
    args = parser.parse_args()
    
    success = convert_annotations_to_bev_format(args.annotations, args.output)
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())

