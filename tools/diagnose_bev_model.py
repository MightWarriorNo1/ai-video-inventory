"""
Diagnose BEV Model - Check checkpoint and normalization parameters.
"""

import torch
import json
import sys
import os
from pathlib import Path

def diagnose_checkpoint(model_path: str):
    """Diagnose BEV model checkpoint."""
    print("=" * 60)
    print("BEV Model Checkpoint Diagnosis")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    
    print(f"\nLoading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print(f"\nCheckpoint keys: {list(checkpoint.keys())}")
    
    # Check normalization parameters
    print("\n" + "-" * 60)
    print("Normalization Parameters:")
    print("-" * 60)
    
    norm_params = None
    if 'normalization_params' in checkpoint:
        norm_params = checkpoint['normalization_params']
        print("Found 'normalization_params' key")
    elif 'normalization' in checkpoint:
        norm_params = checkpoint['normalization']
        print("Found 'normalization' key")
    else:
        print("WARNING: No normalization parameters found!")
        print("This will cause incorrect coordinate predictions.")
    
    if norm_params:
        print(f"  normalize_coords: {norm_params.get('normalize_coords', 'N/A')}")
        print(f"  world_x_min: {norm_params.get('world_x_min', 'N/A')}")
        print(f"  world_x_max: {norm_params.get('world_x_max', 'N/A')}")
        print(f"  world_y_min: {norm_params.get('world_y_min', 'N/A')}")
        print(f"  world_y_max: {norm_params.get('world_y_max', 'N/A')}")
        print(f"  world_x_range: {norm_params.get('world_x_range', 'N/A')}")
        print(f"  world_y_range: {norm_params.get('world_y_range', 'N/A')}")
        
        x_range = norm_params.get('world_x_range', 1.0)
        y_range = norm_params.get('world_y_range', 1.0)
        
        if x_range < 1.0 or y_range < 1.0:
            print("\nWARNING: Normalization ranges are very small!")
            print("This suggests the training data might have issues.")
    
    # Check model info
    print("\n" + "-" * 60)
    print("Model Information:")
    print("-" * 60)
    
    if 'epoch' in checkpoint:
        print(f"  Training epoch: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"  Training loss: {checkpoint['loss']:.6f}")
    if 'input_size' in checkpoint:
        print(f"  Input size: {checkpoint['input_size']}")
    if 'bev_size' in checkpoint:
        print(f"  BEV size: {checkpoint['bev_size']}")
    if 'training_samples' in checkpoint:
        print(f"  Training samples: {checkpoint['training_samples']}")
    if 'unique_images' in checkpoint:
        print(f"  Unique images: {checkpoint['unique_images']}")
    
    # Check GPS reference
    print("\n" + "-" * 60)
    print("GPS Reference:")
    print("-" * 60)
    
    if 'gps_reference' in checkpoint:
        gps_ref = checkpoint['gps_reference']
        print(f"  Latitude: {gps_ref.get('lat', 'N/A')}")
        print(f"  Longitude: {gps_ref.get('lon', 'N/A')}")
    else:
        print("  No GPS reference found")
    
    # Check model state dict
    print("\n" + "-" * 60)
    print("Model State Dict:")
    print("-" * 60)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"  Number of parameters: {len(state_dict)}")
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"  Total parameter count: {total_params:,}")
    else:
        print("  WARNING: No 'model_state_dict' found!")
    
    print("\n" + "=" * 60)
    print("Diagnosis Complete")
    print("=" * 60)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Diagnose BEV model checkpoint')
    parser.add_argument('--model', required=True, help='Path to BEV model checkpoint')
    
    args = parser.parse_args()
    diagnose_checkpoint(args.model)
