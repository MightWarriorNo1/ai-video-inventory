"""
Train Deep Learning-Based BEV Transformation Model

This script trains a CNN-based BEV transformation network using calibration data.
The trained model can replace or complement traditional homography for more robust
image-to-world coordinate transformation.
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
import os
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from app.ai.bev_transformer import BEVNet, BEVDataset, BEVTransformer
from app.gps_utils import gps_to_meters


def load_calibration_data(calib_path: str, image_dir: Optional[str] = None) -> dict:
    """
    Load calibration data from JSON file.
    
    Args:
        calib_path: Path to calibration JSON file
        image_dir: Directory containing images (if different from calib_path dir)
    
    Returns:
        Dictionary with calibration data
    """
    calib_path = Path(calib_path)
    
    with open(calib_path, 'r') as f:
        calib_data = json.load(f)
    
    # Determine image directory
    if image_dir is None:
        # Try to use image_path from calibration data
        if 'image_path' in calib_data:
            image_path = Path(calib_data['image_path'])
            if image_path.exists():
                image_dir = image_path.parent
            else:
                image_dir = calib_path.parent
        else:
            image_dir = calib_path.parent
    
    # Prepare data
    image_points = calib_data.get('image_points', [])
    world_points = calib_data.get('world_points', [])
    
    # Convert to proper format
    image_points = [tuple(p) for p in image_points]
    world_points = [tuple(p) for p in world_points]
    
    # Get homography matrix if available (for faster dense mapping)
    homography_matrix = None
    if 'H' in calib_data:
        import numpy as np
        homography_matrix = np.array(calib_data['H'], dtype=np.float32)
    
    # Get image path
    if 'image_path' in calib_data:
        image_path = Path(calib_data['image_path'])
        if not image_path.exists():
            # Try relative to calib file
            image_path = calib_path.parent / image_path.name
    else:
        # Try to find image in image_dir
        image_files = list(Path(image_dir).glob('*.jpg')) + list(Path(image_dir).glob('*.png'))
        if image_files:
            image_path = image_files[0]
        else:
            raise ValueError(f"No image found in {image_dir}")
    
    return {
        'image_path': str(image_path),
        'image_points': image_points,
        'world_points': world_points,
        'gps_reference': calib_data.get('gps_reference'),
        'gps_points': calib_data.get('gps_points', []),
        'homography_matrix': homography_matrix
    }


def create_training_data(
    calib_data: dict,
    num_augmentations: int = 10
) -> Tuple[List[str], List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Create training data from calibration points.
    
    Args:
        calib_data: Calibration data dictionary
        num_augmentations: Number of augmented samples per calibration point
    
    Returns:
        Tuple of (image_paths, image_points, world_points)
    """
    image_path = calib_data['image_path']
    image_points = calib_data['image_points']
    world_points = calib_data['world_points']
    
    if len(image_points) != len(world_points):
        raise ValueError(f"Mismatch: {len(image_points)} image points vs {len(world_points)} world points")
    
    # Base data: use same image for all points (since they're all from same calibration)
    image_paths = []
    augmented_image_points = []
    augmented_world_points = []
    
    for img_pt, world_pt in zip(image_points, world_points):
        # Add original point
        image_paths.append(image_path)
        augmented_image_points.append(img_pt)
        augmented_world_points.append(world_pt)
        
        # Add augmented points with small random variations
        for _ in range(num_augmentations):
            # Small random offset in image space
            x_offset = np.random.normal(0, 2.0)  # ~2 pixel std
            y_offset = np.random.normal(0, 2.0)
            
            aug_x = img_pt[0] + x_offset
            aug_y = img_pt[1] + y_offset
            
            # Corresponding world offset (assume small local linear approximation)
            # This is approximate - in reality the mapping is non-linear
            world_scale_x = np.random.uniform(0.01, 0.02)  # meters per pixel (rough estimate)
            world_scale_y = np.random.uniform(0.01, 0.02)
            
            aug_world_x = world_pt[0] + x_offset * world_scale_x
            aug_world_y = world_pt[1] + y_offset * world_scale_y
            
            image_paths.append(image_path)
            augmented_image_points.append((aug_x, aug_y))
            augmented_world_points.append((aug_world_x, aug_world_y))
    
    return image_paths, augmented_image_points, augmented_world_points


def train_bev_model(
    calib_path: str,
    output_path: str,
    image_dir: Optional[str] = None,
    epochs: int = 100,
    batch_size: int = 4,
    learning_rate: float = 0.001,
    input_size: Tuple[int, int] = (720, 1280),
    bev_size: Tuple[int, int] = (256, 256),
    device: Optional[str] = None,
    num_augmentations: int = 10
):
    """
    Train BEV transformation model.
    
    Args:
        calib_path: Path to calibration JSON file
        output_path: Path to save trained model
        image_dir: Directory containing calibration images
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        input_size: Input image size (height, width)
        bev_size: BEV output size (height, width)
        device: Device to use ('cuda', 'cpu', or None for auto)
        num_augmentations: Number of augmented samples per calibration point
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Load calibration data
    print(f"Loading calibration data from {calib_path}")
    calib_data = load_calibration_data(calib_path, image_dir)
    
    print(f"Found {len(calib_data['image_points'])} calibration points")
    
    # Create training data
    print("Creating training dataset...")
    image_paths, image_points, world_points = create_training_data(
        calib_data,
        num_augmentations=num_augmentations
    )
    
    print(f"Total training samples: {len(image_paths)}")
    
    # Debug: Print original calibration world points
    print(f"\nOriginal calibration world points:")
    for i, wp in enumerate(calib_data['world_points'][:4]):
        print(f"  Point {i+1}: ({wp[0]:.3f}, {wp[1]:.3f})")
    
    # Create dataset
    dataset = BEVDataset(
        image_paths=image_paths,
        image_points=image_points,
        world_points=world_points,
        image_size=input_size,
        bev_size=bev_size,
        normalize_coords=True,
        homography_matrix=calib_data.get('homography_matrix')
    )
    
    # Verify and save normalization parameters
    print(f"\nNormalization parameters:")
    print(f"  X range: [{dataset.world_x_min:.3f}, {dataset.world_x_max:.3f}] (range: {dataset.world_x_range:.3f})")
    print(f"  Y range: [{dataset.world_y_min:.3f}, {dataset.world_y_max:.3f}] (range: {dataset.world_y_range:.3f})")
    
    # Verify normalization is valid
    if dataset.world_x_range < 1e-6 or dataset.world_y_range < 1e-6:
        print("ERROR: Normalization ranges are too small or zero!")
        print("This will cause incorrect coordinate predictions.")
        print("Check your calibration data.")
        print(f"World points sample: {world_points[:5]}")
        raise ValueError("Invalid normalization parameters")
    
    normalization_params = {
        'normalize_coords': True,
        'world_x_min': float(dataset.world_x_min),
        'world_x_max': float(dataset.world_x_max),
        'world_y_min': float(dataset.world_y_min),
        'world_y_max': float(dataset.world_y_max),
        'world_x_range': float(dataset.world_x_range),
        'world_y_range': float(dataset.world_y_range)
    }
    
    print(f"\nSaving normalization parameters to checkpoint:")
    print(f"  {normalization_params}")
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if device.type == 'cuda' else 0
    )
    
    # Initialize model
    print("Initializing BEV model...")
    model = BEVNet(
        input_size=input_size,
        bev_size=bev_size,
        use_coord_regression=True
    ).to(device)
    
    # Loss function: Use MSE loss for better gradient flow
    # MSE tends to work better for coordinate regression as it penalizes large errors more
    # For coordinate regression, MSE often works better than L1
    
    # Option 1: Standard MSE (simpler, often works well)
    use_weighted_loss = False
    
    if use_weighted_loss:
        # Option 2: Weighted loss to emphasize areas with more variation
        def weighted_mse_loss(pred, target):
            # Calculate per-pixel squared error
            squared_error = (pred - target) ** 2  # (B, 2, H, W)
            
            # Calculate spatial variance in target to find areas with more variation
            # Flatten spatial dimensions for variance calculation
            target_flat = target.view(target.shape[0], target.shape[1], -1)  # (B, 2, H*W)
            target_mean = target_flat.mean(dim=2, keepdim=True)  # (B, 2, 1)
            target_var = ((target_flat - target_mean) ** 2).mean(dim=2, keepdim=True)  # (B, 2, 1)
            
            # Weight by inverse of variance - emphasize learning in areas with more variation
            # Add small epsilon to avoid division by zero
            weight = 1.0 / (target_var.view(target.shape[0], target.shape[1], 1, 1) + 0.1)
            
            # Apply weights and average
            weighted_error = (squared_error * weight).mean()
            return weighted_error
        
        criterion = weighted_mse_loss
        print("Using weighted MSE loss to emphasize spatial variation")
    else:
        criterion = nn.MSELoss()
        print("Using standard MSE loss")
    
    # Optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10
    )
    
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, targets, img_points, world_points_batch in progress_bar:
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / num_batches
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if old_lr != new_lr:
            print(f"Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
        
        print(f"Epoch {epoch+1}/{epochs}: Average Loss = {avg_loss:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"New best model! Loss: {best_loss:.6f}")
            
            # Save checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'input_size': input_size,
                'bev_size': bev_size,
                'normalization': normalization_params
            }
            
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, output_path)
            print(f"Saved model to {output_path}")
    
    print(f"\nTraining complete! Best loss: {best_loss:.6f}")
    print(f"Model saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train BEV Transformation Model')
    parser.add_argument('--calib', required=True, help='Path to calibration JSON file')
    parser.add_argument('--output', required=True, help='Output model checkpoint path')
    parser.add_argument('--image-dir', help='Directory containing calibration images')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--input-size', type=str, default='720,1280', help='Input size as H,W')
    parser.add_argument('--bev-size', type=str, default='256,256', help='BEV size as H,W')
    parser.add_argument('--device', help='Device (cuda/cpu), auto-detect if not specified')
    parser.add_argument('--augmentations', type=int, default=10, help='Augmentations per point')
    
    args = parser.parse_args()
    
    # Parse sizes
    input_size = tuple(map(int, args.input_size.split(',')))
    bev_size = tuple(map(int, args.bev_size.split(',')))
    
    try:
        train_bev_model(
            calib_path=args.calib,
            output_path=args.output,
            image_dir=args.image_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            input_size=input_size,
            bev_size=bev_size,
            device=args.device,
            num_augmentations=args.augmentations
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())







