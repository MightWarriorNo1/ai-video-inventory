"""
Train BEV model using camera frame annotations.

This script trains a BEV (Bird's Eye View) transformation model using
annotated camera frames with GPS coordinates.
"""

import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import os
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from app.ai.bev_transformer import BEVNet, BEVDataset


def load_bev_training_data(data_path: str):
    """Load BEV training data from JSON."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    return {
        'image_paths': data['image_paths'],
        'image_points': [tuple(p) for p in data['image_points']],
        'world_points': [tuple(p) for p in data['world_points']],
        'gps_reference': data.get('gps_reference')
    }


def train_bev_from_camera_frames(
    training_data_path: str,
    output_path: str,
    epochs: int = 100,
    batch_size: int = 4,
    learning_rate: float = 0.001,
    input_size: tuple = (720, 1280),
    bev_size: tuple = (256, 256),
    num_augmentations: int = 5,
    device: str = None
):
    """
    Train BEV model from camera frame annotations.
    
    Args:
        training_data_path: Path to BEV training data JSON file
        output_path: Path to save trained model checkpoint
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        input_size: Input image size (height, width)
        bev_size: BEV output size (height, width)
        num_augmentations: Number of augmented samples per point
        device: Device to use ('cuda', 'cpu', or None for auto)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"{'='*60}")
    print("BEV Model Training")
    print(f"{'='*60}")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Load training data
    print(f"\nLoading training data from {training_data_path}")
    if not os.path.exists(training_data_path):
        raise FileNotFoundError(f"Training data file not found: {training_data_path}")
    
    data = load_bev_training_data(training_data_path)
    
    print(f"Found {len(data['image_paths'])} training samples")
    print(f"From {len(set(data['image_paths']))} unique images")
    
    if len(data['image_paths']) < 4:
        raise ValueError(f"Need at least 4 training samples, got {len(data['image_paths'])}")
    
    # Create augmented dataset
    print(f"\nCreating augmented dataset (augmentations per point: {num_augmentations})...")
    augmented_image_paths = []
    augmented_image_points = []
    augmented_world_points = []
    
    for img_path, img_pt, world_pt in zip(
        data['image_paths'],
        data['image_points'],
        data['world_points']
    ):
        # Original point
        augmented_image_paths.append(img_path)
        augmented_image_points.append(img_pt)
        augmented_world_points.append(world_pt)
        
        # Augmented points with small random variations
        for _ in range(num_augmentations):
            # Small random offset in image space
            x_offset = np.random.normal(0, 3.0)  # ~3 pixel std
            y_offset = np.random.normal(0, 3.0)
            
            aug_x = max(0, min(img_pt[0] + x_offset, input_size[1] - 1))
            aug_y = max(0, min(img_pt[1] + y_offset, input_size[0] - 1))
            
            # Approximate world offset (assume small local linear approximation)
            # This is approximate - in reality the mapping is non-linear
            world_scale = np.random.uniform(0.01, 0.02)  # meters per pixel (rough estimate)
            aug_world_x = world_pt[0] + x_offset * world_scale
            aug_world_y = world_pt[1] + y_offset * world_scale
            
            augmented_image_paths.append(img_path)
            augmented_image_points.append((aug_x, aug_y))
            augmented_world_points.append((aug_world_x, aug_world_y))
    
    print(f"Total training samples (with augmentation): {len(augmented_image_paths)}")
    
    # Resolve image paths to absolute paths
    print("\nResolving image paths...")
    resolved_image_paths = []
    project_root = Path(__file__).parent.parent
    for img_path in augmented_image_paths:
        path_obj = Path(img_path)
        if not path_obj.is_absolute():
            # Try relative to project root
            resolved = project_root / img_path
            if resolved.exists():
                resolved_image_paths.append(str(resolved))
            else:
                # Try as absolute path
                if path_obj.exists():
                    resolved_image_paths.append(str(path_obj.absolute()))
                else:
                    print(f"Warning: Image not found: {img_path}, skipping...")
                    continue
        else:
            if path_obj.exists():
                resolved_image_paths.append(str(path_obj))
            else:
                print(f"Warning: Image not found: {img_path}, skipping...")
                continue
    
    # Filter out corresponding points for missing images
    if len(resolved_image_paths) < len(augmented_image_paths):
        print(f"Warning: {len(augmented_image_paths) - len(resolved_image_paths)} images not found, filtering...")
        # Rebuild lists with only valid paths
        valid_indices = []
        for i, img_path in enumerate(augmented_image_paths):
            path_obj = Path(img_path)
            if not path_obj.is_absolute():
                resolved = project_root / img_path
                if resolved.exists():
                    valid_indices.append(i)
            else:
                if path_obj.exists():
                    valid_indices.append(i)
        
        resolved_image_paths = [resolved_image_paths[i] for i in range(len(resolved_image_paths)) if i < len(valid_indices)]
        augmented_image_points = [augmented_image_points[i] for i in valid_indices]
        augmented_world_points = [augmented_world_points[i] for i in valid_indices]
    
    if len(resolved_image_paths) == 0:
        raise ValueError("No valid images found! Check image paths in training data.")
    
    print(f"Resolved {len(resolved_image_paths)} image paths")
    
    # Create dataset
    print("\nCreating BEV dataset...")
    dataset = BEVDataset(
        image_paths=resolved_image_paths,
        image_points=augmented_image_points,
        world_points=augmented_world_points,
        image_size=input_size,
        bev_size=bev_size,
        normalize_coords=True
    )
    
    # Verify normalization parameters
    print(f"\nNormalization parameters:")
    print(f"  X range: [{dataset.world_x_min:.3f}, {dataset.world_x_max:.3f}] (range: {dataset.world_x_range:.3f}m)")
    print(f"  Y range: [{dataset.world_y_min:.3f}, {dataset.world_y_max:.3f}] (range: {dataset.world_y_range:.3f}m)")
    
    if dataset.world_x_range < 1e-6 or dataset.world_y_range < 1e-6:
        raise ValueError("Normalization ranges are too small! Check your training data.")
    
    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Model
    print("\nInitializing BEV model...")
    model = BEVNet(
        input_size=input_size,
        bev_size=bev_size,
        use_coord_regression=True
    ).to(device)
    
    # Fix in-place ReLU operations to avoid gradient computation errors
    # Replace all ReLU(inplace=True) with ReLU(inplace=False)
    def replace_inplace_relu(module):
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU) and child.inplace:
                # Replace with non-inplace ReLU
                setattr(module, name, nn.ReLU(inplace=False))
            else:
                replace_inplace_relu(child)
    
    replace_inplace_relu(model)
    print("  Fixed in-place ReLU operations to prevent gradient errors")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training for {epochs} epochs...")
    print(f"{'='*60}")
    best_loss = float('inf')
    best_epoch = 0
    
    normalization_params = {
        'normalize_coords': True,
        'world_x_min': float(dataset.world_x_min),
        'world_x_max': float(dataset.world_x_max),
        'world_y_min': float(dataset.world_y_min),
        'world_y_max': float(dataset.world_y_max),
        'world_x_range': float(dataset.world_x_range),
        'world_y_range': float(dataset.world_y_range)
    }
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, targets, img_points, world_points_batch in progress_bar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Zero gradients before forward pass
            optimizer.zero_grad()
            
            # Forward pass
            # Note: The model uses ReLU(inplace=True) which can cause issues
            # We ensure clean forward/backward passes
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Backward pass with error checking
            try:
                loss.backward()
            except RuntimeError as e:
                if "inplace" in str(e).lower():
                    print(f"Warning: In-place operation error, skipping batch. Error: {e}")
                    optimizer.zero_grad()  # Clear gradients
                    continue
                else:
                    raise
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = epoch_loss / num_batches
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch+1}/{epochs}: Average Loss = {avg_loss:.6f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss,
                'input_size': input_size,
                'bev_size': bev_size,
                'normalization': normalization_params,
                'gps_reference': data.get('gps_reference'),
                'training_samples': len(augmented_image_paths),
                'unique_images': len(set(data['image_paths']))
            }
            
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, output_path)
            print(f"  ✓ Saved best model (loss: {best_loss:.6f}) to {output_path}")
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"  Best loss: {best_loss:.6f} (epoch {best_epoch})")
    print(f"  Model saved to: {output_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Train BEV model from camera frame annotations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python tools/train_bev_from_camera_frames.py \\
    --data bev_training_data.json \\
    --output models/bev/lifecam-hd6000-01_bev.pth \\
    --epochs 100 \\
    --batch-size 4 \\
    --lr 0.001 \\
    --input-size 720,1280 \\
    --bev-size 256,256 \\
    --augmentations 5 \\
    --device cuda
        """
    )
    parser.add_argument('--data', required=True, help='BEV training data JSON file')
    parser.add_argument('--output', required=True, help='Output model checkpoint path')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--input-size', type=str, default='720,1280', help='Input size as H,W')
    parser.add_argument('--bev-size', type=str, default='256,256', help='BEV size as H,W')
    parser.add_argument('--augmentations', type=int, default=5, help='Augmentations per point')
    parser.add_argument('--device', help='Device (cuda/cpu), auto-detect if not specified')
    
    args = parser.parse_args()
    
    # Parse sizes
    try:
        input_size = tuple(map(int, args.input_size.split(',')))
        bev_size = tuple(map(int, args.bev_size.split(',')))
    except ValueError:
        print("Error: Invalid size format. Use 'H,W' (e.g., '720,1280')")
        return 1
    
    try:
        train_bev_from_camera_frames(
            training_data_path=args.data,
            output_path=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            input_size=input_size,
            bev_size=bev_size,
            num_augmentations=args.augmentations,
            device=args.device
        )
        return 0
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

