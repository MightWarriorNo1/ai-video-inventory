# test_depth_estimation.py
import sys
sys.path.insert(0, '.')

try:
    from app.depth_estimation import DepthEstimator
    import numpy as np
    import cv2
    
    print("Testing depth estimator initialization...")
    
    # Try MiDaS first
    try:
        estimator = DepthEstimator(model_type="midas")
        print("✓ MiDaS model loaded successfully")
        
        # Test with a dummy image
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        depth_map = estimator.estimate_depth(dummy_img)
        print(f"✓ Depth estimation works! Depth map shape: {depth_map.shape}")
        
    except Exception as e:
        print(f"✗ MiDaS failed: {e}")
        print("Trying Depth Anything...")
        
        estimator = DepthEstimator(model_type="depth_anything")
        print("✓ Depth Anything model loaded successfully")
        
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure PyTorch is installed!")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()