"""
YOLOv8 Detector Wrapper

This module provides a wrapper for YOLOv8 detection using Ultralytics YOLOv8 models.
Uses pre-trained YOLOv8m model from COCO dataset, filtered to only detect trucks (class 7).
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple
import os

# Import check will be done at runtime in __init__ to handle user site-packages
ULTRALYTICS_AVAILABLE = None  # Will be checked at runtime


class YOLOv8Detector:
    """
    YOLOv8 detector wrapper using Ultralytics.
    
    Uses pre-trained YOLOv8m model from COCO dataset.
    Filters detections to only return trucks (class 7).
    """
    
    def __init__(self, model_name: str = "yolov8m.pt", conf_threshold: float = 0.25, 
                 target_class: int = 7, device: str = None):
        """
        Initialize YOLOv8 detector.
        
        Args:
            model_name: YOLOv8 model name (e.g., 'yolov8m.pt') or path to model file
            conf_threshold: Confidence threshold for detections
            target_class: COCO class ID to detect (7 = truck)
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.target_class = target_class  # 7 = truck in COCO
        self.input_size = (640, 640)  # YOLOv8 default input size
        
        # Try to import ultralytics at runtime (handles user site-packages)
        import sys
        import os
        import importlib
        
        # First, try normal import
        try:
            from ultralytics import YOLO
            self.YOLO = YOLO
        except ImportError as e:
            first_error = e
            
            # Try to reload importlib and clear any cached imports
            try:
                importlib.invalidate_caches()
            except:
                pass
            
            # Check if ultralytics directory exists in any path
            import site
            user_site = site.getusersitepackages()
            found_paths = []
            
            # Check all paths in sys.path
            for path in sys.path:
                if os.path.exists(os.path.join(path, 'ultralytics')):
                    found_paths.append(path)
            
            # Also check user site-packages explicitly
            if user_site and os.path.exists(os.path.join(user_site, 'ultralytics')):
                if user_site not in found_paths:
                    found_paths.append(user_site)
            
            # Try common locations
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            common_paths = [
                os.path.expanduser(f'~/.local/lib/python{python_version}/site-packages'),
                os.path.expanduser('~/.local/lib/python3/site-packages'),
            ]
            
            for path in common_paths:
                if os.path.exists(path) and os.path.exists(os.path.join(path, 'ultralytics')):
                    if path not in found_paths:
                        found_paths.append(path)
            
            # Try importing from each found path
            for path in found_paths:
                if path not in sys.path:
                    sys.path.insert(0, path)
                
                try:
                    # Clear module cache for ultralytics
                    modules_to_remove = [k for k in sys.modules.keys() if k.startswith('ultralytics')]
                    for mod in modules_to_remove:
                        del sys.modules[mod]
                    
                    from ultralytics import YOLO
                    self.YOLO = YOLO
                    print(f"Note: Successfully imported ultralytics from: {path}")
                    break
                except ImportError as import_err:
                    # Store the actual error for diagnostics
                    if not hasattr(self, '_last_import_error'):
                        self._last_import_error = str(import_err)
                    continue
                except Exception as other_err:
                    # Store other errors too
                    if not hasattr(self, '_last_import_error'):
                        self._last_import_error = f"{type(other_err).__name__}: {other_err}"
                    continue
            
            # If still not available, try using importlib directly
            if not hasattr(self, 'YOLO'):
                import importlib.util
                for path in found_paths:
                    ultralytics_init = os.path.join(path, "ultralytics", "__init__.py")
                    if os.path.exists(ultralytics_init):
                        try:
                            spec = importlib.util.spec_from_file_location(
                                "ultralytics",
                                ultralytics_init
                            )
                            if spec and spec.loader:
                                module = importlib.util.module_from_spec(spec)
                                sys.modules['ultralytics'] = module
                                spec.loader.exec_module(module)
                                self.YOLO = module.YOLO
                                print(f"Note: Loaded ultralytics using importlib from: {path}")
                                break
                        except Exception as e:
                            # Try to get more info about the error
                            if 'YOLO' not in dir(self):
                                continue
            
            # If still not available, raise error with helpful message
            if not hasattr(self, 'YOLO'):
                import importlib.util
                error_msg = "Ultralytics not available. Install: pip install ultralytics\n"
                error_msg += f"Original import error: {first_error}\n"
                
                # Add last import error if we have one
                if hasattr(self, '_last_import_error'):
                    error_msg += f"Last import attempt error: {self._last_import_error}\n"
                
                error_msg += f"\nPython version: {sys.version}\n"
                error_msg += f"Python executable: {sys.executable}\n"
                
                if found_paths:
                    error_msg += f"\nFound ultralytics in these locations:\n"
                    for path in found_paths:
                        # Check if __init__.py exists
                        init_file = os.path.join(path, "ultralytics", "__init__.py")
                        exists = os.path.exists(init_file)
                        error_msg += f"  - {path} {'(has __init__.py)' if exists else '(missing __init__.py)'}\n"
                    
                    error_msg += f"\nBut import still failed. This may indicate:\n"
                    error_msg += f"  1. Missing dependencies - try: pip install torch torchvision\n"
                    error_msg += f"  2. Corrupted installation - try: pip uninstall ultralytics && pip install ultralytics\n"
                    error_msg += f"  3. Editable install issue - try: pip uninstall ultralytics && pip install --no-editable ultralytics\n"
                    error_msg += f"  4. Check full error above for missing module names\n"
                    
                    # Try to diagnose the actual issue
                    try:
                        # Check if we can at least import the package directory
                        test_path = found_paths[0]
                        test_init = os.path.join(test_path, "ultralytics", "__init__.py")
                        if os.path.exists(test_init):
                            with open(test_init, 'r') as f:
                                first_lines = f.read(500)
                                error_msg += f"\nDebug: First 500 chars of __init__.py:\n{first_lines[:200]}...\n"
                    except:
                        pass
                else:
                    error_msg += f"\nPython path: {sys.path}\n"
                    error_msg += f"User site-packages: {user_site}\n"
                    error_msg += f"\nUltralytics not found in any path. Try:\n"
                    error_msg += f"  pip3 install --user ultralytics\n"
                    error_msg += f"  or\n"
                    error_msg += f"  python3 -m pip install --user ultralytics\n"
                
                raise RuntimeError(error_msg)
        
        # Initialize YOLOv8 model
        # This will download the model if not present
        print(f"Loading YOLOv8 model: {model_name}")
        try:
            self.model = self.YOLO(model_name)
            print(f"✓ YOLOv8 model loaded: {model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLOv8 model {model_name}: {e}")
        
        # Set device
        if device is None:
            # Auto-detect device
            try:
                import torch
                if torch.cuda.is_available():
                    device = 'cuda'
                else:
                    device = 'cpu'
            except ImportError:
                device = 'cpu'
        
        self.device = device
        print(f"Using device: {device}")
        
        # Get class names from model (works for both COCO and fine-tuned models)
        try:
            # YOLOv8 models store class names in model.names
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = self.model.names
                # Convert to dict if it's a list
                if isinstance(self.class_names, list):
                    self.class_names = {i: name for i, name in enumerate(self.class_names)}
            else:
                # Fallback to COCO class names
                self.class_names = {
                    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
                    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
                }
        except Exception:
            # Fallback to COCO class names if model doesn't have names
            self.class_names = {
                0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
                5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            }
        
        # Display target class with actual class name from model
        class_name = self.class_names.get(target_class, f'class_{target_class}')
        print(f"Target class: {target_class} ({class_name})")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for YOLOv8 input.
        
        Note: YOLOv8 handles preprocessing internally, but we keep this
        for interface compatibility.
        
        Args:
            image: BGR image (H, W, 3)
            
        Returns:
            Same image (YOLOv8 handles preprocessing)
        """
        return image
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Run YOLOv8 detection on an image.
        
        Args:
            image: BGR image (H, W, 3)
            
        Returns:
            List of detection dicts with keys: bbox, cls, conf
            Only includes detections for target_class (truck = 7)
        """
        if image is None or image.size == 0:
            return []
        
        # Run YOLOv8 inference
        # YOLOv8 expects RGB images, but cv2 images are BGR
        # The model will handle conversion internally
        results = self.model(
            image,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False  # Disable verbose output
        )
        
        detections = []
        
        # Process results
        # YOLOv8 returns Results object with boxes
        if len(results) > 0:
            result = results[0]  # First (and usually only) result
            
            # Get boxes (xyxy format: x1, y1, x2, y2)
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                # Extract data
                boxes_xyxy = boxes.xyxy.cpu().numpy()  # [N, 4] in xyxy format
                confidences = boxes.conf.cpu().numpy()  # [N]
                class_ids = boxes.cls.cpu().numpy().astype(int)  # [N]
                
                # Filter to only target class (truck = 7)
                mask = class_ids == self.target_class
                boxes_xyxy = boxes_xyxy[mask]
                confidences = confidences[mask]
                class_ids = class_ids[mask]
                
                # Convert to detection dicts
                for i in range(len(boxes_xyxy)):
                    x1, y1, x2, y2 = boxes_xyxy[i]
                    
                    # Clip to image bounds
                    h, w = image.shape[:2]
                    x1 = max(0, min(int(x1), w - 1))
                    y1 = max(0, min(int(y1), h - 1))
                    x2 = max(0, min(int(x2), w - 1))
                    y2 = max(0, min(int(y2), h - 1))
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'cls': int(class_ids[i]),
                        'conf': float(confidences[i])
                    })
        
        return detections

