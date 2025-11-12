"""
TensorRT YOLO Detector Wrapper

This module provides a wrapper for YOLO detection using TensorRT engines.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple
import os

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("Warning: TensorRT or PyCUDA not available. Install: pip install nvidia-tensorrt pycuda")


class TrtEngineYOLO:
    """
    TensorRT YOLO detector wrapper.
    
    Supports YOLOv5/YOLOv8-style detection models with TensorRT optimization.
    """
    
    def __init__(self, engine_path: str, input_size: Tuple[int, int] = (640, 640), conf_threshold: float = 0.35):
        """
        Initialize TensorRT YOLO detector.
        
        Args:
            engine_path: Path to TensorRT engine file (.engine)
            input_size: Input image size (width, height)
            conf_threshold: Confidence threshold for detections
        """
        self.engine_path = engine_path
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
        
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT or PyCUDA not available. Cannot initialize detector.")
        
        # Initialize TensorRT runtime
        self.trt = trt
        self.cuda = cuda
        
        # Create TensorRT logger
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        # Load engine
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        # Create runtime and deserialize engine
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")
        
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
        
        # Get input/output shapes
        self.input_shape = self.engine.get_binding_shape(0)
        self.output_shape = self.engine.get_binding_shape(1)
        
        print(f"Loaded TensorRT engine: {engine_path}")
        print(f"  Input shape: {self.input_shape}")
        print(f"  Output shape: {self.output_shape}")
    
    def _allocate_buffers(self):
        """Allocate GPU buffers for input and output."""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem, 'shape': self.engine.get_binding_shape(binding)})
            else:
                outputs.append({'host': host_mem, 'device': device_mem, 'shape': self.engine.get_binding_shape(binding)})
        
        return inputs, outputs, bindings, stream
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for YOLO input.
        
        Args:
            image: BGR image (H, W, 3)
            
        Returns:
            Preprocessed image tensor (1, 3, H, W) normalized to [0, 1]
        """
        h, w = self.input_size[1], self.input_size[0]
        img_resized = cv2.resize(image, (w, h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))  # HWC -> CHW
        img_batched = np.expand_dims(img_transposed, axis=0)  # Add batch dimension
        return img_batched
    
    def infer(self, preprocessed: np.ndarray) -> np.ndarray:
        """
        Run TensorRT inference.
        
        Args:
            preprocessed: Preprocessed image tensor (1, 3, H, W)
            
        Returns:
            Detection results: [N, 6] array where each row is [x1, y1, x2, y2, conf, cls]
        """
        # Flatten and copy input to host buffer
        input_data = preprocessed.ravel().astype(np.float32)
        np.copyto(self.inputs[0]['host'], input_data)
        
        # Transfer input data to GPU
        self.cuda.memcpy_htod_async(
            self.inputs[0]['device'],
            self.inputs[0]['host'],
            self.stream
        )
        
        # Run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        
        # Transfer predictions back from GPU
        self.cuda.memcpy_dtoh_async(
            self.outputs[0]['host'],
            self.outputs[0]['device'],
            self.stream
        )
        
        # Synchronize
        self.stream.synchronize()
        
        # Get output and reshape
        output = self.outputs[0]['host']
        output_shape = self.outputs[0]['shape']
        
        # Reshape output based on engine output shape
        # YOLOv8 output: [batch, num_detections, 6] or [1, N, 6] where 6 = [x1, y1, x2, y2, conf, cls]
        # YOLOv5 output: [batch, num_boxes, 85] where 85 = [x, y, w, h, conf, 80 classes]
        
        if len(output_shape) == 3:
            # [batch, N, features]
            output = output.reshape(output_shape)
            output = output[0]  # Remove batch dimension
        elif len(output_shape) == 2:
            # [N, features]
            output = output.reshape(output_shape)
        
        # Handle different output formats
        if output.shape[1] == 6:
            # Already in [x1, y1, x2, y2, conf, cls] format (YOLOv8-style)
            detections = output
        elif output.shape[1] == 85:
            # YOLOv5-style: [x, y, w, h, conf, 80 classes]
            # Convert to [x1, y1, x2, y2, conf, cls] format
            boxes = output[:, :4]  # [x, y, w, h]
            confs = output[:, 4:5]  # confidence
            classes = output[:, 5:].argmax(axis=1, keepdims=True)  # class index
            
            # Convert center+size to corner format
            x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2
            
            # Combine into [x1, y1, x2, y2, conf, cls]
            detections = np.column_stack([x1, y1, x2, y2, confs.flatten(), classes.flatten()])
        else:
            # Unknown format, try to extract first 6 columns
            detections = output[:, :6] if output.shape[1] >= 6 else output
        
        # Apply confidence threshold
        detections = detections[detections[:, 4] >= self.conf_threshold]
        
        # Apply NMS (Non-Maximum Suppression)
        if len(detections) > 0:
            detections = self._nms(detections, iou_threshold=0.45)
        
        return detections.astype(np.float32)
    
    def _nms(self, boxes: np.ndarray, iou_threshold: float = 0.45) -> np.ndarray:
        """
        Apply Non-Maximum Suppression to remove overlapping detections.
        
        Args:
            boxes: Detection boxes [N, 6] with [x1, y1, x2, y2, conf, cls]
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Filtered boxes after NMS
        """
        if len(boxes) == 0:
            return boxes
        
        # Sort by confidence (descending)
        order = boxes[:, 4].argsort()[::-1]
        boxes = boxes[order]
        
        keep = []
        while len(boxes) > 0:
            # Keep the box with highest confidence
            keep.append(boxes[0])
            
            if len(boxes) == 1:
                break
            
            # Compute IoU with remaining boxes
            ious = self._compute_iou_batch(boxes[0:1], boxes[1:])
            
            # Remove boxes with IoU > threshold
            mask = ious < iou_threshold
            boxes = boxes[1:][mask]
        
        return np.array(keep) if keep else np.array([]).reshape(0, 6)
    
    def _compute_iou_batch(self, box1: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Compute IoU between one box and multiple boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1[0, 0], box1[0, 1], box1[0, 2], box1[0, 3]
        x1_2, y1_2, x2_2, y2_2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        
        # Intersection
        x1_i = np.maximum(x1_1, x1_2)
        y1_i = np.maximum(y1_1, y1_2)
        x2_i = np.minimum(x2_1, x2_2)
        y2_i = np.minimum(y2_1, y2_2)
        
        inter_area = np.maximum(0, x2_i - x1_i) * np.maximum(0, y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        # IoU
        iou = inter_area / (union_area + 1e-6)
        return iou
    
    def postprocess(self, detections: np.ndarray, original_shape: Tuple[int, int]) -> List[Dict]:
        """
        Post-process detection results and scale to original image size.
        
        Args:
            detections: Raw detection output [N, 6] with [x1, y1, x2, y2, conf, cls]
            original_shape: Original image shape (height, width)
            
        Returns:
            List of detection dicts with keys: bbox, cls, conf
        """
        results = []
        orig_h, orig_w = original_shape
        scale_x = orig_w / self.input_size[0]
        scale_y = orig_h / self.input_size[1]
        
        for det in detections:
            if det[4] < self.conf_threshold:
                continue
            
            x1, y1, x2, y2 = det[0:4]
            # Scale to original image size
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            # Clip to image bounds
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w - 1))
            y2 = max(0, min(y2, orig_h - 1))
            
            results.append({
                'bbox': [x1, y1, x2, y2],
                'cls': int(det[5]),
                'conf': float(det[4])
            })
        
        return results
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Run detection on an image.
        
        Args:
            image: BGR image (H, W, 3)
            
        Returns:
            List of detection dicts with keys: bbox, cls, conf
        """
        original_shape = image.shape[:2]
        preprocessed = self.preprocess(image)
        raw_detections = self.infer(preprocessed)
        detections = self.postprocess(raw_detections, original_shape)
        return detections

