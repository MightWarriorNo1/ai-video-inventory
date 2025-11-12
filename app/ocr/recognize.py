"""
TensorRT OCR Recognizer (CRNN/PP-OCR)

This module provides OCR recognition using TensorRT CRNN/PP-OCR engines.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, List
import os

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("Warning: TensorRT or PyCUDA not available. Install: pip install nvidia-tensorrt pycuda")


class PlateRecognizer:
    """
    TensorRT OCR recognizer wrapper.
    
    Supports CRNN/PP-OCR-style CTC-based text recognition.
    """
    
    def __init__(self, engine_path: str, alphabet_path: str, input_size: Tuple[int, int] = (320, 32)):
        """
        Initialize TensorRT OCR recognizer.
        
        Args:
            engine_path: Path to TensorRT OCR engine file (.engine)
            alphabet_path: Path to alphabet.txt file
            input_size: Input image size (width, height) for OCR
        """
        self.engine_path = engine_path
        self.input_size = input_size
        
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT OCR engine not found: {engine_path}")
        
        # Load alphabet
        with open(alphabet_path, 'r') as f:
            self.alphabet = f.read().strip()
        
        # CTC blank is typically at index 0
        self.blank_idx = 0
        
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT or PyCUDA not available. Cannot initialize OCR.")
        
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
        
        print(f"Loaded TensorRT OCR engine: {engine_path}")
        print(f"  Input shape: {self.input_shape}")
        print(f"  Output shape: {self.output_shape}")
        print(f"Alphabet: {self.alphabet} ({len(self.alphabet)} chars)")
    
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
        Preprocess cropped image for OCR input.
        
        Args:
            image: BGR cropped image (H, W, 3)
            
        Returns:
            Preprocessed image tensor (1, 1, H, W) normalized
        """
        # Resize to input size
        img_resized = cv2.resize(image, self.input_size)
        
        # Convert to grayscale
        if len(img_resized.shape) == 3:
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_resized
        
        # Normalize to [0, 1]
        img_normalized = img_gray.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions: (1, 1, H, W)
        img_batched = np.expand_dims(np.expand_dims(img_normalized, axis=0), axis=0)
        
        return img_batched
    
    def infer(self, preprocessed: np.ndarray) -> np.ndarray:
        """
        Run TensorRT OCR inference.
        
        Args:
            preprocessed: Preprocessed image tensor (1, 1, H, W)
            
        Returns:
            Character probability logits: [T, C] where T is sequence length, C is alphabet size
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
        # CRNN/PP-OCR output: [batch, sequence_length, num_classes] or [sequence_length, num_classes]
        if len(output_shape) == 3:
            # [batch, T, C]
            output = output.reshape(output_shape)
            output = output[0]  # Remove batch dimension -> [T, C]
        elif len(output_shape) == 2:
            # [T, C]
            output = output.reshape(output_shape)
        else:
            # Flatten and reshape to [T, C] assuming C = alphabet size
            total_elements = np.prod(output_shape)
            T = total_elements // len(self.alphabet)
            output = output.reshape(T, len(self.alphabet))
        
        # Apply softmax to get probabilities (if output is logits)
        # Many models output logits, so we apply softmax
        # If your model already outputs probabilities, you can skip this
        exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
        probs = exp_output / (np.sum(exp_output, axis=1, keepdims=True) + 1e-8)
        
        return probs.astype(np.float32)
    
    def ctc_decode(self, logits: np.ndarray) -> Tuple[str, float]:
        """
        Decode CTC output to text string using greedy decoding.
        
        Args:
            logits: Character probability logits [T, C] (already softmaxed)
            
        Returns:
            Tuple of (recognized_text, confidence)
        """
        if logits.shape[0] == 0:
            return "", 0.0
        
        # Greedy CTC decoding
        # Find most likely character at each timestep
        char_indices = np.argmax(logits, axis=1)
        probs = np.max(logits, axis=1)
        
        # Remove blanks and duplicates (CTC decoding)
        decoded_chars = []
        decoded_probs = []
        prev_idx = self.blank_idx
        
        for idx, prob in zip(char_indices, probs):
            # Skip blank tokens
            if idx == self.blank_idx:
                prev_idx = idx
                continue
            
            # Skip duplicate consecutive characters (CTC collapse)
            if idx != prev_idx:
                if idx < len(self.alphabet):
                    decoded_chars.append(self.alphabet[idx])
                    decoded_probs.append(prob)
            
            prev_idx = idx
        
        text = ''.join(decoded_chars)
        
        # Average confidence of decoded characters
        if len(decoded_probs) > 0:
            confidence = float(np.mean(decoded_probs))
        else:
            confidence = 0.0
        
        return text, confidence
    
    def recognize(self, image: np.ndarray) -> Dict[str, any]:
        """
        Recognize text in a cropped image region.
        
        Args:
            image: BGR cropped image (H, W, 3)
            
        Returns:
            Dict with keys: text, conf
        """
        preprocessed = self.preprocess(image)
        logits = self.infer(preprocessed)
        text, conf = self.ctc_decode(logits)
        
        return {
            'text': text.strip(),
            'conf': conf
        }

