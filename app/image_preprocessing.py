"""
Image Preprocessing Module

Provides preprocessing functions to enhance YOLO detection and OCR recognition accuracy.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


class ImagePreprocessor:
    """
    Image preprocessing for YOLO detection and OCR recognition.
    """
    
    def __init__(self, 
                 enable_yolo_preprocessing: bool = True,
                 enable_ocr_preprocessing: bool = True,
                 yolo_strategy: str = "enhanced",
                 ocr_strategy: str = "morphological"):
        """
        Initialize preprocessor.
        
        Args:
            enable_yolo_preprocessing: Enable preprocessing for YOLO detection
            enable_ocr_preprocessing: Enable preprocessing for OCR recognition
            yolo_strategy: YOLO preprocessing strategy ("none", "basic", "enhanced")
            ocr_strategy: OCR preprocessing strategy ("none", "morphological", "clahe", "adaptive", "multi")
        """
        self.enable_yolo_preprocessing = enable_yolo_preprocessing
        self.enable_ocr_preprocessing = enable_ocr_preprocessing
        self.yolo_strategy = yolo_strategy
        self.ocr_strategy = ocr_strategy
    
    def preprocess_for_yolo(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for YOLO detection.
        
        Args:
            image: BGR image (H, W, 3)
            
        Returns:
            Preprocessed BGR image
        """
        if not self.enable_yolo_preprocessing or self.yolo_strategy == "none":
            return image
        
        if self.yolo_strategy == "basic":
            return self._yolo_basic(image)
        elif self.yolo_strategy == "enhanced":
            return self._yolo_enhanced(image)
        else:
            return image
    
    def _yolo_basic(self, image: np.ndarray) -> np.ndarray:
        """Basic YOLO preprocessing: denoising and contrast."""
        # Light denoising
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)
        
        # Slight contrast enhancement
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _yolo_enhanced(self, image: np.ndarray) -> np.ndarray:
        """Enhanced YOLO preprocessing: denoising, contrast, and sharpening."""
        # Denoising
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
        
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel (lightness)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge and convert back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Light sharpening to enhance edges
        kernel_sharp = np.array([[-0.5, -0.5, -0.5],
                                 [-0.5,  5.0, -0.5],
                                 [-0.5, -0.5, -0.5]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharp)
        
        # Blend original and sharpened (70% sharpened, 30% original)
        result = cv2.addWeighted(sharpened, 0.7, enhanced, 0.3, 0)
        
        return result
    
    def preprocess_for_ocr(self, image: np.ndarray) -> List[Dict[str, any]]:
        """
        Preprocess cropped image for OCR recognition.
        Returns multiple preprocessed versions for best result selection.
        Includes support for vertical text detection.
        
        Args:
            image: BGR cropped image (H, W, 3)
            
        Returns:
            List of dicts with keys: 'image', 'method', 'priority'
        """
        if not self.enable_ocr_preprocessing or self.ocr_strategy == "none":
            return [{'image': image, 'method': 'original', 'priority': 1}]
        
        results = []
        
        # Always include original as baseline
        results.append({'image': image, 'method': 'original', 'priority': 1})
        
        # Check if text might be vertical (tall and narrow)
        h, w = image.shape[:2]
        aspect_ratio = h / w if w > 0 else 1.0
        
        # If aspect ratio > 1.5 (tall and narrow), likely vertical text
        # Try rotated versions
        if aspect_ratio > 1.3:
            # Rotate 90 degrees clockwise (to make vertical text horizontal)
            rotated_cw = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            results.append({'image': rotated_cw, 'method': 'rotated-90cw', 'priority': 2})
            
            # Rotate 90 degrees counter-clockwise (alternative orientation)
            rotated_ccw = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            results.append({'image': rotated_ccw, 'method': 'rotated-90ccw', 'priority': 2})
        
        # Also try 180 degrees rotation (upside down text)
        rotated_180 = cv2.rotate(image, cv2.ROTATE_180)
        results.append({'image': rotated_180, 'method': 'rotated-180', 'priority': 1})
        
        if self.ocr_strategy == "morphological":
            results.extend(self._ocr_morphological(image))
        elif self.ocr_strategy == "clahe":
            results.extend(self._ocr_clahe(image))
        elif self.ocr_strategy == "adaptive":
            results.extend(self._ocr_adaptive(image))
        elif self.ocr_strategy == "multi":
            # Use all strategies and pick best
            results.extend(self._ocr_morphological(image))
            results.extend(self._ocr_clahe(image))
            results.extend(self._ocr_adaptive(image))
            
            # Also apply preprocessing to rotated versions if vertical text detected
            if aspect_ratio > 1.3:
                # Apply preprocessing to rotated versions
                rotated_cw = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                rotated_ccw = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # Morphological on rotated
                morph_cw = self._ocr_morphological(rotated_cw)
                for r in morph_cw:
                    r['method'] = f"rotated-90cw-{r['method']}"
                results.extend(morph_cw)
                
                morph_ccw = self._ocr_morphological(rotated_ccw)
                for r in morph_ccw:
                    r['method'] = f"rotated-90ccw-{r['method']}"
                results.extend(morph_ccw)
        
        return results
    
    def _ocr_morphological(self, image: np.ndarray) -> List[Dict[str, any]]:
        """Morphological preprocessing: good for touching characters and noise."""
        results = []
        
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            h, w = gray.shape
            
            # Upscale if too small
            if h < 48:
                scale = 2.5
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            # Bilateral filter for edge-preserving denoising
            smooth = cv2.bilateralFilter(gray, 9, 40, 40)
            
            # Otsu's threshold
            _, thresh = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological opening (removes noise)
            kernel_small = np.ones((2, 2), np.uint8)
            morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small)
            
            # Morphological closing (fills holes)
            kernel_med = np.ones((2, 3), np.uint8)
            morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel_med)
            
            # Normal version
            morph_bgr = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
            results.append({'image': morph_bgr, 'method': 'morphological', 'priority': 2})
            
            # Inverted version
            morph_inv = cv2.bitwise_not(morph)
            morph_inv_bgr = cv2.cvtColor(morph_inv, cv2.COLOR_GRAY2BGR)
            results.append({'image': morph_inv_bgr, 'method': 'morphological-inv', 'priority': 2})
            
        except Exception as e:
            print(f"[ImagePreprocessor] Morphological preprocessing failed: {e}")
        
        return results
    
    def _ocr_clahe(self, image: np.ndarray) -> List[Dict[str, any]]:
        """CLAHE preprocessing: good for low-contrast text."""
        results = []
        
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            h, w = gray.shape
            
            # Upscale if needed
            if h < 48:
                scale = 2.5
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            # Remove border noise (10%)
            h, w = gray.shape
            border_h = max(1, int(h * 0.1))
            border_w = max(1, int(w * 0.1))
            if h > border_h * 2 and w > border_w * 2:
                gray = gray[border_h:-border_h, border_w:-border_w]
            
            # Light denoising
            smooth = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # CLAHE for local contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            clahe_img = clahe.apply(smooth)
            
            # Light sharpening
            kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharp = cv2.filter2D(clahe_img, -1, kernel_sharp)
            
            clahe_bgr = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
            results.append({'image': clahe_bgr, 'method': 'clahe-sharp', 'priority': 3})
            
        except Exception as e:
            print(f"[ImagePreprocessor] CLAHE preprocessing failed: {e}")
        
        return results
    
    def _ocr_adaptive(self, image: np.ndarray) -> List[Dict[str, any]]:
        """Adaptive thresholding: good for varying lighting conditions."""
        results = []
        
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            h, w = gray.shape
            
            # Upscale if needed
            if h < 48:
                scale = 2.5
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            # Remove border (5%)
            h, w = gray.shape
            border = int(min(h, w) * 0.05)
            if border > 0 and h > border * 2 and w > border * 2:
                gray = gray[border:-border, border:-border]
            
            # Bilateral filter
            denoised = cv2.bilateralFilter(gray, 9, 50, 50)
            
            # Adaptive Gaussian threshold
            adaptive = cv2.adaptiveThreshold(
                denoised, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Morphological closing
            kernel = np.ones((2, 2), np.uint8)
            adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
            
            # Normal version
            adaptive_bgr = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
            results.append({'image': adaptive_bgr, 'method': 'adaptive', 'priority': 2})
            
            # Inverted version
            adaptive_inv = cv2.bitwise_not(adaptive)
            adaptive_inv_bgr = cv2.cvtColor(adaptive_inv, cv2.COLOR_GRAY2BGR)
            results.append({'image': adaptive_inv_bgr, 'method': 'adaptive-inv', 'priority': 2})
            
        except Exception as e:
            print(f"[ImagePreprocessor] Adaptive preprocessing failed: {e}")
        
        return results
    
    def select_best_ocr_result(self, ocr_results: List[Dict[str, any]]) -> Dict[str, any]:
        """
        Select best OCR result from multiple preprocessing attempts.
        Handles rotated text results appropriately.
        
        Args:
            ocr_results: List of OCR results with keys: 'text', 'conf', 'method'
            
        Returns:
            Best result dict with keys: 'text', 'conf', 'method', 'is_rotated'
        """
        if not ocr_results:
            return {'text': '', 'conf': 0.0, 'method': 'none', 'is_rotated': False}
        
        # Score each result
        scored_results = []
        for result in ocr_results:
            text = result.get('text', '').strip()
            conf = result.get('conf', 0.0)
            method = result.get('method', 'unknown')
            
            # Skip empty results
            if not text or conf < 0.01:
                continue
            
            # Check if this result came from rotated image
            is_rotated = 'rotated' in method.lower()
            
            # Calculate score
            score = conf
            
            # Bonus for alphanumeric content (trailer IDs usually have both)
            has_letters = any(c.isalpha() for c in text)
            has_numbers = any(c.isdigit() for c in text)
            if has_letters and has_numbers:
                score *= 1.2
            elif has_numbers:
                score *= 1.1
            elif has_letters and len(text) >= 4:
                score *= 1.05
            
            # Bonus for reasonable length (3-12 chars typical for trailer IDs)
            if 3 <= len(text) <= 12:
                score *= 1.15
            elif len(text) >= 4:
                score *= 1.08
            
            # Slight bonus for rotated results (they might be more accurate for vertical text)
            if is_rotated:
                score *= 1.05
            
            scored_results.append({
                'text': text,
                'conf': conf,
                'method': method,
                'score': score,
                'is_rotated': is_rotated
            })
        
        if not scored_results:
            return {'text': '', 'conf': 0.0, 'method': 'none', 'is_rotated': False}
        
        # Return best scored result
        best = max(scored_results, key=lambda x: x['score'])
        return {
            'text': best['text'],
            'conf': best['conf'],
            'method': best['method'],
            'is_rotated': best['is_rotated']
        }

