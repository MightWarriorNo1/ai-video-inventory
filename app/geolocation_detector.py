"""
Geolocation Detector using HuggingFace Models

This module provides GPS coordinate prediction from images using HuggingFace
geolocation models (StreetCLIP, GeoCLIP, etc.) as an alternative to homography.

Note: These models predict where a photo was taken based on visual features,
which is different from homography's pixel-to-GPS mapping. For fixed camera
setups, homography is typically more accurate, but this can serve as:
1. A fallback when homography is unavailable
2. A validation method for homography results
3. An alternative approach for dynamic camera setups
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
from PIL import Image
import cv2
import os


class GeolocationDetector:
    """
    GPS coordinate prediction from images using HuggingFace geolocation models.
    """
    
    def __init__(self, model_name: str = "geolocal/StreetCLIP", use_cuda: bool = True):
        """
        Initialize geolocation detector and load model immediately.
        
        Args:
            model_name: HuggingFace model identifier or local path to fine-tuned model
                       Options: "geolocal/StreetCLIP", "geolocal/GeoCLIP", or local path like "models/finetuned_streetclip"
            use_cuda: Whether to use GPU if available
        """
        self.model_name = model_name
        self.use_cuda = use_cuda
        self.model = None
        self.processor = None
        self.initialized = False
        
        # Candidate locations for StreetCLIP-style models
        # These should be customized based on your facility location
        self.candidate_locations = []  # Text descriptions in training format
        self.location_names = []  # Location names (e.g., "YARD025")
        self.location_coordinates = {}  # Maps location name to (lat, lon)
        
        # Confidence threshold (can be configured)
        self.confidence_threshold = 0.0  # Default: accept all predictions (can be set via config)
        
        # Load model immediately during initialization
        self._load_model()
    
    def _load_model(self):
        """Load the model and processor."""
        if self.initialized:
            return
        
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
            
            print(f"[GeolocationDetector] Loading model: {self.model_name}")
            
            # Check if model_name is a local path or HuggingFace model name
            if os.path.exists(self.model_name) or os.path.isdir(self.model_name):
                # Local path - load from directory
                print(f"[GeolocationDetector] Loading from local path: {self.model_name}")
                self.processor = CLIPProcessor.from_pretrained(self.model_name, use_fast=False)
                self.model = CLIPModel.from_pretrained(self.model_name)
            else:
                # HuggingFace model name
                self.processor = CLIPProcessor.from_pretrained(self.model_name, use_fast=False)
                self.model = CLIPModel.from_pretrained(self.model_name)
            
            # Move to GPU if available and requested
            if self.use_cuda and torch.cuda.is_available():
                self.model = self.model.cuda()
                print(f"[GeolocationDetector] Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                print(f"[GeolocationDetector] Using CPU")
            
            self.model.eval()
            self.initialized = True
            print(f"[GeolocationDetector] Model loaded successfully")
            
        except ImportError:
            raise ImportError(
                "transformers library not installed. "
                "Install with: pip install transformers torch"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load geolocation model: {e}")
    
    def initialize(self):
        """Initialize the model (for backward compatibility - model is now loaded in __init__)."""
        self._load_model()
    
    def set_candidate_locations(self, locations: List[Dict[str, any]]):
        """
        Set candidate locations for location-based models.
        
        Args:
            locations: List of dicts with keys:
                - 'name': Location name (e.g., "Dock 1", "Parking Area A")
                - 'lat': Latitude
                - 'lon': Longitude
                - Optional: 'description': Additional context
        """
        # Store location names and coordinates
        self.location_names = [loc['name'] for loc in locations]
        self.location_coordinates = {
            loc['name']: (loc['lat'], loc['lon'])
            for loc in locations
        }
        
        # Create text descriptions in the SAME format as training
        # Training uses: "A trailer parking spot at latitude {lat:.6f} longitude {lon:.6f}"
        # This is critical - the text format must match what the model was trained on!
        self.candidate_locations = []
        for loc in locations:
            lat = loc['lat']
            lon = loc['lon']
            # Use exact same format as training (see finetune_streetclip.py line 121)
            text = f"A trailer parking spot at latitude {lat:.6f} longitude {lon:.6f}"
            self.candidate_locations.append(text)
        
        print(f"[GeolocationDetector] Set {len(self.candidate_locations)} candidate locations")
        print(f"[GeolocationDetector] Using GPS coordinate text format (matching training format)")
    
    def set_confidence_threshold(self, threshold: float):
        """
        Set confidence threshold for accepting predictions.
        
        Args:
            threshold: Minimum confidence (0.0 to 1.0) to accept a prediction
        """
        self.confidence_threshold = threshold
        print(f"[GeolocationDetector] Confidence threshold set to {threshold:.3f}")
    
    def predict_from_image(self, image: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Predict GPS coordinates from a full image.
        
        This method uses the entire image to predict where it was taken.
        For fixed cameras, this returns the camera's location, not individual
        object locations.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
        
        Returns:
            Tuple of (lat, lon) if prediction successful, None otherwise
        """
        if not self.initialized:
            print("[GeolocationDetector] Warning: Model not initialized, attempting to load...")
            self._load_model()
            if not self.initialized:
                return None
        
        if not self.candidate_locations:
            print("[GeolocationDetector] Warning: No candidate locations set. "
                  "Cannot predict GPS coordinates.")
            return None
        
        try:
            import torch
            
            # Convert BGR to RGB and to PIL Image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Process image and text inputs
            inputs = self.processor(
                text=self.candidate_locations,
                images=pil_image,
                return_tensors="pt",
                padding=True
            )
            
            # Move inputs to device
            if self.use_cuda and torch.cuda.is_available():
                inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            
            # Perform inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Get most probable location
            predicted_idx = probs.argmax().item()
            predicted_text = self.candidate_locations[predicted_idx]
            predicted_location_name = self.location_names[predicted_idx]
            confidence = probs[0][predicted_idx].item()
            
            # Check confidence threshold
            if confidence < self.confidence_threshold:
                print(f"[GeolocationDetector] Low confidence ({confidence:.3f} < {self.confidence_threshold:.3f}), rejecting")
                return None
            
            # Get GPS coordinates for predicted location
            if predicted_location_name in self.location_coordinates:
                lat, lon = self.location_coordinates[predicted_location_name]
                print(f"[GeolocationDetector] Predicted: {predicted_location_name} "
                      f"(confidence: {confidence:.3f}) -> GPS: ({lat:.6f}, {lon:.6f})")
                return (lat, lon)
            else:
                print(f"[GeolocationDetector] Warning: No coordinates for location: {predicted_location_name}")
                return None
                
        except Exception as e:
            print(f"[GeolocationDetector] Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_from_crop(self, crop: Optional[np.ndarray], camera_location: Tuple[float, float],
                        bbox: List[float], image_size: Tuple[int, int], 
                        full_frame: Optional[np.ndarray] = None,
                        max_offset_meters: float = 50.0) -> Optional[Tuple[float, float]]:
        """
        Predict GPS coordinates from a cropped trailer image.
        
        This method uses:
        1. Camera's known location as a base
        2. Bbox position in image to estimate offset
        3. Optionally uses full frame prediction if available
        
        Args:
            crop: Cropped image of the trailer (BGR format) - optional, not currently used
            camera_location: Known GPS location of camera (lat, lon)
            bbox: Bounding box [x1, y1, x2, y2] in original image
            image_size: Original image size (width, height)
            full_frame: Optional full frame image (for better prediction)
            max_offset_meters: Maximum offset in meters from camera location
        
        Returns:
            Tuple of (lat, lon) if prediction successful, None otherwise
        """
        cam_lat, cam_lon = camera_location
        x1, y1, x2, y2 = bbox
        img_width, img_height = image_size
        
        # Method 1: Try full frame prediction if available and candidate locations are set
        # This uses the actual fine-tuned model prediction
        if full_frame is not None and self.candidate_locations:
            try:
                full_frame_gps = self.predict_from_image(full_frame)
                if full_frame_gps:
                    # Model prediction succeeded - use it directly
                    # The model prediction gives us the GPS of the predicted parking spot
                    return full_frame_gps
            except Exception as e:
                # If model prediction fails, fall through to offset calculation
                pass
        
        # Method 2: Fallback to bbox-based offset calculation
        # This is less accurate but provides a rough estimate when model prediction fails
        # Calculate normalized position (0-1) where 0.5 is center
        center_x = (x1 + x2) / 2.0 / img_width
        center_y = (y1 + y2) / 2.0 / img_height
        
        # Calculate offset from center
        # Positive X = right side of image = east
        # Positive Y = bottom of image = south (in image coordinates)
        offset_x_normalized = center_x - 0.5  # -0.5 to +0.5
        offset_y_normalized = center_y - 0.5  # -0.5 to +0.5
        
        # Convert normalized offset to meters
        # Scale based on max_offset_meters
        offset_meters_x = offset_x_normalized * max_offset_meters
        offset_meters_y = -offset_y_normalized * max_offset_meters  # Negative because image Y increases downward
        
        # Convert offset to GPS using camera location as base
        from app.gps_utils import meters_to_gps
        lat, lon = meters_to_gps(offset_meters_x, offset_meters_y, cam_lat, cam_lon)
        
        return (lat, lon)
    
    def validate_homography_result(self, image: np.ndarray, 
                                   homography_gps: Tuple[float, float],
                                   camera_location: Tuple[float, float],
                                   threshold_km: float = 1.0) -> Dict:
        """
        Validate homography GPS result using geolocation model.
        
        Args:
            image: Full camera image
            homography_gps: GPS coordinates from homography (lat, lon)
            camera_location: Known camera location (lat, lon)
            threshold_km: Maximum distance in km for validation
        
        Returns:
            Dict with validation results:
                - 'valid': bool
                - 'distance_km': float
                - 'geolocation_prediction': (lat, lon) or None
                - 'message': str
        """
        if not self.initialized:
            print("[GeolocationDetector] Warning: Model not initialized, attempting to load...")
            self._load_model()
            if not self.initialized:
                return {
                    'valid': False,
                    'distance_km': None,
                    'geolocation_prediction': None,
                    'message': 'Geolocation model not initialized'
                }
        
        # Predict location from image
        predicted_gps = self.predict_from_image(image)
        
        if predicted_gps is None:
            return {
                'valid': False,
                'distance_km': None,
                'geolocation_prediction': None,
                'message': 'Geolocation prediction failed'
            }
        
        # Calculate distance between homography result and camera location
        from app.gps_utils import gps_to_meters
        hom_lat, hom_lon = homography_gps
        cam_lat, cam_lon = camera_location
        
        # Convert both to meters and calculate distance
        x_meters, y_meters = gps_to_meters(hom_lat, hom_lon, cam_lat, cam_lon)
        distance_m = np.sqrt(x_meters**2 + y_meters**2)
        distance_km = distance_m / 1000.0
        
        # Validate: homography result should be close to camera location
        # (since we're in a fixed yard, objects should be within reasonable distance)
        is_valid = distance_km <= threshold_km
        
        return {
            'valid': is_valid,
            'distance_km': distance_km,
            'geolocation_prediction': predicted_gps,
            'homography_gps': homography_gps,
            'camera_location': camera_location,
            'message': f'Homography result is {distance_km:.2f}km from camera. '
                      f'Threshold: {threshold_km}km. Valid: {is_valid}'
        }


def create_geolocation_detector(config: Dict) -> Optional[GeolocationDetector]:
    """
    Factory function to create a geolocation detector from config.
    
    Args:
        config: Configuration dict with keys:
            - 'enabled': bool - Whether to enable geolocation
            - 'model_name': str - HuggingFace model name or local path (optional)
            - 'use_cuda': bool - Whether to use GPU (optional)
            - 'candidate_locations': List[Dict] - Candidate locations (optional)
            - 'confidence_threshold': float - Minimum confidence to accept (optional)
    
    Returns:
        GeolocationDetector instance or None if disabled
    """
    if not config.get('enabled', False):
        return None
    
    model_name = config.get('model_name', 'geolocal/StreetCLIP')
    use_cuda = config.get('use_cuda', True)
    confidence_threshold = config.get('confidence_threshold', 0.0)  # Default: accept all
    
    detector = GeolocationDetector(model_name=model_name, use_cuda=use_cuda)
    
    # Set confidence threshold if provided
    if confidence_threshold > 0.0:
        detector.set_confidence_threshold(confidence_threshold)
    
    # Set candidate locations if provided
    candidate_locations = config.get('candidate_locations', [])
    if candidate_locations:
        detector.set_candidate_locations(candidate_locations)
    
    return detector

