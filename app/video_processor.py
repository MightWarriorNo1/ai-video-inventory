"""
Video Processor for Testing

Processes uploaded video files using the same pipeline as live camera feeds.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Generator, Tuple
from datetime import datetime
import threading


class VideoProcessor:
    """
    Video processor for testing uploaded video files.
    """
    
    def __init__(self, detector, ocr, tracker_factory, spot_resolver, homography=None):
        """
        Initialize video processor.
        
        Args:
            detector: Detection model instance
            ocr: OCR model instance
            tracker_factory: Function that returns a new tracker instance
            spot_resolver: Spot resolver instance
            homography: Optional homography matrix for world projection
        """
        self.detector = detector
        self.ocr = ocr
        self.tracker_factory = tracker_factory
        self.spot_resolver = spot_resolver
        self.homography = homography
        
        # Processing state
        self.processing = False
        self.stop_flag = False
        self.lock = threading.Lock()
        
        # Results storage
        self.processed_frames = {}  # frame_num -> frame
        self.events = []
        self.stats = {
            'frames_processed': 0,
            'detections': 0,
            'tracks': 0,
            'ocr_results': 0
        }
    
        # Last detected nearest trailer (for persistent display)
        self.last_nearest_trailer = None  # {bbox, track_id, conf, text, ocr_conf}
    
    def process_video(self, video_path: str, camera_id: str = "test-video", 
                     detect_every_n: int = 5) -> Generator[Tuple[int, np.ndarray, List[Dict]], None, None]:
        """
        Process a video file.
        
        Args:
            video_path: Path to video file
            camera_id: Camera identifier for events
            detect_every_n: Run detector every N frames
            
        Yields:
            Tuple of (frame_number, processed_frame, events)
        """
        with self.lock:
            if self.processing:
                raise RuntimeError("Video processing already in progress")
            self.processing = True
            self.stop_flag = False
            self.processed_frames = {}
            self.events = []
            self.last_nearest_trailer = None  # Reset persistent display
            self.stats = {
                'frames_processed': 0,
                'detections': 0,
                'tracks': 0,
                'ocr_results': 0
            }
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {video_path}")
            
            print(f"[VideoProcessor] Starting to process video: {video_path}")
            print(f"[VideoProcessor] Detector available: {self.detector is not None}")
            print(f"[VideoProcessor] OCR available: {self.ocr is not None}")
            
            tracker = self.tracker_factory()
            frame_count = 0
            
            while True:
                if self.stop_flag:
                    print(f"[VideoProcessor] Stop flag set, stopping at frame {frame_count}")
                    break
                
                ret, frame = cap.read()
                if not ret:
                    print(f"[VideoProcessor] End of video reached at frame {frame_count}")
                    break
                
                # Process frame
                processed_frame = frame.copy()
                frame_events = []
                
                # Run detector
                detections = []
                try:
                    if self.detector and frame_count % detect_every_n == 0:
                        detections = self.detector.detect(frame)
                        with self.lock:
                            self.stats['detections'] += len(detections)
                        if len(detections) > 0:
                              # Show confidence scores for debugging
                            conf_scores = [f"{d['conf']*100:.1f}%" for d in detections[:5]]
                            print(f"[VideoProcessor] Frame {frame_count}: Found {len(detections)} detections (confidences: {', '.join(conf_scores)})")
                except Exception as e:
                    print(f"[VideoProcessor] Error in detection at frame {frame_count}: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Update tracker
                try:
                    tracks = tracker.update(detections, frame)
                    
                    # Filter out non-trailer objects (false positives like mirrors, signs, etc.)
                    # Trailers have specific characteristics but need LENIENT filters
                    # to account for different viewing angles and distances
                    MIN_TRAILER_WIDTH = 150   # Minimum width in pixels
                    MIN_TRAILER_HEIGHT = 50   # Minimum height in pixels
                    MIN_AREA = 10000          # Minimum area (150*50 = 7500, buffer for safety)
                    MIN_ASPECT_RATIO = 1.2    # Wider than tall (lenient for angled views)
                    MAX_ASPECT_RATIO = 8.0    # Very wide range to catch all trailers
                    
                    filtered_tracks = []
                    for track in tracks:
                        bbox = track['bbox']
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        aspect_ratio = width / (height + 1e-6)  # Avoid division by zero
                        
                        # Apply filters for trailer characteristics
                        if (width >= MIN_TRAILER_WIDTH and 
                            height >= MIN_TRAILER_HEIGHT and 
                            area >= MIN_AREA and
                            MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
                            filtered_tracks.append(track)
                        else:
                            # Log filtered out tracks for debugging
                            if frame_count % 30 == 0:  # Only log occasionally
                                print(f"[VideoProcessor] Filtered Track {track['track_id']}: size={width}x{height}, aspect={aspect_ratio:.2f}")
                    
                    tracks = filtered_tracks
                    
                    with self.lock:
                        self.stats['tracks'] += len(tracks)
                    if len(tracks) > 0:
                        print(f"[VideoProcessor] Frame {frame_count}: Tracking {len(tracks)} valid trailers")
                except Exception as e:
                    print(f"[VideoProcessor] Error in tracking at frame {frame_count}: {e}")
                    import traceback
                    traceback.print_exc()
                    tracks = []
                
                # Find the nearest trailer (largest y2 value = closest to bottom of frame = nearest)
                nearest_track_id = None
                if tracks:
                    max_y2 = -1
                    for track in tracks:
                        bbox = track['bbox']
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        if x2 > x1 and y2 > y1:  # Valid bbox
                            if y2 > max_y2:
                                max_y2 = y2
                                nearest_track_id = track['track_id']
                
                # Process each track
                for track in tracks:
                    try:
                        track_id = track['track_id']
                        bbox = track['bbox']
                        # Get detection confidence from track
                        det_conf = track.get('conf', 0.0)
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        
                        # Ensure bbox is valid
                        if x2 <= x1 or y2 <= y1:
                            continue
                        
                        # Refine bounding box to rear face (focus on back side only)
                        # Rear face is typically in the center portion for wide detections
                        orig_width = x2 - x1
                        orig_height = y2 - y1
                        orig_aspect = orig_width / orig_height if orig_height > 0 else 1.0
                        
                        # Store original bbox for OCR (we'll use refined bbox for display)
                        orig_x1, orig_y1, orig_x2, orig_y2 = x1, y1, x2, y2
                        
                        # If aspect ratio suggests side view (wide), extract center portion for rear face
                        if orig_aspect > 1.5:  # Wide detection (side view)
                            center_x = (x1 + x2) / 2.0
                            rear_width_ratio = 0.65  # Use 65% of width for rear face (centered)
                            rear_width = int(orig_width * rear_width_ratio)
                            rear_x1 = int(center_x - rear_width / 2)
                            rear_x2 = int(center_x + rear_width / 2)
                            # Keep full height
                            rear_y1 = y1
                            rear_y2 = y2
                            
                            # Clip to frame bounds
                            h, w = processed_frame.shape[:2]
                            rear_x1 = max(0, min(rear_x1, w - 1))
                            rear_x2 = max(rear_x1 + 1, min(rear_x2, w - 1))
                            rear_y1 = max(0, min(rear_y1, h - 1))
                            rear_y2 = max(rear_y1 + 1, min(rear_y2, h - 1))
                            
                            # Use refined coordinates for display (rear face)
                            x1, y1, x2, y2 = rear_x1, rear_y1, rear_x2, rear_y2
                        # For narrow/tall detections (front/back view), use original bbox
                        
                        # Calculate trailer dimensions (using refined bbox)
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Draw green bounding box for ALL trailers (rear face)
                        is_nearest = (track_id == nearest_track_id)
                        
                        # Draw bounding box in green for detected trailer rear face (thicker for better visibility)
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            
                        # Draw semi-transparent green overlay on the bounding box (for all trailers)
                        overlay = processed_frame.copy()
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
                        cv2.addWeighted(overlay, 0.1, processed_frame, 0.9, 0, processed_frame)
                        
                        # Draw detection confidence and track ID above the box
                        det_conf_percent = det_conf * 100.0
                        if is_nearest:
                            conf_text = f"Track {track_id} - {det_conf_percent:.1f}% [NEAREST]"
                        else:
                            conf_text = f"Track {track_id} - {det_conf_percent:.1f}%"
                        text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        
                        # Draw black background for text
                        text_x = x1
                        text_y = max(y1 - 10, 20)
                        cv2.rectangle(processed_frame, 
                                     (text_x - 2, text_y - text_size[1] - 2), 
                                     (text_x + text_size[0] + 2, text_y + 2),
                                     (0, 0, 0), -1)
                        
                        # Draw text (green for all trailers)
                        cv2.putText(processed_frame, conf_text, (text_x, text_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Draw trailer size below track ID (only for nearest)
                        if is_nearest:
                            size_text = f"Size: {width}x{height}px"
                            size_text_size = cv2.getTextSize(size_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            size_y = text_y + text_size[1] + 5
                            
                            # Draw black background for size text
                            cv2.rectangle(processed_frame,
                                         (text_x - 2, size_y - size_text_size[1] - 2),
                                         (text_x + size_text_size[0] + 2, size_y + 2),
                                         (0, 0, 0), -1)
                            
                            # Draw size text
                            cv2.putText(processed_frame, size_text, (text_x, size_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Multi-region OCR: Trailer text can appear in different locations
                        # (top - company name/ID, middle - branding, bottom - license plate/numbers)
                        # We'll scan multiple horizontal bands and combine results
                        # Use refined rear face bbox for OCR (where text actually appears)
                        h, w = frame.shape[:2]
                        
                        # Use refined rear face dimensions for OCR regions
                        bbox_width = x2 - x1
                        bbox_height = y2 - y1
                        
                        # Define multiple crop regions to scan (within rear face area)
                        crop_regions = []
                        
                        # Region 1: Full rear face (90% with 5% margins)
                        full_x1 = int(x1 + bbox_width * 0.05)
                        full_x2 = int(x2 - bbox_width * 0.05)
                        full_y1 = int(y1 + bbox_height * 0.05)
                        full_y2 = int(y2 - bbox_height * 0.05)
                        crop_regions.append(('full', full_x1, full_y1, full_x2, full_y2))
                        
                        # Region 2: Top third of rear face (where company names often appear)
                        top_x1 = int(x1 + bbox_width * 0.05)
                        top_x2 = int(x2 - bbox_width * 0.05)
                        top_y1 = int(y1 + bbox_height * 0.05)
                        top_y2 = int(y1 + bbox_height * 0.40)  # Top 35% of rear face
                        crop_regions.append(('top', top_x1, top_y1, top_x2, top_y2))
                        
                        # Region 3: Bottom third of rear face (where numbers/license plates often appear)
                        bot_x1 = int(x1 + bbox_width * 0.05)
                        bot_x2 = int(x2 - bbox_width * 0.05)
                        bot_y1 = int(y1 + bbox_height * 0.60)  # Bottom 40% of rear face
                        bot_y2 = int(y2 - bbox_height * 0.05)
                        crop_regions.append(('bottom', bot_x1, bot_y1, bot_x2, bot_y2))
                        
                        # Debug: print crop regions for nearest trailer
                        if is_nearest and frame_count % 10 == 0:
                            print(f"[VideoProcessor] Frame {frame_count}, Track {track_id}: Crop regions defined")
                            print(f"  Bbox: ({x1}, {y1}) to ({x2}, {y2}), size: {bbox_width}x{bbox_height}")
                            for rname, rx1, ry1, rx2, ry2 in crop_regions:
                                print(f"  Region '{rname}': ({rx1}, {ry1}) to ({rx2}, {ry2})")
                        
                        # Draw OCR region indicators for nearest trailer (ALWAYS draw for debugging)
                        if is_nearest:
                            # Draw all crop regions in different colors for debugging
                            for region_name, rx1, ry1, rx2, ry2 in crop_regions:
                                # Ensure coordinates are within bounds
                                rx1 = max(0, min(rx1, w - 1))
                                ry1 = max(0, min(ry1, h - 1))
                                rx2 = max(rx1 + 1, min(rx2, w))
                                ry2 = max(ry1 + 1, min(ry2, h))
                                
                                # Color coding: full=yellow, top=cyan, bottom=magenta
                                if region_name == 'full':
                                    color = (0, 255, 255)  # Yellow
                                elif region_name == 'top':
                                    color = (255, 255, 0)  # Cyan
                                else:
                                    color = (255, 0, 255)  # Magenta
                                
                                cv2.rectangle(processed_frame, (rx1, ry1), (rx2, ry2), color, 2)  # Thicker for visibility
                        
                        # Run OCR on multiple regions with multiple preprocessing strategies
                        # This ensures we capture text that appears in different locations on the trailer
                        
                        text = ""
                        conf_ocr = 0.0
                        if not self.ocr:
                            if frame_count % 50 == 0:  # Log occasionally
                                print(f"[VideoProcessor] WARNING: OCR engine is not initialized!")
                        elif is_nearest:  # Only run OCR on nearest trailer to save resources
                            if frame_count % 10 == 0:
                                print(f"[VideoProcessor] Frame {frame_count}, Track {track_id}: Starting OCR on {len(crop_regions)} regions")
                            try:
                                all_ocr_results = []  # Store results from all regions and strategies
                                
                                # Process each crop region
                                for region_name, rx1, ry1, rx2, ry2 in crop_regions:
                                    # Ensure coordinates are within bounds
                                    rx1 = max(0, min(rx1, w - 1))
                                    ry1 = max(0, min(ry1, h - 1))
                                    rx2 = max(rx1 + 1, min(rx2, w))
                                    ry2 = max(ry1 + 1, min(ry2, h))
                                    
                                    crop = frame[ry1:ry2, rx1:rx2]
                                    
                                    # Skip if crop is too small
                                    if crop.size == 0 or crop.shape[0] < 15 or crop.shape[1] < 30:
                                        continue
                                    
                                    # Check if text might be vertical (tall and narrow)
                                    h_crop, w_crop = crop.shape[:2]
                                    aspect_ratio = h_crop / w_crop if w_crop > 0 else 1.0
                                    
                                    # Convert to grayscale once for all strategies
                                    if len(crop.shape) == 3:
                                        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                                    else:
                                        crop_gray = crop
                                    
                                    # Strategy 0: Original with minimal processing (baseline)
                                    try:
                                        h_crop, w_crop = crop.shape[:2]
                                        
                                        # Only upscale if text is small
                                        if h_crop < 40 or w_crop < 80:
                                            scale = 2.5
                                            crop_scaled = cv2.resize(crop, None, fx=scale, fy=scale, 
                                                                   interpolation=cv2.INTER_CUBIC)
                                        else:
                                            crop_scaled = crop
                                        
                                        result_orig = self.ocr.recognize(crop_scaled)
                                        all_ocr_results.append({
                                            'text': result_orig.get('text', ''),
                                            'conf': result_orig.get('conf', 0.0),
                                            'method': f'{region_name}-original',
                                            'region': region_name
                                        })
                                    except Exception as e:
                                        print(f"[VideoProcessor] OCR strategy 0 ({region_name}) failed: {e}")
                                    
                                    # Strategy 0v: Vertical text detection (rotate if tall and narrow)
                                    # If aspect ratio > 1.3 (tall and narrow), likely vertical text
                                    if aspect_ratio > 1.3:
                                        try:
                                            # Rotate 90 degrees clockwise (to make vertical text horizontal)
                                            crop_rotated_cw = cv2.rotate(crop_scaled, cv2.ROTATE_90_CLOCKWISE)
                                            result_rot_cw = self.ocr.recognize(crop_rotated_cw)
                                            all_ocr_results.append({
                                                'text': result_rot_cw.get('text', ''),
                                                'conf': result_rot_cw.get('conf', 0.0),
                                                'method': f'{region_name}-rotated-90cw',
                                                'region': region_name
                                            })
                                            
                                            # Rotate 90 degrees counter-clockwise (alternative orientation)
                                            crop_rotated_ccw = cv2.rotate(crop_scaled, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                            result_rot_ccw = self.ocr.recognize(crop_rotated_ccw)
                                            all_ocr_results.append({
                                                'text': result_rot_ccw.get('text', ''),
                                                'conf': result_rot_ccw.get('conf', 0.0),
                                                'method': f'{region_name}-rotated-90ccw',
                                                'region': region_name
                                            })
                                        except Exception as e:
                                            if frame_count % 50 == 0:
                                                print(f"[VideoProcessor] OCR vertical text detection ({region_name}) failed: {e}")
                                    
                                    # Strategy 0b: Morphological preprocessing (character separation)
                                    # This helps separate touching characters and remove small noise
                                    try:
                                        h_crop, w_crop = crop_gray.shape
                                        
                                        # Upscale first
                                        if h_crop < 48:
                                            scale = 2.5
                                            crop_up = cv2.resize(crop_gray, None, fx=scale, fy=scale,
                                                               interpolation=cv2.INTER_CUBIC)
                                        else:
                                            crop_up = crop_gray
                                        
                                        # Apply bilateral filter to reduce noise while keeping edges sharp
                                        crop_smooth = cv2.bilateralFilter(crop_up, 9, 40, 40)
                                        
                                        # Apply Otsu's threshold
                                        _, crop_thresh = cv2.threshold(crop_smooth, 0, 255,
                                                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                        
                                        # Morphological opening to remove small noise
                                        kernel_small = np.ones((2, 2), np.uint8)
                                        crop_morph = cv2.morphologyEx(crop_thresh, cv2.MORPH_OPEN, kernel_small)
                                        
                                        # Morphological closing to fill small holes in characters
                                        kernel_med = np.ones((2, 3), np.uint8)
                                        crop_morph = cv2.morphologyEx(crop_morph, cv2.MORPH_CLOSE, kernel_med)
                                        
                                        # Try both normal and inverted
                                        crop_morph_bgr = cv2.cvtColor(crop_morph, cv2.COLOR_GRAY2BGR)
                                        result_morph = self.ocr.recognize(crop_morph_bgr)
                                        all_ocr_results.append({
                                            'text': result_morph.get('text', ''),
                                            'conf': result_morph.get('conf', 0.0),
                                            'method': f'{region_name}-morphological',
                                            'region': region_name
                                        })
                                        
                                        # Inverted version
                                        crop_morph_inv = cv2.bitwise_not(crop_morph)
                                        crop_morph_inv_bgr = cv2.cvtColor(crop_morph_inv, cv2.COLOR_GRAY2BGR)
                                        result_morph_inv = self.ocr.recognize(crop_morph_inv_bgr)
                                        all_ocr_results.append({
                                            'text': result_morph_inv.get('text', ''),
                                            'conf': result_morph_inv.get('conf', 0.0),
                                            'method': f'{region_name}-morphological-inv',
                                            'region': region_name
                                        })
                                    except Exception as e:
                                        print(f"[VideoProcessor] OCR strategy 0b ({region_name}) failed: {e}")
                                    
                                    # Strategy 1: CLAHE + Border removal (for low-contrast text)
                                    try:
                                        h_crop, w_crop = crop_gray.shape
                                        
                                        # Upscale if needed
                                        if h_crop < 48:
                                            scale = 2.5
                                            crop_up = cv2.resize(crop_gray, None, fx=scale, fy=scale, 
                                                               interpolation=cv2.INTER_CUBIC)
                                        else:
                                            crop_up = crop_gray
                                        
                                        # Remove 10% border to eliminate edge noise
                                        h_up, w_up = crop_up.shape
                                        border_h = max(1, int(h_up * 0.1))
                                        border_w = max(1, int(w_up * 0.1))
                                        if h_up > border_h*2 and w_up > border_w*2:
                                            crop_up = crop_up[border_h:-border_h, border_w:-border_w]
                                        
                                        # Denoise slightly
                                        crop_smooth = cv2.GaussianBlur(crop_up, (3, 3), 0)
                                        
                                        # Apply CLAHE for local contrast enhancement
                                        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
                                        crop_clahe = clahe.apply(crop_smooth)
                                        
                                        # Sharpen slightly
                                        kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                                        crop_clahe = cv2.filter2D(crop_clahe, -1, kernel_sharp)
                                        
                                        crop_clahe_bgr = cv2.cvtColor(crop_clahe, cv2.COLOR_GRAY2BGR)
                                        result_clahe = self.ocr.recognize(crop_clahe_bgr)
                                        all_ocr_results.append({
                                            'text': result_clahe.get('text', ''),
                                            'conf': result_clahe.get('conf', 0.0),
                                            'method': f'{region_name}-clahe-sharp',
                                            'region': region_name
                                        })
                                    except Exception as e:
                                        print(f"[VideoProcessor] OCR strategy 1 ({region_name}) failed: {e}")
                                    
                                    # Strategy 2: Adaptive thresholding (best for varying lighting)
                                    # Handles shadows and uneven illumination
                                    try:
                                        h_crop, w_crop = crop_gray.shape
                                        
                                        # Upscale for better quality
                                        if h_crop < 48:
                                            scale = 2.5
                                            crop_up = cv2.resize(crop_gray, None, fx=scale, fy=scale,
                                                               interpolation=cv2.INTER_CUBIC)
                                        else:
                                            crop_up = crop_gray
                                        
                                        # Remove border noise (5% on each side)
                                        h_up, w_up = crop_up.shape
                                        border = int(min(h_up, w_up) * 0.05)
                                        if border > 0 and h_up > border*2 and w_up > border*2:
                                            crop_up = crop_up[border:-border, border:-border]
                                        
                                        # Denoise with bilateral filter
                                        crop_denoised = cv2.bilateralFilter(crop_up, 9, 50, 50)
                                        
                                        # Adaptive Gaussian thresholding - better for varying lighting
                                        crop_adaptive = cv2.adaptiveThreshold(
                                            crop_denoised, 255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2
                                        )
                                        
                                        # Apply morphological operations to clean up
                                        kernel = np.ones((2, 2), np.uint8)
                                        crop_adaptive = cv2.morphologyEx(crop_adaptive, cv2.MORPH_CLOSE, kernel)
                                        
                                        crop_adaptive_bgr = cv2.cvtColor(crop_adaptive, cv2.COLOR_GRAY2BGR)
                                        result_adaptive = self.ocr.recognize(crop_adaptive_bgr)
                                        all_ocr_results.append({
                                            'text': result_adaptive.get('text', ''),
                                            'conf': result_adaptive.get('conf', 0.0),
                                            'method': f'{region_name}-adaptive',
                                            'region': region_name
                                        })
                                        
                                        # Inverted adaptive threshold
                                        crop_adaptive_inv = cv2.bitwise_not(crop_adaptive)
                                        crop_adaptive_inv_bgr = cv2.cvtColor(crop_adaptive_inv, cv2.COLOR_GRAY2BGR)
                                        result_adaptive_inv = self.ocr.recognize(crop_adaptive_inv_bgr)
                                        all_ocr_results.append({
                                            'text': result_adaptive_inv.get('text', ''),
                                            'conf': result_adaptive_inv.get('conf', 0.0),
                                            'method': f'{region_name}-adaptive-inv',
                                            'region': region_name
                                        })
                                    except Exception as e:
                                        print(f"[VideoProcessor] OCR strategy 2 ({region_name}) failed: {e}")
                                    
                                    # Strategy 3: High-contrast enhancement with dilation
                                    # Good for faded or low-contrast text
                                    try:
                                        h_crop, w_crop = crop_gray.shape
                                        
                                        # Upscale
                                        if h_crop < 48:
                                            scale = 2.5
                                            crop_up = cv2.resize(crop_gray, None, fx=scale, fy=scale,
                                                               interpolation=cv2.INTER_CUBIC)
                                        else:
                                            crop_up = crop_gray
                                        
                                        # Increase contrast dramatically
                                        crop_contrast = cv2.convertScaleAbs(crop_up, alpha=1.5, beta=0)
                                        
                                        # Apply CLAHE for local contrast enhancement
                                        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                                        crop_clahe = clahe.apply(crop_contrast)
                                        
                                        # Threshold
                                        _, crop_high = cv2.threshold(crop_clahe, 0, 255,
                                                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                        
                                        # Dilate slightly to thicken thin characters
                                        kernel_dilate = np.ones((2, 2), np.uint8)
                                        crop_high = cv2.dilate(crop_high, kernel_dilate, iterations=1)
                                        
                                        crop_high_bgr = cv2.cvtColor(crop_high, cv2.COLOR_GRAY2BGR)
                                        result_high = self.ocr.recognize(crop_high_bgr)
                                        all_ocr_results.append({
                                            'text': result_high.get('text', ''),
                                            'conf': result_high.get('conf', 0.0),
                                            'method': f'{region_name}-high-contrast',
                                            'region': region_name
                                        })
                                        
                                        # Inverted version
                                        crop_high_inv = cv2.bitwise_not(crop_high)
                                        crop_high_inv_bgr = cv2.cvtColor(crop_high_inv, cv2.COLOR_GRAY2BGR)
                                        result_high_inv = self.ocr.recognize(crop_high_inv_bgr)
                                        all_ocr_results.append({
                                            'text': result_high_inv.get('text', ''),
                                            'conf': result_high_inv.get('conf', 0.0),
                                            'method': f'{region_name}-high-contrast-inv',
                                            'region': region_name
                                        })
                                    except Exception as e:
                                        print(f"[VideoProcessor] OCR strategy 3 ({region_name}) failed: {e}")
                                
                                # Combine results from all regions intelligently
                                # Group by region to find the best result per region, then combine unique texts
                                
                                # Log all OCR attempts for debugging (every frame for nearest trailer)
                                if is_nearest and frame_count % 5 == 0:  # Log every 5 frames
                                    print(f"[VideoProcessor] Frame {frame_count}, Track {track_id}: {len(all_ocr_results)} OCR attempts from all regions")
                                    for i, result in enumerate(all_ocr_results):
                                        print(f"  {i+1}. {result['method']}: '{result['text']}' (conf: {result['conf']:.3f})")
                                
                                # Find best result per region with relaxed filtering to capture more results
                                region_best = {}  # region_name -> best result for that region
                                MIN_TEXT_LENGTH = 1  # Allow single characters (will be filtered by scoring)
                                MIN_CONFIDENCE = 0.005  # Very low threshold (0.5%) to see what OCR detects
                                
                                for result in all_ocr_results:
                                    # Filter out only completely empty results
                                    if not result['text']:
                                        continue
                                    # Very permissive filtering to see what we're getting
                                    if len(result['text']) < MIN_TEXT_LENGTH:
                                        continue
                                    if result['conf'] < MIN_CONFIDENCE:
                                        continue
                                    
                                    # Check for alphanumeric content
                                    has_letter = any(c.isalpha() for c in result['text'])
                                    has_digit = any(c.isdigit() for c in result['text'])
                                    has_both = has_letter and has_digit
                                    has_alphanumeric = has_letter or has_digit
                                    
                                    # Skip results with no alphanumeric characters
                                    if not has_alphanumeric:
                                        continue
                                    
                                    # Enhanced scoring system
                                    score = result['conf']
                                    
                                    # Trailer IDs usually have both letters and numbers
                                    if has_both:
                                        score += 0.2  # Strong bonus
                                    elif has_digit:
                                        score += 0.1  # Numbers are important
                                    elif has_letter and len(result['text']) >= 4:
                                        score += 0.05  # Some trailers have only letters
                                    
                                    # Length bonus (trailers typically have 3-12 character IDs)
                                    text_len = len(result['text'])
                                    if 3 <= text_len <= 12:
                                        score += 0.15
                                    elif text_len >= 4:
                                        score += 0.08
                                    
                                    # Store score for comparison
                                    result['score'] = score
                                    
                                    # Update best for this region
                                    region_name = result['region']
                                    if region_name not in region_best or score > region_best[region_name]['score']:
                                        region_best[region_name] = result
                                
                                # Combine unique texts from different regions
                                # If we found different text in different regions, combine them
                                unique_texts = {}  # text -> result
                                
                                for region_name, result in region_best.items():
                                    text_found = result['text'].strip()  # Strip whitespace
                                    if not text_found:  # Skip empty after stripping
                                        continue
                                    
                                    if text_found not in unique_texts:
                                        unique_texts[text_found] = result
                                    else:
                                        # Keep the one with higher score
                                        if result['score'] > unique_texts[text_found]['score']:
                                            unique_texts[text_found] = result
                                
                                # Combine all unique texts (e.g., "RZ/IMVZ" from top + "399" from bottom)
                                if unique_texts:
                                    # Sort by score to prioritize best results
                                    sorted_results = sorted(unique_texts.values(), key=lambda x: x['score'], reverse=True)
                                    
                                    # Only combine if we have valid results
                                    valid_texts = []
                                    combined_conf = 0.0
                                    combined_methods = []
                                    
                                    for result in sorted_results:
                                        text_item = result['text'].strip()
                                        if text_item and len(text_item) >= 1:  # Accept single chars for now
                                            valid_texts.append(text_item)
                                            combined_conf = max(combined_conf, result['conf'])
                                            combined_methods.append(result['method'])
                                    
                                    # Join valid texts only
                                    if valid_texts:
                                        text = ' | '.join(valid_texts)
                                        conf_ocr = combined_conf
                                        
                                        print(f"[VideoProcessor] Frame {frame_count}, Track {track_id}: COMBINED OCR '{text}' from {len(valid_texts)} regions")
                                        print(f"  Methods: {', '.join(combined_methods)}")
                                        print(f"  Confidence: {conf_ocr:.3f}")
                                
                                # Accept OCR results if we have text (scoring already filtered quality)
                                if text and len(text) >= 1:  # Accept even single chars now since we're combining regions
                                    with self.lock:
                                        self.stats['ocr_results'] += 1
                                    print(f"[VideoProcessor] Frame {frame_count}, Track {track_id}: OCR ACCEPTED '{text}' (conf: {conf_ocr:.3f})")
                                else:
                                    if is_nearest and not text:
                                        print(f"[VideoProcessor] Frame {frame_count}, Track {track_id}: No valid OCR result (all attempts failed)")
                                    text = ""
                                    conf_ocr = 0.0
                                
                                # Draw OCR text in red ONLY for the nearest trailer
                                if text and is_nearest:
                                    ocr_conf_percent = conf_ocr * 100.0
                                    ocr_text = f"TEXT: {text}"
                                    ocr_conf_text = f"Conf: {ocr_conf_percent:.1f}%"
                                    
                                    # Position text below the bounding box
                                    text_y = min(y2 + 30, h - 30)
                                    
                                    # Draw OCR text with black background for better visibility
                                    ocr_text_size = cv2.getTextSize(ocr_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                                    cv2.rectangle(processed_frame,
                                                 (x1 - 2, text_y - ocr_text_size[1] - 2),
                                                 (x1 + ocr_text_size[0] + 2, text_y + 2),
                                                 (0, 0, 0), -1)
                                    
                                    # Draw OCR text in red
                                    cv2.putText(processed_frame, ocr_text, (x1, text_y),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                    
                                    # Draw confidence below OCR text
                                    conf_y = text_y + 20
                                    ocr_conf_size = cv2.getTextSize(ocr_conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                                    cv2.rectangle(processed_frame,
                                                 (x1 - 2, conf_y - ocr_conf_size[1] - 2),
                                                 (x1 + ocr_conf_size[0] + 2, conf_y + 2),
                                                 (0, 0, 0), -1)
                                    
                                    cv2.putText(processed_frame, ocr_conf_text, (x1, conf_y),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            except Exception as e:
                                print(f"[VideoProcessor] Error in OCR at frame {frame_count}, track {track_id}: {e}")
                                import traceback
                                traceback.print_exc()
                    
                        # Project to world coordinates if homography available
                        world_coords = None
                        if self.homography is not None:
                            try:
                                center_x = (x1 + x2) / 2.0
                                center_y = (y1 + y2) / 2.0
                                point = np.array([[center_x, center_y]], dtype=np.float32)
                                point = np.array([point])
                                projected = cv2.perspectiveTransform(point, self.homography)
                                x_world, y_world = projected[0][0]
                                world_coords = (float(x_world), float(y_world))
                            except Exception as e:
                                print(f"[VideoProcessor] Error in homography projection: {e}")
                        
                        # Resolve parking spot
                        spot = "unknown"
                        method = "no-calibration"
                        if world_coords and self.spot_resolver:
                            try:
                                x_world, y_world = world_coords
                                spot_result = self.spot_resolver.resolve(x_world, y_world)
                                spot = spot_result['spot']
                                method = spot_result['method']
                            except Exception as e:
                                print(f"[VideoProcessor] Error in spot resolution: {e}")
                        
                        # Create event with trailer size information
                        event = {
                            'frame': frame_count,
                            'ts_iso': datetime.utcnow().isoformat(),
                            'camera_id': camera_id,
                            'track_id': track_id,
                            'bbox': bbox,
                            'trailer_width': width,
                            'trailer_height': height,
                            'text': text,
                            'conf': conf_ocr,
                            'det_conf': det_conf,
                            'x_world': world_coords[0] if world_coords else None,
                            'y_world': world_coords[1] if world_coords else None,
                            'spot': spot,
                            'method': method
                        }
                        
                        frame_events.append(event)
                        with self.lock:
                            self.events.append(event)
                        
                        # Update last nearest trailer if this is the nearest one
                        if is_nearest:
                            self.last_nearest_trailer = {
                                'bbox': bbox,
                                'track_id': track_id,
                                'det_conf': det_conf,
                                'text': text,
                                'ocr_conf': conf_ocr,
                                'width': width,
                                'height': height
                            }
                    except Exception as e:
                        print(f"[VideoProcessor] Error processing track at frame {frame_count}: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Draw persistent green box for the last detected nearest trailer
                # This keeps the green box visible even when tracking is temporarily lost
                if self.last_nearest_trailer:
                    trailer = self.last_nearest_trailer
                    x1, y1, x2, y2 = [int(v) for v in trailer['bbox']]
                    width = trailer['width']
                    height = trailer['height']
                    track_id = trailer['track_id']
                    det_conf = trailer['det_conf']
                    text = trailer['text']
                    ocr_conf = trailer['ocr_conf']
                    
                    h, w = processed_frame.shape[:2]
                    
                    # Draw bounding box in green (thicker for better visibility)
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    # Draw semi-transparent green overlay
                    overlay = processed_frame.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
                    cv2.addWeighted(overlay, 0.1, processed_frame, 0.9, 0, processed_frame)
                    
                    # Draw detection confidence and track ID in green above the box
                    det_conf_percent = det_conf * 100.0
                    conf_text = f"Track {track_id} - {det_conf_percent:.1f}% [NEAREST]"
                    text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    
                    # Draw black background for text
                    text_x = x1
                    text_y = max(y1 - 10, 20)
                    cv2.rectangle(processed_frame, 
                                 (text_x - 2, text_y - text_size[1] - 2), 
                                 (text_x + text_size[0] + 2, text_y + 2),
                                 (0, 0, 0), -1)
                    
                    # Draw text
                    cv2.putText(processed_frame, conf_text, (text_x, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Draw trailer size below track ID
                    size_text = f"Size: {width}x{height}px"
                    size_text_size = cv2.getTextSize(size_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    size_y = text_y + text_size[1] + 5
                    
                    # Draw black background for size text
                    cv2.rectangle(processed_frame,
                                 (text_x - 2, size_y - size_text_size[1] - 2),
                                 (text_x + size_text_size[0] + 2, size_y + 2),
                                 (0, 0, 0), -1)
                    
                    # Draw size text
                    cv2.putText(processed_frame, size_text, (text_x, size_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Draw OCR text in red if available
                    if text:
                        ocr_conf_percent = ocr_conf * 100.0
                        ocr_text = f"TEXT: {text}"
                        ocr_conf_text = f"Conf: {ocr_conf_percent:.1f}%"
                        
                        # Position text below the bounding box
                        text_y_pos = min(y2 + 30, h - 30)
                        
                        # Draw OCR text with black background
                        ocr_text_size = cv2.getTextSize(ocr_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                        cv2.rectangle(processed_frame,
                                     (x1 - 2, text_y_pos - ocr_text_size[1] - 2),
                                     (x1 + ocr_text_size[0] + 2, text_y_pos + 2),
                                     (0, 0, 0), -1)
                        
                        # Draw OCR text in red
                        cv2.putText(processed_frame, ocr_text, (x1, text_y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        
                        # Draw confidence below OCR text
                        conf_y = text_y_pos + 20
                        ocr_conf_size = cv2.getTextSize(ocr_conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(processed_frame,
                                     (x1 - 2, conf_y - ocr_conf_size[1] - 2),
                                     (x1 + ocr_conf_size[0] + 2, conf_y + 2),
                                     (0, 0, 0), -1)
                        
                        cv2.putText(processed_frame, ocr_conf_text, (x1, conf_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Store processed frame (always store, even if no detections)
                # Store BEFORE yielding to ensure it's available immediately
                with self.lock:
                    self.processed_frames[frame_count] = processed_frame.copy()
                    self.stats['frames_processed'] = frame_count + 1
                
                # Print progress every 30 frames
                if frame_count % 30 == 0:
                    with self.lock:
                        print(f"[VideoProcessor] Processed {frame_count} frames, detections: {self.stats['detections']}, tracks: {self.stats['tracks']}, OCR: {self.stats['ocr_results']}")
                
                yield frame_count, processed_frame, frame_events
                frame_count += 1
                
                # Small delay to ensure frame is stored before next iteration
                import time
                time.sleep(0.001)  # 1ms delay to allow frame storage
            
            cap.release()
            with self.lock:
                final_stats = self.stats.copy()
            print(f"[VideoProcessor] Finished processing {frame_count} frames")
            print(f"[VideoProcessor] Final stats: {final_stats}")
            print(f"[VideoProcessor] Total frames stored: {len(self.processed_frames)}")
        
        except Exception as e:
            print(f"[VideoProcessor] Fatal error during video processing: {e}")
            import traceback
            traceback.print_exc()
            # Even on error, mark processing as done
            with self.lock:
                self.processing = False
                self.stop_flag = False
        finally:
            # Only set processing to False if we're actually done
            # Keep it True until we've stored at least some frames
            with self.lock:
                if frame_count > 0 or self.stop_flag:
                    self.processing = False
                    self.stop_flag = False
    
    def stop_processing(self):
        """Stop video processing."""
        with self.lock:
            self.stop_flag = True
    
    def is_processing(self) -> bool:
        """Check if video is currently being processed."""
        with self.lock:
            return self.processing
    
    def get_results(self) -> Dict:
        """Get processing results."""
        with self.lock:
            return {
                'frames_processed': self.stats['frames_processed'],
                'detections': self.stats['detections'],
                'tracks': self.stats['tracks'],
                'ocr_results': self.stats['ocr_results'],
                'events': self.events[-100:],  # Last 100 events
                'total_frames_stored': len(self.processed_frames)  # Debug info
            }
    
    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Get a processed frame by frame number."""
        with self.lock:
            return self.processed_frames.get(frame_number)

