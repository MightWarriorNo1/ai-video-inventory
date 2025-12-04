"""
Video Processor for Testing

Processes uploaded video files using the same pipeline as live camera feeds.
"""

import cv2
import numpy as np
import re
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
        
        # Track unique physical trailers by spot + position clustering
        # This prevents counting the same trailer multiple times when tracking is lost/reacquired
        self.unique_tracks = {}  # track_key -> latest_event
        self.unique_track_keys = set()  # Set of unique track keys for counting
        
        # Store OCR results per unique track - collect ALL results, then consolidate
        self.track_ocr_results = {}  # track_key -> list of {text, conf, frame}
        
        # Position clustering: group nearby positions (within 5m) as the same physical location
        # This handles slight movement and position variance
        # Increased tolerance to better handle trailers parked close together
        self.position_clusters = {}  # cluster_id -> {center_x, center_y, count, positions: [(x, y), ...]}
        self.next_cluster_id = 1
    
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
            self.unique_tracks = {}  # (track_id, spot) -> latest_event
            self.unique_track_keys = set()  # Set of unique (track_id, spot) tuples for counting
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
                    # Made more lenient to catch smaller or partially visible trailers
                    MIN_TRAILER_WIDTH = 80    # Reduced from 150 - allow smaller detections
                    MIN_TRAILER_HEIGHT = 30   # Reduced from 50 - allow shorter detections
                    MIN_AREA = 2400           # Reduced from 10000 (80*30 = 2400)
                    MIN_ASPECT_RATIO = 0.5    # More lenient - allow both wide and tall
                    MAX_ASPECT_RATIO = 10.0   # More lenient - allow very wide or tall
                    
                    filtered_tracks = []
                    filtered_count = 0
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
                            filtered_count += 1
                            # Log filtered out tracks for debugging (more frequent)
                            if frame_count % 10 == 0:  # Log more frequently
                                print(f"[VideoProcessor] Filtered Track {track['track_id']}: size={width}x{height}, area={area}, aspect={aspect_ratio:.2f}")
                    
                    # Log filtering summary
                    if len(tracks) > 0 and frame_count % 10 == 0:
                        print(f"[VideoProcessor] Frame {frame_count}: {len(filtered_tracks)}/{len(tracks)} tracks passed filter ({filtered_count} filtered out)")
                    
                    tracks = filtered_tracks
                    
                    # Count unique tracks (track_id + spot combination) - don't count per frame
                    # We'll count unique tracks when events are created
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
                elif frame_count % 30 == 0:  # Log when no tracks available
                    print(f"[VideoProcessor] Frame {frame_count}: No tracks available for OCR")
                
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
                        
                        # Project to world coordinates and resolve spot FIRST (before OCR)
                        # We need spot to identify unique physical trailers
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
                                pass
                        
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
                                pass
                        
                        # Calculate track_key and ocr_cache_key NOW (before OCR processing)
                        # Track unique physical trailers using position clustering
                        if spot != "unknown" and world_coords:
                            x_world, y_world = world_coords
                            
                            # Find or create position cluster (5m tolerance for grouping nearby positions)
                            # Increased from 3m to better handle trailers parked close together
                            cluster_id = None
                            CLUSTER_TOLERANCE = 5.0  # 5 meters
                            
                            for cid, cluster in self.position_clusters.items():
                                # Check if this position is within tolerance of cluster center
                                dist = np.sqrt((x_world - cluster['center_x'])**2 + (y_world - cluster['center_y'])**2)
                                if dist <= CLUSTER_TOLERANCE:
                                    cluster_id = cid
                                    # Add position to cluster
                                    if 'positions' not in cluster:
                                        cluster['positions'] = []
                                    cluster['positions'].append((x_world, y_world))
                                    # Update cluster center (moving average)
                                    cluster['center_x'] = (cluster['center_x'] * cluster['count'] + x_world) / (cluster['count'] + 1)
                                    cluster['center_y'] = (cluster['center_y'] * cluster['count'] + y_world) / (cluster['count'] + 1)
                                    cluster['count'] += 1
                                    break
                            
                            # Create new cluster if none found
                            if cluster_id is None:
                                cluster_id = self.next_cluster_id
                                self.next_cluster_id += 1
                                self.position_clusters[cluster_id] = {
                                    'center_x': x_world,
                                    'center_y': y_world,
                                    'count': 1,
                                    'positions': [(x_world, y_world)]
                                }
                            
                            # Use spot + cluster_id as unique identifier
                            track_key = f"{spot}_cluster_{cluster_id}"
                        elif spot != "unknown":
                            # Use spot only if position unavailable
                            track_key = spot
                        else:
                            # Fallback to track_id if spot is unknown
                            track_key = f"track_{track_id}"
                        
                        # Set OCR cache key to match track_key
                        ocr_cache_key = track_key
                        
                        # Check if we've already run OCR for this unique trailer
                        # We'll collect ALL OCR results per track, then consolidate at the end
                        text = ""
                        conf_ocr = 0.0
                        ocr_already_done = False
                        
                        # Check if we've already run OCR for this track in this frame
                        # (to avoid running OCR multiple times for the same track in the same frame)
                        with self.lock:
                            if ocr_cache_key in self.track_ocr_results:
                                # Check if we've already run OCR for this track in this frame
                                ocr_results_list = self.track_ocr_results[ocr_cache_key]
                                if isinstance(ocr_results_list, list) and len(ocr_results_list) > 0:
                                    # Check if we already have a result for this frame
                                    frame_results = [r for r in ocr_results_list if r.get('frame') == frame_count]
                                    if frame_results:
                                        ocr_already_done = True
                                        # Use the most recent result for this frame
                                        text = frame_results[-1]['text']
                                        conf_ocr = frame_results[-1]['conf']
                        
                        if not ocr_already_done:
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
                            if not self.ocr:
                                if frame_count % 50 == 0:  # Log occasionally
                                    print(f"[VideoProcessor] WARNING: OCR engine is not initialized!")
                            else:
                                # Run OCR for this unique trailer (by spot)
                                if frame_count % 10 == 0:
                                    print(f"[VideoProcessor] Frame {frame_count}, Track {track_id}: Starting OCR on {len(crop_regions)} regions (spot: {spot})")
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
                                        
                                        # Skip if crop is too small to contain readable text
                                        # Text needs to be at least 20px tall and 40px wide to be readable
                                        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 40:
                                            continue
                                        
                                        # Check if crop has sufficient contrast (text needs contrast to be readable)
                                        if len(crop.shape) == 3:
                                            crop_gray_check = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                                        else:
                                            crop_gray_check = crop
                                        std_dev = np.std(crop_gray_check)
                                        # Skip if image is too uniform (no text/features)
                                        if std_dev < 10:  # Very low contrast - likely no text
                                            continue
                                        
                                        # Convert to grayscale once for all strategies
                                        if len(crop.shape) == 3:
                                            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                                        else:
                                            crop_gray = crop
                                        
                                        # Helper function to try OCR with all orientations
                                        def try_ocr_all_orientations(image, base_method_name, region_name):
                                            """Try OCR on image in all 4 orientations (0°, 90°, 180°, 270°)."""
                                            results = []
                                            
                                            # Orientation 0: Original (horizontal text)
                                            try:
                                                result = self.ocr.recognize(image)
                                                if result.get('text', '').strip():
                                                    results.append({
                                                        'text': result.get('text', ''),
                                                        'conf': result.get('conf', 0.0),
                                                        'method': f'{region_name}-{base_method_name}',
                                                        'region': region_name,
                                                        'orientation': 0
                                                    })
                                            except Exception as e:
                                                pass
                                            
                                            # Orientation 1: Rotate 90° clockwise (vertical text → horizontal)
                                            try:
                                                rotated_90cw = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                                                result = self.ocr.recognize(rotated_90cw)
                                                if result.get('text', '').strip():
                                                    results.append({
                                                        'text': result.get('text', ''),
                                                        'conf': result.get('conf', 0.0),
                                                        'method': f'{region_name}-{base_method_name}-rot90cw',
                                                        'region': region_name,
                                                        'orientation': 90
                                                    })
                                            except Exception as e:
                                                pass
                                            
                                            # Orientation 2: Rotate 180° (upside down text)
                                            try:
                                                rotated_180 = cv2.rotate(image, cv2.ROTATE_180)
                                                result = self.ocr.recognize(rotated_180)
                                                if result.get('text', '').strip():
                                                    results.append({
                                                        'text': result.get('text', ''),
                                                        'conf': result.get('conf', 0.0),
                                                        'method': f'{region_name}-{base_method_name}-rot180',
                                                        'region': region_name,
                                                        'orientation': 180
                                                    })
                                            except Exception as e:
                                                pass
                                            
                                            # Orientation 3: Rotate 90° counter-clockwise (vertical text → horizontal, alternative)
                                            try:
                                                rotated_90ccw = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                                result = self.ocr.recognize(rotated_90ccw)
                                                if result.get('text', '').strip():
                                                    results.append({
                                                        'text': result.get('text', ''),
                                                        'conf': result.get('conf', 0.0),
                                                        'method': f'{region_name}-{base_method_name}-rot90ccw',
                                                        'region': region_name,
                                                        'orientation': 270
                                                    })
                                            except Exception as e:
                                                pass
                                            
                                            return results
                                        
                                        # Strategy 0: Original with minimal processing (baseline) - try all orientations
                                        try:
                                            h_crop, w_crop = crop.shape[:2]
                                            
                                            # Enhanced preprocessing for better OCR accuracy
                                            # 1. Upscale small images (text needs to be readable)
                                            # 2. Apply sharpening to enhance text edges
                                            # 3. Improve contrast
                                            
                                            crop_processed = crop.copy()
                                            
                                            # Upscale if too small (preserve aspect ratio)
                                            min_text_height = 80  # Increased for better quality
                                            min_text_width = 120
                                            scale_h = max(1.0, min_text_height / h_crop) if h_crop < min_text_height else 1.0
                                            scale_w = max(1.0, min_text_width / w_crop) if w_crop < min_text_width else 1.0
                                            scale = max(scale_h, scale_w)
                                            
                                            if scale > 1.0:
                                                # Use high-quality upscaling
                                                new_w = int(w_crop * scale)
                                                new_h = int(h_crop * scale)
                                                crop_processed = cv2.resize(crop_processed, (new_w, new_h), 
                                                                          interpolation=cv2.INTER_CUBIC)
                                            
                                            # Apply unsharp masking to sharpen text (helps OCR accuracy)
                                            if len(crop_processed.shape) == 3:
                                                # Convert to grayscale for sharpening, then back to BGR
                                                gray = cv2.cvtColor(crop_processed, cv2.COLOR_BGR2GRAY)
                                            else:
                                                gray = crop_processed
                                            
                                            # Unsharp mask: sharpen = original + (original - blurred) * amount
                                            blurred = cv2.GaussianBlur(gray, (0, 0), 2.0)
                                            sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
                                            
                                            # Convert back to BGR if needed
                                            if len(crop.shape) == 3:
                                                crop_processed = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
                                            else:
                                                crop_processed = sharpened
                                            
                                            # Enhance contrast slightly (helps with faded text)
                                            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
                                            if len(crop_processed.shape) == 3:
                                                lab = cv2.cvtColor(crop_processed, cv2.COLOR_BGR2LAB)
                                                l, a, b = cv2.split(lab)
                                                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                                                l = clahe.apply(l)
                                                crop_processed = cv2.merge([l, a, b])
                                                crop_processed = cv2.cvtColor(crop_processed, cv2.COLOR_LAB2BGR)
                                            else:
                                                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                                                crop_processed = clahe.apply(crop_processed)
                                            
                                            # Try all orientations for processed image
                                            orientation_results = try_ocr_all_orientations(crop_processed, 'original', region_name)
                                            all_ocr_results.extend(orientation_results)
                                        except Exception as e:
                                            if frame_count % 50 == 0:
                                                print(f"[VideoProcessor] OCR strategy 0 ({region_name}) failed: {e}")
                                    
                                    # Strategy 0b: Morphological preprocessing (character separation)
                                    # This helps separate touching characters and remove small noise
                                    try:
                                        h_crop, w_crop = crop_gray.shape
                                        
                                        # Aggressive upscaling for small text
                                        min_text_height = 60
                                        if h_crop < min_text_height:
                                            scale = max(3.0, min_text_height / h_crop)
                                            crop_up = cv2.resize(crop_gray, None, fx=scale, fy=scale,
                                                               interpolation=cv2.INTER_CUBIC)
                                        elif w_crop < 100:
                                            scale = max(2.0, 100 / w_crop)
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
                                        
                                        # Try both normal and inverted, with all orientations
                                        crop_morph_bgr = cv2.cvtColor(crop_morph, cv2.COLOR_GRAY2BGR)
                                        orientation_results = try_ocr_all_orientations(crop_morph_bgr, 'morphological', region_name)
                                        all_ocr_results.extend(orientation_results)
                                        
                                        # Inverted version with all orientations
                                        crop_morph_inv = cv2.bitwise_not(crop_morph)
                                        crop_morph_inv_bgr = cv2.cvtColor(crop_morph_inv, cv2.COLOR_GRAY2BGR)
                                        orientation_results_inv = try_ocr_all_orientations(crop_morph_inv_bgr, 'morphological-inv', region_name)
                                        all_ocr_results.extend(orientation_results_inv)
                                    except Exception as e:
                                        print(f"[VideoProcessor] OCR strategy 0b ({region_name}) failed: {e}")
                                    
                                    # Strategy 1: CLAHE + Border removal (for low-contrast text)
                                    try:
                                        h_crop, w_crop = crop_gray.shape
                                        
                                        # Aggressive upscaling for small text
                                        min_text_height = 60
                                        if h_crop < min_text_height:
                                            scale = max(3.0, min_text_height / h_crop)
                                            crop_up = cv2.resize(crop_gray, None, fx=scale, fy=scale, 
                                                               interpolation=cv2.INTER_CUBIC)
                                        elif w_crop < 100:
                                            scale = max(2.0, 100 / w_crop)
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
                                        orientation_results = try_ocr_all_orientations(crop_clahe_bgr, 'clahe-sharp', region_name)
                                        all_ocr_results.extend(orientation_results)
                                    except Exception as e:
                                        print(f"[VideoProcessor] OCR strategy 1 ({region_name}) failed: {e}")
                                    
                                    # Strategy 2: Adaptive thresholding (best for varying lighting)
                                    # Handles shadows and uneven illumination
                                    try:
                                        h_crop, w_crop = crop_gray.shape
                                        
                                        # Aggressive upscaling for small text
                                        min_text_height = 60
                                        if h_crop < min_text_height:
                                            scale = max(3.0, min_text_height / h_crop)
                                            crop_up = cv2.resize(crop_gray, None, fx=scale, fy=scale,
                                                               interpolation=cv2.INTER_CUBIC)
                                        elif w_crop < 100:
                                            scale = max(2.0, 100 / w_crop)
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
                                        orientation_results = try_ocr_all_orientations(crop_adaptive_bgr, 'adaptive', region_name)
                                        all_ocr_results.extend(orientation_results)
                                        
                                        # Inverted adaptive threshold with all orientations
                                        crop_adaptive_inv = cv2.bitwise_not(crop_adaptive)
                                        crop_adaptive_inv_bgr = cv2.cvtColor(crop_adaptive_inv, cv2.COLOR_GRAY2BGR)
                                        orientation_results_inv = try_ocr_all_orientations(crop_adaptive_inv_bgr, 'adaptive-inv', region_name)
                                        all_ocr_results.extend(orientation_results_inv)
                                    except Exception as e:
                                        print(f"[VideoProcessor] OCR strategy 2 ({region_name}) failed: {e}")
                                    
                                    # Strategy 3: High-contrast enhancement with dilation
                                    # Good for faded or low-contrast text
                                    try:
                                        h_crop, w_crop = crop_gray.shape
                                        
                                        # Aggressive upscaling for small text
                                        min_text_height = 60
                                        if h_crop < min_text_height:
                                            scale = max(3.0, min_text_height / h_crop)
                                            crop_up = cv2.resize(crop_gray, None, fx=scale, fy=scale,
                                                               interpolation=cv2.INTER_CUBIC)
                                        elif w_crop < 100:
                                            scale = max(2.0, 100 / w_crop)
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
                                        orientation_results = try_ocr_all_orientations(crop_high_bgr, 'high-contrast', region_name)
                                        all_ocr_results.extend(orientation_results)
                                        
                                        # Inverted version with all orientations
                                        crop_high_inv = cv2.bitwise_not(crop_high)
                                        crop_high_inv_bgr = cv2.cvtColor(crop_high_inv, cv2.COLOR_GRAY2BGR)
                                        orientation_results_inv = try_ocr_all_orientations(crop_high_inv_bgr, 'high-contrast-inv', region_name)
                                        all_ocr_results.extend(orientation_results_inv)
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
                                            # Store OCR result in list (collect ALL results per track)
                                            if ocr_cache_key not in self.track_ocr_results:
                                                self.track_ocr_results[ocr_cache_key] = []
                                            self.track_ocr_results[ocr_cache_key].append({
                                                'text': text,
                                                'conf': conf_ocr,
                                                'frame': frame_count
                                            })
                                        print(f"[VideoProcessor] Frame {frame_count}, Track {track_id}: OCR ACCEPTED '{text}' (conf: {conf_ocr:.3f}) for {ocr_cache_key}")
                                    else:
                                        if frame_count % 10 == 0:
                                            print(f"[VideoProcessor] Frame {frame_count}, Track {track_id}: No valid OCR result (all attempts failed)")
                                        text = ""
                                        conf_ocr = 0.0
                                        # Store empty result to avoid re-running OCR (use same key as cache check)
                                        with self.lock:
                                            if ocr_cache_key not in self.track_ocr_results:
                                                self.track_ocr_results[ocr_cache_key] = []
                                            self.track_ocr_results[ocr_cache_key].append({
                                                'text': '',
                                                'conf': 0.0,
                                                'frame': frame_count
                                            })
                                except Exception as e:
                                    print(f"[VideoProcessor] Error in OCR at frame {frame_count}, track {track_id}: {e}")
                                    import traceback
                                    traceback.print_exc()
                                    # Store empty result to avoid re-running OCR (use same key as cache check)
                                    with self.lock:
                                        if ocr_cache_key not in self.track_ocr_results:
                                            self.track_ocr_results[ocr_cache_key] = []
                                        self.track_ocr_results[ocr_cache_key].append({
                                            'text': '',
                                            'conf': 0.0,
                                            'frame': frame_count
                                        })
                                
                                # Note: OCR text will be drawn below for all tracks
                        
                        # Draw OCR text in red for ALL tracks (whether OCR was just run or done earlier)
                        if text:
                            ocr_conf_percent = conf_ocr * 100.0
                            ocr_text = f"TEXT: {text}"
                            ocr_conf_text = f"Conf: {ocr_conf_percent:.1f}%"
                            
                            # Position text below the bounding box
                            h, w = processed_frame.shape[:2]
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
                        
                        # Get OCR result for this track (if available)
                        # During processing, we use the most recent result
                        # After consolidation, we use the consolidated result
                        with self.lock:
                            if ocr_cache_key in self.track_ocr_results:
                                ocr_data = self.track_ocr_results[ocr_cache_key]
                                
                                # Check if it's a list (during processing) or dict (after consolidation)
                                if isinstance(ocr_data, list) and len(ocr_data) > 0:
                                    # During processing: use most recent non-empty result
                                    non_empty_results = [r for r in ocr_data if r.get('text', '').strip()]
                                    if non_empty_results:
                                        latest_result = non_empty_results[-1]
                                        text = latest_result['text']
                                        conf_ocr = latest_result['conf']
                                        if frame_count % 30 == 0:  # Log occasionally
                                            print(f"[VideoProcessor] Frame {frame_count}, Track {track_id}: Using OCR for {ocr_cache_key}: '{text}' (conf: {conf_ocr:.3f})")
                                elif isinstance(ocr_data, dict):
                                    # After consolidation: use consolidated result
                                    text = ocr_data.get('text', '')
                                    conf_ocr = ocr_data.get('conf', 0.0)
                        
                        with self.lock:
                            # Check if this is a new unique physical trailer (by spot)
                            is_new_unique_track = track_key not in self.unique_track_keys
                            
                            if is_new_unique_track:
                                # New unique physical trailer - count it and add to set
                                self.unique_track_keys.add(track_key)
                                self.stats['tracks'] += 1
                            
                            # Get existing event to check if OCR text changed
                            existing_event = self.unique_tracks.get(track_key)
                            
                            # Determine if we should add this event to the events list
                            # Only add if:
                            # 1. It's a new unique physical trailer, OR
                            # 2. OCR text has changed (to show OCR progress)
                            should_add_event = False
                            
                            if is_new_unique_track:
                                # New unique physical trailer - always add
                                should_add_event = True
                            elif text and text.strip():  # OCR result available
                                # Check if OCR text has changed from existing event
                                if existing_event:
                                    existing_text = existing_event.get('text', '').strip()
                                    new_text = text.strip()
                                    if existing_text != new_text:
                                        # OCR text changed - update
                                        should_add_event = True
                                else:
                                    # No existing event but we have text - add it
                                    should_add_event = True
                            
                            # Update unique tracks dictionary (keep latest event for each unique physical trailer)
                            # This always happens to keep the latest event
                            self.unique_tracks[track_key] = event
                            
                            # Only add to events list if this is a new unique trailer or OCR text changed
                            if should_add_event:
                                self.events.append(event)
                                frame_events.append(event)
                            # Otherwise, skip adding duplicate event for same physical trailer
                        
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
                
                # Note: Green boxes are already drawn for all tracks in the loop above (lines 244-273)
                # No need to draw persistent boxes here - all active tracks are already displayed
                    
                    # Draw size text
                    cv2.putText(processed_frame, size_text, (text_x, size_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Draw OCR text in red if available
                    if text:
                        ocr_conf_percent = conf_ocr * 100.0
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
            
            # Post-processing: Merge nearby clusters and consolidate OCR results
            print(f"[VideoProcessor] Post-processing: Merging clusters and consolidating OCR...")
            self._merge_nearby_clusters()
            self._consolidate_ocr_results()
            
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
    
    def _merge_nearby_clusters(self):
        """Merge position clusters that are very close together (within 3m) and in the same spot."""
        # This helps reduce overcounting when trailers are parked close together
        # Be conservative - only merge clusters that are very close (3m) and in the same spot
        
        with self.lock:
            # Group clusters by spot first
            clusters_by_spot = {}  # spot -> [cluster_ids]
            for cid in self.position_clusters.keys():
                # Find which spot this cluster belongs to
                spot = None
                for track_key in self.unique_tracks.keys():
                    if f"_cluster_{cid}" in track_key:
                        parts = track_key.split("_cluster_")
                        if len(parts) == 2:
                            spot = parts[0]
                            break
                
                if spot is None:
                    spot = "unknown"
                
                if spot not in clusters_by_spot:
                    clusters_by_spot[spot] = []
                clusters_by_spot[spot].append(cid)
            
            clusters_to_merge = []
            cluster_ids = list(self.position_clusters.keys())
            
            # Find clusters that should be merged
            # Be very conservative - only merge clusters that are very close together (within 3m)
            # and in the same spot. This prevents merging different trailers that happen to be
            # in the same parking spot but are physically separate.
            MERGE_DISTANCE_THRESHOLD = 3.0  # 3 meters - very conservative
            
            for i, cid1 in enumerate(cluster_ids):
                for cid2 in cluster_ids[i+1:]:
                    cluster1 = self.position_clusters[cid1]
                    cluster2 = self.position_clusters[cid2]
                    
                    dist = np.sqrt((cluster1['center_x'] - cluster2['center_x'])**2 + 
                                  (cluster1['center_y'] - cluster2['center_y'])**2)
                    
                    # Check if clusters are in the same spot
                    spot1 = None
                    spot2 = None
                    for track_key in self.unique_tracks.keys():
                        if f"_cluster_{cid1}" in track_key:
                            parts = track_key.split("_cluster_")
                            if len(parts) == 2:
                                spot1 = parts[0]
                        if f"_cluster_{cid2}" in track_key:
                            parts = track_key.split("_cluster_")
                            if len(parts) == 2:
                                spot2 = parts[0]
                    
                    # Only merge if:
                    # 1. Clusters are in the same spot (or both unknown)
                    # 2. Distance is within threshold - indicates same physical trailer
                    # Use larger threshold for same-spot clusters since trailers in the same spot
                    # are likely the same physical trailer, even if positions vary slightly
                    should_merge = False
                    if spot1 == spot2 and spot1 != "unknown":  # Same spot (not unknown)
                        # For same-spot clusters, be conservative to avoid merging different trailers
                        # Strategy: Only merge if one cluster is very small (sparse detection) OR
                        # clusters are very close together (same trailer with position variance)
                        min_count = min(cluster1['count'], cluster2['count'])
                        max_count = max(cluster1['count'], cluster2['count'])
                        
                        # If both clusters have substantial detections (likely different trailers)
                        if min_count >= 5 and max_count >= 10:
                            # Allow merging for position variance (6.5m) - handles same trailer at different positions
                            # This merges clusters 1-2 (6.2m apart) but not clusters 2-3 (8.1m apart)
                            SAME_SPOT_THRESHOLD = 6.5
                        elif min_count < 3:
                            # One cluster is very small (sparse detection) - can merge from further away
                            # Use adaptive threshold based on the larger cluster's size
                            if max_count >= 5:
                                # Small cluster near a medium/large cluster - likely sparse detection (9.5m)
                                # This merges clusters 3-4 (9.1m apart, count=1 and 9)
                                SAME_SPOT_THRESHOLD = 9.5
                            else:
                                # Both clusters are small - be more conservative (7m)
                                SAME_SPOT_THRESHOLD = 7.0
                        else:
                            # One medium-sized cluster - moderate threshold (4m)
                            SAME_SPOT_THRESHOLD = 4.0
                        
                        if dist <= SAME_SPOT_THRESHOLD:
                            should_merge = True
                    elif spot1 == spot2 == "unknown":  # Both unknown spots
                        # For unknown spots, use smaller threshold (5m) to be more conservative
                        if dist <= 5.0:
                            should_merge = True
                    # Never merge clusters from different known spots, even if close
                    
                    if should_merge:
                        # Mark for merging (merge into the cluster with more points)
                        if cluster1['count'] >= cluster2['count']:
                            clusters_to_merge.append((cid2, cid1))  # Merge cid2 into cid1
                        else:
                            clusters_to_merge.append((cid1, cid2))  # Merge cid1 into cid2
            
            # Apply merges (merge into the cluster with more points)
            # First, resolve transitive merges (if A->B and B->C, then A->C)
            # Build initial merge map
            merge_map = {}  # final merge destination for each cluster
            for merge_from, merge_to in clusters_to_merge:
                merge_map[merge_from] = merge_to
            
            # Resolve all transitive merges to find final destinations
            final_merge_map = {}
            for merge_from in merge_map.keys():
                # Follow the chain to find the final destination
                current = merge_from
                visited = set()
                while current in merge_map and current not in visited:
                    visited.add(current)
                    current = merge_map[current]
                final_merge_map[merge_from] = current
            
            # Now apply merges using the resolved destinations
            # Group merges by destination to process efficiently
            merges_by_dest = {}
            for merge_from, merge_to in final_merge_map.items():
                if merge_from == merge_to:
                    continue  # No-op
                if merge_from not in self.position_clusters:
                    continue  # Already deleted
                if merge_to not in self.position_clusters:
                    continue  # Destination doesn't exist
                
                if merge_to not in merges_by_dest:
                    merges_by_dest[merge_to] = []
                merges_by_dest[merge_to].append(merge_from)
            
            # Apply all merges
            merged_into = {}
            for merge_to, merge_froms in merges_by_dest.items():
                if merge_to not in self.position_clusters:
                    continue
                
                cluster_to = self.position_clusters[merge_to]
                
                for merge_from in merge_froms:
                    if merge_from not in self.position_clusters:
                        continue  # Already deleted
                    if merge_from == merge_to:
                        continue  # No-op
                    
                    cluster_from = self.position_clusters[merge_from]
                    
                    # Combine positions
                    if 'positions' in cluster_from:
                        if 'positions' not in cluster_to:
                            cluster_to['positions'] = []
                        cluster_to['positions'].extend(cluster_from['positions'])
                    
                    # Recalculate center (weighted average)
                    total_count = cluster_to['count'] + cluster_from['count']
                    cluster_to['center_x'] = (cluster_to['center_x'] * cluster_to['count'] + 
                                             cluster_from['center_x'] * cluster_from['count']) / total_count
                    cluster_to['center_y'] = (cluster_to['center_y'] * cluster_to['count'] + 
                                             cluster_from['center_y'] * cluster_from['count']) / total_count
                    cluster_to['count'] = total_count
                    
                    # Mark as merged
                    merged_into[merge_from] = merge_to
                    
                    # Delete the merged cluster
                    del self.position_clusters[merge_from]
            
            # Update track keys in unique_tracks and track_ocr_results
            # Find all spots that have clusters that were merged
            spot_cluster_updates = {}  # old_key -> new_key
            
            for merge_from, merge_to in final_merge_map.items():
                # Find all track keys that use the merged cluster
                for track_key in list(self.unique_tracks.keys()):
                    if f"_cluster_{merge_from}" in track_key:
                        # Extract spot from track_key (format: "spot_cluster_id")
                        parts = track_key.split("_cluster_")
                        if len(parts) == 2:
                            spot = parts[0]
                            new_track_key = f"{spot}_cluster_{merge_to}"
                            spot_cluster_updates[track_key] = new_track_key
            
            # Apply updates to unique_tracks
            for old_key, new_key in spot_cluster_updates.items():
                if old_key in self.unique_tracks:
                    # Merge events: keep the most recent one
                    old_event = self.unique_tracks[old_key]
                    if new_key in self.unique_tracks:
                        new_event = self.unique_tracks[new_key]
                        # Keep the most recent event
                        if old_event.get('ts_iso', '') > new_event.get('ts_iso', ''):
                            self.unique_tracks[new_key] = old_event
                    else:
                        self.unique_tracks[new_key] = old_event
                    del self.unique_tracks[old_key]
            
            # Apply updates to track_ocr_results
            for old_key, new_key in spot_cluster_updates.items():
                if old_key in self.track_ocr_results:
                    old_ocr = self.track_ocr_results[old_key]
                    if new_key in self.track_ocr_results:
                        # Merge OCR results: combine lists if both are lists
                        new_ocr = self.track_ocr_results[new_key]
                        if isinstance(old_ocr, list) and isinstance(new_ocr, list):
                            # Combine lists
                            self.track_ocr_results[new_key] = old_ocr + new_ocr
                        elif isinstance(old_ocr, dict) and isinstance(new_ocr, dict):
                            # Both consolidated: keep the one with higher confidence
                            if old_ocr.get('conf', 0.0) > new_ocr.get('conf', 0.0):
                                self.track_ocr_results[new_key] = old_ocr
                    else:
                        self.track_ocr_results[new_key] = old_ocr
                    del self.track_ocr_results[old_key]
            
            # Update unique_track_keys set
            for old_key, new_key in spot_cluster_updates.items():
                if old_key in self.unique_track_keys:
                    self.unique_track_keys.remove(old_key)
                    self.unique_track_keys.add(new_key)
            
            if merged_into:
                print(f"[VideoProcessor] Merged {len(merged_into)} clusters, remaining: {len(self.position_clusters)}")
                if spot_cluster_updates:
                    print(f"[VideoProcessor] Updated {len(spot_cluster_updates)} track keys after cluster merge")
            else:
                print(f"[VideoProcessor] No clusters merged (found {len(cluster_ids)} clusters)")
                # Debug: print cluster positions
                for cid, cluster in self.position_clusters.items():
                    print(f"  Cluster {cid}: center=({cluster['center_x']:.2f}, {cluster['center_y']:.2f}), count={cluster['count']}")
    
    def _consolidate_ocr_results(self):
        """Consolidate all OCR results per track into a single best result."""
        with self.lock:
            consolidated = {}
            
            for track_key, ocr_results_list in self.track_ocr_results.items():
                if not isinstance(ocr_results_list, list) or len(ocr_results_list) == 0:
                    continue
                
                # Filter out empty results
                non_empty_results = [r for r in ocr_results_list if r.get('text', '').strip()]
                
                if not non_empty_results:
                    # No valid OCR results
                    consolidated[track_key] = {
                        'text': '',
                        'conf': 0.0,
                        'frame_first_seen': ocr_results_list[0].get('frame', 0) if ocr_results_list else 0
                    }
                    continue
                
                # Consolidation strategy: Use voting/consensus
                # Count occurrences of each unique text
                text_counts = {}
                text_confidences = {}
                
                for result in non_empty_results:
                    text = result['text'].strip()
                    conf = result.get('conf', 0.0)
                    
                    if text not in text_counts:
                        text_counts[text] = 0
                        text_confidences[text] = []
                    
                    text_counts[text] += 1
                    text_confidences[text].append(conf)
                
                # Find the best text using multiple criteria:
                # 1. Prefer longer alphanumeric text (trailer IDs like "JBHU 249546")
                # 2. Prefer text with alphanumeric patterns (letters + numbers)
                # 3. Prefer higher frequency and confidence
                # 4. Filter out obviously garbage text (single chars, special chars only)
                
                def is_valid_trailer_text(text):
                    """Check if text looks like a valid trailer identifier."""
                    if not text or len(text.strip()) < 2:
                        return False
                    # Should contain at least one letter and one number, or be a known pattern
                    has_letter = any(c.isalpha() for c in text)
                    has_number = any(c.isdigit() for c in text)
                    # Allow patterns like "J.B.HUNT", "JBHU", etc.
                    if has_letter and (has_number or len(text) >= 4):
                        return True
                    # Allow pure numbers if long enough (like "249546")
                    if has_number and len(text) >= 4 and not has_letter:
                        return True
                    return False
                
                def text_quality_score(text):
                    """Calculate quality score for text."""
                    score = 0.0
                    # Length bonus (longer is better, up to a point)
                    score += min(len(text), 20) * 0.1
                    # Alphanumeric bonus
                    has_letter = any(c.isalpha() for c in text)
                    has_number = any(c.isdigit() for c in text)
                    if has_letter and has_number:
                        score += 5.0  # Strong preference for alphanumeric
                    elif has_letter:
                        score += 2.0
                    elif has_number:
                        score += 1.0
                    # Penalize special characters (but allow common ones like ".", "-", " ")
                    special_chars = sum(1 for c in text if not c.isalnum() and c not in " .-")
                    score -= special_chars * 0.5
                    return score
                
                # First, try to find and combine company name + number from different OCR results
                # This should ALWAYS be tried, even if we have valid text, because combining
                # "J.B.HUNT" from one frame with "249546" from another is better than either alone
                company_patterns = ["J.B.HUNT", "JBHUNT", "JBHU", "J.B.HUNTA"]
                company_results = []  # (text, count, conf, pattern)
                number_results = []   # (number, count, conf)
                
                for text, count in text_counts.items():
                    text_upper = text.upper().strip()
                    avg_conf = np.mean(text_confidences[text])
                    
                    # Check if it contains a company name pattern
                    for pattern in company_patterns:
                        if pattern in text_upper:
                            company_results.append((text, count, avg_conf, pattern))
                            break
                    
                    # Extract numbers with better validation
                    # Trailer IDs are typically 6 digits (e.g., "249546")
                    # But can also be 4-8 digits
                    # Extract all number sequences first
                    all_numbers = re.findall(r'\d+', text)
                    
                    # Prefer numbers that are 4-8 digits (typical trailer ID length)
                    # Reject numbers that are too long (likely concatenated wrong)
                    for num in all_numbers:
                        num_len = len(num)
                        if 4 <= num_len <= 8:
                            # Ideal length - full confidence
                            number_results.append((num, count, avg_conf))
                        elif num_len == 3:
                            # Short but might be part of ID - lower confidence
                            number_results.append((num, count, avg_conf * 0.7))
                        elif num_len > 8:
                            # Too long - likely wrong concatenation, skip
                            continue
                        elif num_len < 3:
                            # Too short - skip
                            continue
                    
                    # Also try extracting from strings like "8a727-21700" but be more careful
                    # Only if the original text has a clear pattern
                    if '-' in text or len(re.findall(r'\d+', text)) > 1:
                        # Has separators or multiple number groups - might be formatted like "8a727-21700"
                        # Extract longest sequence of 4-8 digits
                        digits_only = re.sub(r'[^\d]', '', text)
                        if 4 <= len(digits_only) <= 8:
                            # Check if this appears frequently (more likely to be correct)
                            if count >= 2:  # Appeared at least twice
                                number_results.append((digits_only, count, avg_conf * 0.85))
                        elif len(digits_only) > 8:
                            # Too long - likely wrong, but try to find valid sub-sequences
                            # Look for 6-digit sequences (most common trailer ID length)
                            for i in range(len(digits_only) - 5):
                                candidate = digits_only[i:i+6]
                                if candidate and len(candidate) == 6:
                                    number_results.append((candidate, count, avg_conf * 0.8))
                
                # Try to combine company name with number (ALWAYS try this first)
                combined_text = None
                combined_score = -1
                combined_conf = 0.0
                
                if company_results and number_results:
                    # Find best company name (highest count * confidence)
                    best_company = max(company_results, key=lambda x: x[1] * (1.0 + x[2]))
                    
                    # Find best number with better validation
                    # Prefer numbers that:
                    # 1. Are 6 digits (most common trailer ID length)
                    # 2. Appeared frequently
                    # 3. Have high confidence
                    def number_score(num_tuple):
                        num, num_count, num_conf = num_tuple
                        num_len = len(num)
                        # Base score
                        score = num_count * (1.0 + num_conf)
                        # Bonus for 6 digits (most common)
                        if num_len == 6:
                            score *= 1.5
                        elif num_len == 5 or num_len == 7:
                            score *= 1.2
                        elif num_len == 4 or num_len == 8:
                            score *= 1.1
                        # Penalty for very long numbers (likely wrong)
                        if num_len > 8:
                            score *= 0.5
                        return score
                    
                    best_number = max(number_results, key=number_score)
                    
                    # Validate the combination makes sense
                    num_len = len(best_number[0])
                    if 4 <= num_len <= 8:  # Valid trailer ID length
                        combined_text = f"{best_company[3]} {best_number[0]}"
                        combined_conf = (best_company[2] + best_number[2]) / 2.0
                        combined_score = (best_company[1] + best_number[1]) * (1.0 + combined_conf) + text_quality_score(combined_text) * 5.0
                    else:
                        # Number is invalid length - don't combine
                        combined_text = None
                        combined_score = -1
                        combined_conf = 0.0
                
                # Now score all individual text results
                best_text = None
                best_score = -1
                best_avg_conf = 0.0
                
                for text, count in text_counts.items():
                    avg_conf = np.mean(text_confidences[text])
                    
                    # Base score from frequency and confidence
                    base_score = count * (1.0 + avg_conf)
                    
                    # Quality bonus for valid trailer text
                    quality_bonus = 0.0
                    if is_valid_trailer_text(text):
                        quality_bonus = text_quality_score(text) * 2.0  # Strong bonus for valid text
                    else:
                        # Small penalty for invalid text
                        quality_bonus = -2.0
                    
                    # Combined score
                    score = base_score + quality_bonus
                    
                    if score > best_score:
                        best_score = score
                        best_text = text
                        best_avg_conf = avg_conf
                
                # Use combined text if it's better than the best individual text
                if combined_text and combined_score > best_score:
                    best_text = combined_text
                    best_score = combined_score
                    best_avg_conf = combined_conf
                    print(f"[VideoProcessor] OCR consolidation for {track_key}: Combined '{combined_text}' (score: {combined_score:.2f})")
                elif combined_text:
                    print(f"[VideoProcessor] OCR consolidation for {track_key}: Combined '{combined_text}' (score: {combined_score:.2f}) < best '{best_text}' (score: {best_score:.2f})")
                
                # Store consolidated result
                consolidated[track_key] = {
                    'text': best_text if best_text else '',
                    'conf': best_avg_conf,
                    'frame_first_seen': min(r.get('frame', 0) for r in ocr_results_list),
                    'total_results': len(ocr_results_list),
                    'unique_texts': len(text_counts)
                }
            
            # Replace the list-based storage with consolidated results
            self.track_ocr_results = consolidated
            
            print(f"[VideoProcessor] Consolidated OCR results for {len(consolidated)} tracks")
    
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
            # Get unique tracks count (based on unique track keys after cluster merging)
            unique_tracks_count = len(self.unique_track_keys)
            
            # Get deduplicated events with consolidated OCR results
            deduplicated_events = {}
            for track_key, event in self.unique_tracks.items():
                # Update event with consolidated OCR result if available
                if track_key in self.track_ocr_results:
                    ocr_data = self.track_ocr_results[track_key]
                    if isinstance(ocr_data, dict):
                        # Use consolidated OCR result
                        event['text'] = ocr_data.get('text', '')
                        event['conf'] = ocr_data.get('conf', 0.0)
                
                # Use track_key as the deduplication key (already includes spot + cluster)
                if track_key not in deduplicated_events:
                    deduplicated_events[track_key] = event
                else:
                    # Compare timestamps and keep the most recent
                    existing_ts = deduplicated_events[track_key].get('ts_iso', '')
                    new_ts = event.get('ts_iso', '')
                    if new_ts > existing_ts:
                        deduplicated_events[track_key] = event
            
            # Convert to list and sort by timestamp (most recent first)
            unique_events = list(deduplicated_events.values())
            unique_events.sort(key=lambda x: x.get('ts_iso', ''), reverse=True)
            
            # Take last 100 (most recent)
            unique_events = unique_events[:100]
            
            # Count unique OCR results (non-empty)
            unique_ocr_count = sum(1 for track_key, ocr_data in self.track_ocr_results.items() 
                                 if isinstance(ocr_data, dict) and ocr_data.get('text', '').strip())
            
            return {
                'frames_processed': self.stats['frames_processed'],
                'detections': self.stats['detections'],
                'tracks': unique_tracks_count,  # Use unique tracks count after cluster merging
                'ocr_results': unique_ocr_count,  # Count of unique tracks with OCR results
                'events': unique_events,  # Deduplicated events with consolidated OCR
                'total_frames_stored': len(self.processed_frames)  # Debug info
            }
    
    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Get a processed frame by frame number."""
        with self.lock:
            return self.processed_frames.get(frame_number)

