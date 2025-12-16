"""
Video Processor for Testing

Processes uploaded video files using the same pipeline as live camera feeds.
"""

import cv2
import numpy as np
import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Generator, Tuple
from datetime import datetime
import threading
import gc

# Try to import torch for GPU memory management
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class VideoProcessor:
    """
    Video processor for testing uploaded video files.
    """
    
    def __init__(self, detector, ocr, tracker_factory, spot_resolver, homography=None, preprocessor=None, gps_reference=None):
        """
        Initialize video processor.
        
        Args:
            detector: Detection model instance (should be YOLOv8Detector for YOLOv8m COCO truck detection)
            ocr: OCR model instance
            tracker_factory: Function that returns a new tracker instance
            spot_resolver: Spot resolver instance
            homography: Optional homography matrix for world projection
            preprocessor: Optional ImagePreprocessor instance for OCR preprocessing
            gps_reference: Optional dict with 'lat' and 'lon' keys for GPS conversion
        """
        self.detector = detector
        self.ocr = ocr
        self.tracker_factory = tracker_factory
        self.spot_resolver = spot_resolver
        self.homography = homography
        self.preprocessor = preprocessor
        self.gps_reference = gps_reference  # Dict with 'lat' and 'lon' keys
        
        # Log detector type for verification
        if self.detector is not None:
            detector_type = type(self.detector).__name__
            print(f"[VideoProcessor] Using detector: {detector_type}")
            if hasattr(self.detector, 'model_name'):
                print(f"[VideoProcessor] Detector model: {self.detector.model_name}")
            if hasattr(self.detector, 'target_class'):
                # Get actual class name from model
                class_name = "unknown"
                if hasattr(self.detector, 'class_names'):
                    class_name = self.detector.class_names.get(self.detector.target_class, f'class_{self.detector.target_class}')
                print(f"[VideoProcessor] Detector target class: {self.detector.target_class} ({class_name})")
        
        # Check if OCR is oLmOCR (needs GPU memory cleanup)
        self.is_olmocr = ocr is not None and 'OlmOCRRecognizer' in type(ocr).__name__
        
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
        
        # Two-stage processing: Save cropped trailers for batch OCR
        self.save_crops = True  # Enable crop saving mode
        self.crops_dir = None  # Will be set per video
        self.saved_crops = []  # List of saved crop metadata: [{crop_path, frame_count, track_id, bbox, ...}]
        self.crop_counter = 0  # Counter for unique crop filenames
        self.crop_deduplication = {}  # track_key -> crop_path (ONE crop per unique physical trailer)
        
        # Position clustering: group nearby positions (within 5m) as the same physical location
        # This handles slight movement and position variance
        # Increased tolerance to better handle trailers parked close together
        self.position_clusters = {}  # cluster_id -> {center_x, center_y, count, positions: [(x, y), ...]}
        self.next_cluster_id = 1
    
        # Last detected nearest trailer (for persistent display)
        self.last_nearest_trailer = None  # {bbox, track_id, conf, text, ocr_conf}
    
    def _cleanup_gpu_memory(self):
        """
        Clean up GPU memory after OCR operations to prevent accumulation.
        This is critical when processing multiple frames in sequence.
        """
        if not TORCH_AVAILABLE or not self.is_olmocr:
            return
        
        if torch.cuda.is_available():
            try:
                # Wait for all GPU operations to complete
                torch.cuda.synchronize()
                # Clear CUDA cache
                torch.cuda.empty_cache()
                # Force Python garbage collection
                gc.collect()
                # Clear cache again after GC
                torch.cuda.empty_cache()
            except Exception:
                # Silently fail - don't interrupt video processing
                pass
    
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
            # Reset position clusters for new video processing
            self.position_clusters = {}  # cluster_id -> {center_x, center_y, count, positions: [(x, y), ...]}
            self.next_cluster_id = 1
            # Reset OCR results tracking
            self.track_ocr_results = {}  # track_key -> list of {text, conf, frame}
            
            # Two-stage processing: Initialize crop saving
            import os
            from pathlib import Path
            video_name = Path(video_path).stem
            self.crops_dir = Path("out/crops") / camera_id / video_name
            self.crops_dir.mkdir(parents=True, exist_ok=True)
            self.saved_crops = []
            self.crop_counter = 0
            self.crop_deduplication = {}  # track_key -> crop_path (ONE crop per unique physical trailer)
            print(f"[VideoProcessor] Crop directory: {self.crops_dir}")
            
            self.stats = {
                'frames_processed': 0,
                'detections': 0,
                'tracks': 0,
                'ocr_results': 0,
                'crops_saved': 0
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
                        try:
                            all_detections = self.detector.detect(frame)
                        except RuntimeError as e:
                            if 'CUDA' in str(e) or 'cuda' in str(e).lower():
                                print(f"[VideoProcessor] CUDA error in detection at frame {frame_count}: {e}")
                                # Try to clear CUDA cache
                                try:
                                    import torch
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                        torch.cuda.synchronize()
                                except:
                                    pass
                                # Continue with empty detections for this frame
                                all_detections = []
                            else:
                                print(f"[VideoProcessor] Runtime error in detection at frame {frame_count}: {e}")
                                all_detections = []
                        except Exception as e:
                            print(f"[VideoProcessor] Error in detection at frame {frame_count}: {e}")
                            import traceback
                            traceback.print_exc()
                            all_detections = []
                        
                        # Filter out very low confidence detections before tracking
                        # This prevents false positive tracks from low-quality detections
                        MIN_DETECTION_CONFIDENCE = 0.10  # Minimum confidence to create a track
                        detections = [d for d in all_detections if d.get('conf', 0.0) >= MIN_DETECTION_CONFIDENCE]
                        
                        filtered_low_conf = len(all_detections) - len(detections)
                        if filtered_low_conf > 0:
                            print(f"[VideoProcessor] Frame {frame_count}: Filtered {filtered_low_conf} low-confidence detections (< {MIN_DETECTION_CONFIDENCE})")
                        
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
                    MIN_ASPECT_RATIO = 0.15   # More lenient - allow very tall/narrow trailers (was 0.5)
                    MAX_ASPECT_RATIO = 10.0   # More lenient - allow very wide or tall
                    
                    filtered_tracks = []
                    filtered_count = 0
                    for track in tracks:
                        bbox = track['bbox']
                        
                        # Validate bbox before processing - skip if contains NaN or Inf
                        if not bbox or len(bbox) != 4:
                            filtered_count += 1
                            continue
                        
                        # Check for NaN or Inf values
                        if any(not np.isfinite(v) for v in bbox):
                            filtered_count += 1
                            if frame_count % 10 == 0:
                                print(f"[VideoProcessor] Filtered Track {track['track_id']}: invalid bbox (NaN/Inf): {bbox}")
                            continue
                        
                        try:
                            x1, y1, x2, y2 = [int(v) for v in bbox]
                        except (ValueError, OverflowError) as e:
                            filtered_count += 1
                            if frame_count % 10 == 0:
                                print(f"[VideoProcessor] Filtered Track {track['track_id']}: cannot convert bbox to int: {bbox}, error: {e}")
                            continue
                        
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
                    
                    # Additional deduplication: Merge tracks with very high IoU (likely same trailer)
                    # This helps catch cases where ByteTrack creates duplicate tracks for the same trailer
                    tracks = self._deduplicate_overlapping_tracks(tracks, frame_count)
                    
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
                        # Validate bbox before processing
                        if not bbox or len(bbox) != 4 or any(not np.isfinite(v) for v in bbox):
                            continue
                        try:
                            x1, y1, x2, y2 = [int(v) for v in bbox]
                        except (ValueError, OverflowError):
                            continue
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
                        
                        # Validate bbox before processing
                        if not bbox or len(bbox) != 4 or any(not np.isfinite(v) for v in bbox):
                            continue
                        try:
                            x1, y1, x2, y2 = [int(v) for v in bbox]
                        except (ValueError, OverflowError):
                            continue
                        
                        # Ensure bbox is valid
                        if x2 <= x1 or y2 <= y1:
                            continue
                        
                        # Store original bbox (which is the full YOLO detection)
                        # This is the HIGH QUALITY bbox from YOLO that captures the whole trailer backside
                        orig_x1, orig_y1, orig_x2, orig_y2 = x1, y1, x2, y2
                        
                        # For DISPLAY: Clip bbox to frame boundaries (to prevent drawing errors)
                        # But for CROPS: Use original bbox to preserve full quality
                        frame_height, frame_width = frame.shape[:2]
                        display_x1 = max(0, min(x1, frame_width - 1))
                        display_y1 = max(0, min(y1, frame_height - 1))
                        display_x2 = max(display_x1 + 1, min(x2, frame_width))
                        display_y2 = max(display_y1 + 1, min(y2, frame_height))
                        
                        # Re-validate display bbox after clipping
                        if display_x2 <= display_x1 or display_y2 <= display_y1:
                            continue
                        
                        # For CROPS: Use original YOLO bbox (full quality, captures whole trailer)
                        # Only clip crop bbox if it's completely outside frame (to prevent errors)
                        crop_x1 = max(0, orig_x1)  # Only clip negative values
                        crop_y1 = max(0, orig_y1)  # Only clip negative values
                        crop_x2 = min(orig_x2, frame_width)  # Only clip if exceeds frame
                        crop_y2 = min(orig_y2, frame_height)  # Only clip if exceeds frame
                        
                        # Validate crop bbox
                        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                            continue
                        
                        # Calculate trailer dimensions (using original bbox for accuracy)
                        width = orig_x2 - orig_x1
                        height = orig_y2 - orig_y1
                        
                        # Draw bounding box for ALL trucks (stable detection display)
                        is_nearest = (track_id == nearest_track_id)
                        track_state = track.get('state', 'tracked')
                        
                        # Color coding:
                        # - Green: Tracked (confirmed detection)
                        # - Yellow: Lost but within buffer (predicted position - maintains stability)
                        if track_state == 'tracked':
                            box_color = (0, 255, 0)  # Green for confirmed
                            thickness = 3
                        else:  # lost
                            box_color = (0, 255, 255)  # Yellow for predicted (maintains stability)
                            thickness = 2
                        
                        # Draw bounding box (thicker for better visibility)
                        # Use display bbox (clipped) for drawing to prevent drawing errors
                        cv2.rectangle(processed_frame, (display_x1, display_y1), (display_x2, display_y2), box_color, thickness)
                            
                        # Draw semi-transparent overlay on the bounding box
                        overlay = processed_frame.copy()
                        cv2.rectangle(overlay, (display_x1, display_y1), (display_x2, display_y2), box_color, -1)
                        cv2.addWeighted(overlay, 0.1, processed_frame, 0.9, 0, processed_frame)
                        
                        # Draw detection confidence and track ID above the box
                        det_conf_percent = det_conf * 100.0
                        if is_nearest:
                            conf_text = f"Track {track_id} - {det_conf_percent:.1f}% [NEAREST]"
                        else:
                            conf_text = f"Track {track_id} - {det_conf_percent:.1f}%"
                        
                        # Add state indicator for lost tracks
                        if track_state == 'lost':
                            conf_text += " [PRED]"
                        text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        
                        # Draw black background for text
                        text_x = display_x1
                        text_y = max(display_y1 - 10, 20)
                        cv2.rectangle(processed_frame, 
                                     (text_x - 2, text_y - text_size[1] - 2), 
                                     (text_x + text_size[0] + 2, text_y + 2),
                                     (0, 0, 0), -1)
                        
                        # Draw text (use same color as box)
                        text_color = box_color
                        cv2.putText(processed_frame, conf_text, (text_x, text_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                        
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
                        # Use original bbox center for more accurate world coordinates
                        world_coords_meters = None
                        world_coords_gps = None
                        if self.homography is not None:
                            try:
                                center_x = (orig_x1 + orig_x2) / 2.0
                                center_y = (orig_y1 + orig_y2) / 2.0
                                point = np.array([[center_x, center_y]], dtype=np.float32)
                                point = np.array([point])
                                projected = cv2.perspectiveTransform(point, self.homography)
                                x_world, y_world = projected[0][0]
                                world_coords_meters = (float(x_world), float(y_world))
                                
                                # Convert to GPS if reference available
                                if self.gps_reference:
                                    from app.gps_utils import meters_to_gps
                                    lat, lon = meters_to_gps(x_world, y_world, 
                                                           self.gps_reference['lat'], 
                                                           self.gps_reference['lon'])
                                    world_coords_gps = (float(lat), float(lon))
                            except Exception as e:
                                pass
                        
                        # Resolve parking spot (spot resolver expects meters, not GPS)
                        spot = "unknown"
                        method = "no-calibration"
                        if world_coords_meters and self.spot_resolver:
                            try:
                                x_world, y_world = world_coords_meters
                                spot_result = self.spot_resolver.resolve(x_world, y_world)
                                spot = spot_result['spot']
                                method = spot_result['method']
                            except Exception as e:
                                pass
                        
                        # Use GPS coordinates for output if available, otherwise use meters
                        world_coords = world_coords_gps if world_coords_gps else world_coords_meters
                        
                        # Calculate track_key and ocr_cache_key NOW (before OCR processing)
                        # Track unique physical trailers using position clustering
                        if spot != "unknown" and world_coords:
                            x_world, y_world = world_coords
                            
                            # Find or create position cluster (tolerance for grouping nearby positions)
                            # Use 5m tolerance to group detections of the same trailer while keeping different trailers separate
                            cluster_id = None
                            CLUSTER_TOLERANCE = 5.0  # 5 meters - balance between grouping same trailer and separating different trailers
                            
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
                            
                            # Use spot + cluster_id + track_id as unique identifier
                            # Including track_id helps distinguish different trailers in the same spot/cluster
                            # Clusters are merged incrementally (every 30 frames), so cluster_id
                            # should already reflect merged clusters by the time we save crops
                            # track_id helps prevent different trailers from sharing the same crop
                            track_key = f"{spot}_cluster_{cluster_id}_track_{track_id}"
                        elif spot != "unknown":
                            # Use spot + track_id if position unavailable
                            track_key = f"{spot}_track_{track_id}"
                        else:
                            # Fallback to track_id if spot is unknown
                            track_key = f"track_{track_id}"
                        
                        # TWO-STAGE APPROACH: Save cropped trailers for batch OCR (no inline OCR)
                        # Stage 1: Save crops during video processing (fast)
                        # Stage 2: Run OCR on saved crops after video processing (batch)
                        
                        text = ""
                        conf_ocr = 0.0
                        
                        # Crop the trailer region using ORIGINAL YOLO bbox (full quality)
                        # This ensures we capture the complete trailer backside with maximum quality
                        # Only minimal clipping applied to prevent out-of-bounds errors
                        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                        if crop.size == 0:
                            continue
                        
                        # OPTIMIZATION: Deduplication - save ONLY ONE crop per unique physical trailer
                        # We crop tracks (after ByteTrack), not detections
                        # Example: 103 detections → 4 unique trailers → 4 crops (one per unique trailer)
                        # Uses track_key (spot + cluster) for deduplication, not track_id (which can change)
                        import hashlib
                        # Use original bbox for hash to maintain consistency
                        bbox_hash = hashlib.md5(f"{orig_x1}_{orig_y1}_{orig_x2}_{orig_y2}_{width}_{height}".encode()).hexdigest()[:8]
                        
                        # Check if we've already saved a crop for this track_key (unique physical trailer)
                        # track_key is stable (spot + cluster), unlike track_id which can change
                        # This ensures we only save ONE crop per unique physical trailer
                        # 
                        # Important: Before saving, check if this cluster was merged into another.
                        # If merged, use the merged cluster's track_key to avoid saving duplicate crops.
                        should_save_crop = True
                        
                        # Resolve to final merged track_key (if cluster was merged)
                        # This ensures we only save one crop per final merged cluster
                        # However, we still want to preserve track_id to distinguish different trailers
                        final_track_key = track_key
                        if "_cluster_" in track_key:
                            # Check if this cluster_id still exists (hasn't been merged)
                            if cluster_id not in self.position_clusters:
                                # Cluster was merged - find the merged cluster's track_key
                                # But preserve track_id to distinguish different trailers
                                merged_base_key = self._resolve_final_track_key(spot, cluster_id)
                                # Extract track_id from original track_key and append to merged key
                                if "_track_" in track_key:
                                    track_id_part = track_key.split("_track_")[-1]
                                    final_track_key = f"{merged_base_key.split('_track_')[0]}_track_{track_id_part}"
                                else:
                                    final_track_key = merged_base_key
                        
                        # CRITICAL: Deduplicate crops by track_id to ensure one crop per unique track
                        # Each track_id represents a different trailer, so we save one crop per track_id
                        # Use track_id-based key for deduplication to ensure we get crops for all unique tracks
                        crop_dedup_key = None
                        if cluster_id is not None:
                            # Use track_id in the deduplication key to ensure one crop per unique track
                            crop_dedup_key = f"{spot}_cluster_{cluster_id}_track_{track_id}"
                        elif spot != "unknown":
                            crop_dedup_key = f"{spot}_track_{track_id}"
                        else:
                            crop_dedup_key = f"track_{track_id}"
                        
                        # Keep cluster_key for logging/display purposes (without track_id)
                        cluster_key = None
                        if cluster_id is not None:
                            cluster_key = f"{spot}_cluster_{cluster_id}"  # Base key without track_id for display
                        
                        # Check if we already have a crop for this track_id
                        # We'll check later if we should replace it with a better crop
                        if crop_dedup_key and crop_dedup_key in self.crop_deduplication:
                            # Crop exists - we'll check if this one is better later
                            pass
                        
                        if cluster_key:
                            # Only save crops for "stable" clusters (seen multiple times)
                            # This prevents saving crops for one-off detections that create new clusters
                            # A cluster is considered stable if it has been seen at least 2 times
                            # Lowered from 3 to 2 to ensure all valid clusters (count >= 2) get crops
                            MIN_CLUSTER_COUNT_FOR_CROP = 2
                            
                            # Check stability using the ORIGINAL cluster_id (the one that was just updated)
                            # This is the cluster that actually exists and has the current count
                            if cluster_id in self.position_clusters:
                                cluster_info = self.position_clusters[cluster_id]
                                cluster_count = cluster_info.get('count', 0)
                                if cluster_count < MIN_CLUSTER_COUNT_FOR_CROP:
                                    # Cluster not stable yet - skip saving crop
                                    should_save_crop = False
                                    # Always log stability skips (not just every 30 frames) for debugging
                                    print(f"[VideoProcessor] Frame {frame_count}: Skipping crop for {track_key} (track_id={track_id}, cluster_id={cluster_id}) - cluster not stable yet (count={cluster_count} < {MIN_CLUSTER_COUNT_FOR_CROP})")
                            else:
                                # Cluster doesn't exist (was merged) - skip saving
                                should_save_crop = False
                                if frame_count % 30 == 0:
                                    print(f"[VideoProcessor] Frame {frame_count}: Skipping crop for {track_key} (track_id={track_id}, cluster_id={cluster_id}) - cluster was merged")
                        else:
                            # No cluster_id means no world_coords or spot unknown - skip saving
                            should_save_crop = False
                            if frame_count % 30 == 0:
                                print(f"[VideoProcessor] Frame {frame_count}: Skipping crop for {track_key} (track_id={track_id}) - no cluster_id (missing world_coords or spot)")
                        
                        # Filter out narrow/partial crops - only save full backside crops
                        # Minimum requirements for a good crop (full trailer backside)
                        MIN_CROP_WIDTH = 200   # Minimum width for full backside (was allowing 80-100px narrow slices)
                        MIN_CROP_HEIGHT = 150  # Minimum height for full backside
                        MIN_CROP_AREA = 30000  # Minimum area (200*150 = 30000)
                        
                        if should_save_crop:
                            # Check crop size requirements
                            if width < MIN_CROP_WIDTH or height < MIN_CROP_HEIGHT or (width * height) < MIN_CROP_AREA:
                                should_save_crop = False
                                if frame_count % 30 == 0:
                                    print(f"[VideoProcessor] Frame {frame_count}: Skipping crop for {track_key} (track_id={track_id}) - crop too small ({width}x{height}, area={width*height}) - need at least {MIN_CROP_WIDTH}x{MIN_CROP_HEIGHT} (area={MIN_CROP_AREA})")
                            
                            # Check if crop is significantly clipped (partial trailer near edge)
                            # Calculate how much of the original bbox is visible in the crop
                            original_width = orig_x2 - orig_x1
                            original_height = orig_y2 - orig_y1
                            original_area = original_width * original_height
                            
                            crop_width = crop_x2 - crop_x1
                            crop_height = crop_y2 - crop_y1
                            crop_area = crop_width * crop_height
                            
                            if original_area > 0:
                                visible_ratio = crop_area / original_area
                                # Require at least 90% of the original bbox to be visible (not clipped)
                                # Increased from 85% to 90% to be more strict about partial crops
                                # This ensures we only save full backside crops, not partial ones
                                MIN_VISIBLE_RATIO = 0.90
                                
                                # Also check if dimensions are significantly reduced (indicates clipping)
                                width_ratio = crop_width / original_width if original_width > 0 else 1.0
                                height_ratio = crop_height / original_height if original_height > 0 else 1.0
                                
                                # Check aspect ratio - trailer backsides should be roughly square or taller than wide
                                # Very wide crops (aspect ratio > 1.8) are likely placards or partial views
                                aspect_ratio = crop_width / crop_height if crop_height > 0 else 0
                                MAX_ASPECT_RATIO = 1.8  # Reject crops wider than 1.8:1 (trailer backsides are typically < 1.5:1)
                                
                                if visible_ratio < MIN_VISIBLE_RATIO or width_ratio < MIN_VISIBLE_RATIO or height_ratio < MIN_VISIBLE_RATIO:
                                    should_save_crop = False
                                    if frame_count % 30 == 0:
                                        print(f"[VideoProcessor] Frame {frame_count}: Skipping crop for {track_key} (track_id={track_id}) - crop is partially clipped (visible={visible_ratio:.1%}, width={width_ratio:.1%}, height={height_ratio:.1%}, original={original_width}x{original_height}, crop={crop_width}x{crop_height})")
                                elif aspect_ratio > MAX_ASPECT_RATIO:
                                    should_save_crop = False
                                    if frame_count % 30 == 0:
                                        print(f"[VideoProcessor] Frame {frame_count}: Skipping crop for {track_key} (track_id={track_id}) - crop is too wide (aspect={aspect_ratio:.2f} > {MAX_ASPECT_RATIO}, size={crop_width}x{crop_height}) - likely placard or partial view")
                        
                        # Save crop if it's a new unique track and save_crops is enabled
                        if should_save_crop and self.save_crops:
                            try:
                                # CRITICAL: Don't save if crop already exists for this track_id (unless significantly better)
                                # This prevents multiple crops for the same track_id, but allows different track_ids to have crops
                                if crop_dedup_key and crop_dedup_key in self.crop_deduplication:
                                    # Find existing crop metadata
                                    existing_crop = None
                                    for crop_meta in self.saved_crops:
                                        if crop_meta.get('track_key') == crop_dedup_key:
                                            existing_crop = crop_meta
                                            break
                                    
                                    if existing_crop:
                                        existing_area = existing_crop.get('width', 0) * existing_crop.get('height', 0)
                                        existing_conf = existing_crop.get('det_conf', 0.0)
                                        current_area = width * height
                                        
                                        # Only replace if current crop is significantly better (larger area or much higher confidence)
                                        # Prefer larger area (full backside) over slightly higher confidence
                                        replace_existing = False
                                        if current_area > existing_area * 1.2:  # 20% larger
                                            replace_existing = True
                                        elif current_area >= existing_area * 0.9 and det_conf > existing_conf + 0.15:  # Similar size but much higher confidence
                                            replace_existing = True
                                        
                                        if replace_existing:
                                            # Remove old crop file
                                            old_crop_path = self.crop_deduplication[crop_dedup_key]
                                            if os.path.exists(old_crop_path):
                                                try:
                                                    os.remove(old_crop_path)
                                                    print(f"[VideoProcessor] Frame {frame_count}: Replacing crop for {crop_dedup_key} with better quality crop (existing: {existing_area}px²@{existing_conf:.2f}, current: {current_area}px²@{det_conf:.2f})")
                                                except Exception as e:
                                                    print(f"[VideoProcessor] Error removing old crop {old_crop_path}: {e}")
                                            
                                            # Remove old metadata
                                            self.saved_crops = [c for c in self.saved_crops if c.get('track_key') != crop_dedup_key]
                                            # Remove from crop_deduplication (will be re-added below)
                                            del self.crop_deduplication[crop_dedup_key]
                                        else:
                                            # Crop already exists and new one isn't better - skip saving
                                            should_save_crop = False
                                            if frame_count % 30 == 0:
                                                print(f"[VideoProcessor] Frame {frame_count}: Skipping crop for {crop_dedup_key} - crop already exists (existing: {existing_area}px²@{existing_conf:.2f}, current: {current_area}px²@{det_conf:.2f})")
                                
                                if not should_save_crop:
                                    # Skip saving - crop already exists or not better
                                    continue
                                
                                self.crop_counter += 1
                                crop_filename = f"crop_{frame_count:06d}_track{track_id:03d}_{bbox_hash}.jpg"
                                crop_path = self.crops_dir / crop_filename
                                
                                # Save crop image
                                cv2.imwrite(str(crop_path), crop)
                                
                                # Save metadata
                                crop_metadata = {
                                    'crop_path': str(crop_path),
                                    'crop_filename': crop_filename,
                                    'frame_count': frame_count,
                                    'track_id': track_id,
                                    'bbox': [crop_x1, crop_y1, crop_x2, crop_y2],  # Actual crop coordinates
                                    'bbox_original': [orig_x1, orig_y1, orig_x2, orig_y2],  # Original YOLO bbox
                                    'bbox_display': [display_x1, display_y1, display_x2, display_y2],  # Display bbox (clipped)
                                    'det_conf': det_conf,
                                    'width': width,
                                    'height': height,
                                    'area': width * height,  # Store area for comparison
                                    'spot': spot,
                                    'world_coords': world_coords,
                                    'track_key': crop_dedup_key,  # Use crop_dedup_key (with track_id) for deduplication
                                    'bbox_hash': bbox_hash,
                                    'timestamp': datetime.utcnow().isoformat(),
                                    'camera_id': camera_id
                                }
                                
                                self.saved_crops.append(crop_metadata)
                                # Store by crop_dedup_key (includes track_id) to ensure one crop per unique track_id
                                # This allows different track_ids to have separate crops, even if they're in the same cluster
                                self.crop_deduplication[crop_dedup_key] = str(crop_path)
                                
                                with self.lock:
                                    self.stats['crops_saved'] += 1
                                
                                if frame_count % 30 == 0 or len(self.saved_crops) <= 10:
                                    print(f"[VideoProcessor] Frame {frame_count}: Saved crop for {crop_dedup_key} (track_id={track_id}, size={width}x{height}, area={width*height}, conf={det_conf:.2f}) -> {crop_filename} (total unique tracks: {len(self.crop_deduplication)})")
                                    
                            except Exception as e:
                                print(f"[VideoProcessor] Error saving crop at frame {frame_count}: {e}")
                        
                        # TWO-STAGE APPROACH: Skip OCR during video processing (will be done in batch later)
                        # Set empty OCR results for now - OCR will be done in Stage 2 (batch processing)
                        text = ""
                        conf_ocr = 0.0
                        
                        # Note: OCR text drawing is skipped during Stage 1 (no OCR results yet)
                        # OCR results will be added in Stage 2 and can be matched back to detections
                        
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
                            'x_world': world_coords_meters[0] if world_coords_meters else None,
                            'y_world': world_coords_meters[1] if world_coords_meters else None,
                            'lat': world_coords_gps[0] if world_coords_gps else None,
                            'lon': world_coords_gps[1] if world_coords_gps else None,
                            'spot': spot,
                            'method': method
                        }
                        
                        # TWO-STAGE: OCR results will be empty during Stage 1
                        # OCR will be processed in Stage 2 (batch) and matched back using metadata
                        # For now, text and conf_ocr remain empty (set above)
                        
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
                            # Note: OCR text checking is skipped in Stage 1 (no OCR results yet)
                            # In Stage 2, OCR results can be matched back and events updated
                            
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
                
                # Note: Green boxes are already drawn for all tracks in the loop above
                # OCR text drawing is skipped during Stage 1 (no OCR results yet)
                
                # Store processed frame (always store, even if no detections)
                # Store BEFORE yielding to ensure it's available immediately
                with self.lock:
                    self.processed_frames[frame_count] = processed_frame.copy()
                    self.stats['frames_processed'] = frame_count + 1
                
                # Merge clusters periodically during processing (not just at the end)
                # This ensures crop deduplication uses merged cluster IDs, matching final unique trailer count
                # Merge every 30 frames to keep clusters up-to-date as we process
                if frame_count > 0 and frame_count % 30 == 0:
                    # Merge clusters incrementally so track_key uses final merged cluster IDs
                    # This ensures we save crops based on merged clusters, matching final unique trailer count
                    self._merge_nearby_clusters()
                
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
            
            # Clean up duplicate crops after merging (keep only crops for final merged clusters)
            if self.save_crops:
                self._cleanup_merged_crops()
            
            # Save crop metadata to JSON file for batch OCR processing
            if self.save_crops and len(self.saved_crops) > 0:
                import json
                metadata_path = self.crops_dir / "crops_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(self.saved_crops, f, indent=2)
                print(f"[VideoProcessor] Saved {len(self.saved_crops)} crop metadata to {metadata_path}")
                print(f"[VideoProcessor] Crops directory: {self.crops_dir}")
                print(f"[VideoProcessor] Ready for batch OCR processing")
            
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
    
    def _resolve_final_track_key(self, spot: str, cluster_id: int) -> str:
        """
        Resolve the final merged track_key base (without track_id) for a given spot and cluster_id.
        If clusters have been merged, returns the base track_key with the final merged cluster ID.
        Otherwise, returns the base track_key.
        Note: This returns only the base (spot_cluster_id), track_id should be appended separately.
        """
        base_key = f"{spot}_cluster_{cluster_id}"
        
        # First, check if this cluster_id still exists in position_clusters
        # If it exists, it hasn't been merged, so return the base track_key
        if cluster_id in self.position_clusters:
            return base_key
        
        # Cluster was merged (doesn't exist in position_clusters anymore)
        # Find which cluster it merged into by checking unique_tracks or crop_deduplication
        # Look for track_keys with the same spot that exist in position_clusters
        # (these are the final merged clusters)
        for existing_track_key in self.crop_deduplication.keys():
            if existing_track_key.startswith(f"{spot}_cluster_"):
                # Extract cluster_id from track_key (format: spot_cluster_id_track_trackid or spot_cluster_id)
                parts = existing_track_key.split("_cluster_")
                if len(parts) >= 2:
                    try:
                        # Extract cluster_id (may be followed by _track_)
                        cluster_part = parts[1].split("_track_")[0]
                        existing_cluster_id = int(cluster_part)
                        # Check if this cluster_id still exists (it's the final merged cluster)
                        if existing_cluster_id in self.position_clusters:
                            # Return base key (without track_id part)
                            return f"{spot}_cluster_{existing_cluster_id}"
                    except ValueError:
                        pass
        
        # Also check unique_tracks
        for existing_track_key in self.unique_tracks.keys():
            if existing_track_key.startswith(f"{spot}_cluster_"):
                parts = existing_track_key.split("_cluster_")
                if len(parts) >= 2:
                    try:
                        # Extract cluster_id (may be followed by _track_)
                        cluster_part = parts[1].split("_track_")[0]
                        existing_cluster_id = int(cluster_part)
                        if existing_cluster_id in self.position_clusters:
                            # Return base key (without track_id part)
                            return f"{spot}_cluster_{existing_cluster_id}"
                    except ValueError:
                        pass
        
        # Fallback: return base track_key (cluster might not have been merged yet)
        return base_key
    
    def _cleanup_merged_crops(self):
        """
        Remove duplicate crops after cluster merging.
        Keep only the best crop for each final merged cluster, delete crops for clusters that were merged.
        Also filter out crops that are too small (narrow/partial detections).
        """
        if not self.save_crops or len(self.saved_crops) == 0:
            return
        
        with self.lock:
            # Get all final cluster IDs (clusters that still exist after merging)
            final_cluster_ids = set(self.position_clusters.keys())
            
            # Minimum crop size requirements (same as during processing)
            MIN_CROP_WIDTH = 200
            MIN_CROP_HEIGHT = 150
            MIN_CROP_AREA = 30000
            
            # Build map of merged clusters (old_id -> final_id) using final_merge_map from _merge_nearby_clusters
            # This helps us find crops from clusters that were merged
            merged_cluster_map = {}  # old_cluster_id -> final_cluster_id
            
            # Use the final_merge_map stored by _merge_nearby_clusters if available
            if hasattr(self, 'final_merge_map') and self.final_merge_map:
                # Use the stored merge map to find final clusters
                for crop_metadata in self.saved_crops:
                    track_key = crop_metadata.get('track_key', '')
                    if '_cluster_' in track_key:
                        parts = track_key.split('_cluster_')
                        if len(parts) >= 2:
                            try:
                                old_cluster_id = int(parts[1])
                                spot = parts[0]
                                
                                # Check if this cluster was merged
                                if old_cluster_id in self.final_merge_map:
                                    # Cluster was merged - get final cluster ID
                                    final_cluster_id = self.final_merge_map[old_cluster_id]
                                    # Only use if final cluster still exists
                                    if final_cluster_id in final_cluster_ids:
                                        merged_cluster_map[old_cluster_id] = final_cluster_id
                                    else:
                                        # Final cluster was also merged or filtered - skip this crop
                                        continue
                                elif old_cluster_id in final_cluster_ids:
                                    # Cluster wasn't merged and still exists
                                    merged_cluster_map[old_cluster_id] = old_cluster_id
                            except ValueError:
                                pass
            else:
                # Fallback: build map from saved crops (less accurate but works if merge map not available)
                for crop_metadata in self.saved_crops:
                    track_key = crop_metadata.get('track_key', '')
                    if '_cluster_' in track_key:
                        parts = track_key.split('_cluster_')
                        if len(parts) >= 2:
                            try:
                                old_cluster_id = int(parts[1])
                                spot = parts[0]
                                
                                # If cluster still exists, it's the final cluster
                                if old_cluster_id in final_cluster_ids:
                                    merged_cluster_map[old_cluster_id] = old_cluster_id
                                else:
                                    # Cluster was merged - try to find final cluster by spot
                                    for final_cid in final_cluster_ids:
                                        final_key = f"{spot}_cluster_{final_cid}"
                                        if final_key in self.crop_deduplication:
                                            merged_cluster_map[old_cluster_id] = final_cid
                                            break
                            except ValueError:
                                pass
            
            # Group crops by track_id (not cluster) to keep one crop per unique track
            crops_by_track = {}  # track_id -> [crop_metadata]
            
            for crop_metadata in self.saved_crops:
                track_key = crop_metadata.get('track_key', '')
                track_id = crop_metadata.get('track_id')
                
                # Extract track_id from track_key if not in metadata
                if track_id is None and '_track_' in track_key:
                    try:
                        track_id_part = track_key.split('_track_')[-1]
                        track_id = int(track_id_part)
                    except (ValueError, IndexError):
                        pass
                
                # Use track_id as the grouping key (one crop per unique track_id)
                if track_id is not None:
                    # Filter out small/partial crops
                    width = crop_metadata.get('width', 0)
                    height = crop_metadata.get('height', 0)
                    area = width * height
                    
                    # Check if crop is significantly clipped (partial trailer near edge)
                    bbox_original = crop_metadata.get('bbox_original', [])
                    bbox_crop = crop_metadata.get('bbox', [])
                    
                    is_valid = True
                    if len(bbox_original) == 4 and len(bbox_crop) == 4:
                        orig_x1, orig_y1, orig_x2, orig_y2 = bbox_original
                        crop_x1, crop_y1, crop_x2, crop_y2 = bbox_crop
                        original_area = (orig_x2 - orig_x1) * (orig_y2 - orig_y1)
                        crop_area = (crop_x2 - crop_x1) * (crop_y2 - crop_y1)
                        
                        if original_area > 0:
                            visible_ratio = crop_area / original_area
                            original_width = orig_x2 - orig_x1
                            original_height = orig_y2 - orig_y1
                            crop_width = crop_x2 - crop_x1
                            crop_height = crop_y2 - crop_y1
                            
                            width_ratio = crop_width / original_width if original_width > 0 else 1.0
                            height_ratio = crop_height / original_height if original_height > 0 else 1.0
                            
                            # Require at least 90% of the original bbox to be visible (area, width, and height)
                            # Increased from 85% to 90% to be more strict about partial crops
                            MIN_VISIBLE_RATIO = 0.90
                            if visible_ratio < MIN_VISIBLE_RATIO or width_ratio < MIN_VISIBLE_RATIO or height_ratio < MIN_VISIBLE_RATIO:
                                is_valid = False
                    
                    if is_valid and width >= MIN_CROP_WIDTH and height >= MIN_CROP_HEIGHT and area >= MIN_CROP_AREA:
                        if track_id not in crops_by_track:
                            crops_by_track[track_id] = []
                        crops_by_track[track_id].append(crop_metadata)
            
            # For each track_id, keep only the best crop (largest area, highest confidence)
            crops_to_keep = []
            crops_to_remove = []
            
            for track_id, track_crops in crops_by_track.items():
                if len(track_crops) == 0:
                    continue
                
                # Find best crop: prioritize area (full backside), then confidence
                best_crop = max(track_crops, key=lambda c: (
                    c.get('area', c.get('width', 0) * c.get('height', 0)),  # Primary: area
                    c.get('det_conf', 0.0)  # Secondary: confidence
                ))
                
                crops_to_keep.append(best_crop)
                
                # Mark others for removal
                for crop in track_crops:
                    if crop != best_crop:
                        crops_to_remove.append(crop)
            
            # Delete crop files for removed crops
            for crop_metadata in crops_to_remove:
                crop_path = crop_metadata.get('crop_path')
                if crop_path:
                    # Handle both string and Path objects
                    if isinstance(crop_path, str):
                        crop_path_obj = Path(crop_path)
                    else:
                        crop_path_obj = crop_path
                    
                    if crop_path_obj.exists():
                        try:
                            crop_path_obj.unlink()
                            print(f"[VideoProcessor] Removed duplicate/small crop: {crop_path_obj}")
                        except Exception as e:
                            print(f"[VideoProcessor] Error removing crop {crop_path_obj}: {e}")
                
                # Remove from crop_deduplication
                track_key = crop_metadata.get('track_key', '')
                if track_key in self.crop_deduplication:
                    del self.crop_deduplication[track_key]
            
            # Additional cleanup: Find and remove any orphaned crop files on disk for the same track_ids
            # This handles cases where cluster merges created files with old cluster_ids that weren't tracked
            if self.crops_dir.exists():
                # Normalize kept crop paths to Path objects for comparison
                kept_crop_paths = set()
                for c in crops_to_keep:
                    crop_path = c.get('crop_path')
                    if crop_path:
                        kept_crop_paths.add(Path(crop_path).resolve())
                
                kept_track_ids = {c.get('track_id') for c in crops_to_keep if c.get('track_id') is not None}
                
                # Scan directory for crop files
                for crop_file in self.crops_dir.glob('crop_*_track*.jpg'):
                    if crop_file.resolve() in kept_crop_paths:
                        continue  # This is a kept crop, skip
                    
                    # Extract track_id from filename (format: crop_XXXXXX_trackYYY_*.jpg)
                    try:
                        filename_parts = crop_file.stem.split('_track')
                        if len(filename_parts) >= 2:
                            track_id_str = filename_parts[1].split('_')[0]
                            file_track_id = int(track_id_str)
                            
                            # If this file belongs to a track_id we're keeping, but it's not the kept file, delete it
                            if file_track_id in kept_track_ids:
                                try:
                                    crop_file.unlink()
                                    print(f"[VideoProcessor] Removed orphaned crop file for track_id {file_track_id}: {crop_file.name}")
                                except Exception as e:
                                    print(f"[VideoProcessor] Error removing orphaned crop {crop_file}: {e}")
                    except (ValueError, IndexError):
                        # Can't parse track_id from filename, skip
                        pass
            
            # Update saved_crops to only include kept crops
            self.saved_crops = crops_to_keep
            
            # Update crop_deduplication to only include kept crops
            self.crop_deduplication = {
                c.get('track_key'): c.get('crop_path')
                for c in crops_to_keep
                if c.get('track_key') and c.get('crop_path')
            }
            
            if len(crops_to_remove) > 0:
                print(f"[VideoProcessor] Cleaned up {len(crops_to_remove)} duplicate/small crops, kept {len(crops_to_keep)} best quality crops (one per unique track_id)")
            
            # After cleanup, check if any track_ids are missing crops
            # This handles cases where track_ids had crops that were filtered out during cleanup
            track_ids_with_crops = set()
            for crop in crops_to_keep:
                track_id = crop.get('track_id')
                if track_id is not None:
                    track_ids_with_crops.add(track_id)
            
            # Find track_ids that need crops (from events but don't have crops)
            track_ids_needing_crops = []
            for track_key, event in self.unique_tracks.items():
                track_id = event.get('track_id')
                if track_id is not None and track_id not in track_ids_with_crops:
                    # Check if this track_id is in a valid cluster
                    if '_cluster_' in track_key:
                        parts = track_key.split('_cluster_')
                        if len(parts) >= 2:
                            try:
                                cluster_part = parts[1].split('_track_')[0] if '_track_' in parts[1] else parts[1]
                                cluster_id = int(cluster_part)
                                # Only create crop if cluster is valid (in final_cluster_ids)
                                if cluster_id in final_cluster_ids:
                                    spot = parts[0]
                                    if (track_id, spot, cluster_id) not in track_ids_needing_crops:
                                        track_ids_needing_crops.append((track_id, spot, cluster_id))
                            except ValueError:
                                pass
            
            # Try to create crops for missing track_ids from unique_tracks events
            if track_ids_needing_crops and len(self.processed_frames) > 0:
                print(f"[VideoProcessor] Found {len(track_ids_needing_crops)} track_ids without crops, attempting to create crops from stored frames...")
                for track_id, spot, cluster_id in track_ids_needing_crops:
                    crop_dedup_key = f"{spot}_cluster_{cluster_id}_track_{track_id}"
                    
                    # Find the best event for this track_id
                    # Strategy: First try to find fully visible crops, then fall back to best available
                    # Prefer: 1) Fully visible (not near edges), 2) Largest area, 3) Highest confidence
                    best_event_fully_visible = None
                    best_score_fully_visible = -1
                    best_event_any = None
                    best_score_any = -1
                    
                    for track_key, event in self.unique_tracks.items():
                        # Match by track_id and cluster_id
                        event_track_id = event.get('track_id')
                        if event_track_id == track_id:
                            # Check if this event belongs to the same cluster
                            if f"_cluster_{cluster_id}" in track_key:
                                parts = track_key.split("_cluster_")
                                if len(parts) >= 2:
                                    cluster_part = parts[1].split("_track_")[0] if "_track_" in parts[1] else parts[1]
                                    try:
                                        if int(cluster_part) == cluster_id:
                                            # Calculate score: area * confidence
                                            width = event.get('trailer_width', 0)
                                            height = event.get('trailer_height', 0)
                                            area = width * height
                                            conf = event.get('det_conf', 0.0)
                                            
                                            # Check if bbox is near frame edges (likely partial)
                                            frame_num = event.get('frame', 0)
                                            is_fully_visible = False
                                            edge_penalty = 1.0
                                            
                                            if frame_num in self.processed_frames:
                                                frame = self.processed_frames[frame_num]
                                                h, w = frame.shape[:2]
                                                bbox = event.get('bbox', [])
                                                if len(bbox) == 4:
                                                    x1, y1, x2, y2 = bbox
                                                    # Use both percentage and absolute thresholds
                                                    EDGE_THRESHOLD_PCT = 0.05
                                                    EDGE_THRESHOLD_PX = 100
                                                    
                                                    # Check if bbox extends beyond frame (definitely partial - reject)
                                                    if x1 < 0 or x2 >= w or y1 < 0 or y2 >= h:
                                                        continue  # Skip this event entirely
                                                    
                                                    near_left = x1 < max(w * EDGE_THRESHOLD_PCT, EDGE_THRESHOLD_PX)
                                                    near_right = x2 > w - max(w * EDGE_THRESHOLD_PCT, EDGE_THRESHOLD_PX)
                                                    near_top = y1 < max(h * EDGE_THRESHOLD_PCT, EDGE_THRESHOLD_PX)
                                                    near_bottom = y2 > h - max(h * EDGE_THRESHOLD_PCT, EDGE_THRESHOLD_PX)
                                                    edges_near = sum([near_left, near_right, near_top, near_bottom])
                                                    
                                                    # Fully visible if not near any edges
                                                    is_fully_visible = (edges_near == 0)
                                                    
                                                    # Penalize crops near edges (prefer fully visible)
                                                    if edges_near >= 2:
                                                        edge_penalty = 0.1  # Very heavy penalty for near 2+ edges
                                                    elif edges_near == 1:
                                                        edge_penalty = 0.5  # Heavy penalty for near 1 edge
                                                    
                                                    score = area * (1.0 + conf) * edge_penalty
                                                else:
                                                    score = area * (1.0 + conf)
                                            else:
                                                score = area * (1.0 + conf)
                                            
                                            # Track best fully visible and best overall
                                            if is_fully_visible and score > best_score_fully_visible:
                                                best_score_fully_visible = score
                                                best_event_fully_visible = event
                                            
                                            if score > best_score_any:
                                                best_score_any = score
                                                best_event_any = event
                                    except ValueError:
                                        pass
                    
                    # Try all events for this track_id, sorted by quality (fully visible first, then by score)
                    # This ensures we find a valid crop even if the best one fails validation
                    all_events = []
                    if best_event_fully_visible:
                        all_events.append((best_event_fully_visible, True))  # (event, is_fully_visible)
                    # Add all other events for this track_id sorted by score
                    for track_key, event in self.unique_tracks.items():
                        # Match by track_id and cluster_id
                        event_track_id = event.get('track_id')
                        if event_track_id == track_id:
                            # Check if this event belongs to the same cluster
                            if f"_cluster_{cluster_id}" in track_key:
                                parts = track_key.split("_cluster_")
                                if len(parts) >= 2:
                                    cluster_part = parts[1].split("_track_")[0] if "_track_" in parts[1] else parts[1]
                                    try:
                                        if int(cluster_part) == cluster_id:
                                            # Skip if already added as fully visible
                                            if event != best_event_fully_visible:
                                                all_events.append((event, False))
                                    except ValueError:
                                        pass
                    
                    # Sort by: 1) fully visible, 2) score (area * confidence)
                    def event_score(event_tuple):
                        event, is_vis = event_tuple
                        width = event.get('trailer_width', 0)
                        height = event.get('trailer_height', 0)
                        area = width * height
                        conf = event.get('det_conf', 0.0)
                        return (1 if is_vis else 0, area * (1.0 + conf))
                    
                    all_events.sort(key=event_score, reverse=True)
                    
                    # Try each event until we find one that passes validation
                    crop_created = False
                    for event, is_fully_visible in all_events:
                        if crop_created:
                            break
                        
                        is_fallback = not is_fully_visible
                        frame_num = event.get('frame', 0)
                        if frame_num in self.processed_frames:
                            try:
                                frame = self.processed_frames[frame_num]
                                bbox = event.get('bbox', [])
                                
                                if len(bbox) == 4:
                                    x1, y1, x2, y2 = bbox
                                    width = x2 - x1
                                    height = y2 - y1
                                    
                                    # Check if crop meets minimum requirements
                                    MIN_CROP_WIDTH = 200
                                    MIN_CROP_HEIGHT = 150
                                    MIN_CROP_AREA = 30000
                                    
                                    if width >= MIN_CROP_WIDTH and height >= MIN_CROP_HEIGHT and (width * height) >= MIN_CROP_AREA:
                                        # Crop the frame
                                        h, w = frame.shape[:2]
                                        crop_x1 = max(0, int(x1))
                                        crop_y1 = max(0, int(y1))
                                        crop_x2 = min(w, int(x2))
                                        crop_y2 = min(h, int(y2))
                                        
                                        # Check if crop is significantly clipped (partial trailer near edge)
                                        # Calculate how much of the original bbox is visible in the crop
                                        original_width = x2 - x1
                                        original_height = y2 - y1
                                        original_area = original_width * original_height
                                        
                                        crop_width = crop_x2 - crop_x1
                                        crop_height = crop_y2 - crop_y1
                                        crop_area = crop_width * crop_height
                                        
                                        # Check visibility ratio (must be at least 90% to avoid partial crops)
                                        MIN_VISIBLE_RATIO = 0.90
                                        is_valid_crop = True
                                        visible_ratio = 1.0
                                        width_ratio = 1.0
                                        height_ratio = 1.0
                                        
                                        if original_area > 0:
                                            visible_ratio = crop_area / original_area
                                            width_ratio = crop_width / original_width if original_width > 0 else 1.0
                                            height_ratio = crop_height / original_height if original_height > 0 else 1.0
                                            
                                            # Also check if bbox is too close to frame edges (likely partial)
                                            # Use both percentage (5%) and absolute distance (100px) thresholds
                                            EDGE_THRESHOLD_PCT = 0.05
                                            EDGE_THRESHOLD_PX = 100  # Absolute pixel threshold
                                            
                                            # Check if bbox extends beyond frame boundaries (definitely partial)
                                            extends_beyond_left = x1 < 0
                                            extends_beyond_right = x2 >= w
                                            extends_beyond_top = y1 < 0
                                            extends_beyond_bottom = y2 >= h
                                            
                                            # Check if bbox is very close to edges
                                            near_left_edge = x1 < max(w * EDGE_THRESHOLD_PCT, EDGE_THRESHOLD_PX)
                                            near_right_edge = x2 > w - max(w * EDGE_THRESHOLD_PCT, EDGE_THRESHOLD_PX)
                                            near_top_edge = y1 < max(h * EDGE_THRESHOLD_PCT, EDGE_THRESHOLD_PX)
                                            near_bottom_edge = y2 > h - max(h * EDGE_THRESHOLD_PCT, EDGE_THRESHOLD_PX)
                                            
                                            # Check aspect ratio - trailer backsides should be roughly square or taller than wide
                                            # Very wide crops (aspect ratio > 1.8) are likely placards or partial views
                                            aspect_ratio = crop_width / crop_height if crop_height > 0 else 0
                                            MAX_ASPECT_RATIO = 1.8  # Reject crops wider than 1.8:1 (trailer backsides are typically < 1.5:1)
                                            MAX_ASPECT_RATIO_FALLBACK = 2.5  # More lenient for fallback (accept up to 2.5:1 if it's the only option for a valid cluster)
                                            
                                            # Always reject these (no exceptions):
                                            # 1. Bbox extends beyond frame boundaries (definitely partial)
                                            # 2. Visibility ratio too low (clipped)
                                            if extends_beyond_left or extends_beyond_right or extends_beyond_top or extends_beyond_bottom:
                                                is_valid_crop = False
                                            elif visible_ratio < MIN_VISIBLE_RATIO or width_ratio < MIN_VISIBLE_RATIO or height_ratio < MIN_VISIBLE_RATIO:
                                                is_valid_crop = False
                                            # For fallback cases (no fully visible crop available), be more lenient:
                                            # Accept wider aspect ratios and crops near one edge (if not clipped)
                                            # This ensures valid clusters (count >= 2) get at least one crop
                                            elif is_fallback:
                                                # For fallback, prioritize: not clipped > aspect ratio
                                                # If crop is not clipped (100% visible), accept even if wide (up to 3.0:1)
                                                # This handles cases where a trailer is detected but only partially visible
                                                if visible_ratio >= 0.99 and width_ratio >= 0.99 and height_ratio >= 0.99:
                                                    # Fully visible (not clipped) - accept even if wide (up to 3.0:1)
                                                    if aspect_ratio > 3.0:
                                                        is_valid_crop = False  # Too wide even for fallback
                                                    # Otherwise accept (it's fully visible, just wide)
                                                # If clipped, be more strict
                                                elif aspect_ratio > MAX_ASPECT_RATIO_FALLBACK:
                                                    is_valid_crop = False  # Too wide and clipped
                                                # Reject if near 2+ edges AND clipped (definitely partial)
                                                elif (near_left_edge + near_right_edge + near_top_edge + near_bottom_edge >= 2) and (width_ratio < 0.95 or height_ratio < 0.95):
                                                    is_valid_crop = False
                                                # Reject if near horizontal edge AND width is significantly clipped (partial)
                                                elif (near_left_edge or near_right_edge) and (width_ratio < 0.90):
                                                    is_valid_crop = False
                                                # Reject if near vertical edge AND height is significantly clipped (partial)
                                                elif (near_top_edge or near_bottom_edge) and (height_ratio < 0.90):
                                                    is_valid_crop = False
                                                # Otherwise, allow it (better than no crop for a valid cluster)
                                            # For non-fallback (we have fully visible options), be strict:
                                            # Reject if aspect ratio too wide
                                            elif aspect_ratio > MAX_ASPECT_RATIO:
                                                is_valid_crop = False
                                            # Reject if near any horizontal edge (trailer backsides should be centered)
                                            elif near_left_edge or near_right_edge:
                                                is_valid_crop = False
                                            # Reject if near multiple edges
                                            elif (near_left_edge + near_right_edge + near_top_edge + near_bottom_edge >= 2):
                                                is_valid_crop = False
                                            # Reject if near edge AND clipped
                                            elif (near_left_edge or near_right_edge) and (width_ratio < 0.95):
                                                is_valid_crop = False
                                            elif (near_top_edge or near_bottom_edge) and (height_ratio < 0.95):
                                                is_valid_crop = False
                                        
                                        if not is_valid_crop:
                                            aspect_info = f", aspect={aspect_ratio:.2f}" if original_area > 0 else ""
                                            print(f"[VideoProcessor] Skipping partial crop for {crop_dedup_key} from frame {frame_num} - visible={visible_ratio:.1%}, width={width_ratio:.1%}, height={height_ratio:.1%}{aspect_info}, near_edges={near_left_edge + near_right_edge + near_top_edge + near_bottom_edge}, is_fallback={is_fallback}")
                                            continue
                                        
                                        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                                        
                                        if crop.size > 0:
                                            # Save crop
                                            import hashlib
                                            bbox_str = f"{x1}_{y1}_{x2}_{y2}"
                                            bbox_hash = hashlib.md5(bbox_str.encode()).hexdigest()[:8]
                                            event_track_id = event.get('track_id', track_id)
                                            crop_filename = f"crop_{frame_num:06d}_track{event_track_id:03d}_{bbox_hash}.jpg"
                                            crop_path = self.crops_dir / crop_filename
                                            
                                            cv2.imwrite(str(crop_path), crop)
                                            
                                            crop_metadata = {
                                                'crop_path': str(crop_path),
                                                'crop_filename': crop_filename,
                                                'frame_count': frame_num,
                                                'track_id': event_track_id,
                                                'bbox': [crop_x1, crop_y1, crop_x2, crop_y2],
                                                'bbox_original': [x1, y1, x2, y2],
                                                'bbox_display': [crop_x1, crop_y1, crop_x2, crop_y2],
                                                'det_conf': event.get('det_conf', 0.0),
                                                'width': width,
                                                'height': height,
                                                'area': width * height,
                                                'spot': spot,
                                                'world_coords': (event.get('x_world'), event.get('y_world')),
                                                'track_key': crop_dedup_key,  # Use crop_dedup_key (with track_id)
                                                'bbox_hash': bbox_hash,
                                                'timestamp': datetime.utcnow().isoformat(),
                                                'camera_id': event.get('camera_id', 'test-video')
                                            }
                                            
                                            crops_to_keep.append(crop_metadata)
                                            self.crop_deduplication[crop_dedup_key] = str(crop_path)
                                            print(f"[VideoProcessor] Created missing crop for {crop_dedup_key} from frame {frame_num} (size={width}x{height}, area={width*height}, conf={event.get('det_conf', 0.0):.2f})")
                                            crop_created = True
                            except Exception as e:
                                print(f"[VideoProcessor] Error creating crop for {crop_dedup_key}: {e}")
                    
                    if not crop_created:
                        print(f"[VideoProcessor] Could not create valid crop for {crop_dedup_key} - all available detections were too partial or invalid")
                
                # Update saved_crops with newly created crops
                self.saved_crops = crops_to_keep
    
    def _deduplicate_overlapping_tracks(self, tracks: List[Dict], frame_count: int) -> List[Dict]:
        """
        Remove duplicate tracks that have very high IoU overlap (likely same trailer).
        This helps catch cases where ByteTrack creates multiple tracks for the same physical trailer.
        
        Args:
            tracks: List of track dicts with bbox, track_id, etc.
            
        Returns:
            Deduplicated list of tracks (keeps the track with highest confidence)
        """
        if len(tracks) <= 1:
            return tracks
        
        def compute_iou(bbox1, bbox2):
            """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
            
            # Calculate intersection
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)
            
            if x2_i <= x1_i or y2_i <= y1_i:
                return 0.0
            
            intersection = (x2_i - x1_i) * (y2_i - y1_i)
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - intersection
            
            if union == 0:
                return 0.0
            
            return intersection / union
        
        # IoU threshold for considering tracks as duplicates
        # Lowered from 0.85 to 0.70 to catch more duplicate tracks
        # Trailers at different angles or partially occluded may have lower IoU but still be the same trailer
        IOU_THRESHOLD = 0.70
        
        # Track which tracks to keep (by index)
        keep_indices = set(range(len(tracks)))
        
        # Compare all pairs of tracks
        for i in range(len(tracks)):
            if i not in keep_indices:
                continue
            for j in range(i + 1, len(tracks)):
                if j not in keep_indices:
                    continue
                
                bbox_i = tracks[i]['bbox']
                bbox_j = tracks[j]['bbox']
                iou = compute_iou(bbox_i, bbox_j)
                
                if iou > IOU_THRESHOLD:
                    # Very high overlap - likely same trailer
                    # Keep the track with higher confidence
                    conf_i = tracks[i].get('conf', 0.0)
                    conf_j = tracks[j].get('conf', 0.0)
                    
                    if conf_i >= conf_j:
                        keep_indices.discard(j)  # Remove j, keep i
                        if frame_count % 30 == 0:
                            print(f"[VideoProcessor] Frame {frame_count}: Deduplicated overlapping tracks {tracks[i].get('track_id', '?')} and {tracks[j].get('track_id', '?')} (IoU={iou:.2f}), keeping track {tracks[i].get('track_id', '?')}")
                    else:
                        keep_indices.discard(i)  # Remove i, keep j
                        if frame_count % 30 == 0:
                            print(f"[VideoProcessor] Frame {frame_count}: Deduplicated overlapping tracks {tracks[i].get('track_id', '?')} and {tracks[j].get('track_id', '?')} (IoU={iou:.2f}), keeping track {tracks[j].get('track_id', '?')}")
        
        # Return only kept tracks
        return [tracks[i] for i in sorted(keep_indices)]
    
    def _merge_nearby_clusters(self):
        """Merge position clusters that are very close together (within 3m) and in the same spot."""
        # This helps reduce overcounting when trailers are parked close together
        # Be conservative - only merge clusters that are very close (3m) and in the same spot
        
        with self.lock:
            # Group clusters by spot first
            clusters_by_spot = {}  # spot -> [cluster_ids]
            for cid in self.position_clusters.keys():
                # Find which spot this cluster belongs to
                # Check both unique_tracks (old format with track_id) and crop_deduplication (new format without track_id)
                spot = None
                
                # Check unique_tracks (may have _track_ in key)
                for track_key in self.unique_tracks.keys():
                    if f"_cluster_{cid}" in track_key:
                        parts = track_key.split("_cluster_")
                        if len(parts) >= 2:
                            spot = parts[0]
                            # Verify cluster_id matches (handle format with or without track_id)
                            cluster_part = parts[1].split("_track_")[0] if "_track_" in parts[1] else parts[1]
                            try:
                                if int(cluster_part) == cid:
                                    break
                            except ValueError:
                                pass
                
                # Also check crop_deduplication (new format: spot_cluster_id without track_id)
                if spot is None:
                    for cluster_key in self.crop_deduplication.keys():
                        if f"_cluster_{cid}" in cluster_key:
                            parts = cluster_key.split("_cluster_")
                            if len(parts) >= 2:
                                spot = parts[0]
                                try:
                                    cluster_part = parts[1]
                                    if int(cluster_part) == cid:
                                        break
                                except ValueError:
                                    pass
                
                if spot is None:
                    spot = "unknown"
                
                if spot not in clusters_by_spot:
                    clusters_by_spot[spot] = []
                clusters_by_spot[spot].append(cid)
            
            clusters_to_merge = []
            cluster_ids = list(self.position_clusters.keys())
            
            # Find clusters that should be merged
            # Be VERY conservative - only merge clusters that are VERY close together (within 3m)
            # This ensures we only merge duplicate detections of the SAME trailer, not different trailers
            # Different trailers can be in the same spot but should NOT be merged
            MERGE_DISTANCE_THRESHOLD = 3.0  # 3 meters - very conservative to prevent merging different trailers
            
            for i, cid1 in enumerate(cluster_ids):
                for cid2 in cluster_ids[i+1:]:
                    cluster1 = self.position_clusters[cid1]
                    cluster2 = self.position_clusters[cid2]
                    
                    dist = np.sqrt((cluster1['center_x'] - cluster2['center_x'])**2 + 
                                  (cluster1['center_y'] - cluster2['center_y'])**2)
                    
                    # Check if clusters are in the same spot
                    # Check both unique_tracks and crop_deduplication to find spots
                    spot1 = None
                    spot2 = None
                    
                    # Check unique_tracks (may have _track_ in key)
                    for track_key in self.unique_tracks.keys():
                        if f"_cluster_{cid1}" in track_key:
                            parts = track_key.split("_cluster_")
                            if len(parts) >= 2:
                                # Verify cluster_id matches
                                cluster_part = parts[1].split("_track_")[0] if "_track_" in parts[1] else parts[1]
                                try:
                                    if int(cluster_part) == cid1:
                                        spot1 = parts[0]
                                except ValueError:
                                    pass
                        if f"_cluster_{cid2}" in track_key:
                            parts = track_key.split("_cluster_")
                            if len(parts) >= 2:
                                # Verify cluster_id matches
                                cluster_part = parts[1].split("_track_")[0] if "_track_" in parts[1] else parts[1]
                                try:
                                    if int(cluster_part) == cid2:
                                        spot2 = parts[0]
                                except ValueError:
                                    pass
                    
                    # Also check crop_deduplication (new format: spot_cluster_id without track_id)
                    if spot1 is None:
                        for cluster_key in self.crop_deduplication.keys():
                            if f"_cluster_{cid1}" in cluster_key:
                                parts = cluster_key.split("_cluster_")
                                if len(parts) >= 2:
                                    try:
                                        if int(parts[1]) == cid1:
                                            spot1 = parts[0]
                                            break
                                    except ValueError:
                                        pass
                    
                    if spot2 is None:
                        for cluster_key in self.crop_deduplication.keys():
                            if f"_cluster_{cid2}" in cluster_key:
                                parts = cluster_key.split("_cluster_")
                                if len(parts) >= 2:
                                    try:
                                        if int(parts[1]) == cid2:
                                            spot2 = parts[0]
                                            break
                                    except ValueError:
                                        pass
                    
                    # Only merge if:
                    # 1. Clusters are VERY close together (within 3m) - indicates same physical trailer
                    # 2. AND one cluster is small (likely duplicate detection) OR both are in same spot
                    # This prevents merging different trailers that happen to be in the same spot
                    should_merge = False
                    min_count = min(cluster1['count'], cluster2['count'])
                    max_count = max(cluster1['count'], cluster2['count'])
                    
                    # Primary condition: Must be very close (within 3m) to be considered same trailer
                    # OR if one cluster is small and in same spot, use larger threshold (likely duplicate)
                    if spot1 == spot2 and spot1 != "unknown":
                        # Same spot - more likely to be same trailer if close
                        # Be more aggressive: trailers in the same spot are likely the same physical trailer
                        if min_count < 5:  # One cluster is small - likely duplicate detection
                            # Small clusters in same spot: merge if within 10m (very lenient for duplicates)
                            # This handles cases where a trailer creates multiple clusters due to tracking issues
                            if dist <= 10.0:
                                should_merge = True
                        elif min_count < 10 and max_count >= 15:
                            # One medium-sized cluster near a large one in same spot
                            # Merge if within 8m (more lenient for same spot)
                            if dist <= 8.0:
                                should_merge = True
                        elif max_count >= 15:  # Both substantial clusters in same spot
                            # For substantial clusters in same spot, use 15m threshold (very aggressive)
                            # Trailers in the same spot are very likely to be the same physical trailer
                            # Even if they're far apart, they're probably the same trailer moving
                            if dist <= 15.0:
                                should_merge = True
                                print(f"[VideoProcessor] Merging substantial clusters {cid1} and {cid2} in same spot {spot1} (dist={dist:.2f}m, counts={cluster1['count']}/{cluster2['count']})")
                        elif dist <= MERGE_DISTANCE_THRESHOLD:
                            # Both smaller clusters in same spot - use strict 3m threshold
                            should_merge = True
                    elif dist <= MERGE_DISTANCE_THRESHOLD:
                        # Different spots or unknown - use strict 3m threshold
                        if min_count < 5:  # One cluster is small - likely duplicate detection
                            should_merge = True
                    elif spot1 == spot2 == "unknown":  # Both unknown spots
                        # For unknown spots, merge if one is small (likely false positive)
                        # But still use conservative distance threshold
                        if min_count < 5:
                            # Small cluster - merge if within 3m (same as main threshold)
                            if dist <= 3.0:
                                should_merge = True
                        else:
                            # Both substantial - be very conservative (2.5m)
                            if dist <= 2.5:
                                should_merge = True
                    elif spot1 != spot2 and (min_count < 3):  # Different spots, but one is very small
                        # Very small cluster in different spot - might be false positive or misclassified
                        # Only merge if VERY close (2m) - likely same trailer misclassified
                        if dist <= 2.0:
                            should_merge = True
                    # Don't merge clusters from different known spots if both are substantial
                    
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
            
            # Store final_merge_map for use in cleanup
            self.final_merge_map = final_merge_map
            
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
                        # Extract spot and track_id from track_key
                        # Format: "spot_cluster_id" or "spot_cluster_id_track_trackid"
                        parts = track_key.split("_cluster_")
                        if len(parts) >= 2:
                            spot = parts[0]
                            # Check if track_id is present
                            cluster_and_track = parts[1]
                            if "_track_" in cluster_and_track:
                                track_id_part = cluster_and_track.split("_track_")[-1]
                                new_track_key = f"{spot}_cluster_{merge_to}_track_{track_id_part}"
                            else:
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
            
            # Apply updates to crop_deduplication (CRITICAL for two-stage processing)
            # When clusters merge, update crop_deduplication to use merged cluster IDs
            # This ensures we only save one crop per final unique trailer
            # Also update saved_crops metadata to use merged cluster IDs
            for old_key, new_key in spot_cluster_updates.items():
                if old_key in self.crop_deduplication:
                    # If new_key already has a crop, keep it (first saved wins)
                    # Otherwise, move the crop path to new_key
                    if new_key not in self.crop_deduplication:
                        self.crop_deduplication[new_key] = self.crop_deduplication[old_key]
                    # Remove old_key entry (merged into new_key)
                    del self.crop_deduplication[old_key]
                
                # Also update saved_crops metadata to use merged cluster IDs
                for crop_meta in self.saved_crops:
                    if crop_meta.get('track_key') == old_key:
                        crop_meta['track_key'] = new_key
            
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
            
            # NOTE: We do NOT consolidate tracks with different track_ids that end up in the same cluster.
            # Multiple track_ids in the same cluster can represent different physical trailers in the same spot.
            # The track_id from the tracker is the source of truth for distinguishing different trailers.
            # Cluster merging only handles spatial proximity, not track identity.
            
            if merged_into:
                print(f"[VideoProcessor] Merged {len(merged_into)} clusters, remaining: {len(self.position_clusters)}")
                if spot_cluster_updates:
                    print(f"[VideoProcessor] Updated {len(spot_cluster_updates)} track keys after cluster merge")
            else:
                print(f"[VideoProcessor] No clusters merged (found {len(cluster_ids)} clusters)")
            
            # Filter out small clusters (likely false positives) and unrealistic positions
            # This is done AFTER merging to catch any remaining small clusters
            # Lowered to 2 to allow trailers that are detected infrequently
            MIN_CLUSTER_COUNT = 2  # Minimum detections to consider a cluster valid (lowered from 5)
            MAX_REALISTIC_X = 50.0  # Maximum realistic X position (meters)
            MAX_REALISTIC_Y = 100.0  # Maximum realistic Y position (meters)
            MIN_REALISTIC_X = -20.0  # Minimum realistic X position (meters)
            MIN_REALISTIC_Y = 0.0  # Minimum realistic Y position (meters)
            
            clusters_to_remove = []
            for cid, cluster in self.position_clusters.items():
                should_remove = False
                reason = []
                
                # Check cluster size
                if cluster['count'] < MIN_CLUSTER_COUNT:
                    should_remove = True
                    reason.append(f"count={cluster['count']} < {MIN_CLUSTER_COUNT}")
                
                # Check position bounds
                x, y = cluster['center_x'], cluster['center_y']
                if x > MAX_REALISTIC_X or x < MIN_REALISTIC_X:
                    should_remove = True
                    reason.append(f"x={x:.2f} out of bounds")
                if y > MAX_REALISTIC_Y or y < MIN_REALISTIC_Y:
                    should_remove = True
                    reason.append(f"y={y:.2f} out of bounds")
                
                if should_remove:
                    clusters_to_remove.append((cid, reason))
            
            # Remove invalid clusters
            if clusters_to_remove:
                print(f"[VideoProcessor] Filtering out {len(clusters_to_remove)} invalid clusters:")
                for cid, reasons in clusters_to_remove:
                    cluster = self.position_clusters[cid]
                    print(f"  Removing Cluster {cid}: center=({cluster['center_x']:.2f}, {cluster['center_y']:.2f}), count={cluster['count']} - {', '.join(reasons)}")
                    
                    # Remove cluster
                    del self.position_clusters[cid]
                    
                    # Remove associated track keys (handle both old and new formats)
                    keys_to_remove = []
                    for track_key in list(self.unique_tracks.keys()):
                        # Check if this track_key uses the removed cluster
                        # Format: "spot_cluster_cid" or "spot_cluster_cid_track_trackid"
                        if f"_cluster_{cid}" in track_key:
                            # Extract cluster_id from track_key to verify it matches
                            parts = track_key.split("_cluster_")
                            if len(parts) >= 2:
                                cluster_part = parts[1].split("_track_")[0] if "_track_" in parts[1] else parts[1]
                                try:
                                    if int(cluster_part) == cid:
                                        keys_to_remove.append(track_key)
                                except ValueError:
                                    # Invalid format, but contains cluster string, so remove it
                                    keys_to_remove.append(track_key)
                    
                    for track_key in keys_to_remove:
                        if track_key in self.unique_tracks:
                            del self.unique_tracks[track_key]
                        if track_key in self.unique_track_keys:
                            self.unique_track_keys.remove(track_key)
                        if track_key in self.crop_deduplication:
                            del self.crop_deduplication[track_key]
                        if track_key in self.track_ocr_results:
                            del self.track_ocr_results[track_key]
                
                print(f"[VideoProcessor] After filtering: {len(self.position_clusters)} valid clusters remaining")
            
            # Debug: print remaining cluster positions
            if len(self.position_clusters) > 0:
                print(f"[VideoProcessor] Valid clusters:")
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
            # Get unique tracks count based on final valid clusters (most accurate)
            # This ensures the count matches the actual number of unique physical trailers
            # Handle case where position_clusters might not be initialized yet
            if not hasattr(self, 'position_clusters') or self.position_clusters is None:
                self.position_clusters = {}
            
            # Get deduplicated events with consolidated OCR results
            # Only include events for valid clusters (those that still exist)
            valid_cluster_ids = set(self.position_clusters.keys())
            deduplicated_events = {}
            
            # Process events and filter by valid clusters
            for track_key, event in self.unique_tracks.items():
                # Only include events for valid clusters (filter out deleted/merged clusters)
                if "_cluster_" in track_key:
                    # Extract cluster_id from track_key
                    parts = track_key.split("_cluster_")
                    if len(parts) == 2:
                        try:
                            cluster_part = parts[1].split("_track_")[0] if "_track_" in parts[1] else parts[1]
                            cluster_id = int(cluster_part)
                            # Skip if cluster was filtered out or merged
                            if cluster_id not in valid_cluster_ids:
                                continue
                        except ValueError:
                            pass  # Invalid format, include it anyway
                
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
            
            # Count unique track_ids based on saved crops (matches the crop saving logic)
            # This ensures the track count matches the number of crops saved (one per unique track_id)
            # After cleanup, saved_crops contains the final crops that were kept
            unique_tracks_count = 0
            if hasattr(self, 'saved_crops') and self.saved_crops:
                # Count unique track_ids from saved crops (after cleanup)
                unique_track_ids_from_crops = set()
                for crop_meta in self.saved_crops:
                    track_id = crop_meta.get('track_id')
                    # Fallback: extract track_id from track_key if not in metadata
                    if track_id is None:
                        track_key = crop_meta.get('track_key', '')
                        if '_track_' in track_key:
                            try:
                                # Extract track_id from track_key (format: spot_cluster_id_track_trackid)
                                parts = track_key.split('_track_')
                                if len(parts) >= 2:
                                    track_id = int(parts[-1])  # Last part is track_id
                            except (ValueError, IndexError):
                                pass
                    if track_id is not None:
                        unique_track_ids_from_crops.add(track_id)
                unique_tracks_count = len(unique_track_ids_from_crops)
            elif hasattr(self, 'crop_deduplication') and self.crop_deduplication:
                # Fallback: count from crop_deduplication if saved_crops not available
                # Extract track_ids from track_keys in crop_deduplication
                unique_track_ids_from_crops = set()
                for track_key in self.crop_deduplication.keys():
                    if '_track_' in track_key:
                        try:
                            # Extract track_id from track_key (format: spot_cluster_id_track_trackid)
                            parts = track_key.split('_track_')
                            if len(parts) >= 2:
                                track_id = int(parts[-1])  # Last part is track_id
                                unique_track_ids_from_crops.add(track_id)
                        except (ValueError, IndexError):
                            pass
                unique_tracks_count = len(unique_track_ids_from_crops)
            else:
                # Fallback: count from deduplicated events if no crops saved
                unique_track_ids = set()
                for event in deduplicated_events.values():
                    if 'track_id' in event:
                        unique_track_ids.add(event['track_id'])
                unique_tracks_count = len(unique_track_ids) if unique_track_ids else len(deduplicated_events)
            
            # Convert to list and sort by timestamp (most recent first)
            unique_events = list(deduplicated_events.values())
            unique_events.sort(key=lambda x: x.get('ts_iso', ''), reverse=True)
            
            # Take last 100 (most recent)
            unique_events = unique_events[:100]
            
            # Count unique OCR results (non-empty) for valid clusters only
            unique_ocr_count = 0
            for track_key, ocr_data in self.track_ocr_results.items():
                if isinstance(ocr_data, dict) and ocr_data.get('text', '').strip():
                    # Only count if cluster is still valid
                    if "_cluster_" in track_key:
                        parts = track_key.split("_cluster_")
                        if len(parts) == 2:
                            try:
                                cluster_id = int(parts[1])
                                if cluster_id in valid_cluster_ids:
                                    unique_ocr_count += 1
                            except ValueError:
                                unique_ocr_count += 1  # Include if can't parse
                    else:
                        unique_ocr_count += 1  # Include non-cluster keys
            
            return {
                'frames_processed': self.stats['frames_processed'],
                'detections': self.stats['detections'],
                'tracks': unique_tracks_count,  # Count of unique track_ids detected (accurate count of different trailers)
                'ocr_results': unique_ocr_count,  # Count of unique tracks with OCR results
                'events': unique_events,  # Deduplicated events with consolidated OCR
                'total_frames_stored': len(self.processed_frames)  # Debug info
            }
    
    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Get a processed frame by frame number."""
        with self.lock:
            return self.processed_frames.get(frame_number)

