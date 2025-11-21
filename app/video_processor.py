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
                    # Trailers have specific characteristics:
                    # 1. Minimum size (trailers are large objects)
                    # 2. Aspect ratio (wider than tall, typically 2:1 to 5:1)
                    MIN_TRAILER_WIDTH = 150   # Minimum width in pixels
                    MIN_TRAILER_HEIGHT = 50   # Minimum height in pixels
                    MIN_AREA = 10000          # Minimum area (150*50 = 7500, use higher for safety)
                    MIN_ASPECT_RATIO = 1.5    # Width/Height ratio (trailers are wider than tall)
                    MAX_ASPECT_RATIO = 6.0    # Maximum aspect ratio to avoid detecting long thin objects
                    
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
                        
                        # Calculate trailer dimensions
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Only draw green bounding box and overlay for the NEAREST trailer
                        is_nearest = (track_id == nearest_track_id)
                        
                        if is_nearest:
                            # Draw bounding box in green for detected trailer (thicker for better visibility)
                            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            
                            # Draw semi-transparent green overlay on the bounding box
                            overlay = processed_frame.copy()
                            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
                            cv2.addWeighted(overlay, 0.1, processed_frame, 0.9, 0, processed_frame)
                            
                            # Draw detection confidence and track ID in green above the box with background
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
                        else:
                            # Draw subtle gray bounding box for other detected trailers
                            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                            
                            # Draw small track ID in gray
                            det_conf_percent = det_conf * 100.0
                            conf_text = f"T{track_id}"
                            cv2.putText(processed_frame, conf_text, (x1, max(y1 - 5, 15)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
                        
                        # Crop region for OCR - focus on areas where text/license plates appear
                        # On trailers, text/numbers are typically on the front face, center-left area
                        h, w = frame.shape[:2]
                        
                        # Calculate focused crop region (center-left 70% width, center 50% height)
                        # This reduces background noise and focuses on text areas
                        bbox_width = x2 - x1
                        bbox_height = y2 - y1
                        
                        # Focus on center-left area where trailer text typically appears
                        text_crop_x1 = int(x1 + bbox_width * 0.1)   # Start 10% from left
                        text_crop_x2 = int(x1 + bbox_width * 0.8)   # End at 80% (70% width)
                        text_crop_y1 = int(y1 + bbox_height * 0.25)  # Start 25% from top
                        text_crop_y2 = int(y1 + bbox_height * 0.75)  # End at 75% (50% height)
                        
                        # Ensure coordinates are within frame bounds
                        text_crop_x1 = max(0, min(text_crop_x1, w - 1))
                        text_crop_y1 = max(0, min(text_crop_y1, h - 1))
                        text_crop_x2 = max(text_crop_x1 + 1, min(text_crop_x2, w))
                        text_crop_y2 = max(text_crop_y1 + 1, min(text_crop_y2, h))
                        
                        crop = frame[text_crop_y1:text_crop_y2, text_crop_x1:text_crop_x2]
                        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                            continue
                        
                        # Draw OCR region indicator for nearest trailer (helps debugging)
                        if is_nearest:
                            cv2.rectangle(processed_frame, (text_crop_x1, text_crop_y1), 
                                        (text_crop_x2, text_crop_y2), (0, 255, 255), 1)
                        
                        # Run OCR with multiple preprocessing strategies
                        # Strategy 1: Original image (best for clear text with good contrast)
                        # Strategy 2: CLAHE enhancement (best for low contrast)
                        # Strategy 3: Light preprocessing (balanced approach)
                        
                        text = ""
                        conf_ocr = 0.0
                        if self.ocr:
                            try:
                                ocr_results = []
                                
                                # Strategy 1: Try original crop first (often best for numbers)
                                try:
                                    result_orig = self.ocr.recognize(crop)
                                    ocr_results.append({
                                        'text': result_orig.get('text', ''),
                                        'conf': result_orig.get('conf', 0.0),
                                        'method': 'original'
                                    })
                                except Exception as e:
                                    print(f"[VideoProcessor] OCR strategy 1 failed: {e}")
                                
                                # Strategy 2: CLAHE enhancement only (preserve details, enhance contrast)
                                try:
                                    if len(crop.shape) == 3:
                                        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                                    else:
                                        crop_gray = crop
                                    
                                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                                    crop_clahe = clahe.apply(crop_gray)
                                    crop_clahe_bgr = cv2.cvtColor(crop_clahe, cv2.COLOR_GRAY2BGR)
                                    
                                    result_clahe = self.ocr.recognize(crop_clahe_bgr)
                                    ocr_results.append({
                                        'text': result_clahe.get('text', ''),
                                        'conf': result_clahe.get('conf', 0.0),
                                        'method': 'clahe'
                                    })
                                except Exception as e:
                                    print(f"[VideoProcessor] OCR strategy 2 failed: {e}")
                                
                                # Strategy 3: Light denoising + sharpening (good for slightly blurry images)
                                try:
                                    if len(crop.shape) == 3:
                                        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                                    else:
                                        crop_gray = crop
                                    
                                    # Denoise slightly
                                    crop_denoised = cv2.fastNlMeansDenoising(crop_gray, None, 10, 7, 21)
                                    
                                    # Sharpen to make text edges crisper
                                    kernel = np.array([[-1,-1,-1],
                                                      [-1, 9,-1],
                                                      [-1,-1,-1]])
                                    crop_sharp = cv2.filter2D(crop_denoised, -1, kernel)
                                    crop_sharp_bgr = cv2.cvtColor(crop_sharp, cv2.COLOR_GRAY2BGR)
                                    
                                    result_sharp = self.ocr.recognize(crop_sharp_bgr)
                                    ocr_results.append({
                                        'text': result_sharp.get('text', ''),
                                        'conf': result_sharp.get('conf', 0.0),
                                        'method': 'sharpened'
                                    })
                                except Exception as e:
                                    print(f"[VideoProcessor] OCR strategy 3 failed: {e}")
                                
                                # Select the best result based on:
                                # 1. Presence of both letters AND numbers (trailer IDs usually have both)
                                # 2. Higher confidence
                                # 3. Longer text length
                                best_result = None
                                for result in ocr_results:
                                    if not result['text']:
                                        continue
                                    
                                    has_letter = any(c.isalpha() for c in result['text'])
                                    has_digit = any(c.isdigit() for c in result['text'])
                                    has_both = has_letter and has_digit
                                    
                                    # Scoring: prioritize results with both letters and numbers
                                    score = result['conf']
                                    if has_both:
                                        score += 0.15  # Bonus for having both
                                    score += len(result['text']) * 0.01  # Small bonus for length
                                    
                                    if best_result is None or score > best_result['score']:
                                        best_result = {
                                            'text': result['text'],
                                            'conf': result['conf'],
                                            'method': result['method'],
                                            'score': score
                                        }
                                
                                if best_result:
                                    text = best_result['text']
                                    conf_ocr = best_result['conf']
                                    if frame_count % 10 == 0:  # Log occasionally
                                        print(f"[VideoProcessor] Best OCR method: {best_result['method']} for '{text}'")
                                
                                # Filter out very low confidence results
                                MIN_OCR_CONF = 0.01  # 1% minimum confidence
                                if text and conf_ocr >= MIN_OCR_CONF:
                                    with self.lock:
                                        self.stats['ocr_results'] += 1
                                    print(f"[VideoProcessor] Frame {frame_count}, Track {track_id}: OCR result '{text}' (conf: {conf_ocr:.2f})")
                                else:
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
                    except Exception as e:
                        print(f"[VideoProcessor] Error processing track at frame {frame_count}: {e}")
                        import traceback
                        traceback.print_exc()
                
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

