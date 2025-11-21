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
                            print(f"[VideoProcessor] Frame {frame_count}: Found {len(detections)} detections")
                except Exception as e:
                    print(f"[VideoProcessor] Error in detection at frame {frame_count}: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Update tracker
                try:
                    tracks = tracker.update(detections, frame)
                    with self.lock:
                        self.stats['tracks'] += len(tracks)
                    if len(tracks) > 0:
                        print(f"[VideoProcessor] Frame {frame_count}: Tracking {len(tracks)} objects")
                except Exception as e:
                    print(f"[VideoProcessor] Error in tracking at frame {frame_count}: {e}")
                    import traceback
                    traceback.print_exc()
                    tracks = []
                
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
                        
                        # Draw bounding box in green for detected trailer (thicker for better visibility)
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        
                        # Draw semi-transparent green overlay on the bounding box
                        overlay = processed_frame.copy()
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
                        cv2.addWeighted(overlay, 0.1, processed_frame, 0.9, 0, processed_frame)
                        
                        # Draw detection confidence and track ID in green above the box with background
                        det_conf_percent = det_conf * 100.0
                        conf_text = f"Track {track_id} - {det_conf_percent:.1f}%"
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
                        
                        # Crop region for OCR
                        # Ensure coordinates are within frame bounds
                        h, w = frame.shape[:2]
                        x1_crop = max(0, min(x1, w - 1))
                        y1_crop = max(0, min(y1, h - 1))
                        x2_crop = max(x1_crop + 1, min(x2, w))
                        y2_crop = max(y1_crop + 1, min(y2, h))
                        
                        crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                        if crop.size == 0:
                            continue
                        
                        # Run OCR
                        text = ""
                        conf_ocr = 0.0
                        if self.ocr:
                            try:
                                ocr_result = self.ocr.recognize(crop)
                                text = ocr_result.get('text', '')
                                conf_ocr = ocr_result.get('conf', 0.0)
                                
                                if text:
                                    with self.lock:
                                        self.stats['ocr_results'] += 1
                                    print(f"[VideoProcessor] Frame {frame_count}, Track {track_id}: OCR result '{text}' (conf: {conf_ocr:.2f})")
                                
                                # Draw OCR text in red with confidence percentage (prominent display)
                                if text:
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

