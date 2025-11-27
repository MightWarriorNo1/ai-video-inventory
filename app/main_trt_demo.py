"""
Main Application Loop - Trailer Vision Edge

End-to-end processing pipeline:
1. Camera ingestion (RTSP/USB)
2. Detection (YOLO TensorRT)
3. Tracking (ByteTrack)
4. OCR (CRNN/PP-OCR TensorRT)
5. Homography projection (image -> world)
6. Spot resolution (GeoJSON polygons)
7. Logging, publishing, metrics
"""

import os
import sys
import yaml
import json
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import signal
import threading

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Application modules
from app.rtsp import open_stream, frame_generator
from app.ai.detector_trt import TrtEngineYOLO
from app.ai.tracker_bytetrack import ByteTrackWrapper
from app.ocr.recognize import PlateRecognizer
from app.spot_resolver import SpotResolver
from app.csv_logger import CSVLogger
from app.media_rotator import MediaRotator
from app.event_bus import MultiPublisher
from app.uploader import UploadManager
from app.metrics_server import MetricsServer
from app.video_processor import VideoProcessor
from app.image_preprocessing import ImagePreprocessor
import requests


class TrailerVisionApp:
    """
    Main application class for trailer vision processing.
    """
    
    def __init__(self, config_path: str = "config/cameras.yaml"):
        """
        Initialize application.
        
        Args:
            config_path: Path to cameras.yaml configuration
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.running = False
        
        # Initialize components
        self.detector = None
        self.ocr = None
        self.spot_resolver = None
        self.csv_logger = None
        self.media_rotator = None
        self.publisher = None
        self.uploader = None
        self.metrics_server = None
        self.preprocessor = None
        
        # Per-camera trackers and homographies
        self.trackers = {}
        self.homographies = {}
        
        # Metrics tracking
        self.camera_metrics = {}
        
        # Frame storage for video streaming (thread-safe)
        self.latest_frames = {}
        self.frame_lock = threading.Lock()
        
        self._initialize_components()
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _initialize_components(self):
        """Initialize all application components."""
        globals_cfg = self.config.get('globals', {})
        
        # Initialize detector
        detector_path = "models/trailer_detector.engine"
        if os.path.exists(detector_path):
            self.detector = TrtEngineYOLO(
                detector_path,
                conf_threshold=globals_cfg.get('detector_conf', 0.35)
            )
        else:
            print(f"Warning: Detector engine not found: {detector_path}")
        
        # Initialize OCR
        ocr_path = "models/ocr_crnn.engine"
        alphabet_path = "app/ocr/alphabet.txt"
        if os.path.exists(ocr_path) and os.path.exists(alphabet_path):
            self.ocr = PlateRecognizer(ocr_path, alphabet_path)
        else:
            print(f"Warning: OCR engine not found: {ocr_path}")
        
        # Initialize spot resolver
        spots_path = "config/spots.geojson"
        if os.path.exists(spots_path):
            self.spot_resolver = SpotResolver(spots_path)
        else:
            print(f"Warning: Spots GeoJSON not found: {spots_path}")
        
        # Initialize uploader
        self.uploader = UploadManager()
        
        # Initialize CSV logger
        self.csv_logger = CSVLogger(uploader=self.uploader)
        
        # Initialize media rotator
        keep_screenshots = globals_cfg.get('keep_screenshots', 10)
        self.media_rotator = MediaRotator(
            keep_last=keep_screenshots,
            uploader=self.uploader
        )
        
        # Initialize event publisher
        self.publisher = MultiPublisher()
        
        # REST ingest client (if enabled)
        self.ingest_enabled = os.getenv('INGEST_ENABLED', 'false').lower() == 'true'
        self.ingest_url = os.getenv('INGEST_URL', 'http://localhost:8000/events')
        
        # Initialize metrics server
        metrics_port = int(os.getenv('METRICS_PORT', '8080'))
        self.metrics_server = MetricsServer(port=metrics_port, csv_logger=self.csv_logger)
        self.metrics_server.start()
        
        # Initialize image preprocessor
        preproc_cfg = globals_cfg.get('preprocessing', {})
        self.preprocessor = ImagePreprocessor(
            enable_yolo_preprocessing=preproc_cfg.get('enable_yolo', True),
            enable_ocr_preprocessing=preproc_cfg.get('enable_ocr', True),
            yolo_strategy=preproc_cfg.get('yolo_strategy', 'enhanced'),
            ocr_strategy=preproc_cfg.get('ocr_strategy', 'multi')
        )
        print(f"[TrailerVisionApp] Image preprocessing enabled:")
        print(f"  YOLO: {self.preprocessor.enable_yolo_preprocessing} ({self.preprocessor.yolo_strategy})")
        print(f"  OCR: {self.preprocessor.enable_ocr_preprocessing} ({self.preprocessor.ocr_strategy})")
        
        # Load homographies for each camera
        for camera in self.config.get('cameras', []):
            camera_id = camera['id']
            calib_path = f"config/calib/{camera_id}_h.json"
            if os.path.exists(calib_path):
                with open(calib_path, 'r') as f:
                    calib_data = json.load(f)
                    self.homographies[camera_id] = np.array(calib_data['H'])
                print(f"Loaded homography for {camera_id}")
            else:
                print(f"Warning: Homography not found for {camera_id}: {calib_path}")
        
        # Initialize trackers for each camera
        # Use lower track threshold to match detector confidence
        globals_cfg = self.config.get('globals', {})
        track_thresh = globals_cfg.get('detector_conf', 0.20)
        
        for camera in self.config.get('cameras', []):
            camera_id = camera['id']
            self.trackers[camera_id] = ByteTrackWrapper(track_thresh=track_thresh)
            self.camera_metrics[camera_id] = {
                'frames_processed': 0,
                'fps_ema': 0.0,
                'last_publish': None
            }
        
        # Initialize video processor for testing
        def create_tracker():
            return ByteTrackWrapper(track_thresh=track_thresh)
        
        # Get homography for first camera (or None) for video processing
        test_homography = None
        cameras_list = self.config.get('cameras', [])
        if cameras_list:
            first_camera_id = cameras_list[0]['id']
            if first_camera_id in self.homographies:
                test_homography = self.homographies[first_camera_id]
        
        print(f"[TrailerVisionApp] Initializing video processor:")
        print(f"  - Detector: {'Available' if self.detector else 'NOT AVAILABLE'}")
        print(f"  - OCR: {'Available' if self.ocr else 'NOT AVAILABLE'}")
        print(f"  - Spot Resolver: {'Available' if self.spot_resolver else 'NOT AVAILABLE'}")
        print(f"  - Homography: {'Available' if test_homography is not None else 'NOT AVAILABLE'}")
        
        try:
            self.video_processor = VideoProcessor(
                detector=self.detector,
                ocr=self.ocr,
                tracker_factory=create_tracker,
                spot_resolver=self.spot_resolver,
                homography=test_homography
            )
            print(f"[TrailerVisionApp] Video processor created successfully")
        except Exception as e:
            print(f"[TrailerVisionApp] ERROR: Failed to create video processor: {e}")
            import traceback
            traceback.print_exc()
            self.video_processor = None
        
        # Update metrics server with video processor and frame storage
        if self.metrics_server:
            self.metrics_server.video_processor = self.video_processor
            self.metrics_server.frame_storage = self
            print(f"[TrailerVisionApp] Video processor assigned to metrics server: {self.metrics_server.video_processor is not None}")
        else:
            print(f"[TrailerVisionApp] ERROR: Metrics server not initialized!")
    
    def _project_to_world(self, camera_id: str, x_img: float, y_img: float) -> Optional[tuple]:
        """
        Project image coordinates to world coordinates using homography.
        
        Args:
            camera_id: Camera identifier
            x_img: Image X coordinate
            y_img: Image Y coordinate
            
        Returns:
            Tuple of (x_world, y_world) or None if homography not available
        """
        if camera_id not in self.homographies:
            return None
        
        H = self.homographies[camera_id]
        point = np.array([[x_img, y_img]], dtype=np.float32)
        point = np.array([point])
        
        # Project using homography
        projected = cv2.perspectiveTransform(point, H)
        x_world, y_world = projected[0][0]
        
        return (float(x_world), float(y_world))
    
    def _process_frame(self, camera_id: str, frame: np.ndarray, frame_count: int) -> None:
        """
        Process a single frame for a camera.
        
        Args:
            camera_id: Camera identifier
            frame: BGR image frame
            frame_count: Current frame number
        """
        # Store latest frame for video streaming (thread-safe)
        with self.frame_lock:
            self.latest_frames[camera_id] = frame.copy()
        
        globals_cfg = self.config.get('globals', {})
        detect_every_n = globals_cfg.get('detect_every_n', 5)
        save_frames = globals_cfg.get('save_frames', False)
        
        tracker = self.trackers[camera_id]
        metrics = self.camera_metrics[camera_id]
        
        # Preprocess frame for YOLO if enabled
        processed_frame = frame
        if self.preprocessor and self.preprocessor.enable_yolo_preprocessing:
            processed_frame = self.preprocessor.preprocess_for_yolo(frame)
        
        # Run detector every N frames
        detections = []
        if self.detector and frame_count % detect_every_n == 0:
            detections = self.detector.detect(processed_frame)
        
        # Update tracker
        tracks = tracker.update(detections, frame)
        
        # Process each track
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            x1, y1, x2, y2 = bbox
            
            # Crop region for OCR
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            # Run OCR with preprocessing
            text = ""
            conf_ocr = 0.0
            ocr_method = "none"
            if self.ocr:
                if self.preprocessor and self.preprocessor.enable_ocr_preprocessing:
                    # Get multiple preprocessed versions
                    preprocessed_crops = self.preprocessor.preprocess_for_ocr(crop)
                    
                    # Try OCR on each preprocessed version
                    ocr_results = []
                    for prep in preprocessed_crops:
                        try:
                            result = self.ocr.recognize(prep['image'])
                            if result.get('text', '').strip():
                                ocr_results.append({
                                    'text': result.get('text', ''),
                                    'conf': result.get('conf', 0.0),
                                    'method': prep['method']
                                })
                        except Exception as e:
                            if frame_count % 50 == 0:  # Log occasionally
                                print(f"[TrailerVisionApp] OCR error with {prep['method']}: {e}")
                    
                    # Select best result
                    if ocr_results:
                        best_result = self.preprocessor.select_best_ocr_result(ocr_results)
                        text = best_result['text']
                        conf_ocr = best_result['conf']
                        ocr_method = best_result['method']
                        # Note: best_result may also include 'is_rotated' flag if needed
                    else:
                        # Fallback: try original crop if all preprocessing failed
                        try:
                            ocr_result = self.ocr.recognize(crop)
                            text = ocr_result.get('text', '')
                            conf_ocr = ocr_result.get('conf', 0.0)
                            ocr_method = 'original-fallback'
                        except Exception as e:
                            if frame_count % 50 == 0:
                                print(f"[TrailerVisionApp] OCR fallback failed: {e}")
                else:
                    # Original OCR without preprocessing
                    ocr_result = self.ocr.recognize(crop)
                    text = ocr_result['text']
                    conf_ocr = ocr_result['conf']
                    ocr_method = 'original'
            
            # Project center to world coordinates
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            world_coords = self._project_to_world(camera_id, center_x, center_y)
            
            # Resolve parking spot
            spot = "unknown"
            method = "no-calibration"
            if world_coords and self.spot_resolver:
                x_world, y_world = world_coords
                spot_result = self.spot_resolver.resolve(x_world, y_world)
                spot = spot_result['spot']
                method = spot_result['method']
            
            # Create event
            event = {
                'ts_iso': datetime.utcnow().isoformat(),
                'camera_id': camera_id,
                'track_id': track_id,
                'bbox': bbox,
                'text': text,
                'conf': conf_ocr,
                'ocr_method': ocr_method,  # Track which preprocessing method was used
                'x_world': world_coords[0] if world_coords else None,
                'y_world': world_coords[1] if world_coords else None,
                'spot': spot,
                'method': method
            }
            
            # Log to CSV
            self.csv_logger.log(event)
            
            # Publish to event buses
            self.publisher.publish(event)
            
            # POST to REST ingest API if enabled
            if self.ingest_enabled:
                try:
                    requests.post(
                        self.ingest_url,
                        json=event,
                        timeout=1.0
                    )
                except Exception as e:
                    print(f"Error posting to ingest API: {e}")
            
            # Update last publish time
            metrics['last_publish'] = datetime.utcnow()
        
        # Save screenshot if enabled
        if save_frames and len(tracks) > 0:
            # Save frame with first track
            track_id = tracks[0]['track_id'] if tracks else None
            self.media_rotator.save_frame(camera_id, frame, track_id)
        
        # Update metrics
        metrics['frames_processed'] += 1
        
        # Update FPS (simple EMA)
        # In production, use proper time-based FPS calculation
        if metrics['frames_processed'] == 1:
            metrics['fps_ema'] = 30.0  # Initial estimate
        else:
            alpha = 0.1
            metrics['fps_ema'] = alpha * 30.0 + (1 - alpha) * metrics['fps_ema']
        
        # Update metrics server
        queue_depth = self.publisher.get_queue_depth()
        self.metrics_server.update_camera_metrics(
            camera_id,
            metrics['fps_ema'],
            metrics['frames_processed'],
            metrics['last_publish'],
            queue_depth
        )
    
    def _process_camera(self, camera: Dict):
        """
        Process a single camera stream.
        
        Args:
            camera: Camera configuration dict
        """
        camera_id = camera['id']
        rtsp_url = camera['rtsp_url']
        width = camera.get('width', 1920)
        height = camera.get('height', 1080)
        fps_cap = camera.get('fps_cap', 30)
        
        globals_cfg = self.config.get('globals', {})
        use_gstreamer = globals_cfg.get('use_gstreamer', False)
        
        print(f"Opening stream for {camera_id}: {rtsp_url}")
        cap = open_stream(rtsp_url, width, height, fps_cap, use_gstreamer)
        
        if cap is None:
            print(f"Failed to open stream for {camera_id}")
            return
        
        frame_count = 0
        
        try:
            for ret, frame in frame_generator(cap):
                if not ret or frame is None:
                    break
                
                if not self.running:
                    break
                
                self._process_frame(camera_id, frame, frame_count)
                frame_count += 1
                
        except KeyboardInterrupt:
            print(f"Interrupted processing for {camera_id}")
        except Exception as e:
            print(f"Error processing {camera_id}: {e}")
        finally:
            cap.release()
            print(f"Closed stream for {camera_id}")
    
    def run(self):
        """Run the main application loop."""
        print("Starting Trailer Vision Edge application...")
        self.running = True
        
        # Process each camera in a separate thread (or sequentially for simplicity)
        import threading
        
        threads = []
        for camera in self.config.get('cameras', []):
            thread = threading.Thread(
                target=self._process_camera,
                args=(camera,),
                daemon=True
            )
            thread.start()
            threads.append(thread)
        
        # Wait for threads (or handle signals)
        try:
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.stop()
    
    def stop(self):
        """Stop the application and cleanup."""
        print("Stopping application...")
        self.running = False
        
        # Stop publishers
        if self.publisher:
            self.publisher.stop()
        
        # Stop uploader
        if self.uploader:
            self.uploader.stop()
        
        # Close CSV logger
        if self.csv_logger:
            self.csv_logger.close()
        
        # Stop metrics server
        if self.metrics_server:
            self.metrics_server.stop()
        
        print("Application stopped.")


def main():
    """Main entry point."""
    # Initialize CUDA context in main thread before loading TensorRT engines
    # This ensures all worker threads can access the same context
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        # Get the primary context created by autoinit
        primary_ctx = cuda.Context.get_current()
        if primary_ctx is not None:
            # Store it globally so worker threads can access it
            import sys
            # Store in both module names for compatibility
            if '__main__' in sys.modules:
                sys.modules['__main__']._cuda_primary_context = primary_ctx
            # Also store in the actual module
            if 'app.main_trt_demo' in sys.modules:
                sys.modules['app.main_trt_demo']._cuda_primary_context = primary_ctx
            # Also store in this module's globals
            globals()['_cuda_primary_context'] = primary_ctx
            print(f"CUDA context initialized in main thread (via autoinit), context: {primary_ctx}")
        else:
            print("Warning: CUDA context initialization returned None")
    except Exception as e:
        print(f"Warning: Failed to initialize CUDA context: {e}")
        # Continue anyway - might work with autoinit in worker threads
    
    app = TrailerVisionApp()
    
    # Handle signals
    def signal_handler(sig, frame):
        app.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run application
    app.run()


if __name__ == '__main__':
    main()

