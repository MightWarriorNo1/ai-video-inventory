"""
Main Application Loop - Trailer Vision Edge

End-to-end processing pipeline:
1. Camera ingestion (RTSP/USB)
2. Detection (YOLO TensorRT)
3. Tracking (ByteTrack)
4. OCR (TrOCR/PaddleOCR/CRNN TensorRT)
5. BEV projection (image -> world using deep learning)
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
import gc
import time

# Try to import torch for GPU memory management
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Application modules
from app.rtsp import open_stream, frame_generator
from app.ai.detector_trt import TrtEngineYOLO
from app.ai.detector_yolov8 import YOLOv8Detector
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
from app.gps_sensor import GPSSensor
from app.video_recorder import VideoRecorder
from app.processing_queue import ProcessingQueueManager
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
        self.processing_queue = None  # Processing queue manager for automated workflow
        
        # Per-camera trackers and BEV projectors
        self.trackers = {}
        self.camera_3d_projectors = {}  # camera_id -> Camera3DProjector (priority)
        self.bev_projectors = {}  # camera_id -> BEVProjector (fallback)
        
        # Check if OCR is oLmOCR (needs GPU memory cleanup)
        self.is_olmocr = False  # Will be set after OCR initialization
        
        # Frame storage for camera feed display (thread-safe)
        import threading
        self.frame_lock = threading.Lock()
        self.latest_frames = {}  # camera_id -> latest frame
        
        # Metrics tracking
        self.camera_metrics = {}
        
        # Frame storage for video streaming (thread-safe)
        self.latest_frames = {}
        self.frame_lock = threading.Lock()
        
        # OCR optimization: Cache results per track_id to avoid re-processing
        self.ocr_cache = {}  # (camera_id, track_id) -> {'text': str, 'conf': float, 'frame': int, 'last_updated': int}
        self.ocr_cache_max_age = 30  # Re-run OCR every 30 frames if no good result
        self.ocr_min_confidence = 0.5  # Only cache results with confidence >= this
        self.ocr_run_every_n_frames = 10  # Run OCR every N frames for existing tracks (if not cached)
        self.ocr_cache_max_size = 100  # Maximum cache entries to prevent memory leaks
        
        # GPS sensor and video recorder
        self.gps_sensor = None
        self.video_recorder = None
        
        # Graceful shutdown state tracking
        self.recording_stopped = False  # True when recording stopped but processing continues
        self.shutdown_lock = threading.Lock()  # Lock for thread-safe state access
        
        self._initialize_components()
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _initialize_components(self):
        """Initialize all application components."""
        globals_cfg = self.config.get('globals', {})
        
        # Initialize detector - prefer fine-tuned YOLO model over TensorRT engine
        fine_tuned_model_path = "runs/detect/truck_detector_finetuned/weights/best.pt"
        engine_path = "models/trailer_detector.engine"
        
        if os.path.exists(fine_tuned_model_path):
            # Use fine-tuned YOLO model
            print(f"Loading fine-tuned YOLO model: {fine_tuned_model_path}")
            try:
                self.detector = YOLOv8Detector(
                    model_name=fine_tuned_model_path,
                    conf_threshold=globals_cfg.get('detector_conf', 0.35),
                    target_class=0  # Fine-tuned model is single-class (trailer = class 0)
                )
                print(f"✓ Fine-tuned YOLO model loaded successfully")
            except Exception as e:
                print(f"Error loading fine-tuned YOLO model: {e}")
                print(f"Falling back to TensorRT engine...")
                self.detector = None
        else:
            print(f"Fine-tuned model not found: {fine_tuned_model_path}")
            self.detector = None
        
        # Fallback to TensorRT engine if fine-tuned model not available or failed
        if self.detector is None and os.path.exists(engine_path):
            print(f"Loading TensorRT engine: {engine_path}")
            try:
                self.detector = TrtEngineYOLO(
                    engine_path,
                    conf_threshold=globals_cfg.get('detector_conf', 0.35)
                )
                print(f"✓ TensorRT engine loaded successfully")
            except Exception as e:
                print(f"Error loading TensorRT engine: {e}")
                self.detector = None
        
        if self.detector is None:
            print(f"Warning: No detector available. Tried:")
            print(f"  - Fine-tuned model: {fine_tuned_model_path}")
            print(f"  - TensorRT engine: {engine_path}")
        
        # Initialize OCR - Lazy loading: Don't load OCR initially to save memory
        # OCR will be loaded on-demand when explicitly requested (not automatically)
        # This prevents memory issues when processing large videos
        self.ocr = None
        self.is_olmocr = False
        print(f"[TrailerVisionApp] OCR loading deferred - will be loaded on-demand when requested")
        
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
        self.metrics_server = MetricsServer(
            port=metrics_port, 
            csv_logger=self.csv_logger,
            frame_storage=self  # Pass self for camera feed display
        )
        self.metrics_server.start()
        
        # Initialize GPS sensor
        try:
            gps_port = os.getenv('GPS_PORT', None)
            # Try common GPS ports if not specified
            if gps_port is None:
                # Try common Linux ports
                for port in ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0']:
                    if os.path.exists(port):
                        gps_port = port
                        break
                # Try Windows COM ports if Linux ports not found
                if gps_port is None:
                    import platform
                    if platform.system() == 'Windows':
                        for port in ['COM3', 'COM4', 'COM5']:
                            gps_port = port
                            break
            
            if gps_port:
                self.gps_sensor = GPSSensor(port=gps_port, baudrate=4800)
                self.gps_sensor.start()
                print(f"[TrailerVisionApp] GPS sensor initialized on {gps_port} at 4800 baud")
            else:
                print(f"[TrailerVisionApp] No GPS port found, GPS sensor disabled")
                self.gps_sensor = None
        except Exception as e:
            print(f"[TrailerVisionApp] Failed to initialize GPS sensor: {e}")
            self.gps_sensor = None
        
        # Initialize video recorder with 45-second auto-chunking
        # Callback will be set after processing queue is initialized
        self.video_recorder = VideoRecorder(
            output_dir="out/recordings",
            gps_sensor=self.gps_sensor,
            chunk_duration_seconds=45.0,  # 45-second chunks
            on_chunk_saved=None  # Will be set after processing queue initialization
        )
        print(f"[TrailerVisionApp] Video recorder initialized with 45-second auto-chunking")
        
        # Initialize image preprocessor
        # IMPORTANT: Disable rotation for oLmOCR and EasyOCR because they handle rotation internally.
        # Pre-rotating images causes double rotation issues.
        preproc_cfg = globals_cfg.get('preprocessing', {})
        ocr_type = str(type(self.ocr)) if self.ocr is not None else ""
        is_olmocr = 'OlmOCRRecognizer' in ocr_type
        is_easyocr = 'EasyOCRRecognizer' in ocr_type
        # Disable rotation for oLmOCR and EasyOCR (both handle rotation internally)
        enable_rotation = not (is_olmocr or is_easyocr)
        
        self.preprocessor = ImagePreprocessor(
            enable_yolo_preprocessing=preproc_cfg.get('enable_yolo', True),
            enable_ocr_preprocessing=preproc_cfg.get('enable_ocr', True),
            yolo_strategy=preproc_cfg.get('yolo_strategy', 'enhanced'),
            ocr_strategy=preproc_cfg.get('ocr_strategy', 'multi'),
            enable_rotation=enable_rotation
        )
        print(f"[TrailerVisionApp] Image preprocessing enabled:")
        print(f"  YOLO: {self.preprocessor.enable_yolo_preprocessing} ({self.preprocessor.yolo_strategy})")
        print(f"  OCR: {self.preprocessor.enable_ocr_preprocessing} ({self.preprocessor.ocr_strategy})")
        if is_olmocr:
            rotation_msg = "Disabled (oLmOCR handles rotation internally)"
        elif is_easyocr:
            rotation_msg = "Disabled (EasyOCR handles rotation internally)"
        else:
            rotation_msg = "Enabled"
        print(f"  Rotation: {rotation_msg}")
        
        # Load 3D projectors and BEV projectors for each camera
        self.gps_references = {}  # Store GPS reference for each camera
        from app.bev_utils import BEVProjector
        
        for camera in self.config.get('cameras', []):
            camera_id = camera['id']
            
            # Priority 1: Load 3D projector if available
            calib_3d_path = f"config/calib/{camera_id}_3d.json"
            gps_reference = None
            
            if os.path.exists(calib_3d_path):
                try:
                    from app.camera_3d_projection import Camera3DProjector
                    self.camera_3d_projectors[camera_id] = Camera3DProjector.from_calibration_file(calib_3d_path)
                    gps_msg = ""
                    if self.camera_3d_projectors[camera_id].gps_reference:
                        gps_ref = self.camera_3d_projectors[camera_id].gps_reference
                        gps_reference = gps_ref
                        gps_msg = f"with GPS reference: ({gps_ref['lat']:.6f}, {gps_ref['lon']:.6f})"
                        self.gps_references[camera_id] = gps_ref
                    print(f"Loaded 3D projector for {camera_id} {gps_msg}")
                except Exception as e:
                    print(f"Error loading 3D projector for {camera_id}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Fallback: Also check homography calibration file for GPS reference if not set
            if not gps_reference:
                calib_path = f"config/calib/{camera_id}_h.json"
                if os.path.exists(calib_path):
                    with open(calib_path, 'r') as f:
                        calib_data = json.load(f)
                        # Load GPS reference if available
                        if 'gps_reference' in calib_data:
                            gps_ref = calib_data['gps_reference']
                            gps_reference = {
                                'lat': gps_ref['lat'],
                                'lon': gps_ref['lon']
                            }
                            self.gps_references[camera_id] = gps_reference
            
            # Priority 2: Load BEV transformer if configured (fallback)
            bev_model_path = f"models/bev/{camera_id}_bev.pth"
            if os.path.exists(bev_model_path):
                try:
                    # Get input size from camera config or use default
                    input_size = (camera.get('height', 720), camera.get('width', 1280))
                    
                    self.bev_projectors[camera_id] = BEVProjector(
                        model_path=bev_model_path,
                        gps_reference=gps_reference,
                        input_size=input_size
                    )
                    gps_msg = f"with GPS reference: ({gps_reference['lat']:.6f}, {gps_reference['lon']:.6f})" if gps_reference else "(no GPS reference - using local meters)"
                    print(f"Loaded BEV projector for {camera_id} {gps_msg}")
                except Exception as e:
                    print(f"Error loading BEV projector for {camera_id}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Warning: BEV model not found for {camera_id}: {bev_model_path}")
        
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
        
        # Get BEV projector and GPS reference for first camera (or None) for video processing
        test_bev_projector = None
        test_gps_reference = None
        cameras_list = self.config.get('cameras', [])
        if cameras_list:
            first_camera_id = cameras_list[0]['id']
            if first_camera_id in self.bev_projectors:
                test_bev_projector = self.bev_projectors[first_camera_id]
                if first_camera_id in self.gps_references:
                    test_gps_reference = self.gps_references[first_camera_id]
        
        print(f"[TrailerVisionApp] Initializing video processor:")
        print(f"  - Detector: {'Available' if self.detector else 'NOT AVAILABLE'}")
        print(f"  - OCR: {'Available' if self.ocr else 'NOT AVAILABLE'}")
        print(f"  - Spot Resolver: {'Available' if self.spot_resolver else 'NOT AVAILABLE'}")
        # Check for 3D projector
        test_3d_projector = None
        if cameras_list:
            first_camera_id = cameras_list[0]['id']
            if first_camera_id in self.camera_3d_projectors:
                test_3d_projector = self.camera_3d_projectors[first_camera_id]
        
        print(f"  - 3D Projector: {'Available' if test_3d_projector is not None else 'NOT AVAILABLE'}")
        print(f"  - BEV Projector: {'Available' if test_bev_projector is not None else 'NOT AVAILABLE'}")
        if test_gps_reference:
            print(f"  - GPS Reference: ({test_gps_reference['lat']:.6f}, {test_gps_reference['lon']:.6f})")
        
        try:
            # Determine camera_id for homography loading (use first camera or "test-video")
            test_camera_id = "test-video"
            if cameras_list:
                test_camera_id = cameras_list[0]['id']
            
            self.video_processor = VideoProcessor(
                preprocessor=self.preprocessor,
                detector=self.detector,
                ocr=self.ocr,
                tracker_factory=create_tracker,
                spot_resolver=self.spot_resolver,
                bev_projector=test_bev_projector,
                gps_reference=test_gps_reference,
                camera_id=test_camera_id
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
        
        # Processing queue will be initialized when Start Application is called
        # This allows lazy loading of OCR to save memory until needed
        self.processing_queue = None
        print(f"[TrailerVisionApp] Processing queue will be initialized when application starts")
    
    def initialize_assets(self):
        """
        Initialize all assets required for automated processing.
        This includes OCR model loading and processing queue setup.
        
        Returns:
            Dict with 'success', 'message', and 'assets_loaded' status
        """
        assets_loaded = {
            'detector': self.detector is not None,
            'ocr': self.ocr is not None,
            'video_processor': self.video_processor is not None,
            'processing_queue': self.processing_queue is not None
        }
        
        try:
            # 1. Check detector
            if not self.detector:
                return {
                    'success': False,
                    'message': 'Detector not available. Please check model files.',
                    'assets_loaded': assets_loaded
                }
            
            # 2. Load OCR if not already loaded
            if self.ocr is None:
                print(f"[TrailerVisionApp] Loading OCR for automated processing...")
                self._initialize_ocr()
                assets_loaded['ocr'] = self.ocr is not None
            
            if not self.ocr:
                return {
                    'success': False,
                    'message': 'OCR model failed to load. Please check OCR model files.',
                    'assets_loaded': assets_loaded
                }
            
            # 3. Ensure video processor is available
            if not self.video_processor:
                return {
                    'success': False,
                    'message': 'Video processor not available.',
                    'assets_loaded': assets_loaded
                }
            
            # 4. Initialize processing queue if not already initialized
            if self.processing_queue is None:
                print(f"[TrailerVisionApp] Initializing processing queue...")
                
                # Define callbacks for extensibility
                def on_video_complete(video_path, crops_dir, results):
                    """Called when video processing completes."""
                    print(f"[TrailerVisionApp] Video processing complete: {Path(video_path).name}")
                    print(f"  - Crops directory: {crops_dir}")
                    # Extensibility point: Add server upload or other processing here
                    if results:
                        self.upload_to_server(video_path, crops_dir, {'type': 'video_processing', 'results': results})
                
                def on_ocr_complete(video_path, crops_dir, ocr_results):
                    """Called when OCR processing completes."""
                    print(f"[TrailerVisionApp] OCR processing complete: {Path(video_path).name}")
                    print(f"  - Processed {len(ocr_results)} crops with OCR")
                    # Extensibility point: Add server upload here
                    self.upload_to_server(video_path, crops_dir, {'type': 'ocr_complete', 'ocr_results': ocr_results})
                
                try:
                    self.processing_queue = ProcessingQueueManager(
                        video_processor=self.video_processor,
                        ocr=self.ocr,
                        preprocessor=self.preprocessor,
                        on_video_complete=on_video_complete,
                        on_ocr_complete=on_ocr_complete
                    )
                    print(f"[TrailerVisionApp] Processing queue manager initialized")
                    assets_loaded['processing_queue'] = True
                    
                    # Set up video recorder callback for auto-processing
                    def on_chunk_saved(video_path, gps_log_path):
                        """Called when a video chunk is saved."""
                        # Extract camera_id from video path
                        video_name = Path(video_path).stem
                        # Format: camera_id_timestamp_chunkXXXX
                        parts = video_name.split('_')
                        camera_id = parts[0] if parts else "unknown"
                        
                        print(f"[TrailerVisionApp] Chunk saved: {Path(video_path).name}")
                        print(f"  - Queueing for processing...")
                        
                        # Queue video processing
                        if self.processing_queue:
                            self.processing_queue.queue_video_processing(
                                video_path=video_path,
                                camera_id=camera_id,
                                gps_log_path=gps_log_path,
                                detect_every_n=5,
                                detection_mode='trailer'
                            )
                    
                    # Update video recorder with callback
                    self.video_recorder.on_chunk_saved = on_chunk_saved
                    print(f"[TrailerVisionApp] Video recorder callback configured for auto-processing")
                    
                except Exception as e:
                    print(f"[TrailerVisionApp] ERROR: Failed to initialize processing queue: {e}")
                    import traceback
                    traceback.print_exc()
                    return {
                        'success': False,
                        'message': f'Failed to initialize processing queue: {str(e)}',
                        'assets_loaded': assets_loaded
                    }
            
            # All assets loaded successfully
            return {
                'success': True,
                'message': 'All assets initialized successfully',
                'assets_loaded': assets_loaded
            }
            
        except Exception as e:
            print(f"[TrailerVisionApp] ERROR: Failed to initialize assets: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'message': f'Failed to initialize assets: {str(e)}',
                'assets_loaded': assets_loaded
            }
    
    def _get_gps_reference(self, camera_id: str) -> Optional[Dict[str, float]]:
        """
        Get GPS reference point for coordinate conversion.
        
        Priority: Live GPS sensor > Static calibration GPS reference
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            Dict with 'lat' and 'lon' keys, or None if no GPS reference available
        """
        # Priority 1: Use live GPS sensor if available and has data
        if self.gps_sensor:
            try:
                current_gps = self.gps_sensor.get_current_gps()
                if current_gps and current_gps.get('lat') and current_gps.get('lon'):
                    return {
                        'lat': current_gps['lat'],
                        'lon': current_gps['lon']
                    }
            except Exception as e:
                # GPS sensor error - fall back to static reference
                pass
        
        # Priority 2: Fall back to static GPS reference from calibration
        if camera_id in self.gps_references:
            return self.gps_references[camera_id]
        
        return None
    
    def _project_to_world(self, camera_id: str, x_img: float, y_img: float, return_gps: bool = True, frame: Optional[np.ndarray] = None) -> Optional[tuple]:
        """
        Project image coordinates to world coordinates.
        
        Priority: 3D Projection > Homography > BEV Transformer
        
        Args:
            camera_id: Camera identifier
            x_img: Image X coordinate
            y_img: Image Y coordinate
            return_gps: If True and GPS reference available, return GPS coordinates (lat, lon).
                       If False or no GPS reference, return local meters (x, y).
            frame: Optional frame image (required for BEV projection, not needed for 3D/homography)
            
        Returns:
            Tuple of (lat, lon) if return_gps=True and GPS reference available,
            Tuple of (x_world, y_world) in meters otherwise,
            or None if no projector available
        """
        # Priority 1: Use 3D projection if available
        if camera_id in self.camera_3d_projectors:
            try:
                projector = self.camera_3d_projectors[camera_id]
                x_world, y_world = projector.pixel_to_world(
                    x_img, y_img,
                    plane_z=0.0,  # Project to ground plane
                    return_gps=return_gps
                )
                return (x_world, y_world)
            except Exception as e:
                print(f"Warning: 3D projection failed for {camera_id}: {e}")
                # Fall through to other methods
        
        # Priority 2: Use BEV transformer if available
        if camera_id in self.bev_projectors:
            if frame is None:
                # Try to get latest frame from storage
                if camera_id in self.latest_frames:
                    frame = self.latest_frames[camera_id]
                else:
                    print(f"Warning: No frame available for BEV projection for camera {camera_id}")
                    return None
            
            bev_projector = self.bev_projectors[camera_id]
            
            # Project using BEV transformer
            x_world, y_world = bev_projector.project_to_world(
                frame,
                x_img,
                y_img,
                return_gps=False  # Get meters first, convert to GPS below if needed
            )
        
        # Convert to GPS if requested and reference available
        if return_gps:
            gps_ref = self._get_gps_reference(camera_id)
            if gps_ref:
                from app.gps_utils import meters_to_gps
                lat, lon = meters_to_gps(x_world, y_world, gps_ref['lat'], gps_ref['lon'])
                return (float(lat), float(lon))
        
        return (float(x_world), float(y_world))
    
    def _cleanup_gpu_memory(self, force: bool = False):
        """
        Clean up GPU memory after OCR operations to prevent accumulation.
        This is critical when processing multiple frames in sequence.
        
        Args:
            force: If True, always cleanup. If False, cleanup only periodically.
        """
        if not TORCH_AVAILABLE or not self.is_olmocr:
            return
        
        # OPTIMIZATION: Only cleanup periodically to reduce overhead
        if not force:
            if not hasattr(self, '_ocr_call_count'):
                self._ocr_call_count = 0
            self._ocr_call_count += 1
            # Only cleanup every 5 OCR operations to reduce overhead
            if self._ocr_call_count % 5 != 0:
                return
        
        if torch.cuda.is_available():
            try:
                # Wait for all GPU operations to complete
                torch.cuda.synchronize()
                # Clear CUDA cache
                torch.cuda.empty_cache()
                # Force Python garbage collection (less frequently)
                if force or (hasattr(self, '_ocr_call_count') and self._ocr_call_count % 10 == 0):
                    gc.collect()
                # Clear cache again after GC
                torch.cuda.empty_cache()
                gc.collect()
                # Clear cache again after GC
                torch.cuda.empty_cache()
            except Exception:
                # Silently fail - don't interrupt frame processing
                pass
    
    def _initialize_ocr(self):
        """
        Initialize OCR model (lazy loading).
        This is called after YOLO video processing is complete to save memory.
        """
        if self.ocr is not None:
            print(f"[TrailerVisionApp] OCR already initialized, skipping...")
            return
        
        globals_cfg = self.config.get('globals', {})
        
        print(f"[TrailerVisionApp] Initializing OCR...")
        
        # Initialize OCR - Priority: oLmOCR > EasyOCR > TrOCR > PaddleOCR English-only > Multilingual > Legacy CRNN
        # oLmOCR (Qwen3-VL/Qwen2.5-VL) is recommended for best accuracy, especially for vertical text and complex layouts
        self.ocr = None
        
        # Try oLmOCR first (best accuracy for vertical text, complex layouts, and multi-language support)
        try:
            from app.ocr.olmocr_recognizer import OlmOCRRecognizer
            # Use Qwen3-VL-4B-Instruct for best OCR performance (32 languages, 4B params)
            # For edge devices with limited memory, consider Qwen2.5-VL-3B-Instruct
            # OPTIMIZATION: Enable fast preprocessing for better speed
            # Set fast_preprocessing=True for 2-3x faster preprocessing (slightly less enhancement)
            self.ocr = OlmOCRRecognizer(
                model_name="Qwen/Qwen3-VL-4B-Instruct",
                use_gpu=True,
                fast_preprocessing=globals_cfg.get('ocr_fast_preprocessing', False)  # Enable via config
            )
            self.is_olmocr = True  # Mark as oLmOCR for GPU memory cleanup
            print(f"✓ Using oLmOCR (Qwen3-VL-4B-Instruct) - recommended for high accuracy, especially vertical text")
        except ImportError:
            print("oLmOCR not available (pip3 install transformers torch qwen-vl-utils), trying EasyOCR...")
        except Exception as e:
            print(f"oLmOCR initialization failed: {e}, trying EasyOCR...")
        
        # Try EasyOCR if oLmOCR failed (self.ocr is still None)
        if self.ocr is None:
            try:
                from app.ocr.easyocr_recognizer import EasyOCRRecognizer
                self.ocr = EasyOCRRecognizer(languages=['en'], gpu=True)
                print(f"✓ Using EasyOCR (fallback - good accuracy for printed text)")
            except ImportError:
                print("EasyOCR not available (pip3 install easyocr), trying TrOCR...")
            except Exception as e:
                print(f"EasyOCR initialization failed: {e}, trying TrOCR...")
        
        # Only try TrOCR if oLmOCR and EasyOCR failed (self.ocr is still None)
        if self.ocr is None:
            # Try TrOCR (transformer-based, good accuracy for printed text)
            # Check for full TrOCR model first, then encoder-only
            trocr_engine_paths = [
                "models/trocr_full.engine",  # Full encoder-decoder model (preferred)
                "models/trocr.engine",        # Encoder-only or full model
            ]
            
            trocr_loaded = False
            for trocr_path in trocr_engine_paths:
                if os.path.exists(trocr_path):
                    try:
                        from app.ocr.trocr_recognizer import TrOCRRecognizer
                        # Try to find tokenizer in standard locations
                        tokenizer_paths = [
                            "models/trocr_base_printed",
                            "models/trocr-base-printed",
                        ]
                        model_dir = None
                        for path in tokenizer_paths:
                            if os.path.exists(path):
                                model_dir = path
                                break
                        
                        self.ocr = TrOCRRecognizer(trocr_path, model_dir=model_dir)
                        print(f"✓ Using TrOCR model: {trocr_path} (fallback - oLmOCR/EasyOCR not available)")
                        trocr_loaded = True
                        break
                    except ImportError:
                        print("TrOCR not available (transformers not installed), trying other OCR models...")
                        break
                    except Exception as e:
                        print(f"TrOCR initialization failed ({trocr_path}): {e}, trying other OCR models...")
                        continue
        
        # Fallback to PaddleOCR models if oLmOCR, EasyOCR, and TrOCR not available
        if self.ocr is None:
            ocr_path = None
            alphabet_path = None
            input_size = (320, 48)
            
            # Try PaddleOCR English-only (best for English/number text)
            if os.path.exists("models/paddleocr_rec_english.engine") and os.path.exists("app/ocr/ppocr_keys_en.txt"):
                ocr_path = "models/paddleocr_rec_english.engine"
                alphabet_path = "app/ocr/ppocr_keys_en.txt"
                print("Using PaddleOCR English-only model (fallback)")
            # Fallback to PaddleOCR multilingual
            elif os.path.exists("models/paddleocr_rec.engine") and os.path.exists("app/ocr/ppocr_keys_v1.txt"):
                ocr_path = "models/paddleocr_rec.engine"
                alphabet_path = "app/ocr/ppocr_keys_v1.txt"
                print("Using PaddleOCR multilingual model (fallback)")
            # Fallback to legacy CRNN engine
            elif os.path.exists("models/ocr_crnn.engine") and os.path.exists("app/ocr/alphabet.txt"):
                ocr_path = "models/ocr_crnn.engine"
                alphabet_path = "app/ocr/alphabet.txt"
                input_size = None  # CRNN uses default size
                print("Using legacy CRNN model (fallback)")
            
            if ocr_path and alphabet_path:
                if "paddleocr" in ocr_path.lower():
                    self.ocr = PlateRecognizer(ocr_path, alphabet_path, input_size=input_size)
                else:
                    self.ocr = PlateRecognizer(ocr_path, alphabet_path)
                print(f"Loaded OCR engine: {ocr_path} with alphabet: {alphabet_path}")
        
        if self.ocr is None:
            print(f"Warning: No OCR engine found. Tried:")
            print(f"  - oLmOCR (recommended - install: pip3 install transformers torch qwen-vl-utils)")
            print(f"  - EasyOCR (fallback - install: pip3 install easyocr)")
            print(f"  - models/trocr.engine (TrOCR)")
            print(f"  - models/paddleocr_rec_english.engine (English-only)")
            print(f"  - models/paddleocr_rec.engine (multilingual)")
            print(f"  - models/ocr_crnn.engine (legacy)")
        else:
            # Update video processor OCR if it exists
            if hasattr(self, 'video_processor') and self.video_processor is not None:
                self.video_processor.ocr = self.ocr
                # Update is_olmocr flag in video processor
                self.video_processor.is_olmocr = self.is_olmocr
                print(f"[TrailerVisionApp] Updated video processor with OCR")
    
    def _unload_detector(self):
        """
        Unload YOLO detector to free GPU memory.
        This is called after YOLO video processing is complete.
        """
        if self.detector is None:
            print(f"[TrailerVisionApp] Detector already unloaded, skipping...")
            return
        
        print(f"[TrailerVisionApp] Unloading YOLO detector to free GPU memory...")
        
        # Clean up detector based on type
        detector_type = type(self.detector).__name__
        
        if detector_type == "YOLOv8Detector":
            # YOLOv8 uses PyTorch/Ultralytics
            try:
                if hasattr(self.detector, 'model'):
                    # Move model to CPU and delete
                    if hasattr(self.detector.model, 'to'):
                        self.detector.model = self.detector.model.to('cpu')
                    del self.detector.model
                del self.detector
            except Exception as e:
                print(f"[TrailerVisionApp] Warning: Error unloading YOLOv8 detector: {e}")
        elif detector_type == "TrtEngineYOLO":
            # TensorRT engine - delete the engine
            try:
                if hasattr(self.detector, 'engine'):
                    del self.detector.engine
                if hasattr(self.detector, 'context'):
                    del self.detector.context
                del self.detector
            except Exception as e:
                print(f"[TrailerVisionApp] Warning: Error unloading TensorRT detector: {e}")
        
        self.detector = None
        
        # Clean up GPU memory
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"[TrailerVisionApp] Warning: Error cleaning GPU memory: {e}")
        
        print(f"[TrailerVisionApp] YOLO detector unloaded successfully")
    
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
            all_detections = self.detector.detect(processed_frame)
            
            # OPTIMIZATION: Filter to only trailer class (class 0) before tracking
            # This ensures OCR only runs on confirmed trailers, not false positives
            # Your YOLO model is single-class (trailer_back = class 0)
            trailer_class_id = 0  # Trailer class ID from your YOLO model
            for det in all_detections:
                # Check if detection is a trailer (class 0)
                if det.get('cls', -1) == trailer_class_id:
                    detections.append(det)
                # Optional: Log filtered detections for debugging
                # elif frame_count % 100 == 0:
                #     print(f"[TrailerVisionApp] Filtered non-trailer detection: cls={det.get('cls', -1)}, conf={det.get('conf', 0.0):.2f}")
        
        # Update tracker (now only contains trailer detections)
        tracks = tracker.update(detections, frame)
        
        # Process each track (all should be trailers now)
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            x1, y1, x2, y2 = bbox
            
            # OPTIMIZATION: Double-check this is a trailer before OCR (safety check)
            # Since we filter detections before tracking, all tracks should be trailers
            # This is a safety check in case tracker preserves class info
            track_cls = track.get('cls', -1)
            trailer_class_id = 0  # Trailer class ID
            
            # Skip OCR if this is explicitly not a trailer (safety check)
            # Note: If tracker doesn't preserve 'cls', this check is skipped (track_cls = -1)
            if track_cls != -1 and track_cls != trailer_class_id:
                if frame_count % 50 == 0:  # Log occasionally
                    print(f"[TrailerVisionApp] Skipping OCR for non-trailer track {track_id}: cls={track_cls}")
                continue  # Skip this track - not a trailer
            
            # Refine bounding box to rear face (focus on back side only)
            # This prevents OCR from detecting text on the sides of trailers
            orig_width = x2 - x1
            orig_height = y2 - y1
            orig_aspect = orig_width / orig_height if orig_height > 0 else 1.0
            
            # If aspect ratio suggests side view (wide), extract center portion for rear face
            # For wide detections (side view), the rear face is typically in the center
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
                h, w = frame.shape[:2]
                rear_x1 = max(0, min(rear_x1, w - 1))
                rear_x2 = max(rear_x1 + 1, min(rear_x2, w - 1))
                rear_y1 = max(0, min(rear_y1, h - 1))
                rear_y2 = max(rear_y1 + 1, min(rear_y2, h - 1))
                
                # Use refined coordinates for OCR (rear face only)
                x1, y1, x2, y2 = rear_x1, rear_y1, rear_x2, rear_y2
            # For narrow/tall detections (front/back view), use original bbox (already focused)
            
            # Crop region for OCR (now focused on rear face)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            # OPTIMIZATION: Check OCR cache first
            cache_key = (camera_id, track_id)
            should_run_ocr = True
            text = ""
            conf_ocr = 0.0
            ocr_method = "cached"
            
            if cache_key in self.ocr_cache:
                cached_result = self.ocr_cache[cache_key]
                cache_age = frame_count - cached_result['last_updated']
                
                # Use cached result if:
                # 1. It has good confidence (>= min_confidence)
                # 2. It's not too old (within max_age frames)
                # 3. Or we have a valid text result
                if (cached_result['conf'] >= self.ocr_min_confidence and 
                    cache_age < self.ocr_cache_max_age) or cached_result['text']:
                    text = cached_result['text']
                    conf_ocr = cached_result['conf']
                    should_run_ocr = False
                # Re-run OCR if cache is stale or low confidence
                elif cache_age >= self.ocr_cache_max_age:
                    should_run_ocr = True
                # For existing tracks, only run OCR periodically
                elif frame_count % self.ocr_run_every_n_frames != 0:
                    should_run_ocr = False
                    text = cached_result['text']  # Use cached even if low confidence
                    conf_ocr = cached_result['conf']
            
            # OPTIMIZATION: Skip OCR on very small crops (likely false positives)
            crop_area = (x2 - x1) * (y2 - y1)
            min_crop_area = 1000  # Minimum pixels for OCR (e.g., 32x32 = 1024)
            if crop_area < min_crop_area:
                should_run_ocr = False
                if cache_key not in self.ocr_cache:
                    text = ""
                    conf_ocr = 0.0
            
            # Run OCR with preprocessing (only if needed)
            if self.ocr and should_run_ocr:
                # OPTIMIZATION: Resize large crops before OCR to speed up processing
                # Large images take much longer to process
                h_crop, w_crop = crop.shape[:2]
                max_dimension = 640  # Maximum dimension for OCR (balance speed vs accuracy)
                if max(h_crop, w_crop) > max_dimension:
                    scale = max_dimension / max(h_crop, w_crop)
                    new_w = int(w_crop * scale)
                    new_h = int(h_crop * scale)
                    crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                if self.preprocessor and self.preprocessor.enable_ocr_preprocessing:
                    # Get multiple preprocessed versions
                    preprocessed_crops = self.preprocessor.preprocess_for_ocr(crop)
                    
                    # OPTIMIZATION: Try OCR on each preprocessed version with early exit
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
                                # OPTIMIZATION: Early exit if we get high confidence result
                                if result.get('conf', 0.0) >= 0.85:
                                    break
                        except Exception as e:
                            if frame_count % 50 == 0:  # Log occasionally
                                print(f"[TrailerVisionApp] OCR error with {prep['method']}: {e}")
                    
                    # Select best result
                    if ocr_results:
                        best_result = self.preprocessor.select_best_ocr_result(ocr_results)
                        text = best_result['text']
                        conf_ocr = best_result['conf']
                        ocr_method = best_result['method']
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
                            text = ""
                            conf_ocr = 0.0
                            ocr_method = 'error'
                else:
                    # Original OCR without preprocessing
                    try:
                        ocr_result = self.ocr.recognize(crop)
                        text = ocr_result['text']
                        conf_ocr = ocr_result['conf']
                        ocr_method = 'original'
                    except Exception as e:
                        if frame_count % 50 == 0:
                            print(f"[TrailerVisionApp] OCR error: {e}")
                        text = ""
                        conf_ocr = 0.0
                        ocr_method = 'error'
                
                # OPTIMIZATION: Cache OCR result (only cleanup GPU memory once after all OCR attempts)
                self.ocr_cache[cache_key] = {
                    'text': text,
                    'conf': conf_ocr,
                    'frame': frame_count,
                    'last_updated': frame_count
                }
                
                # OPTIMIZATION: Cleanup old cache entries to prevent memory leaks
                if len(self.ocr_cache) > self.ocr_cache_max_size:
                    # Remove oldest entries (by last_updated)
                    sorted_cache = sorted(self.ocr_cache.items(), key=lambda x: x[1]['last_updated'])
                    entries_to_remove = len(self.ocr_cache) - self.ocr_cache_max_size
                    for key, _ in sorted_cache[:entries_to_remove]:
                        del self.ocr_cache[key]
                
                # Clean GPU memory after OCR (only once, not after each preprocessing attempt)
                self._cleanup_gpu_memory()
            
            # Calculate ground contact point for trailer using learned linear model
            from app.bbox_to_image_coords_advanced import calculate_image_coords_from_bbox_with_config
            
            bbox = [float(x1), float(y1), float(x2), float(y2)]
            
            # Priority: If 3D projector available, use its bbox_to_ground_coords method
            # which properly accounts for trailer elevation
            if camera_id in self.camera_3d_projectors:
                try:
                    x_world, y_world = self.camera_3d_projectors[camera_id].bbox_to_ground_coords(
                        bbox,
                        method="backside_projection",
                        trailer_height=2.6  # Typical trailer back height in meters
                    )
                    world_coords_meters = (float(x_world), float(y_world))
                    # Image coords for reference (bottom-center)
                    center_x = (x1 + x2) / 2.0
                    bottom_y = float(y2)
                    image_coords = [float(center_x), float(bottom_y)]
                except Exception as e:
                    print(f"[TrailerVisionApp] Error in 3D bbox projection: {e}")
                    # Fall through to standard method
                    ground_x, ground_y = calculate_image_coords_from_bbox_with_config(bbox)
                    image_coords = [float(ground_x), float(ground_y)]
                    world_coords_meters = self._project_to_world(camera_id, ground_x, ground_y, return_gps=False, frame=frame)
            else:
                # Standard method: calculate image coords then project
                ground_x, ground_y = calculate_image_coords_from_bbox_with_config(bbox)
                image_coords = [float(ground_x), float(ground_y)]
                
                # Get world coordinates in meters first (for spot resolution)
                # Note: frame should be available in the calling context
                world_coords_meters = self._project_to_world(camera_id, ground_x, ground_y, return_gps=False, frame=frame)
            
            # Resolve parking spot (uses meters, same as GeoJSON)
            spot = "unknown"
            method = "no-calibration"
            if world_coords_meters and self.spot_resolver:
                x_world, y_world = world_coords_meters
                spot_result = self.spot_resolver.resolve(x_world, y_world)
                spot = spot_result['spot']
                method = spot_result['method']
            
            # Convert to GPS coordinates for output (if GPS reference available)
            # Uses live GPS sensor if available, otherwise falls back to static calibration reference
            world_coords_gps = None
            if world_coords_meters:
                gps_ref = self._get_gps_reference(camera_id)
                if gps_ref:
                    from app.gps_utils import meters_to_gps
                    x_meters, y_meters = world_coords_meters
                    lat, lon = meters_to_gps(x_meters, y_meters, gps_ref['lat'], gps_ref['lon'])
                    world_coords_gps = (lat, lon)
            
            # Use GPS coordinates if available, otherwise use meters
            world_coords_output = world_coords_gps if world_coords_gps else world_coords_meters
            
            # Create event
            event = {
                'ts_iso': datetime.utcnow().isoformat(),
                'camera_id': camera_id,
                'track_id': track_id,
                'bbox': bbox,
                'image_coords': image_coords,  # Calculated image coordinates (ground contact point)
                'text': text,
                'conf': conf_ocr,
                'ocr_method': ocr_method,  # Track which preprocessing method was used
                'x_world': world_coords_output[0] if world_coords_output else None,
                'y_world': world_coords_output[1] if world_coords_output else None,
                'lat': world_coords_gps[0] if world_coords_gps else None,  # GPS latitude
                'lon': world_coords_gps[1] if world_coords_gps else None,  # GPS longitude
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
        
        # CAMERA FEED PROCESSING - Only for display, no full processing
        # Enable camera feeds for dashboard display (demo stage)
        # Full processing is disabled, but we need to stream camera feeds for dashboard
        print("Camera feed display enabled (demo stage).")
        print("Full processing disabled - only displaying camera feeds.")
        
        # Process each camera in a separate thread for display only
        import threading
        
        threads = []
        for camera in self.config.get('cameras', []):
            thread = threading.Thread(
                target=self._process_camera_display_only,
                args=(camera,),
                daemon=True
            )
            thread.start()
            threads.append(thread)
        
        # Keep application running for video processing and camera display
        try:
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.stop()
    
    def _process_camera_display_only(self, camera: Dict):
        """
        Process camera feed for display only (no full processing).
        This just captures frames and makes them available for streaming.
        
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
        
        print(f"Opening camera stream for display: {camera_id}: {rtsp_url}")
        cap = open_stream(rtsp_url, width, height, fps_cap, use_gstreamer)
        
        if cap is None:
            print(f"Failed to open stream for {camera_id}")
            # Still register camera in metrics so dashboard shows it (even if stream failed)
            if self.metrics_server:
                self.metrics_server.update_camera_metrics(
                    camera_id,
                    fps_ema=0.0,
                    frames_processed_count=0,
                    last_publish=datetime.utcnow(),
                    queue_depth=0
                )
            return
        
        # Store the camera capture for streaming
        if not hasattr(self, 'camera_captures'):
            self.camera_captures = {}
        self.camera_captures[camera_id] = cap
        
        frame_count = 0
        
        # Register camera in metrics immediately so dashboard shows it
        if self.metrics_server:
            self.metrics_server.update_camera_metrics(
                camera_id,
                fps_ema=0.0,  # Will update when frames start coming
                frames_processed_count=0,
                last_publish=datetime.utcnow(),
                queue_depth=0
            )
        
        try:
            for ret, frame in frame_generator(cap):
                if not ret or frame is None:
                    break
                
                if not self.running:
                    break
                
                # Store latest frame for streaming (no processing)
                with self.frame_lock:
                    self.latest_frames[camera_id] = frame.copy()
                
                # Write frame to video recorder if recording
                if self.video_recorder and self.video_recorder.is_recording():
                    self.video_recorder.write_frame(frame)
                
                frame_count += 1
                
                # Update camera metrics periodically for dashboard display (every 30 frames = ~1 second)
                if frame_count % 30 == 0 and self.metrics_server:
                    self.metrics_server.update_camera_metrics(
                        camera_id,
                        fps_ema=30.0,  # Display only, estimate 30 FPS
                        frames_processed_count=frame_count,
                        last_publish=datetime.utcnow(),
                        queue_depth=0
                    )
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.033)  # ~30 FPS
                
        except KeyboardInterrupt:
            print(f"Interrupted display stream for {camera_id}")
        except Exception as e:
            print(f"Error displaying {camera_id}: {e}")
        finally:
            if camera_id in self.camera_captures:
                del self.camera_captures[camera_id]
            cap.release()
            print(f"Closed display stream for {camera_id}")
    
    def start_recording(self, camera_id: str = None) -> Dict:
        """
        Start video recording with GPS logging.
        
        Args:
            camera_id: Camera ID to record (uses first camera if None)
            
        Returns:
            Dict with 'success', 'message', and optional 'video_path', 'gps_log_path'
        """
        if not self.video_recorder:
            return {
                'success': False,
                'message': 'Video recorder not initialized'
            }
        
        if self.video_recorder.is_recording():
            return {
                'success': False,
                'message': 'Already recording. Stop current recording first.'
            }
        
        # Get camera config
        cameras = self.config.get('cameras', [])
        if not cameras:
            return {
                'success': False,
                'message': 'No cameras configured'
            }
        
        # Use specified camera or first camera
        camera = None
        if camera_id:
            camera = next((c for c in cameras if c['id'] == camera_id), None)
            if not camera:
                return {
                    'success': False,
                    'message': f'Camera {camera_id} not found'
                }
        else:
            camera = cameras[0]
        
        camera_id = camera['id']
        width = camera.get('width', 1920)
        height = camera.get('height', 1080)
        fps = camera.get('fps_cap', 30)
        
        # Start recording
        success = self.video_recorder.start_recording(camera_id, width, height, fps)
        
        if success:
            return {
                'success': True,
                'message': f'Recording started for camera {camera_id}',
                'camera_id': camera_id
            }
        else:
            return {
                'success': False,
                'message': 'Failed to start recording'
            }
    
    def stop_recording(self) -> Dict:
        """
        Stop video recording but continue processing remaining videos.
        Recording stops immediately, but video processing and OCR continue in background.
        
        Returns:
            Dict with 'success', 'message', 'video_path', 'gps_log_path', 'processing_ongoing'
        """
        if not self.video_recorder:
            return {
                'success': False,
                'message': 'Video recorder not initialized'
            }
        
        with self.shutdown_lock:
            if self.recording_stopped:
                # Already stopped, check if processing is still ongoing
                is_processing = self.is_processing_ongoing()
                return {
                    'success': True,
                    'message': 'Recording already stopped',
                    'processing_ongoing': is_processing
                }
        
        if not self.video_recorder.is_recording():
            return {
                'success': False,
                'message': 'Not currently recording'
            }
        
        # Stop recording
        video_path, gps_log_path = self.video_recorder.stop_recording()
        
        # Mark recording as stopped (but processing continues)
        with self.shutdown_lock:
            self.recording_stopped = True
        
        # Check if there are remaining videos to process
        is_processing = self.is_processing_ongoing()
        processing_status = self.get_processing_status()
        
        message = 'Recording stopped'
        if is_processing:
            video_queue = processing_status.get('video_queue_size', 0)
            ocr_queue = processing_status.get('ocr_queue_size', 0)
            message += f'. Processing {video_queue} video(s) and {ocr_queue} OCR job(s) in background'
        
        return {
            'success': True,
            'message': message,
            'video_path': video_path,
            'gps_log_path': gps_log_path,
            'processing_ongoing': is_processing,
            'processing_status': processing_status
        }
    
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self.video_recorder.is_recording() if self.video_recorder else False
    
    def is_processing_ongoing(self) -> bool:
        """Check if video processing or OCR is still ongoing."""
        if not self.processing_queue:
            return False
        
        status = self.processing_queue.get_status()
        return (status.get('video_queue_size', 0) > 0 or 
                status.get('ocr_queue_size', 0) > 0 or
                status.get('processing_video', False) or
                status.get('processing_ocr', False))
    
    def is_gracefully_shutting_down(self) -> bool:
        """Check if recording is stopped but processing is still ongoing."""
        with self.shutdown_lock:
            return self.recording_stopped and self.is_processing_ongoing()
    
    def get_processing_status(self) -> Dict:
        """
        Get processing queue status.
        
        Returns:
            Dict with processing queue status information
        """
        if self.processing_queue:
            return self.processing_queue.get_status()
        return {
            'processing_video': False,
            'processing_ocr': False,
            'video_queue_size': 0,
            'ocr_queue_size': 0,
            'stats': {
                'videos_queued': 0,
                'videos_processed': 0,
                'ocr_jobs_queued': 0,
                'ocr_jobs_processed': 0,
                'errors': 0
            }
        }
    
    def upload_to_server(self, video_path: str, crops_dir: str, data: Dict):
        """
        Upload processed data to server.
        
        This is an extensibility point - implement your server upload logic here.
        Called automatically after OCR processing completes.
        
        Args:
            video_path: Path to processed video file
            crops_dir: Directory containing processed crops
            data: Processed data (OCR results, events, etc.)
        """
        # TODO: Implement server upload logic
        # Example:
        # import requests
        # response = requests.post('https://your-server.com/api/upload', json=data)
        # return response.status_code == 200
        
        print(f"[TrailerVisionApp] Server upload hook called (not implemented)")
        print(f"  - Video: {Path(video_path).name}")
        print(f"  - Crops: {crops_dir}")
        print(f"  - Data keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
        pass
    
    def stop(self):
        """Stop the application and cleanup."""
        print("Stopping application...")
        self.running = False
        
        # Stop recording if active
        if self.video_recorder and self.video_recorder.is_recording():
            self.video_recorder.stop_recording()
        
        # Stop processing queue
        if self.processing_queue:
            self.processing_queue.stop()
        
        # Stop GPS sensor
        if self.gps_sensor:
            self.gps_sensor.stop()
        
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

