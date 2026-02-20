"""
Main Application Loop - Trailer Vision Edge

End-to-end processing pipeline:
1. Camera ingestion (RTSP/USB)
2. Detection (YOLO TensorRT)
3. Tracking (ByteTrack)
4. OCR (TrOCR/PaddleOCR/CRNN TensorRT)
5. GPS coordinates from GPS sensor/log files
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
from typing import Dict, Optional, List
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
from app.video_frame_db import VideoFrameDB
from app.app_logger import setup_logging, get_logger
import requests

log = get_logger(__name__)


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
        setup_logging()
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
        self.video_frame_db = None  # Database for storing video frame records
        
        # Per-camera trackers
        self.trackers = {}
        self.camera_3d_projectors = {}  # camera_id -> Camera3DProjector (priority)
        
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
        
        # Detection mode for automatic pipeline (car vs trailer); can be overridden in config
        self.detection_mode = 'trailer'
        self._initialize_components()
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _initialize_components(self):
        """Initialize all application components."""
        globals_cfg = self.config.get('globals', {})
        # Detection mode for recording/auto pipeline: 'car' or 'trailer'
        dm = (globals_cfg.get('detection_mode') or 'trailer').strip().lower()
        self.detection_mode = dm if dm in ('car', 'trailer') else 'trailer'
        log.info(f"[TrailerVisionApp] Detection mode: {self.detection_mode}")
        
        # Initialize detector - prefer fine-tuned YOLO model over TensorRT engine
        fine_tuned_model_path = "runs/detect/truck_detector_finetuned/weights/best.pt"
        engine_path = "models/trailer_detector.engine"
        
        if os.path.exists(fine_tuned_model_path):
            # Use fine-tuned YOLO model
            log.info(f"Loading fine-tuned YOLO model: {fine_tuned_model_path}")
            try:
                self.detector = YOLOv8Detector(
                    model_name=fine_tuned_model_path,
                    conf_threshold=globals_cfg.get('detector_conf', 0.35),
                    target_class=0  # Fine-tuned model is single-class (trailer = class 0)
                )
                log.info(f"✓ Fine-tuned YOLO model loaded successfully")
            except Exception as e:
                log.info(f"Error loading fine-tuned YOLO model: {e}")
                log.info(f"Falling back to TensorRT engine...")
                self.detector = None
        else:
            log.info(f"Fine-tuned model not found: {fine_tuned_model_path}")
            self.detector = None
        
        # Fallback to TensorRT engine if fine-tuned model not available or failed
        if self.detector is None and os.path.exists(engine_path):
            log.info(f"Loading TensorRT engine: {engine_path}")
            try:
                self.detector = TrtEngineYOLO(
                    engine_path,
                    conf_threshold=globals_cfg.get('detector_conf', 0.35)
                )
                log.info(f"✓ TensorRT engine loaded successfully")
            except Exception as e:
                log.info(f"Error loading TensorRT engine: {e}")
                self.detector = None
        
        if self.detector is None:
            log.warning("No detector available. Tried:")
            log.info(f"  - Fine-tuned model: {fine_tuned_model_path}")
            log.info(f"  - TensorRT engine: {engine_path}")
        
        # Initialize OCR - Lazy loading: Don't load OCR initially to save memory
        # OCR will be loaded on-demand when explicitly requested (not automatically)
        # This prevents memory issues when processing large videos
        self.ocr = None
        self.is_olmocr = False
        log.info(f"[TrailerVisionApp] OCR loading deferred - will be loaded on-demand when requested")
        
        # Initialize spot resolver
        spots_path = "config/spots.geojson"
        if os.path.exists(spots_path):
            self.spot_resolver = SpotResolver(spots_path)
        else:
            log.warning("Spots GeoJSON not found: %s", spots_path)
        
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
                log.info(f"[TrailerVisionApp] GPS sensor initialized on {gps_port} at 4800 baud")
            else:
                log.info(f"[TrailerVisionApp] No GPS port found, GPS sensor disabled")
                self.gps_sensor = None
        except Exception as e:
            log.warning("[TrailerVisionApp] Failed to initialize GPS sensor: %s", e)
            self.gps_sensor = None
        
        # Initialize video recorder with 45-second auto-chunking
        # Callback will be set after processing queue is initialized
        self.video_recorder = VideoRecorder(
            output_dir="out/recordings",
            gps_sensor=self.gps_sensor,
            chunk_duration_seconds=30.0,  # 30-second chunks
            on_chunk_saved=None  # Will be set after processing queue initialization
        )
        log.info(f"[TrailerVisionApp] Video recorder initialized with 45-second auto-chunking")
        
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
        log.info(f"[TrailerVisionApp] Image preprocessing enabled:")
        log.info(f"  YOLO: {self.preprocessor.enable_yolo_preprocessing} ({self.preprocessor.yolo_strategy})")
        log.info(f"  OCR: {self.preprocessor.enable_ocr_preprocessing} ({self.preprocessor.ocr_strategy})")
        if is_olmocr:
            rotation_msg = "Disabled (oLmOCR handles rotation internally)"
        elif is_easyocr:
            rotation_msg = "Disabled (EasyOCR handles rotation internally)"
        else:
            rotation_msg = "Enabled"
        log.info(f"  Rotation: {rotation_msg}")
        
        # Load 3D projectors for each camera (GPS coordinates come from GPS sensor/log files)
        self.gps_references = {}  # Store GPS reference for each camera
        
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
                    log.info(f"Loaded 3D projector for {camera_id} {gps_msg}")
                except Exception as e:
                    log.info(f"Error loading 3D projector for {camera_id}: {e}")
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
        
        # Get GPS reference for first camera (or None) for video processing
        # Note: GPS coordinates come from GPS sensor/log files, not from projection
        test_gps_reference = None
        cameras_list = self.config.get('cameras', [])
        if cameras_list:
            first_camera_id = cameras_list[0]['id']
            if first_camera_id in self.gps_references:
                test_gps_reference = self.gps_references[first_camera_id]
        
        log.info(f"[TrailerVisionApp] Initializing video processor:")
        log.info(f"  - Detector: {'Available' if self.detector else 'NOT AVAILABLE'}")
        log.info(f"  - OCR: {'Available' if self.ocr else 'NOT AVAILABLE'}")
        log.info(f"  - Spot Resolver: {'Available' if self.spot_resolver else 'NOT AVAILABLE'}")
        # Check for 3D projector
        test_3d_projector = None
        if cameras_list:
            first_camera_id = cameras_list[0]['id']
            if first_camera_id in self.camera_3d_projectors:
                test_3d_projector = self.camera_3d_projectors[first_camera_id]
        
        log.info(f"  - 3D Projector: {'Available' if test_3d_projector is not None else 'NOT AVAILABLE'}")
        log.info(f"  - GPS Method: GPS Sensor/Log Files (BEV projection removed)")
        if test_gps_reference:
            log.info(f"  - GPS Reference: ({test_gps_reference['lat']:.6f}, {test_gps_reference['lon']:.6f})")
        
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
                bev_projector=None,  # BEV projection removed - using GPS sensor/log files
                gps_reference=test_gps_reference,
                camera_id=test_camera_id
            )
            log.info(f"[TrailerVisionApp] Video processor created successfully")
        except Exception as e:
            log.error("[TrailerVisionApp] Failed to create video processor: %s", e)
            import traceback
            traceback.print_exc()
            self.video_processor = None
        
        # Update metrics server with video processor and frame storage
        if self.metrics_server:
            self.metrics_server.video_processor = self.video_processor
            self.metrics_server.frame_storage = self
            log.info(f"[TrailerVisionApp] Video processor assigned to metrics server: {self.metrics_server.video_processor is not None}")
        else:
            log.error("[TrailerVisionApp] Metrics server not initialized!")
        
        # Processing queue will be initialized when Start Application is called
        # This allows lazy loading of OCR to save memory until needed
        self.processing_queue = None
        log.info(f"[TrailerVisionApp] Processing queue will be initialized when application starts")
        
        # Initialize database for video frame records
        try:
            self.video_frame_db = VideoFrameDB(db_path="data/video_frames.db")
            log.info(f"[TrailerVisionApp] Video frame database initialized")
        except Exception as e:
            log.warning("[TrailerVisionApp] Failed to initialize database: %s", e)
            self.video_frame_db = None
        
        # Periodic upload of processed records to AWS (only processed data; delete from SQLite after success)
        self._upload_thread = None
        self._upload_thread_stop = threading.Event()
        self._upload_status_lock = threading.Lock()
        upload_url_set = bool(os.getenv("EDGE_UPLOAD_URL") or os.getenv("AWS_DASHBOARD_API_URL"))
        self.upload_status = {
            "enabled": upload_url_set and self.video_frame_db is not None,
            "is_uploading": False,
            "last_run_at": None,
            "last_result": None,
            "last_batch_count": 0,
            "last_deleted_count": 0,
            "last_error": None,
            "total_uploaded": 0,
            "last_response_status": None,
            "last_response_body": None,
            "config_message": "" if (upload_url_set and self.video_frame_db) else (
                "EDGE_UPLOAD_URL (or AWS_DASHBOARD_API_URL) not set. Add to .env to enable upload."
                if not upload_url_set else "Video frame database not available."
            ),
        }
        if upload_url_set and self.video_frame_db:
            self._upload_thread = threading.Thread(target=self._upload_processed_loop, daemon=True, name="UploadProcessedRecords")
            self._upload_thread.start()
            log.info("[TrailerVisionApp] Upload processed-records thread started (interval 60s)")
    
    def _upload_processed_loop(self):
        """Background loop: upload processed records to AWS and delete from SQLite after success."""
        interval = max(30, int(os.getenv("EDGE_UPLOAD_INTERVAL_SECONDS", "60")))
        while not self._upload_thread_stop.wait(timeout=interval):
            if not self.running:
                continue
            try:
                self.upload_processed_records_and_delete()
            except Exception as e:
                log.warning("[TrailerVisionApp] Upload processed records error: %s", e)
    
    def upload_processed_records_and_delete(self):
        """
        Fetch processed records (is_processed=1) from local SQLite, upload to AWS,
        and delete them from SQLite on success. Only processed data is uploaded.
        """
        upload_url = os.getenv("EDGE_UPLOAD_URL") or os.getenv("AWS_DASHBOARD_API_URL")
        if not upload_url or not self.video_frame_db:
            return
        upload_url = upload_url.rstrip("/")
        ingest_url = f"{upload_url}/api/ingest/video-frame-records"
        api_key = os.getenv("DASHBOARD_API_KEY")
        device_id = os.getenv("EDGE_DEVICE_ID", "")
        batch_size = min(200, max(1, int(os.getenv("EDGE_UPLOAD_BATCH_SIZE", "100"))))
        
        records = self.video_frame_db.get_all_records(limit=batch_size, offset=0, is_processed=True, camera_id=None)
        if not records:
            with self._upload_status_lock:
                self.upload_status["last_run_at"] = datetime.utcnow().isoformat() + "Z"
                self.upload_status["last_result"] = "skipped"
                self.upload_status["last_batch_count"] = 0
                self.upload_status["last_error"] = None
            return
        with self._upload_status_lock:
            self.upload_status["is_uploading"] = True
            self.upload_status["last_run_at"] = datetime.utcnow().isoformat() + "Z"
            self.upload_status["last_error"] = None
        # Build payload records (image_url filled by server when we send multipart with inline images)
        payload_records = []
        ids_to_delete = []
        for r in records:
            ids_to_delete.append(r["id"])
            rec = {
                "licence_plate_trailer": r.get("licence_plate_trailer"),
                "latitude": r.get("latitude"),
                "longitude": r.get("longitude"),
                "speed": r.get("speed"),
                "barrier": r.get("barrier"),
                "confidence": r.get("confidence"),
                "image_path": r.get("image_path"),
                "camera_id": r.get("camera_id"),
                "video_path": r.get("video_path"),
                "frame_number": r.get("frame_number"),
                "track_id": r.get("track_id"),
                "timestamp": r.get("timestamp") or r.get("created_on"),
                "created_on": r.get("created_on"),
                "is_processed": True,
                "assigned_spot_id": r.get("assigned_spot_id"),
                "assigned_spot_name": r.get("assigned_spot_name"),
                "assigned_distance_ft": r.get("assigned_distance_ft"),
                "processed_comment": r.get("processed_comment"),
                "image_url": None,
            }
            for key in ("timestamp", "created_on"):
                if hasattr(rec.get(key), "isoformat"):
                    rec[key] = rec[key].isoformat()
            payload_records.append(rec)
        # Single combined request: multipart with "records" (JSON) + "image_0", "image_1", ... for each record with a local image
        payload_json = json.dumps({"records": payload_records, "device_id": device_id or None})
        files = [("records", (None, payload_json, "application/json"))]
        for i, r in enumerate(records):
            img_path = r.get("image_path")
            if img_path and Path(img_path).is_file():
                try:
                    name = Path(img_path).name or "crop.jpg"
                    files.append((f"image_{i}", (name, open(img_path, "rb"), "image/jpeg")))
                except Exception as e:
                    log.debug("[TrailerVisionApp] Image open skipped for index %s: %s", i, e)
        headers = {}
        if api_key:
            headers["X-API-Key"] = api_key
        if device_id:
            headers["X-Device-ID"] = device_id
        try:
            resp = requests.post(ingest_url, files=files, headers=headers, timeout=90)
            response_status = resp.status_code
            response_body = (resp.text or "")[:500]
            if resp.ok:
                deleted = self.video_frame_db.delete_records_by_ids(ids_to_delete)
                log.info("[TrailerVisionApp] Uploaded %s processed records to AWS and deleted %s from SQLite", len(payload_records), deleted)
                # Delete uploaded image files and empty crop folders from device
                self._delete_uploaded_images_and_crop_folders(records)
                with self._upload_status_lock:
                    self.upload_status["last_result"] = "success"
                    self.upload_status["last_batch_count"] = len(payload_records)
                    self.upload_status["last_deleted_count"] = deleted
                    self.upload_status["total_uploaded"] = self.upload_status.get("total_uploaded", 0) + deleted
                    self.upload_status["last_error"] = None
                    self.upload_status["last_response_status"] = response_status
                    self.upload_status["last_response_body"] = response_body
            else:
                err_msg = f"{resp.status_code}: {(resp.text or '')[:200]}"
                log.warning("[TrailerVisionApp] Upload processed records failed: %s", err_msg)
                with self._upload_status_lock:
                    self.upload_status["last_result"] = "failed"
                    self.upload_status["last_batch_count"] = len(payload_records)
                    self.upload_status["last_deleted_count"] = 0
                    self.upload_status["last_error"] = err_msg
                    self.upload_status["last_response_status"] = response_status
                    self.upload_status["last_response_body"] = response_body
        except Exception as e:
            log.warning("[TrailerVisionApp] Upload processed records error: %s", e)
            with self._upload_status_lock:
                self.upload_status["last_result"] = "failed"
                self.upload_status["last_batch_count"] = len(payload_records)
                self.upload_status["last_deleted_count"] = 0
                self.upload_status["last_error"] = str(e)
                self.upload_status["last_response_status"] = None
                self.upload_status["last_response_body"] = None
        finally:
            with self._upload_status_lock:
                self.upload_status["is_uploading"] = False
            for part in files[1:]:
                if len(part) >= 2 and hasattr(part[1], "__iter__") and not isinstance(part[1], (str, bytes)):
                    try:
                        f = part[1][1]
                        if hasattr(f, "close"):
                            f.close()
                    except Exception:
                        pass
    
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
            
            # 2. Load OCR for per-video pipeline (OCR runs after each video)
            if self.ocr is None:
                log.info(f"[TrailerVisionApp] Loading OCR for automated processing...")
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
                log.info(f"[TrailerVisionApp] Initializing processing queue...")
                
                # Define callbacks for extensibility
                def on_video_complete(video_path, crops_dir, results):
                    """Called when video processing completes."""
                    log.info(f"[TrailerVisionApp] Video processing complete: {Path(video_path).name}")
                    log.info(f"  - Crops directory: {crops_dir}")
                    # Extensibility point: Add server upload or other processing here
                    if results:
                        self.upload_to_server(video_path, crops_dir, {'type': 'video_processing', 'results': results})
                
                def on_ocr_complete(video_path, crops_dir, ocr_results):
                    """Called when OCR processing completes. Only processed data is uploaded (via periodic upload thread)."""
                    log.info(f"[TrailerVisionApp] OCR processing complete: {Path(video_path).name}")
                    log.info(f"  - Processed {len(ocr_results)} crops with OCR")
                    
                    # Store results in database (as per diagram requirement)
                    if self.video_frame_db and ocr_results:
                        self._store_ocr_results_in_db(video_path, crops_dir, ocr_results)
                    
                    # Upload to AWS is done only for processed records (after data processor assigns spots)
                    # See upload_processed_records_and_delete() and the periodic upload thread.
                    
                    # Delete video, crops, and GPS log permanently after processing is done
                    self._delete_processed_video_assets(video_path, crops_dir)
                
                try:
                    self.processing_queue = ProcessingQueueManager(
                        video_processor=self.video_processor,
                        ocr=self.ocr,
                        preprocessor=self.preprocessor,
                        on_video_complete=on_video_complete,
                        on_ocr_complete=on_ocr_complete,
                        defer_ocr=False,
                        on_video_queue_drained=None
                    )
                    log.info(f"[TrailerVisionApp] Processing queue manager initialized (OCR per video)")
                    assets_loaded['processing_queue'] = True
                    
                    # Set up video recorder callback for auto-processing
                    def on_chunk_saved(video_path, gps_log_path):
                        """Called when a video chunk is saved."""
                        # Extract camera_id from video path
                        video_name = Path(video_path).stem
                        # Format: camera_id_timestamp_chunkXXXX
                        parts = video_name.split('_')
                        camera_id = parts[0] if parts else "unknown"
                        
                        log.info(f"[TrailerVisionApp] Chunk saved: {Path(video_path).name}")
                        log.info(f"  - Queueing for processing...")
                        
                        # Queue video processing (use app detection_mode so car/trailer matches config)
                        if self.processing_queue:
                            self.processing_queue.queue_video_processing(
                                video_path=video_path,
                                camera_id=camera_id,
                                gps_log_path=gps_log_path,
                                detect_every_n=5,
                                detection_mode=getattr(self, 'detection_mode', 'trailer')
                            )
                    
                    # Update video recorder with callback
                    self.video_recorder.on_chunk_saved = on_chunk_saved
                    log.info(f"[TrailerVisionApp] Video recorder callback configured for auto-processing")
                    
                except Exception as e:
                    log.error("[TrailerVisionApp] Failed to initialize processing queue: %s", e)
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
            log.error("[TrailerVisionApp] Failed to initialize assets: %s", e)
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
        
        Priority: 3D Projection > Homography
        Note: BEV projection has been removed - GPS coordinates come from GPS sensor/log files.
        
        Args:
            camera_id: Camera identifier
            x_img: Image X coordinate
            y_img: Image Y coordinate
            return_gps: If True and GPS reference available, return GPS coordinates (lat, lon).
                       If False or no GPS reference, return local meters (x, y).
            frame: Optional frame image (not used, kept for API compatibility)
            
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
                log.warning("3D projection failed for %s: %s", camera_id, e)
                # Fall through - return None if no projection available
        
        # Note: BEV projection removed - GPS coordinates come from GPS sensor/log files
        # If 3D projection is not available, return None
        # GPS coordinates should be obtained from GPS log files during video processing
        return None
    
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
            log.info(f"[TrailerVisionApp] OCR already initialized, skipping...")
            return
        
        globals_cfg = self.config.get('globals', {})
        
        log.info(f"[TrailerVisionApp] Initializing OCR...")
        
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
            log.info(f"✓ Using oLmOCR (Qwen3-VL-4B-Instruct) - recommended for high accuracy, especially vertical text")
        except ImportError:
            log.info("oLmOCR not available (pip3 install transformers torch qwen-vl-utils), trying EasyOCR...")
        except Exception as e:
            log.info(f"oLmOCR initialization failed: {e}, trying EasyOCR...")
        
        # Try EasyOCR if oLmOCR failed (self.ocr is still None)
        if self.ocr is None:
            try:
                from app.ocr.easyocr_recognizer import EasyOCRRecognizer
                self.ocr = EasyOCRRecognizer(languages=['en'], gpu=True)
                log.info(f"✓ Using EasyOCR (fallback - good accuracy for printed text)")
            except ImportError:
                log.info("EasyOCR not available (pip3 install easyocr), trying TrOCR...")
            except Exception as e:
                log.info(f"EasyOCR initialization failed: {e}, trying TrOCR...")
        
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
                        log.info(f"✓ Using TrOCR model: {trocr_path} (fallback - oLmOCR/EasyOCR not available)")
                        trocr_loaded = True
                        break
                    except ImportError:
                        log.info("TrOCR not available (transformers not installed), trying other OCR models...")
                        break
                    except Exception as e:
                        log.info(f"TrOCR initialization failed ({trocr_path}): {e}, trying other OCR models...")
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
                log.info("Using PaddleOCR English-only model (fallback)")
            # Fallback to PaddleOCR multilingual
            elif os.path.exists("models/paddleocr_rec.engine") and os.path.exists("app/ocr/ppocr_keys_v1.txt"):
                ocr_path = "models/paddleocr_rec.engine"
                alphabet_path = "app/ocr/ppocr_keys_v1.txt"
                log.info("Using PaddleOCR multilingual model (fallback)")
            # Fallback to legacy CRNN engine
            elif os.path.exists("models/ocr_crnn.engine") and os.path.exists("app/ocr/alphabet.txt"):
                ocr_path = "models/ocr_crnn.engine"
                alphabet_path = "app/ocr/alphabet.txt"
                input_size = None  # CRNN uses default size
                log.info("Using legacy CRNN model (fallback)")
            
            if ocr_path and alphabet_path:
                if "paddleocr" in ocr_path.lower():
                    self.ocr = PlateRecognizer(ocr_path, alphabet_path, input_size=input_size)
                else:
                    self.ocr = PlateRecognizer(ocr_path, alphabet_path)
                log.info(f"Loaded OCR engine: {ocr_path} with alphabet: {alphabet_path}")
        
        if self.ocr is None:
            log.warning("No OCR engine found. Tried:")
            log.info(f"  - oLmOCR (recommended - install: pip3 install transformers torch qwen-vl-utils)")
            log.info(f"  - EasyOCR (fallback - install: pip3 install easyocr)")
            log.info(f"  - models/trocr.engine (TrOCR)")
            log.info(f"  - models/paddleocr_rec_english.engine (English-only)")
            log.info(f"  - models/paddleocr_rec.engine (multilingual)")
            log.info(f"  - models/ocr_crnn.engine (legacy)")
        else:
            # Update video processor OCR if it exists
            if hasattr(self, 'video_processor') and self.video_processor is not None:
                self.video_processor.ocr = self.ocr
                # Update is_olmocr flag in video processor
                self.video_processor.is_olmocr = self.is_olmocr
                log.info(f"[TrailerVisionApp] Updated video processor with OCR")
    
    def _unload_detector(self):
        """
        Unload YOLO detector to free GPU memory.
        This is called after YOLO video processing is complete.
        """
        if self.detector is None:
            log.info(f"[TrailerVisionApp] Detector already unloaded, skipping...")
            return
        
        log.info(f"[TrailerVisionApp] Unloading YOLO detector to free GPU memory...")
        
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
                log.warning("[TrailerVisionApp] Error unloading YOLOv8 detector: %s", e)
        elif detector_type == "TrtEngineYOLO":
            # TensorRT engine - delete the engine
            try:
                if hasattr(self.detector, 'engine'):
                    del self.detector.engine
                if hasattr(self.detector, 'context'):
                    del self.detector.context
                del self.detector
            except Exception as e:
                log.warning("[TrailerVisionApp] Error unloading TensorRT detector: %s", e)
        
        self.detector = None
        
        # Clean up GPU memory
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                log.warning("[TrailerVisionApp] Error cleaning GPU memory: %s", e)
        
        log.info(f"[TrailerVisionApp] YOLO detector unloaded successfully")
    
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
                #     log.info(f"[TrailerVisionApp] Filtered non-trailer detection: cls={det.get('cls', -1)}, conf={det.get('conf', 0.0):.2f}")
        
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
                    log.info(f"[TrailerVisionApp] Skipping OCR for non-trailer track {track_id}: cls={track_cls}")
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
                                log.info(f"[TrailerVisionApp] OCR error with {prep['method']}: {e}")
                    
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
                                log.info(f"[TrailerVisionApp] OCR fallback failed: {e}")
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
                            log.info(f"[TrailerVisionApp] OCR error: {e}")
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
                    log.info(f"[TrailerVisionApp] Error in 3D bbox projection: {e}")
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
            
            # Log to CSV - COMMENTED OUT: Data is now stored in database instead
            # self.csv_logger.log(event)
            
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
                    log.info(f"Error posting to ingest API: {e}")
            
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
        
        log.info(f"Opening stream for {camera_id}: {rtsp_url}")
        cap = open_stream(rtsp_url, width, height, fps_cap, use_gstreamer)
        
        if cap is None:
            log.warning("Failed to open stream for %s", camera_id)
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
            log.info(f"Interrupted processing for {camera_id}")
        except Exception as e:
            log.info(f"Error processing {camera_id}: {e}")
        finally:
            cap.release()
            log.info(f"Closed stream for {camera_id}")
    
    def run(self):
        """Run the main application loop."""
        log.info("Starting Trailer Vision Edge application...")
        self.running = True
        
        # CAMERA FEED PROCESSING - Only for display, no full processing
        # Enable camera feeds for dashboard display (demo stage)
        # Full processing is disabled, but we need to stream camera feeds for dashboard
        log.info("Camera feed display enabled (demo stage).")
        log.info("Full processing disabled - only displaying camera feeds.")
        
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
            log.info("\nShutting down...")
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
        
        log.info(f"Opening camera stream for display: {camera_id}: {rtsp_url}")
        cap = open_stream(rtsp_url, width, height, fps_cap, use_gstreamer)
        
        if cap is None:
            log.warning("Failed to open stream for %s", camera_id)
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
            log.info(f"Interrupted display stream for {camera_id}")
        except Exception as e:
            log.info(f"Error displaying {camera_id}: {e}")
        finally:
            if camera_id in self.camera_captures:
                del self.camera_captures[camera_id]
            cap.release()
            log.info(f"Closed display stream for {camera_id}")
    
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
        
        # Reset app state so we are no longer "gracefully shutting down" (new recording session)
        with self.shutdown_lock:
            self.recording_stopped = False
        # Notify queue that recording started (reset deferred-OCR state for this session)
        if self.processing_queue and getattr(self.processing_queue, 'notify_recording_started', None):
            self.processing_queue.notify_recording_started()
        
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
        
        # Notify queue so deferred OCR can run when video queue drains
        if self.processing_queue and getattr(self.processing_queue, 'notify_recording_stopped', None):
            self.processing_queue.notify_recording_stopped()
        
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
    
    def _run_deferred_ocr(self, pending_jobs: List[Dict]):
        """
        Run OCR once on all pending crop directories (called when recording stopped and video queue drained).
        Loads OCR on first use, then processes each job and stores/uploads results.
        """
        if not pending_jobs:
            log.info("[TrailerVisionApp] Deferred OCR: no pending jobs")
            return
        log.info(f"[TrailerVisionApp] Deferred OCR: loading OCR and processing {len(pending_jobs)} crop directory(ies)")
        # Load OCR if not already loaded (only now to avoid GPU use during capture/video processing)
        if self.ocr is None:
            self._initialize_ocr()
        if not self.ocr:
            log.error("[TrailerVisionApp] Deferred OCR: failed to load OCR, skipping")
            return
        from app.batch_ocr_processor import BatchOCRProcessor
        for i, job in enumerate(pending_jobs):
            video_path = job.get('video_path', '')
            crops_dir = job.get('crops_dir', '')
            camera_id = job.get('camera_id', '')
            if not crops_dir or not Path(crops_dir).exists():
                log.warning(f"[TrailerVisionApp] Deferred OCR: skip missing crops_dir {crops_dir}")
                continue
            try:
                batch_processor = BatchOCRProcessor(self.ocr, self.preprocessor)
                ocr_results = batch_processor.process_crops_directory(crops_dir)
                combined_results = batch_processor.match_ocr_to_detections(crops_dir, ocr_results)
                if self.video_frame_db and combined_results:
                    self._store_ocr_results_in_db(video_path, crops_dir, combined_results)
                # Upload to AWS is only for processed records (periodic upload thread)
                log.info(f"[TrailerVisionApp] Deferred OCR: completed {i+1}/{len(pending_jobs)} {Path(crops_dir).name}")
            except Exception as e:
                log.exception(f"[TrailerVisionApp] Deferred OCR error for {crops_dir}: {e}")
            finally:
                self._cleanup_gpu_memory()
        log.info("[TrailerVisionApp] Deferred OCR: all jobs completed")
    
    def _store_ocr_results_in_db(self, video_path: str, crops_dir: str, ocr_results: List[Dict]):
        """
        Store OCR results in database (as per diagram requirement).
        
        Args:
            video_path: Path to source video
            crops_dir: Directory containing crops
            ocr_results: List of combined OCR results with GPS data
        """
        if not self.video_frame_db:
            return
        
        stored_count = 0
        for result in ocr_results:
            try:
                # Extract data from combined result
                licence_plate = result.get('ocr_text', '').strip()
                if not licence_plate:
                    continue  # Skip records without OCR text
                
                # Get GPS coordinates: combined results have 'lat'/'lon' from crop metadata (from GPS log during video processing)
                latitude = result.get('lat')
                longitude = result.get('lon')
                if latitude is None or longitude is None:
                    world_coords = result.get('world_coords')
                    if isinstance(world_coords, (list, tuple)) and len(world_coords) >= 2:
                        latitude, longitude = world_coords[0], world_coords[1]
                    elif isinstance(world_coords, dict):
                        latitude = world_coords.get('lat') or world_coords.get('x_world')
                        longitude = world_coords.get('lon') or world_coords.get('y_world')
                
                # Get other fields
                timestamp_str = result.get('timestamp', '')
                confidence = result.get('ocr_conf', 0.0)
                image_path = result.get('crop_path', '')
                camera_id = result.get('camera_id', '')
                frame_number = result.get('frame_count', 0)
                track_id = result.get('track_id')
                
                # Speed and barrier from GPS (if available in result)
                speed = result.get('speed')
                barrier = result.get('barrier') or result.get('heading')
                
                # Convert timestamp
                timestamp = None
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    except:
                        pass
                
                # Store in database
                if latitude is not None and longitude is not None:
                    self.video_frame_db.insert_frame_record(
                        licence_plate_trailer=licence_plate,
                        latitude=float(latitude),
                        longitude=float(longitude),
                        speed=speed,
                        barrier=barrier,
                        confidence=confidence,
                        image_path=image_path,
                        camera_id=camera_id,
                        video_path=video_path,
                        frame_number=frame_number,
                        track_id=track_id,
                        timestamp=timestamp
                    )
                    stored_count += 1
            except Exception as e:
                log.info(f"[TrailerVisionApp] Error storing OCR result in database: {e}")
        
        log.info(f"[TrailerVisionApp] Stored {stored_count} records in database")
    
    def _delete_uploaded_images_and_crop_folders(self, records: List[Dict]):
        """
        After successful upload, delete local image files and empty crop folders for the uploaded records.
        """
        deleted_files = 0
        dirs_to_check = set()
        for r in records:
            img_path = r.get("image_path")
            if not img_path:
                continue
            p = Path(img_path)
            if p.is_file():
                try:
                    p.unlink()
                    deleted_files += 1
                except Exception as e:
                    log.warning("[TrailerVisionApp] Failed to delete image %s: %s", img_path, e)
            if p.parent and p.parent != p:
                dirs_to_check.add(p.parent)
        # Remove empty crop directories (and parents up to out/ or cwd)
        try:
            stop_at = Path("out").resolve() if Path("out").exists() else Path.cwd()
        except Exception:
            stop_at = Path.cwd()
        for dir_entry in sorted(dirs_to_check, key=lambda x: len(x.parts), reverse=True):
            try:
                current = Path(dir_entry).resolve()
                while current.exists() and current.is_dir() and current != stop_at:
                    if any(current.iterdir()):
                        break
                    current.rmdir()
                    log.debug("[TrailerVisionApp] Removed empty crop dir: %s", current)
                    current = current.parent
            except Exception as e:
                log.debug("[TrailerVisionApp] Skip removing dir %s: %s", dir_entry, e)
        if deleted_files or dirs_to_check:
            log.info("[TrailerVisionApp] Deleted %s image file(s) and cleaned empty crop folder(s)", deleted_files)

    def _delete_processed_video_assets(self, video_path: str, crops_dir: str):
        """
        Permanently delete video file and GPS log after processing is complete.
        Crops folder is kept. Called from on_ocr_complete once DB and upload are done.
        """
        video_p = Path(video_path)
        # 1. Delete video file
        if video_p.is_file():
            try:
                video_p.unlink()
                log.info(f"[TrailerVisionApp] Deleted video: {video_p.name}")
            except Exception as e:
                log.warning(f"[TrailerVisionApp] Failed to delete video {video_path}: {e}")
        # 2. Delete GPS log (naming: same stem as video + _gps.json)
        gps_p = video_p.parent / f"{video_p.stem}_gps.json"
        if gps_p.is_file():
            try:
                gps_p.unlink()
                log.info(f"[TrailerVisionApp] Deleted GPS log: {gps_p.name}")
            except Exception as e:
                log.warning(f"[TrailerVisionApp] Failed to delete GPS log {gps_p}: {e}")
    
    def upload_to_server(self, video_path: str, crops_dir: str, data: Dict):
        """
        Optional hook for uploading data to AWS. Only video_processing type is sent here.
        Processed records (after data processor assigns spots) are uploaded by
        upload_processed_records_and_delete() in the periodic upload thread.
        """
        if data.get("type") == "ocr_complete":
            # Processed data is uploaded only after data processor runs; see upload_processed_records_and_delete
            return
        # Optional: handle type == 'video_processing' if needed
        if data.get("type") != "video_processing" or not data.get("results"):
            return
        log.debug("[TrailerVisionApp] Server upload hook: video_processing (optional)")
    
    def stop(self):
        """Stop the application and cleanup."""
        log.info("Stopping application...")
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
        
        # Stop upload processed-records thread
        if getattr(self, "_upload_thread_stop", None) is not None:
            self._upload_thread_stop.set()
        if getattr(self, "_upload_thread", None) is not None and self._upload_thread.is_alive():
            self._upload_thread.join(timeout=5.0)
        
        log.info("Application stopped.")


def main():
    """Main entry point."""
    setup_logging()
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
            log.info("CUDA context initialized in main thread (via autoinit), context: %s", primary_ctx)
        else:
            log.warning("CUDA context initialization returned None")
    except Exception as e:
        log.warning("Failed to initialize CUDA context: %s", e)
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

