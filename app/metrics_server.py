"""
Metrics Server with Web Dashboard

Flask server exposing JSON metrics, Prometheus metrics, health check,
events API, and static web dashboard files.
"""

import os
import json
import cv2
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from flask import Flask, jsonify, send_from_directory, request, Response
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from werkzeug.utils import secure_filename
import threading
import time

from app.app_logger import get_logger, setup_logging

log = get_logger(__name__)

# Prometheus metrics
frames_processed = Counter('trailer_vision_frames_processed_total', 'Total frames processed', ['camera_id'])
detections_total = Counter('trailer_vision_detections_total', 'Total detections', ['camera_id'])
events_logged = Counter('trailer_vision_events_logged_total', 'Total events logged')
fps_gauge = Gauge('trailer_vision_fps', 'Frames per second (EMA)', ['camera_id'])
queue_depth_gauge = Gauge('trailer_vision_queue_depth', 'Event queue depth')
last_publish_time = Gauge('trailer_vision_last_publish_timestamp', 'Last publish timestamp', ['camera_id'])

# Application metrics registry
metrics_registry = {
    'cameras': {}  # camera_id -> {fps_ema, frames_processed, last_publish, queue_depth}
}


class MetricsServer:
    """
    Metrics server with web dashboard.
    """
    
    def __init__(self, port: int = 8080, csv_logger=None, frame_storage=None, video_processor=None):
        """
        Initialize metrics server.
        
        Args:
            port: HTTP server port
            csv_logger: Optional CSV logger for events API
            frame_storage: Optional frame storage object (TrailerVisionApp instance)
            video_processor: Optional video processor instance
        """
        setup_logging()
        self.port = port
        self.csv_logger = csv_logger
        self.frame_storage = frame_storage
        self.video_processor = video_processor
        self.app = Flask(__name__, static_folder='../web', static_url_path='')
        self._setup_error_handlers()
        self._setup_routes()
        self.thread = None
        self.running = False
        
        # Video upload directory
        self.upload_dir = Path(tempfile.gettempdir()) / 'trailer_vision_uploads'
        self.upload_dir.mkdir(exist_ok=True)
        
        # Processing status tracking
        self.processing_status = {
            'status': 'idle',  # 'idle', 'processing_video', 'processing_ocr', 'completed', 'error'
            'message': '',
            'video_processing_complete': False,
            'ocr_processing_complete': False
        }
        self.status_lock = threading.Lock()
        
        # Data processor service (lazy: created on first use or when loading CSV)
        self._data_processor_service = None
    
    def _setup_error_handlers(self):
        """Setup Flask error handlers."""
        from werkzeug.exceptions import NotFound
        
        @self.app.errorhandler(NotFound)
        def handle_not_found(e):
            """Handle 404 Not Found errors gracefully."""
            # Don't log 404s for static files as errors - they're expected
            return jsonify({'error': 'Not found'}), 404
        
        @self.app.errorhandler(Exception)
        def handle_exception(e):
            """Handle all other exceptions."""
            # Skip logging for NotFound exceptions (already handled above)
            from werkzeug.exceptions import NotFound
            if isinstance(e, NotFound):
                return jsonify({'error': 'Not found'}), 404
            
            import traceback
            error_msg = str(e)
            traceback_str = traceback.format_exc()
            log.exception("Unhandled Exception: %s", error_msg)
            return jsonify({'error': 'Internal server error', 'message': error_msg}), 500
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        # API routes must be defined BEFORE catch-all routes
        # to ensure they are matched first
        
        @self.app.route('/metrics.json')
        def metrics_json():
            """Get metrics as JSON."""
            return jsonify(self._get_metrics())
        
        @self.app.route('/metrics')
        def metrics_prometheus():
            """Get metrics in Prometheus format."""
            return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}
        
        @self.app.route('/healthz')
        def healthz():
            """Health check endpoint."""
            return jsonify({'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()})
        
        @self.app.route('/events')
        def events():
            """Get recent events from CSV."""
            camera_id = request.args.get('camera_id')
            limit = int(request.args.get('limit', 100))
            
            events = self._get_recent_events(camera_id, limit)
            return jsonify({'events': events, 'count': len(events)})
        
        @self.app.route('/stream/<camera_id>')
        def stream_camera(camera_id):
            """MJPEG stream for a camera."""
            return Response(
                self._generate_mjpeg(camera_id),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        
        @self.app.route('/api/start-recording', methods=['POST'])
        def start_recording():
            """Start video recording with GPS logging."""
            try:
                data = request.get_json() or {}
                camera_id = data.get('camera_id')
                
                # Get app instance from frame_storage
                app = self.frame_storage if hasattr(self, 'frame_storage') else None
                if not app or not hasattr(app, 'start_recording'):
                    return jsonify({'error': 'Recording not available'}), 500
                
                result = app.start_recording(camera_id=camera_id)
                
                if result.get('success'):
                    return jsonify(result), 200
                else:
                    return jsonify(result), 400
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/stop-recording', methods=['POST'])
        def stop_recording():
            """Stop video recording and return paths."""
            try:
                # Get app instance from frame_storage
                app = self.frame_storage if hasattr(self, 'frame_storage') else None
                if not app or not hasattr(app, 'stop_recording'):
                    return jsonify({'error': 'Recording not available'}), 500
                
                result = app.stop_recording()
                
                if result.get('success'):
                    return jsonify(result), 200
                else:
                    return jsonify(result), 400
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/recording-status', methods=['GET'])
        def recording_status():
            """Get current recording status."""
            try:
                # Get app instance from frame_storage
                app = self.frame_storage if hasattr(self, 'frame_storage') else None
                if not app or not hasattr(app, 'is_recording'):
                    return jsonify({'recording': False, 'error': 'Recording not available'}), 200
                
                is_recording = app.is_recording()
                return jsonify({'recording': is_recording}), 200
            except Exception as e:
                return jsonify({'recording': False, 'error': str(e)}), 200
        
        @self.app.route('/api/upload-video', methods=['POST'])
        def upload_video():
            """Upload video file for processing."""
            if 'video' not in request.files:
                return jsonify({'error': 'No video file provided'}), 400
            
            file = request.files['video']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            unique_id = str(uuid.uuid4())
            file_path = self.upload_dir / f"{unique_id}_{filename}"
            file.save(str(file_path))
            
            return jsonify({
                'video_id': unique_id,
                'filename': filename,
                'status': 'uploaded'
            })
        
        @self.app.route('/api/process-video/<video_id>', methods=['POST'])
        def process_video(video_id):
            """Start processing uploaded video."""
            log.info(f"[MetricsServer] process_video called with video_id: {video_id}")
            try:
                log.info(f"[MetricsServer] Checking video_processor: {self.video_processor is not None}")
                if self.video_processor is None:
                    log.error("Video processor not available")
                    return jsonify({'error': 'Video processor not available'}), 500
                
                # Check if already processing
                try:
                    is_processing = self.video_processor.is_processing()
                    log.info(f"[MetricsServer] Video processor is_processing: {is_processing}")
                    if is_processing:
                        log.error("Video processing already in progress")
                        return jsonify({'error': 'Video processing already in progress'}), 400
                except Exception as e:
                    log.warning("Error checking is_processing: %s", e)
                    import traceback
                    traceback.print_exc()
                    return jsonify({'error': f'Error checking processing status: {str(e)}'}), 500
                
                # Find video file
                try:
                    log.info(f"[MetricsServer] Looking for video file with pattern: {video_id}_*")
                    log.info(f"[MetricsServer] Upload directory: {self.upload_dir}")
                    log.info(f"[MetricsServer] Upload directory exists: {self.upload_dir.exists()}")
                    video_files = list(self.upload_dir.glob(f"{video_id}_*"))
                    log.info(f"[MetricsServer] Found {len(video_files)} matching files")
                    if not video_files:
                        log.error("Video file not found for ID: %s", video_id)
                        all_files = list(self.upload_dir.glob('*'))
                        log.info(f"[MetricsServer] All files in directory ({len(all_files)}): {[str(f.name) for f in all_files]}")
                        return jsonify({'error': 'Video file not found'}), 404
                    
                    video_path = str(video_files[0])
                    log.info(f"[MetricsServer] Found video file: {video_path}")
                    
                    # Look for corresponding GPS log file
                    video_path_obj = Path(video_path)
                    gps_log_path = None
                    
                    # Extract camera ID, date, and time from video filename
                    # Format: {video_id}_{camera_id}_{date}_{time}.mp4
                    # Example: 5b8e7e9a-b5f8-4ff2-ad12-b7f9b837b223_lifecam-hd6000-01_20260116_174341.mp4
                    video_stem = video_path_obj.stem
                    parts = video_stem.split('_')
                    camera_id = None
                    date_str = None
                    time_str = None
                    
                    if len(parts) >= 3:
                        # Try to find camera_id, date, and time in filename
                        # Look for pattern: {video_id}_{camera_id}_{date}_{time}
                        # Find the date part (8 digits: YYYYMMDD) and time part (6 digits: HHMMSS)
                        for i in range(1, len(parts) - 1):
                            # Check if this part looks like a date (YYYYMMDD)
                            if len(parts[i]) == 8 and parts[i].isdigit():
                                # Camera ID is everything before the date
                                camera_id = '_'.join(parts[1:i])  # Skip video_id (parts[0])
                                date_str = parts[i]
                                # Check if next part is time (6 digits: HHMMSS)
                                if i + 1 < len(parts) and len(parts[i+1]) == 6 and parts[i+1].isdigit():
                                    time_str = parts[i+1]
                                break
                    
                    # Try to find GPS log with same base name (in same directory)
                    possible_gps_paths = [
                        video_path_obj.parent / f"{video_path_obj.stem}_gps.json",
                        video_path_obj.parent / f"{video_path_obj.stem.replace('_', '_')}_gps.json"
                    ]
                    
                    # Also check out/recordings directory for GPS log files
                    recordings_dir = Path("out/recordings")
                    if recordings_dir.exists() and camera_id and date_str:
                        # Priority 1: Try exact match with time component
                        if time_str:
                            exact_match = recordings_dir / f"{camera_id}_{date_str}_{time_str}_gps.json"
                            if exact_match.exists():
                                possible_gps_paths.insert(0, exact_match)  # Highest priority
                                log.info(f"[MetricsServer] Found exact GPS log match: {exact_match}")
                            else:
                                # Priority 2: Find closest timestamp match
                                # Look for all GPS log files matching camera_id and date
                                gps_pattern = f"{camera_id}_{date_str}_*_gps.json"
                                matching_gps_files = list(recordings_dir.glob(gps_pattern))
                                if matching_gps_files:
                                    # Extract timestamps and find the closest match
                                    def extract_time_from_filename(filename):
                                        """Extract time string (HHMMSS) from GPS log filename."""
                                        name = filename.stem  # Remove .json extension
                                        parts = name.split('_')
                                        # Find the part that looks like time (6 digits)
                                        for part in parts:
                                            if len(part) == 6 and part.isdigit():
                                                return part
                                        return None
                                    
                                    # Convert video time to seconds for comparison
                                    video_time_sec = int(time_str[:2]) * 3600 + int(time_str[2:4]) * 60 + int(time_str[4:6])
                                    
                                    best_match = None
                                    min_time_diff = float('inf')
                                    
                                    for gps_file in matching_gps_files:
                                        gps_time_str = extract_time_from_filename(gps_file)
                                        if gps_time_str:
                                            gps_time_sec = int(gps_time_str[:2]) * 3600 + int(gps_time_str[2:4]) * 60 + int(gps_time_str[4:6])
                                            time_diff = abs(gps_time_sec - video_time_sec)
                                            if time_diff < min_time_diff:
                                                min_time_diff = time_diff
                                                best_match = gps_file
                                    
                                    if best_match:
                                        # Only use if within 5 minutes (300 seconds)
                                        if min_time_diff <= 300:
                                            possible_gps_paths.insert(0, best_match)
                                            log.info(f"[MetricsServer] Found closest GPS log match (time diff: {min_time_diff}s): {best_match}")
                                        else:
                                            log.info(f"[MetricsServer] GPS log files found but time difference too large ({min_time_diff}s > 300s), using closest anyway")
                                            possible_gps_paths.append(best_match)
                                    else:
                                        # Fallback: use first matching file
                                        possible_gps_paths.append(matching_gps_files[0])
                                        log.info(f"[MetricsServer] Found GPS log in recordings directory (no time match): {matching_gps_files[0]}")
                                else:
                                    # Try without time component (just camera_id and date)
                                    gps_pattern_alt = f"{camera_id}_{date_str}_gps.json"
                                    matching_gps_files_alt = list(recordings_dir.glob(gps_pattern_alt))
                                    if matching_gps_files_alt:
                                        possible_gps_paths.append(matching_gps_files_alt[0])
                                        log.info(f"[MetricsServer] Found GPS log in recordings directory (no time): {matching_gps_files_alt[0]}")
                        else:
                            # No time component in video filename, just match by camera_id and date
                            gps_pattern = f"{camera_id}_{date_str}_*_gps.json"
                            matching_gps_files = list(recordings_dir.glob(gps_pattern))
                            if matching_gps_files:
                                possible_gps_paths.append(matching_gps_files[0])
                                log.info(f"[MetricsServer] Found GPS log in recordings directory: {matching_gps_files[0]}")
                            else:
                                # Try without time component
                                gps_pattern_alt = f"{camera_id}_{date_str}_gps.json"
                                matching_gps_files_alt = list(recordings_dir.glob(gps_pattern_alt))
                                if matching_gps_files_alt:
                                    possible_gps_paths.append(matching_gps_files_alt[0])
                                    log.info(f"[MetricsServer] Found GPS log in recordings directory (no time): {matching_gps_files_alt[0]}")
                    
                    # Try all possible paths in priority order
                    for gps_path in possible_gps_paths:
                        if gps_path.exists():
                            gps_log_path = str(gps_path)
                            log.info(f"[MetricsServer] Found GPS log file: {gps_log_path}")
                            break
                    
                    if not gps_log_path:
                        log.info(f"[MetricsServer] No GPS log file found for video (this is OK if video was not recorded with GPS)")
                        log.info(f"[MetricsServer] Searched in: {video_path_obj.parent} and {recordings_dir if recordings_dir.exists() else 'N/A'}")
                        if camera_id and date_str:
                            log.info(f"[MetricsServer] Looking for GPS log with camera_id={camera_id}, date={date_str}, time={time_str if time_str else 'N/A'}")
                except Exception as e:
                    log.error("Failed to find video file: %s", e)
                    import traceback
                    traceback.print_exc()
                    return jsonify({'error': f'Failed to find video file: {str(e)}'}), 500
                
                # Get detect_every_n and detection_mode parameters safely (must be done in request context)
                detect_every_n = 20  # default
                detection_mode = 'trailer'  # default
                try:
                    if request.is_json:
                        try:
                            json_data = request.get_json(silent=True)
                            if json_data:
                                if 'detect_every_n' in json_data:
                                    detect_every_n = int(json_data['detect_every_n'])
                                if 'detection_mode' in json_data:
                                    detection_mode = str(json_data['detection_mode']).lower()
                                    if detection_mode not in ['trailer', 'car']:
                                        detection_mode = 'trailer'  # fallback to trailer
                        except (ValueError, TypeError) as e:
                            log.info(f"[MetricsServer] Warning: Invalid parameter value, using defaults: {e}")
                except Exception as e:
                    log.info(f"[MetricsServer] Warning: Failed to parse request JSON, using defaults: {e}")
                
                # Capture video_path, GPS log path, detect_every_n, and detection_mode for background thread
                # (Flask request context is not available in background threads)
                captured_video_path = video_path
                captured_gps_log_path = gps_log_path if 'gps_log_path' in locals() else None
                captured_detect_every_n = detect_every_n
                captured_detection_mode = detection_mode
                
                # Reset status before starting new processing
                with self.status_lock:
                    self.processing_status['status'] = 'idle'
                    self.processing_status['message'] = ''
                    self.processing_status['video_processing_complete'] = False
                    self.processing_status['ocr_processing_complete'] = False
                
                # Start processing in background thread
                def process_in_background():
                    try:
                        import traceback
                        log.info(f"[MetricsServer] Starting video processing: {captured_video_path}, detect_every_n={captured_detect_every_n}, detection_mode={captured_detection_mode}")
                        
                        # Update status: video processing started
                        with self.status_lock:
                            self.processing_status['status'] = 'processing_video'
                            self.processing_status['message'] = f'Processing video ({captured_detection_mode} mode)...'
                            self.processing_status['video_processing_complete'] = False
                            self.processing_status['ocr_processing_complete'] = False
                        
                        # Create detector based on detection mode
                        original_detector = self.video_processor.detector
                        try:
                            from app.ai.detector_yolov8 import YOLOv8Detector
                            import os
                            
                            if captured_detection_mode == 'car':
                                # Use COCO pre-trained model for car detection (class 2 = car)
                                log.info(f"[MetricsServer] Creating car detector (COCO class 2)")
                                car_detector = YOLOv8Detector(
                                    model_name="yolov8m.pt",  # COCO pre-trained model
                                    conf_threshold=0.25,
                                    target_class=2  # COCO class 2 = car
                                )
                                self.video_processor.detector = car_detector
                                log.info(f"[MetricsServer] Car detector loaded successfully")
                            else:  # trailer mode
                                # Use fine-tuned model if available, otherwise COCO truck (class 7)
                                fine_tuned_model_path = "runs/detect/truck_detector_finetuned/weights/best.pt"
                                if os.path.exists(fine_tuned_model_path):
                                    log.info(f"[MetricsServer] Creating trailer detector (fine-tuned model)")
                                    trailer_detector = YOLOv8Detector(
                                        model_name=fine_tuned_model_path,
                                        conf_threshold=0.35,
                                        target_class=0  # Fine-tuned model is single-class (trailer = class 0)
                                    )
                                    self.video_processor.detector = trailer_detector
                                    log.info(f"[MetricsServer] Fine-tuned trailer detector loaded successfully")
                                else:
                                    # Fallback to COCO truck detection
                                    log.info(f"[MetricsServer] Creating trailer detector (COCO class 7 = truck)")
                                    trailer_detector = YOLOv8Detector(
                                        model_name="yolov8m.pt",
                                        conf_threshold=0.25,
                                        target_class=7  # COCO class 7 = truck
                                    )
                                    self.video_processor.detector = trailer_detector
                                    log.info(f"[MetricsServer] COCO truck detector loaded successfully")
                        except Exception as e:
                            log.info(f"[MetricsServer] Warning: Failed to create detector for {captured_detection_mode} mode: {e}")
                            log.info(f"[MetricsServer] Using original detector")
                            # Keep original detector if new one fails
                        
                        frame_count = 0
                        last_results_check = 0
                        
                        # Update video processor with GPS log path if available
                        if captured_gps_log_path and hasattr(self.video_processor, 'gps_log_path'):
                            self.video_processor.gps_log_path = captured_gps_log_path
                            # Reload GPS log
                            if hasattr(self.video_processor, 'gps_log'):
                                from app.gps_sensor import load_gps_log
                                self.video_processor.gps_log = load_gps_log(captured_gps_log_path)
                                log.info(f"[MetricsServer] Loaded GPS log for video processing: {captured_gps_log_path}")
                        
                        for frame_num, processed_frame, events in self.video_processor.process_video(
                            captured_video_path, camera_id="test-video", detect_every_n=captured_detect_every_n
                        ):
                            frame_count += 1
                            
                            # Check if we should stop
                            if not self.video_processor.is_processing():
                                log.info(f"[MetricsServer] Processing stopped at frame {frame_num}")
                                break
                            
                            # Log progress every 30 frames
                            if frame_count - last_results_check >= 30:
                                results = self.video_processor.get_results()
                                log.info(f"[MetricsServer] Progress: {results['frames_processed']} frames, {results['detections']} detections, {results['tracks']} tracks, {results['ocr_results']} OCR results")
                                last_results_check = frame_count
                        
                        # Final results check
                        final_results = self.video_processor.get_results()
                        log.info(f"[MetricsServer] Video processing completed: {frame_count} frames processed")
                        log.info(f"[MetricsServer] Final results: {final_results}")
                        
                        # Restore original detector
                        if 'original_detector' in locals() and original_detector is not None:
                            self.video_processor.detector = original_detector
                            log.info(f"[MetricsServer] Restored original detector")
                        
                        # Update status: video processing complete
                        # OCR will not be loaded automatically - user must trigger it manually if needed
                        # YOLO detector is kept loaded (not unloaded automatically)
                        with self.status_lock:
                            self.processing_status['status'] = 'completed'
                            self.processing_status['message'] = 'Video processing completed. Crops saved. OCR processing available on demand.'
                            self.processing_status['video_processing_complete'] = True
                            self.processing_status['ocr_processing_complete'] = False  # OCR not run automatically
                        
                        log.info(f"[MetricsServer] Video processing complete. Crops saved to: {self.video_processor.crops_dir if self.video_processor.crops_dir else 'N/A'}")
                        log.info(f"[MetricsServer] OCR processing is available on demand (not run automatically)")
                        
                    except Exception as e:
                        log.error("Error processing video: %s", e)
                        import traceback
                        traceback.print_exc()
                        # Restore original detector on error
                        if 'original_detector' in locals() and original_detector is not None:
                            try:
                                self.video_processor.detector = original_detector
                                log.info(f"[MetricsServer] Restored original detector after error")
                            except:
                                pass
                        # Make sure processing flag is cleared on error
                        try:
                            self.video_processor.stop_processing()
                        except:
                            pass
                        # Update status: error
                        with self.status_lock:
                            self.processing_status['status'] = 'error'
                            self.processing_status['message'] = f'Processing failed: {str(e)}'
                            self.processing_status['video_processing_complete'] = False
                            self.processing_status['ocr_processing_complete'] = False
                
                thread = threading.Thread(target=process_in_background, daemon=True)
                thread.start()
                
                log.info(f"[MetricsServer] Video processing thread started for video_id: {video_id}")
                return jsonify({'status': 'processing_started', 'video_id': video_id})
                
            except Exception as e:
                log.exception("FATAL ERROR in process_video endpoint: %s", e)
                import traceback
                traceback.print_exc()
                return jsonify({'error': f'Internal server error: {str(e)}'}), 500
        
        @self.app.route('/api/stop-processing', methods=['POST'])
        def stop_processing():
            """Stop current video processing."""
            if self.video_processor is None:
                return jsonify({'error': 'Video processor not available'}), 500
            
            self.video_processor.stop_processing()
            return jsonify({'status': 'stopped'})
        
        @self.app.route('/api/processing-results', methods=['GET'])
        def get_processing_results():
            """Get current processing results."""
            if self.video_processor is None:
                return jsonify({'error': 'Video processor not available'}), 500
            
            results = self.video_processor.get_results()
            results['processing'] = self.video_processor.is_processing()
            
            # Add processing status
            with self.status_lock:
                results['processing_status'] = self.processing_status.copy()
            
            return jsonify(results)
        
        @self.app.route('/api/processing-status', methods=['GET'])
        def get_processing_status():
            """Get current processing status for dashboard."""
            with self.status_lock:
                return jsonify(self.processing_status.copy())
        
        # ========== DEBUG ENDPOINTS ==========
        
        @self.app.route('/api/debug/start-auto-recording', methods=['POST'])
        def debug_start_auto_recording():
            """Debug: Start auto recording with 45-second chunking."""
            try:
                data = request.get_json() or {}
                camera_id = data.get('camera_id')
                
                app = self.frame_storage if hasattr(self, 'frame_storage') else None
                if not app or not hasattr(app, 'start_recording'):
                    return jsonify({'error': 'Recording not available'}), 500
                
                result = app.start_recording(camera_id=camera_id)
                
                if result.get('success'):
                    return jsonify({
                        'success': True,
                        'message': 'Auto recording started with 45-second chunking',
                        'camera_id': result.get('camera_id')
                    }), 200
                else:
                    return jsonify(result), 400
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/debug/stop-auto-recording', methods=['POST'])
        def debug_stop_auto_recording():
            """Debug: Stop auto recording."""
            try:
                app = self.frame_storage if hasattr(self, 'frame_storage') else None
                if not app or not hasattr(app, 'stop_recording'):
                    return jsonify({'error': 'Recording not available'}), 500
                
                result = app.stop_recording()
                
                if result.get('success'):
                    return jsonify({
                        'success': True,
                        'message': 'Auto recording stopped',
                        'result': result
                    }), 200
                else:
                    return jsonify(result), 400
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/debug/stop-video-processing', methods=['POST'])
        def debug_stop_video_processing():
            """Debug: Stop current video processing and clear pending video jobs."""
            try:
                app = self.frame_storage if hasattr(self, 'frame_storage') else None
                if not app or not hasattr(app, 'video_processor'):
                    return jsonify({'error': 'Video processor not available'}), 500
                
                if not app.video_processor:
                    return jsonify({'error': 'Video processor not initialized'}), 500
                
                # Stop current video job (sets stop flag so loop exits)
                if hasattr(app.video_processor, 'stop_processing'):
                    app.video_processor.stop_processing()
                
                # Clear pending video jobs so they are not processed
                if hasattr(app, 'processing_queue') and app.processing_queue:
                    app.processing_queue.clear_video_queue()
                
                return jsonify({
                    'success': True,
                    'message': 'Video processing stopped'
                }), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/debug/stop-ocr-processing', methods=['POST'])
        def debug_stop_ocr_processing():
            """Debug: Stop current OCR processing and clear OCR queue."""
            try:
                app = self.frame_storage if hasattr(self, 'frame_storage') else None
                if not app or not hasattr(app, 'processing_queue'):
                    return jsonify({'error': 'Processing queue not available'}), 500
                
                if not app.processing_queue:
                    return jsonify({'error': 'Processing queue not initialized'}), 500
                
                # Request current OCR job to stop (worker checks this between crops)
                app.processing_queue.request_ocr_stop()
                # Clear pending OCR jobs
                cleared_jobs = app.processing_queue.clear_ocr_queue()
                
                return jsonify({
                    'success': True,
                    'message': f'OCR processing stopped. Queue cleared ({cleared_jobs} job(s) removed).',
                    'cleared_jobs': cleared_jobs
                }), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/debug/start-video-processing', methods=['POST'])
        def debug_start_video_processing():
            """Debug: Manually trigger video processing on a video file or all videos in out/recordings."""
            try:
                data = request.get_json() or {}
                video_path = data.get('video_path')
                process_all = data.get('process_all', False)
                camera_id = data.get('camera_id', 'test-video')
                gps_log_path = data.get('gps_log_path')
                detect_every_n = data.get('detect_every_n', 5)
                detection_mode = data.get('detection_mode', 'trailer')
                
                app = self.frame_storage if hasattr(self, 'frame_storage') else None
                
                # For debug mode, we need to ensure we have a processing queue
                # If app exists but processing queue doesn't, create one on the fly
                processing_queue = None
                
                if app:
                    # Debug flow: same as application pipeline - OCR after each video (defer_ocr=False)
                    if not hasattr(app, 'video_processor') or not app.video_processor:
                        return jsonify({
                            'error': 'Video processor not available',
                            'hint': 'Please start the application first to initialize the video processor.'
                        }), 500
                    if not getattr(app, 'ocr', None):
                        if hasattr(app, '_initialize_ocr'):
                            app._initialize_ocr()
                        if not app.ocr:
                            return jsonify({
                                'error': 'OCR not available',
                                'hint': 'Please start the application first to load OCR, or check OCR model files.'
                            }), 500

                    def _on_debug_ocr_complete(video_path, crops_dir, ocr_results):
                        """Store OCR results in database when debug processing completes OCR for a job."""
                        app_ref = self.frame_storage if hasattr(self, 'frame_storage') else None
                        if not app_ref or not getattr(app_ref, 'video_frame_db', None) or not ocr_results:
                            return
                        try:
                            if hasattr(app_ref, '_store_ocr_results_in_db'):
                                app_ref._store_ocr_results_in_db(video_path, crops_dir, ocr_results)
                                log.info("[MetricsServer] Debug OCR: stored results in database for %s", Path(video_path).name)
                        except Exception as e:
                            log.exception("[MetricsServer] Debug OCR: failed to store results in database: %s", e)

                    # Use or create a processing queue (OCR per video, same as application)
                    if hasattr(app, 'processing_queue') and app.processing_queue and not getattr(app.processing_queue, 'defer_ocr', True):
                        processing_queue = app.processing_queue
                        if processing_queue.on_ocr_complete is None:
                            processing_queue.on_ocr_complete = _on_debug_ocr_complete
                    else:
                        log.info("[MetricsServer] Creating debug processing queue (OCR per video)")
                        try:
                            from app.processing_queue import ProcessingQueueManager
                            processing_queue = ProcessingQueueManager(
                                video_processor=app.video_processor,
                                ocr=app.ocr,
                                preprocessor=getattr(app, 'preprocessor', None),
                                on_video_complete=None,
                                on_ocr_complete=_on_debug_ocr_complete,
                                defer_ocr=False,
                                on_video_queue_drained=None
                            )
                            app.processing_queue = processing_queue
                            log.info("[MetricsServer] Debug processing queue created")
                        except Exception as e:
                            log.info(f"[MetricsServer] Failed to create debug processing queue: {e}")
                            import traceback
                            traceback.print_exc()
                            return jsonify({
                                'error': f'Failed to initialize processing queue: {str(e)}',
                                'hint': 'Make sure the video processor and OCR are initialized (start the application).'
                            }), 500
                else:
                    # No app available - check if we can use video_processor from metrics_server
                    if self.video_processor:
                        # We have a video processor but need OCR and other components
                        return jsonify({
                            'error': 'Application not available. Cannot create processing queue without full application context.',
                            'hint': 'Please start the application first to initialize all components.'
                        }), 500
                    else:
                        return jsonify({
                            'error': 'Application not available. Please start the application first.',
                            'hint': 'Click "Start Application" button to initialize the processing queue.'
                        }), 500
                
                if not processing_queue:
                    return jsonify({
                        'error': 'Processing queue not available and could not be created.',
                        'hint': 'Please start the application first to initialize all components.'
                    }), 500
                
                # Process all videos in out/recordings folder (or out/recording)
                if process_all or not video_path:
                    # Try both singular and plural folder names
                    recordings_dir = None
                    for folder_name in ["out/recordings", "out/recording"]:
                        test_dir = Path(folder_name)
                        if test_dir.exists():
                            recordings_dir = test_dir
                            break
                    
                    if not recordings_dir or not recordings_dir.exists():
                        return jsonify({
                            'error': f'Recordings directory not found. Checked: out/recordings and out/recording',
                            'checked_paths': ['out/recordings', 'out/recording']
                        }), 404
                    
                    # Find all video files (search recursively in subdirectories too)
                    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
                    video_files = []
                    for ext in video_extensions:
                        # Search in root directory
                        video_files.extend(recordings_dir.glob(f'*{ext}'))
                        # Search recursively in subdirectories
                        video_files.extend(recordings_dir.rglob(f'*{ext}'))
                    
                    # Remove duplicates (in case same file matches both patterns)
                    video_files = list(set(video_files))
                    
                    log.info(f"[MetricsServer] Found {len(video_files)} video file(s) in {recordings_dir}")
                    if video_files:
                        log.info(f"[MetricsServer] Video files: {[f.name for f in video_files[:5]]}{'...' if len(video_files) > 5 else ''}")
                    
                    if not video_files:
                        return jsonify({
                            'success': False,
                            'message': f'No video files found in {recordings_dir}',
                            'videos_queued': 0,
                            'searched_directory': str(recordings_dir)
                        }), 200
                    
                    # Queue all videos for processing
                    queued_count = 0
                    errors = []
                    for vid_path in sorted(video_files):
                        try:
                            # Try to find corresponding GPS log
                            # Look in the same directory as the video file first, then in root
                            gps_log = None
                            video_stem = vid_path.stem
                            video_dir = vid_path.parent
                            
                            # Look for GPS log with same name (check video's directory first, then root)
                            gps_log_paths = [
                                video_dir / f"{video_stem}.json",  # Same directory as video
                                video_dir / f"{video_stem}_gps.json",  # Same directory as video
                                recordings_dir / f"{video_stem}.json",  # Root directory
                                recordings_dir / f"{video_stem}_gps.json",  # Root directory
                            ]
                            for gps_path in gps_log_paths:
                                if gps_path.exists():
                                    gps_log = str(gps_path)
                                    log.info(f"[MetricsServer] Found GPS log for {vid_path.name}: {gps_path.name}")
                                    break
                            
                            # Extract camera_id from filename (format: camera_id_timestamp_chunkXXXX)
                            parts = video_stem.split('_')
                            vid_camera_id = parts[0] if parts else camera_id
                            
                            processing_queue.queue_video_processing(
                                video_path=str(vid_path),
                                camera_id=vid_camera_id,
                                gps_log_path=gps_log,
                                detect_every_n=detect_every_n,
                                detection_mode=detection_mode
                            )
                            queued_count += 1
                            log.info(f"[MetricsServer] Queued video {queued_count}/{len(video_files)}: {vid_path.name} (camera: {vid_camera_id})")
                        except Exception as e:
                            error_msg = f"Error queueing {vid_path.name}: {str(e)}"
                            errors.append(error_msg)
                            log.info(f"[MetricsServer] {error_msg}")
                            import traceback
                            traceback.print_exc()
                    
                    if errors:
                        return jsonify({
                            'success': queued_count > 0,
                            'message': f'Queued {queued_count} video(s) for processing, {len(errors)} error(s)',
                            'videos_queued': queued_count,
                            'errors': errors
                        }), 200 if queued_count > 0 else 500
                    
                    log.info(f"[MetricsServer] Successfully queued {queued_count} video(s) for processing")
                    return jsonify({
                        'success': True,
                        'message': f'Queued {queued_count} video(s) for processing',
                        'videos_queued': queued_count,
                        'total_found': len(video_files),
                        'directory': str(recordings_dir)
                    }), 200
                
                # Process single video
                if not video_path:
                    return jsonify({'error': 'video_path is required when process_all is false'}), 400
                
                # Queue video processing
                processing_queue.queue_video_processing(
                    video_path=video_path,
                    camera_id=camera_id,
                    gps_log_path=gps_log_path,
                    detect_every_n=detect_every_n,
                    detection_mode=detection_mode
                )
                
                return jsonify({
                    'success': True,
                    'message': f'Video processing queued: {Path(video_path).name}',
                    'video_path': video_path
                }), 200
                
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                log.exception("ERROR in debug_start_video_processing: %s", e)
                log.info(f"[MetricsServer] Traceback:\n{error_trace}")
                return jsonify({
                    'error': str(e),
                    'traceback': error_trace
                }), 500
        
        @self.app.route('/api/debug/start-ocr-processing', methods=['POST'])
        def debug_start_ocr_processing():
            """Debug: Manually trigger OCR processing on a crops directory or all crops in out/crops."""
            try:
                data = request.get_json() or {}
                crops_dir = data.get('crops_dir')
                process_all = data.get('process_all', False)
                video_path = data.get('video_path', '')
                camera_id = data.get('camera_id', 'test-video')
                
                app = self.frame_storage if hasattr(self, 'frame_storage') else None
                if not app or not hasattr(app, 'processing_queue'):
                    return jsonify({'error': 'Processing queue not available'}), 500
                
                if not app.processing_queue:
                    return jsonify({'error': 'Processing queue not initialized'}), 500
                
                # Process all crops in out/crops folder
                if process_all or not crops_dir:
                    crops_base_dir = Path("out/crops")
                    if not crops_base_dir.exists():
                        return jsonify({'error': f'Crops directory not found: {crops_base_dir}'}), 404
                    
                    # Find all crop directories (each video has its own directory)
                    crop_dirs = []
                    for camera_dir in crops_base_dir.iterdir():
                        if camera_dir.is_dir():
                            for video_dir in camera_dir.iterdir():
                                if video_dir.is_dir():
                                    # Check if directory has image files
                                    image_files = list(video_dir.glob('*.jpg')) + list(video_dir.glob('*.png'))
                                    if image_files:
                                        crop_dirs.append((str(video_dir), camera_dir.name))
                    
                    if not crop_dirs:
                        return jsonify({
                            'success': False,
                            'message': 'No crop directories found in out/crops',
                            'ocr_jobs_queued': 0
                        }), 200
                    
                    # Queue all crop directories for OCR
                    queued_count = 0
                    for crop_path, vid_camera_id in crop_dirs:
                        app.processing_queue._queue_ocr_job(
                            video_path=video_path or crop_path,
                            crops_dir=crop_path,
                            camera_id=vid_camera_id
                        )
                        queued_count += 1
                    
                    return jsonify({
                        'success': True,
                        'message': f'Queued {queued_count} crop directory(ies) for OCR processing',
                        'ocr_jobs_queued': queued_count
                    }), 200
                
                # Process single crops directory
                if not crops_dir:
                    return jsonify({'error': 'crops_dir is required when process_all is false'}), 400
                
                # Manually queue OCR job
                app.processing_queue._queue_ocr_job(
                    video_path=video_path or crops_dir,
                    crops_dir=crops_dir,
                    camera_id=camera_id
                )
                
                return jsonify({
                    'success': True,
                    'message': f'OCR processing queued: {Path(crops_dir).name}',
                    'crops_dir': crops_dir
                }), 200
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/debug/trigger-upload', methods=['POST'])
        def debug_trigger_upload():
            """Debug: Trigger upload of processed records to AWS once (same as background thread)."""
            try:
                app = self.frame_storage if hasattr(self, 'frame_storage') else None
                if not app or not hasattr(app, 'upload_processed_records_and_delete'):
                    return jsonify({'success': False, 'error': 'Upload not available'}), 500
                if not getattr(app, 'upload_status', {}).get('enabled'):
                    return jsonify({
                        'success': False,
                        'error': 'Upload disabled (EDGE_UPLOAD_URL not set or video frame DB not available)'
                    }), 400
                app.upload_processed_records_and_delete()
                if hasattr(app, '_upload_status_lock'):
                    with app._upload_status_lock:
                        status = dict(getattr(app, 'upload_status', {}))
                else:
                    status = dict(getattr(app, 'upload_status', {}))
                status['thread_alive'] = (
                    getattr(app, '_upload_thread', None) is not None and app._upload_thread.is_alive()
                )
                return jsonify({
                    'success': True,
                    'message': 'Upload triggered',
                    'upload_status': status
                }), 200
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/debug/processing-queue-status', methods=['GET'])
        def debug_processing_queue_status():
            """Debug: Get processing queue status."""
            try:
                app = self.frame_storage if hasattr(self, 'frame_storage') else None
                if not app or not hasattr(app, 'processing_queue'):
                    return jsonify({
                        'error': 'Processing queue not available',
                        'available': False
                    }), 200
                
                if not app.processing_queue:
                    return jsonify({
                        'error': 'Processing queue not initialized',
                        'available': False
                    }), 200
                
                status = app.processing_queue.get_status()
                return jsonify({
                    'available': True,
                    'status': status
                }), 200
                
            except Exception as e:
                return jsonify({'error': str(e), 'available': False}), 500
        
        @self.app.route('/api/start-application', methods=['POST'])
        def start_application():
            """Start the automated application workflow (load assets + recording + processing)."""
            try:
                data = request.get_json() or {}
                camera_id = data.get('camera_id')
                detection_mode = (data.get('detection_mode') or 'trailer').strip().lower()
                if detection_mode not in ('car', 'trailer'):
                    detection_mode = 'trailer'
                
                app = self.frame_storage if hasattr(self, 'frame_storage') else None
                if not app:
                    return jsonify({'error': 'Application not available'}), 500
                
                app.detection_mode = detection_mode
                log.info("[MetricsServer] Start application: detection_mode=%s", detection_mode)
                
                # Check if already running
                if app.is_recording():
                    return jsonify({
                        'success': False,
                        'message': 'Application is already running'
                    }), 400
                
                # Step 1: Initialize assets (OCR, processing queue, etc.)
                log.info(f"[MetricsServer] Initializing assets for automated processing...")
                assets_result = app.initialize_assets()
                
                if not assets_result.get('success'):
                    return jsonify({
                        'success': False,
                        'message': f'Failed to initialize assets: {assets_result.get("message", "Unknown error")}',
                        'assets_loaded': assets_result.get('assets_loaded', {})
                    }), 500
                
                log.info(f"[MetricsServer] Assets initialized successfully")
                
                # Step 2: Start auto recording (this will automatically trigger processing)
                result = app.start_recording(camera_id=camera_id)
                
                if result.get('success'):
                    return jsonify({
                        'success': True,
                        'message': 'Application started successfully',
                        'camera_id': result.get('camera_id'),
                        'workflow': 'Assets loaded (no OCR yet) → Recording → Video processing per chunk → When stop: OCR once on all crops → DB/upload',
                        'assets_loaded': assets_result.get('assets_loaded', {})
                    }), 200
                else:
                    return jsonify(result), 400
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/stop-application', methods=['POST'])
        def stop_application():
            """Stop the automated application workflow."""
            try:
                app = self.frame_storage if hasattr(self, 'frame_storage') else None
                if not app:
                    return jsonify({'error': 'Application not available'}), 500
                
                # Stop recording
                result = app.stop_recording()
                
                return jsonify({
                    'success': True,
                    'message': 'Application stopped successfully',
                    'result': result
                }), 200
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/application-status', methods=['GET'])
        def get_application_status():
            """Get comprehensive application status."""
            try:
                app = self.frame_storage if hasattr(self, 'frame_storage') else None
                if not app:
                    return jsonify({
                        'running': False,
                        'error': 'Application not available'
                    }), 200
                
                is_recording = app.is_recording()
                
                # Get camera_id from video recorder if recording
                camera_id = None
                if hasattr(app, 'video_recorder') and app.video_recorder:
                    if app.video_recorder.is_recording() and hasattr(app.video_recorder, 'camera_id'):
                        camera_id = app.video_recorder.camera_id
                
                # Check if gracefully shutting down (recording stopped but processing ongoing)
                is_gracefully_shutting_down = False
                is_processing_ongoing = False
                if hasattr(app, 'is_gracefully_shutting_down'):
                    is_gracefully_shutting_down = app.is_gracefully_shutting_down()
                if hasattr(app, 'is_processing_ongoing'):
                    is_processing_ongoing = app.is_processing_ongoing()
                
                # Get processing queue status if available
                queue_status = None
                if hasattr(app, 'processing_queue') and app.processing_queue:
                    queue_status = app.processing_queue.get_status()
                
                # Get AWS upload status (include thread_alive so dashboard can show "background thread working")
                upload_status = None
                if hasattr(app, 'upload_status') and hasattr(app, '_upload_status_lock'):
                    with app._upload_status_lock:
                        upload_status = dict(app.upload_status)
                    upload_status['thread_alive'] = (
                        getattr(app, '_upload_thread', None) is not None and app._upload_thread.is_alive()
                    )
                
                return jsonify({
                    'running': is_recording,
                    'recording': is_recording,
                    'camera_id': camera_id,
                    'gracefully_shutting_down': is_gracefully_shutting_down,
                    'processing_ongoing': is_processing_ongoing,
                    'queue_status': queue_status,
                    'processing_queue_available': hasattr(app, 'processing_queue') and app.processing_queue is not None,
                    'upload_status': upload_status
                }), 200
                    
            except Exception as e:
                return jsonify({
                    'running': False,
                    'error': str(e)
                }), 200
        
        @self.app.route('/api/upload-status', methods=['GET'])
        def get_upload_status():
            """Get AWS upload status (thread, config, last run, success/failed/skipped)."""
            try:
                app = self.frame_storage if hasattr(self, 'frame_storage') else None
                if not app or not hasattr(app, 'upload_status'):
                    return jsonify({
                        'enabled': False,
                        'thread_alive': False,
                        'config_message': 'Application or upload status not available.',
                        'is_uploading': False,
                        'last_run_at': None,
                        'last_result': None,
                        'last_batch_count': 0,
                        'last_deleted_count': 0,
                        'last_error': None,
                        'total_uploaded': 0
                    }), 200
                with app._upload_status_lock:
                    out = dict(app.upload_status)
                out['thread_alive'] = (
                    getattr(app, '_upload_thread', None) is not None and app._upload_thread.is_alive()
                )
                return jsonify(out), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/processed-frame/<int:frame_number>')
        def get_processed_frame(frame_number):
            """Get a specific processed frame."""
            if self.video_processor is None:
                return jsonify({'error': 'Video processor not available'}), 500
            
            frame = self.video_processor.get_frame(frame_number)
            if frame is None:
                return jsonify({'error': 'Frame not found'}), 404
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                return jsonify({'error': 'Failed to encode frame'}), 500
            
            return Response(buffer.tobytes(), mimetype='image/jpeg')
        
        @self.app.route('/api/processed-video-stream')
        def stream_processed_video():
            """Stream processed video frames as MJPEG."""
            if self.video_processor is None:
                return jsonify({'error': 'Video processor not available'}), 500
            
            return Response(
                self._generate_processed_mjpeg(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        
        @self.app.route('/api/dashboard/data', methods=['GET'])
        def get_dashboard_data():
            """Get dashboard data from video_frames.db (fallback: combined_results.json)."""
            date_str = request.args.get('date')
            log.info("[MetricsServer] GET /api/dashboard/data request date=%s", date_str)
            try:
                data = self._get_dashboard_data_from_db(date_str=date_str)
                if data is None:
                    data = self._get_dashboard_data_from_json(date_str=date_str)
                    log.info("[MetricsServer] GET /api/dashboard/data response source=json kpis.trailersOnYard=%s",
                             data.get('kpis', {}).get('trailersOnYard', {}).get('value'))
                else:
                    log.info("[MetricsServer] GET /api/dashboard/data response source=db kpis.trailersOnYard=%s",
                             data.get('kpis', {}).get('trailersOnYard', {}).get('value'))
                return jsonify(data)
            except Exception as e:
                log.exception("[MetricsServer] Error getting dashboard data: %s", e)
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/dashboard/events', methods=['GET'])
        def get_dashboard_events():
            """Get events from video_frames.db (fallback: combined_results.json)."""
            limit = int(request.args.get('limit', 1000))
            date_str = request.args.get('date')
            log.info("[MetricsServer] GET /api/dashboard/events request limit=%s date=%s", limit, date_str)
            try:
                events = self._get_events_from_db(limit, date_str=date_str)
                if events is None:
                    events = self._get_events_from_json(limit, date_str=date_str)
                    log.info("[MetricsServer] GET /api/dashboard/events response source=json count=%s", len(events))
                else:
                    log.info("[MetricsServer] GET /api/dashboard/events response source=db count=%s", len(events))
                return jsonify({'events': events, 'count': len(events)})
            except Exception as e:
                log.exception("[MetricsServer] Error getting dashboard events: %s", e)
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/cameras', methods=['GET'])
        def get_cameras():
            """Get list of cameras from cameras.yaml with status."""
            force_check = request.args.get('force', 'false').lower() == 'true'
            log.info("[MetricsServer] GET /api/cameras request force=%s", force_check)
            try:
                cameras = self._get_cameras_with_status(force_check=force_check)
                log.info("[MetricsServer] GET /api/cameras response count=%s", len(cameras))
                return jsonify({'cameras': cameras})
            except Exception as e:
                log.exception("[MetricsServer] Error getting cameras: %s", e)
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/inventory', methods=['GET'])
        def get_inventory():
            """Get inventory data from video_frames.db (fallback: combined_results.json)."""
            log.info("[MetricsServer] GET /api/inventory request")
            try:
                data = self._get_inventory_from_db()
                if data is None:
                    data = self._get_inventory_from_json()
                    log.info("[MetricsServer] GET /api/inventory response source=json trailers=%s",
                             len(data.get('trailers', [])))
                else:
                    log.info("[MetricsServer] GET /api/inventory response source=db trailers=%s",
                             len(data.get('trailers', [])))
                return jsonify(data)
            except Exception as e:
                log.exception("[MetricsServer] Error getting inventory data: %s", e)
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/yard-view', methods=['GET'])
        def get_yard_view():
            """Get yard view (spots/lanes) from video_frames.db."""
            log.info("[MetricsServer] GET /api/yard-view request")
            try:
                data = self._get_yard_view_from_db()
                log.info("[MetricsServer] GET /api/yard-view response spots=%s lanes=%s",
                         len(data.get('spots', [])), len(data.get('lanes', [])))
                return jsonify(data)
            except Exception as e:
                log.exception("[MetricsServer] Error getting yard view: %s", e)
                return jsonify({'error': str(e), 'spots': [], 'lanes': []}), 500
        
        @self.app.route('/api/reports', methods=['GET'])
        def get_reports():
            """Get reports (daily/weekly/monthly) from video_frames.db."""
            log.info("[MetricsServer] GET /api/reports request")
            try:
                data = self._get_reports_from_db()
                log.info("[MetricsServer] GET /api/reports response ok")
                return jsonify(data)
            except Exception as e:
                log.exception("[MetricsServer] Error getting reports: %s", e)
                return jsonify({'error': str(e)}), 500
        
        def _get_data_processor_service():
            """Get or create the data processor service (lazy init)."""
            if self._data_processor_service is None:
                from app.video_frame_db import VideoFrameDB
                from app.data_processor_service import DataProcessorService
                db = VideoFrameDB(db_path="data/video_frames.db")
                self._data_processor_service = DataProcessorService(db, logger=log)
                geojson_path = Path("config/spots.geojson")
                if geojson_path.exists():
                    self._data_processor_service.load_parking_spots_from_geojson(str(geojson_path))
            return self._data_processor_service
        
        @self.app.route('/api/data-processor/status', methods=['GET'])
        def data_processor_status():
            """Get data processor status (running, spots loaded)."""
            try:
                svc = _get_data_processor_service()
                stats = svc.get_statistics()
                return jsonify({
                    'running': svc.running,
                    'parking_spots_loaded': len(svc.parking_spots),
                    **stats
                }), 200
            except Exception as e:
                log.exception("Data processor status error: %s", e)
                return jsonify({'error': str(e), 'running': False, 'parking_spots_loaded': 0}), 500
        
        @self.app.route('/api/data-processor/load-csv', methods=['POST'])
        def data_processor_load_csv():
            """Upload a CSV file and load reference parking spots; then run one processing cycle."""
            try:
                if 'file' not in request.files and 'csv' not in request.files:
                    return jsonify({'success': False, 'error': 'No file part; use form field "file" or "csv"'}), 400
                f = request.files.get('file') or request.files.get('csv')
                if not f or f.filename == '':
                    return jsonify({'success': False, 'error': 'No file selected'}), 400
                if not (f.filename or '').lower().endswith('.csv'):
                    return jsonify({'success': False, 'error': 'File must be a CSV'}), 400
                filename = secure_filename(f.filename) or 'spots.csv'
                csv_path = self.upload_dir / f"data_processor_{uuid.uuid4().hex}_{filename}"
                f.save(str(csv_path))
                try:
                    svc = _get_data_processor_service()
                    svc.load_parking_spots_from_csv(str(csv_path))
                    spots_count = len(svc.parking_spots)
                    if spots_count == 0:
                        return jsonify({
                            'success': False,
                            'error': 'No valid parking spots found in CSV. Expected columns: id, name, latitude, longitude (or lat, lon)'
                        }), 200
                    result = svc.run_all()
                    processed = result.get('processed', 0)
                    log.info(
                        "Data processor load-csv: loaded %d spots, processed %d record(s).",
                        spots_count, processed
                    )
                    return jsonify({
                        'success': True,
                        'parking_spots_loaded': spots_count,
                        'processed': processed,
                        'message': f'Loaded {spots_count} spots and processed {processed} record(s).'
                    }), 200
                finally:
                    if csv_path.exists():
                        try:
                            csv_path.unlink()
                        except OSError:
                            pass
            except Exception as e:
                log.exception("Data processor load CSV error: %s", e)
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/data-processor/run', methods=['POST'])
        def data_processor_run():
            """Run one processing cycle with current parking spots."""
            try:
                svc = _get_data_processor_service()
                if not svc.parking_spots:
                    return jsonify({'success': False, 'error': 'No parking spots loaded. Load a CSV first.'}), 200
                result = svc.run_once()
                return jsonify({
                    'success': result.get('success', True),
                    'processed': result.get('processed', 0),
                    'message': f"Processed {result.get('processed', 0)} record(s)."
                }), 200
            except Exception as e:
                log.exception("Data processor run error: %s", e)
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/data-processor/start', methods=['POST'])
        def data_processor_start():
            """Start the data processor service (runs periodically)."""
            try:
                svc = _get_data_processor_service()
                if not svc.parking_spots:
                    return jsonify({'success': False, 'error': 'No parking spots loaded. Load a CSV first.'}), 200
                svc.start()
                return jsonify({'success': True, 'message': 'Data processor started.'}), 200
            except Exception as e:
                log.exception("Data processor start error: %s", e)
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/data-processor/stop', methods=['POST'])
        def data_processor_stop():
            """Stop the data processor service."""
            try:
                if self._data_processor_service is None:
                    return jsonify({'success': True, 'message': 'Data processor was not running.'}), 200
                self._data_processor_service.stop()
                return jsonify({'success': True, 'message': 'Data processor stopped.'}), 200
            except Exception as e:
                log.exception("Data processor stop error: %s", e)
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/video-frame-records', methods=['GET'])
        def get_video_frame_records():
            """Get video frame records from video_frames.db."""
            limit = request.args.get('limit', default=50, type=int)
            offset = request.args.get('offset', default=0, type=int)
            is_processed = request.args.get('is_processed', default=None, type=str)
            camera_id = request.args.get('camera_id', default=None, type=str)
            log.info("[MetricsServer] GET /api/video-frame-records limit=%s offset=%s is_processed=%s camera_id=%s",
                     limit, offset, is_processed, camera_id)
            try:
                from app.video_frame_db import VideoFrameDB
                db = VideoFrameDB(db_path="data/video_frames.db")
                is_processed_bool = None
                if is_processed is not None:
                    is_processed_bool = is_processed.lower() == 'true'
                records = db.get_all_records(
                    limit=limit,
                    offset=offset,
                    is_processed=is_processed_bool,
                    camera_id=camera_id if camera_id else None
                )
                stats = db.get_statistics()
                log.info("[MetricsServer] GET /api/video-frame-records response records=%s total=%s",
                         len(records), stats.get('total', 0))
                return jsonify({
                    'records': records,
                    'stats': stats,
                    'limit': limit,
                    'offset': offset,
                    'total': stats.get('total', 0)
                })
            except Exception as e:
                log.exception("[MetricsServer] Error getting video frame records: %s", e)
                return jsonify({'error': str(e)}), 500
        
        # Static file routes must be LAST (after all API routes)
        # to avoid catching API requests
        
        @self.app.route('/')
        def index():
            """Serve dashboard index.html."""
            web_dir = Path(__file__).parent.parent / 'web'
            return send_from_directory(str(web_dir), 'index.html')
        
        @self.app.route('/<path:path>')
        def static_files(path):
            """Serve static files (JS, CSS). Excludes /api/* paths."""
            # Don't serve API paths as static files
            if path.startswith('api/'):
                return jsonify({'error': 'API endpoint not found'}), 404
            
            web_dir = Path(__file__).parent.parent / 'web'
            file_path = web_dir / path
            
            # Check if file exists before trying to serve it
            if not file_path.exists() or not file_path.is_file():
                # File doesn't exist - return 404 (this will be handled by NotFound handler)
                from werkzeug.exceptions import NotFound
                raise NotFound()
            
            # File exists - serve it
            return send_from_directory(str(web_dir), path)
    
    def _get_video_frame_db(self):
        """Get VideoFrameDB instance (uses data/video_frames.db)."""
        from app.video_frame_db import VideoFrameDB
        db_path = Path(__file__).parent.parent / "data" / "video_frames.db"
        return VideoFrameDB(db_path=str(db_path))
    
    def _filter_records_by_date(self, records: List[Dict], date_str: Optional[str]) -> List[Dict]:
        """Filter records by created_on date (YYYY-MM-DD). If date_str is None, use today."""
        from datetime import date as date_type
        if not date_str:
            date_str = datetime.utcnow().strftime('%Y-%m-%d')
        try:
            filter_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            return records
        out = []
        for r in records:
            created = r.get('created_on') or r.get('timestamp') or ''
            if not created:
                continue
            try:
                if 'T' in str(created):
                    rec_date = datetime.fromisoformat(str(created).replace('Z', '+00:00')).date()
                else:
                    rec_date = datetime.strptime(str(created)[:10], '%Y-%m-%d').date()
                if rec_date == filter_date:
                    out.append(r)
            except (ValueError, TypeError):
                continue
        return out
    
    def _get_dashboard_data_from_db(self, date_str: str = None) -> Optional[Dict]:
        """Build dashboard KPIs/charts from video_frames.db. Returns None if DB has no data."""
        try:
            db = self._get_video_frame_db()
            stats = db.get_statistics()
            if stats.get('total', 0) == 0:
                return None
            records = db.get_all_records(limit=2000, offset=0, is_processed=None, camera_id=None)
            records = self._filter_records_by_date(records, date_str)
            if not records and date_str:
                return None
            if not records:
                records = db.get_all_records(limit=500, offset=0, is_processed=None, camera_id=None)
            from collections import defaultdict
            unique_tracks = set(r.get('track_id') for r in records if r.get('track_id') is not None)
            unique_spots = set()
            for r in records:
                s = (r.get('assigned_spot_name') or r.get('processed_comment') or '').strip()
                if s and s.lower() != 'unknown':
                    unique_spots.add(s)
            total_detections = len(records)
            ocr_ok = sum(1 for r in records if (r.get('licence_plate_trailer') or '').strip())
            ocr_accuracy = (ocr_ok / total_detections * 100) if total_detections else 0
            low_conf = sum(1 for r in records if (r.get('confidence') or 0) < 0.5)
            cameras = self._get_cameras_with_status()
            online = sum(1 for c in cameras if c.get('status') == 'online')
            degraded = sum(1 for c in cameras if c.get('status') == 'degraded')
            hourly = defaultdict(lambda: {'total': 0, 'total_det': 0.0, 'total_ocr': 0.0, 'ocr_n': 0})
            for r in records:
                created = r.get('created_on') or r.get('timestamp') or ''
                if created:
                    try:
                        dt = datetime.fromisoformat(str(created).replace('Z', '+00:00'))
                        key = dt.strftime('%H:00')
                        hourly[key]['total'] += 1
                        conf = float(r.get('confidence') or 0)
                        hourly[key]['total_det'] += conf
                        if (r.get('licence_plate_trailer') or '').strip():
                            hourly[key]['total_ocr'] += conf
                            hourly[key]['ocr_n'] += 1
                    except (ValueError, TypeError):
                        pass
            accuracy_chart = []
            for h in range(24):
                key = f'{h:02d}:00'
                d = hourly.get(key, {'total': 0, 'total_det': 0.0, 'total_ocr': 0.0, 'ocr_n': 0})
                det_pct = (d['total_det'] / d['total'] * 100) if d['total'] else 0
                ocr_pct = (d['total_ocr'] / d['ocr_n'] * 100) if d['ocr_n'] else 0
                accuracy_chart.append({'time': key, 'detection': round(det_pct, 1), 'ocr': round(ocr_pct, 1)})
            spot_counts = defaultdict(int)
            for r in records:
                s = (r.get('assigned_spot_name') or '').strip() or (r.get('processed_comment') or '').strip()
                if s and s.lower() != 'unknown':
                    spot_counts[s] += 1
            yard_util = [{'lane': k, 'utilization': v} for k, v in sorted(spot_counts.items())]
            return {
                'kpis': {
                    'trailersOnYard': {'value': len(unique_tracks), 'change': f'+{len(unique_tracks)}', 'icon': '🚛'},
                    'newDetections24h': {'value': total_detections, 'ocrAccuracy': f'{ocr_accuracy:.1f}%', 'icon': '📈'},
                    'anomalies': {'value': low_conf, 'description': f'{low_conf} low confidence', 'icon': '⚠️'},
                    'camerasOnline': {'value': online, 'degraded': degraded, 'icon': '📷'},
                },
                'queueStatus': {'ingestQ': 0, 'ocrQ': 0, 'pubQ': 0},
                'accuracyChart': accuracy_chart,
                'yardUtilization': yard_util,
                'cameraHealth': cameras,
            }
        except Exception as e:
            log.debug("[MetricsServer] _get_dashboard_data_from_db failed: %s", e)
            return None
    
    def _get_events_from_db(self, limit: int, date_str: str = None) -> Optional[List[Dict]]:
        """Get events from video_frames.db. Returns None on error or empty DB."""
        try:
            db = self._get_video_frame_db()
            stats = db.get_statistics()
            if stats.get('total', 0) == 0:
                return []
            records = db.get_all_records(limit=min(limit, 2000), offset=0, is_processed=None, camera_id=None)
            records = self._filter_records_by_date(records, date_str)
            events = []
            for r in records:
                ts = r.get('timestamp') or r.get('created_on') or ''
                events.append({
                    'ts_iso': ts,
                    'camera_id': r.get('camera_id') or 'N/A',
                    'track_id': r.get('track_id') if r.get('track_id') is not None else 'N/A',
                    'text': (r.get('licence_plate_trailer') or '').strip() or '',
                    'conf': float(r.get('confidence') or 0),
                    'spot': (r.get('assigned_spot_name') or '').strip() or 'unknown',
                    'ocr_conf': float(r.get('confidence') or 0),
                    'lat': r.get('latitude'),
                    'lon': r.get('longitude'),
                })
            events.sort(key=lambda x: x.get('ts_iso', ''), reverse=True)
            return events[:limit]
        except Exception as e:
            log.debug("[MetricsServer] _get_events_from_db failed: %s", e)
            return None
    
    def _get_inventory_from_db(self) -> Optional[Dict]:
        """Get inventory (trailers + stats) from video_frames.db. Returns None on error."""
        from collections import defaultdict
        try:
            db = self._get_video_frame_db()
            stats = db.get_statistics()
            if stats.get('total', 0) == 0:
                return None
            records = db.get_all_records(limit=1000, offset=0, is_processed=None, camera_id=None)
            by_track = defaultdict(list)
            for r in records:
                tid = r.get('track_id')
                if tid is not None:
                    by_track[tid].append(r)
            trailers = []
            for track_id, rlist in by_track.items():
                r = max(rlist, key=lambda x: (x.get('created_on') or x.get('timestamp') or ''))
                spot = (r.get('assigned_spot_name') or '').strip() or 'N/A'
                status = 'Parked' if r.get('is_processed') or (spot and spot != 'N/A') else 'In Transit'
                trailers.append({
                    'id': f"T{track_id}" if track_id is not None else f"R{r.get('id', 0)}",
                    'plate': (r.get('licence_plate_trailer') or '').strip() or 'N/A',
                    'spot': spot if spot != 'N/A' else 'N/A',
                    'status': status,
                    'detectedAt': r.get('created_on') or r.get('timestamp') or '',
                    'ocrConfidence': float(r.get('confidence') or 0),
                    'lat': r.get('latitude'),
                    'lon': r.get('longitude'),
                })
            total = len(trailers)
            parked = sum(1 for t in trailers if t['status'] == 'Parked')
            anomalies = sum(1 for rec in records if (rec.get('confidence') or 0) < 0.5)
            return {
                'trailers': trailers,
                'stats': {'total': total, 'parked': parked, 'inTransit': total - parked, 'anomalies': anomalies},
            }
        except Exception as e:
            log.debug("[MetricsServer] _get_inventory_from_db failed: %s", e)
            return None
    
    def _get_yard_view_from_db(self) -> Dict:
        """Get yard view spots/lanes from video_frames.db (assigned spots + occupancy)."""
        try:
            db = self._get_video_frame_db()
            records = db.get_all_records(limit=500, offset=0, is_processed=None, camera_id=None)
            spot_to_latest = {}
            for r in records:
                sid = (r.get('assigned_spot_id') or '').strip() or (r.get('assigned_spot_name') or '').strip()
                if not sid:
                    continue
                created = r.get('created_on') or r.get('timestamp') or ''
                if sid not in spot_to_latest or (spot_to_latest[sid].get('created_on') or '') < created:
                    spot_to_latest[sid] = r
            spots = []
            for sid, r in spot_to_latest.items():
                name = (r.get('assigned_spot_name') or sid).strip()
                lane = name.split('-')[0] if name else 'A'
                spots.append({
                    'id': sid,
                    'lane': lane,
                    'row': 1,
                    'occupied': True,
                    'trailerId': f"T{r.get('track_id')}" if r.get('track_id') is not None else None,
                    'plate': (r.get('licence_plate_trailer') or '').strip() or None,
                })
            lanes = sorted(set(s['lane'] for s in spots)) if spots else ['A', 'B', 'C', 'D', 'Dock']
            return {'spots': spots, 'lanes': lanes}
        except Exception as e:
            log.debug("[MetricsServer] _get_yard_view_from_db failed: %s", e)
            return {'spots': [], 'lanes': ['A', 'B', 'C', 'D', 'Dock']}
    
    def _get_reports_from_db(self) -> Dict:
        """Get report stats (daily/weekly/monthly) from video_frames.db."""
        try:
            db = self._get_video_frame_db()
            stats = db.get_statistics()
            all_records = db.get_all_records(limit=5000, offset=0, is_processed=None, camera_id=None)
            total = len(all_records)
            ocr_ok = sum(1 for r in all_records if (r.get('licence_plate_trailer') or '').strip())
            ocr_pct = (ocr_ok / total * 100) if total else 0
            anomalies = sum(1 for r in all_records if (r.get('confidence') or 0) < 0.5)
            today = datetime.utcnow().strftime('%Y-%m-%d')
            week_start = (datetime.utcnow() - __import__('datetime').timedelta(days=7)).strftime('%Y-%m-%d')
            month_start = (datetime.utcnow() - __import__('datetime').timedelta(days=30)).strftime('%Y-%m-%d')
            daily_count = len(self._filter_records_by_date(all_records, today))
            weekly_records = [r for r in all_records if (r.get('created_on') or '')[:10] >= week_start]
            monthly_records = [r for r in all_records if (r.get('created_on') or '')[:10] >= month_start]
            return {
                'daily': {
                    'date': today,
                    'totalDetections': daily_count,
                    'ocrAccuracy': round(ocr_pct, 1),
                    'anomalies': anomalies,
                    'avgProcessingTime': 86,
                },
                'weekly': {
                    'week': f'Last 7 days',
                    'totalDetections': len(weekly_records),
                    'ocrAccuracy': round(ocr_pct, 1),
                    'anomalies': anomalies,
                    'avgProcessingTime': 92,
                },
                'monthly': {
                    'month': datetime.utcnow().strftime('%B %Y'),
                    'totalDetections': len(monthly_records),
                    'ocrAccuracy': round(ocr_pct, 1),
                    'anomalies': anomalies,
                    'avgProcessingTime': 88,
                },
            }
        except Exception as e:
            log.debug("[MetricsServer] _get_reports_from_db failed: %s", e)
            return {
                'daily': {'date': '', 'totalDetections': 0, 'ocrAccuracy': 0, 'anomalies': 0, 'avgProcessingTime': 0},
                'weekly': {'week': '', 'totalDetections': 0, 'ocrAccuracy': 0, 'anomalies': 0, 'avgProcessingTime': 0},
                'monthly': {'month': '', 'totalDetections': 0, 'ocrAccuracy': 0, 'anomalies': 0, 'avgProcessingTime': 0},
            }
    
    def _get_dashboard_data_from_json(self, date_str: str = None) -> Dict:
        """Get dashboard data aggregated from combined_results.json files.
        
        Args:
            date_str: Optional date string in YYYY-MM-DD format to filter by. 
                     If None, defaults to today's date.
        """
        base_dir = Path(__file__).parent.parent / 'out' / 'crops' / 'test-video'
        
        # Parse date filter
        filter_date = None
        if date_str:
            try:
                filter_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                log.info(f"[MetricsServer] Invalid date format: {date_str}, using today")
                filter_date = datetime.now().date()
        else:
            filter_date = datetime.now().date()
        
        # Find all combined_results.json files
        combined_files = []
        if base_dir.exists():
            for folder in base_dir.iterdir():
                if folder.is_dir():
                    json_file = folder / 'combined_results.json'
                    if json_file.exists():
                        combined_files.append(json_file)
        
        if not combined_files:
            # Return empty/default data structure
            return {
                'kpis': {
                    'trailersOnYard': {'value': 0, 'change': '+0', 'icon': '🚛'},
                    'newDetections24h': {'value': 0, 'ocrAccuracy': '0%', 'icon': '📈'},
                    'anomalies': {'value': 0, 'description': 'No anomalies detected', 'icon': '⚠️'},
                    'camerasOnline': {'value': 0, 'degraded': 0, 'icon': '📷'}
                },
                'queueStatus': {'ingestQ': 0, 'ocrQ': 0, 'pubQ': 0},
                'accuracyChart': [],
                'yardUtilization': [],
                'cameraHealth': []
            }
        
        # Read and aggregate all combined results
        all_results = []
        for json_file in combined_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    if isinstance(results, list):
                        all_results.extend(results)
            except Exception as e:
                log.info(f"[MetricsServer] Error reading {json_file}: {e}")
                continue
        
        if not all_results:
            return {
                'kpis': {
                    'trailersOnYard': {'value': 0, 'change': '+0', 'icon': '🚛'},
                    'newDetections24h': {'value': 0, 'ocrAccuracy': '0%', 'icon': '📈'},
                    'anomalies': {'value': 0, 'description': 'No anomalies detected', 'icon': '⚠️'},
                    'camerasOnline': {'value': 0, 'degraded': 0, 'icon': '📷'}
                },
                'queueStatus': {'ingestQ': 0, 'ocrQ': 0, 'pubQ': 0},
                'accuracyChart': [],
                'yardUtilization': [],
                'cameraHealth': []
            }
        
        # Filter results by date if specified
        filtered_results = []
        for result in all_results:
            timestamp = result.get('timestamp', '')
            if timestamp:
                try:
                    result_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).date()
                    if result_date == filter_date:
                        filtered_results.append(result)
                except:
                    # If we can't parse timestamp, skip this result when filtering by date
                    if not date_str:
                        filtered_results.append(result)
            else:
                # If no timestamp, only include when not filtering by date
                if not date_str:
                    filtered_results.append(result)
        
        # Use filtered results for calculations
        all_results = filtered_results

        # Calculate KPIs
        unique_tracks = set()
        unique_spots = set()
        ocr_successful = 0
        total_detections = len(all_results)
        low_conf_detections = 0

        for result in all_results:
            track_id = result.get('track_id')
            if track_id:
                unique_tracks.add(track_id)

            spot = result.get('spot', '')
            if spot and spot != 'unknown':
                unique_spots.add(spot)

            ocr_text = result.get('ocr_text', '')
            if ocr_text and ocr_text.strip():
                ocr_successful += 1

            det_conf = result.get('det_conf', 0.0)
            if isinstance(det_conf, (int, float)) and det_conf < 0.5:
                low_conf_detections += 1
        
        # Calculate OCR accuracy
        ocr_accuracy = (ocr_successful / total_detections * 100) if total_detections > 0 else 0
        
        # Get camera health
        cameras = self._get_cameras_with_status()
        online_cameras = sum(1 for c in cameras if c.get('status') == 'online')
        degraded_cameras = sum(1 for c in cameras if c.get('status') == 'degraded')
        
        # Build accuracy chart data (filtered by date)
        accuracy_data = []
        # Group by hour for chart, filter by selected date
        from collections import defaultdict
        hourly_data = defaultdict(lambda: {
            'total': 0, 
            'total_det_conf': 0.0,
            'total_ocr_conf': 0.0,
            'ocr_attempts': 0  # Count of detections where OCR was actually attempted
        })
        
        for result in all_results:
            timestamp = result.get('timestamp', '')
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    result_date = dt.date()
                    
                    # Only include results from the selected date (already filtered above, but double-check)
                    if result_date != filter_date:
                        continue
                    
                    hour_key = dt.strftime('%H:00')
                    hourly_data[hour_key]['total'] += 1
                    
                    # For detection accuracy, use the actual confidence value
                    # Detection accuracy = average confidence of detections
                    det_conf = result.get('det_conf', 0.0)
                    if isinstance(det_conf, (int, float)):
                        det_conf_float = float(det_conf)
                        hourly_data[hour_key]['total_det_conf'] += det_conf_float
                    
                    # For OCR accuracy, use the OCR confidence value
                    # Only count detections where OCR was actually attempted
                    ocr_conf = result.get('ocr_conf', 0.0)
                    if isinstance(ocr_conf, (int, float)):
                        ocr_conf_float = float(ocr_conf)
                        # Only count if OCR was attempted (confidence > 0 or text exists)
                        ocr_text = result.get('ocr_text', '') or result.get('text', '')
                        if ocr_conf_float > 0 or (ocr_text and ocr_text.strip()):
                            hourly_data[hour_key]['total_ocr_conf'] += ocr_conf_float
                            hourly_data[hour_key]['ocr_attempts'] += 1
                except Exception as e:
                    # Skip invalid timestamps
                    continue
        
        # Generate data for all 24 hours of today (even if no data)
        # This ensures the chart shows the full day
        for hour in range(24):
            hour_key = f"{hour:02d}:00"
            data = hourly_data.get(hour_key, {
                'total': 0, 
                'total_det_conf': 0.0,
                'total_ocr_conf': 0.0,
                'ocr_attempts': 0
            })
            
            # Calculate detection accuracy (average detection confidence * 100)
            if data['total'] > 0:
                avg_det_conf = data['total_det_conf'] / data['total']
                detection_accuracy = avg_det_conf * 100
            else:
                detection_accuracy = 0
            
            # Calculate OCR accuracy (average OCR confidence * 100)
            # Only average over detections where OCR was actually attempted
            # This gives a more accurate representation of OCR quality
            if data['ocr_attempts'] > 0:
                avg_ocr_conf = data['total_ocr_conf'] / data['ocr_attempts']
                ocr_accuracy = avg_ocr_conf * 100
            elif data['total'] > 0:
                # If there are detections but no OCR attempts, accuracy is 0%
                ocr_accuracy = 0
            else:
                # No data for this hour
                ocr_accuracy = 0
            
            accuracy_data.append({
                'time': hour_key,
                'detection': round(detection_accuracy, 1),
                'ocr': round(ocr_accuracy, 1)
            })
        
        # Build yard utilization data
        spot_counts = defaultdict(int)
        for result in all_results:
            spot = result.get('spot', '')
            if spot and spot != 'unknown':
                spot_counts[spot] += 1
        
        utilization_data = []
        for spot, count in sorted(spot_counts.items()):
            utilization_data.append({
                'lane': spot,
                'utilization': count
            })
        
        return {
            'kpis': {
                'trailersOnYard': {
                    'value': len(unique_tracks),
                    'change': f'+{len(unique_tracks)}',
                    'icon': '🚛'
                },
                'newDetections24h': {
                    'value': total_detections,
                    'ocrAccuracy': f'{ocr_accuracy:.1f}%',
                    'icon': '📈'
                },
                'anomalies': {
                    'value': low_conf_detections,
                    'description': f'{low_conf_detections} low confidence detections',
                    'icon': '⚠️'
                },
                'camerasOnline': {
                    'value': online_cameras,
                    'degraded': degraded_cameras,
                    'icon': '📷'
                }
            },
            'queueStatus': {
                'ingestQ': 0,
                'ocrQ': 0,
                'pubQ': 0
            },
            'accuracyChart': accuracy_data,  # Direct array, not wrapped in 'data'
            'yardUtilization': utilization_data,  # Direct array, not wrapped in 'data'
            'cameraHealth': cameras
        }
    
    def _get_events_from_json(self, limit: int = 1000, date_str: str = None) -> List[Dict]:
        """Get events from combined_results.json files.
        
        Args:
            limit: Maximum number of events to return
            date_str: Optional date string in YYYY-MM-DD format to filter by.
                     If None, returns all events (up to limit).
        """
        base_dir = Path(__file__).parent.parent / 'out' / 'crops' / 'test-video'
        
        # Parse date filter
        filter_date = None
        if date_str:
            try:
                filter_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                log.info(f"[MetricsServer] Invalid date format: {date_str}, returning all events")
                filter_date = None
        
        # Find all combined_results.json files
        combined_files = []
        if base_dir.exists():
            for folder in base_dir.iterdir():
                if folder.is_dir():
                    json_file = folder / 'combined_results.json'
                    if json_file.exists():
                        combined_files.append(json_file)
        
        all_events = []
        for json_file in combined_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    if isinstance(results, list):
                        for result in results:
                            # Filter by date if specified
                            if filter_date:
                                timestamp = result.get('timestamp', '')
                                if timestamp:
                                    try:
                                        result_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).date()
                                        if result_date != filter_date:
                                            continue
                                    except:
                                        continue
                                else:
                                    continue  # Skip events without timestamp when filtering by date
                            
                            # Transform to event format expected by frontend
                            # Frontend expects 'text' for OCR text, not 'ocr_text'
                            ocr_text = result.get('ocr_text', '') or result.get('text', '')
                            
                            # Extract GPS coordinates - check direct fields first, then world_coords
                            lat = result.get('lat')
                            lon = result.get('lon')
                            world_coords = result.get('world_coords')
                            
                            # Extract from world_coords if lat/lon not directly available
                            # world_coords format: [lat, lon] when GPS coordinates are available
                            if (lat is None or lon is None) and world_coords:
                                if isinstance(world_coords, (list, tuple)) and len(world_coords) >= 2:
                                    try:
                                        coord1, coord2 = float(world_coords[0]), float(world_coords[1])
                                        # Check if coordinates look like GPS (lat/lon) vs meters
                                        # GPS: lat is -90 to 90, lon is -180 to 180
                                        # Meters: typically much larger values (hundreds or thousands)
                                        if -90.0 <= coord1 <= 90.0 and -180.0 <= coord2 <= 180.0:
                                            lat = coord1
                                            lon = coord2
                                        # Also check if world_coords might be stored as [lon, lat] (less common)
                                        elif -90.0 <= coord2 <= 90.0 and -180.0 <= coord1 <= 180.0:
                                            lat = coord2
                                            lon = coord1
                                    except (ValueError, TypeError) as e:
                                        # If conversion fails, leave lat/lon as None
                                        pass
                            
                            event = {
                                'ts_iso': result.get('timestamp', ''),
                                'camera_id': result.get('camera_id', 'N/A'),
                                'track_id': result.get('track_id', 'N/A'),
                                'text': ocr_text,  # Frontend expects 'text' for OCR result
                                'conf': result.get('det_conf', 0.0),
                                'spot': result.get('spot', 'unknown'),
                                'ocr_conf': result.get('ocr_conf', 0.0),
                                'frame_count': result.get('frame_count', 0),
                                'bbox': result.get('bbox', []),
                                'world_coords': world_coords,
                                'lat': lat,
                                'lon': lon,
                                # Keep original fields for reference
                                'ocr_text': ocr_text,
                                'det_conf': result.get('det_conf', 0.0)
                            }
                            all_events.append(event)
            except Exception as e:
                log.info(f"[MetricsServer] Error reading {json_file}: {e}")
                continue
        
        # Sort by timestamp (most recent first)
        all_events.sort(key=lambda x: x.get('ts_iso', ''), reverse=True)
        
        # Return limited results
        return all_events[:limit]
    
    def _get_inventory_from_json(self) -> Dict:
        """Get inventory data from combined_results.json files."""
        base_dir = Path(__file__).parent.parent / 'out' / 'crops' / 'test-video'
        
        # Find all combined_results.json files
        combined_files = []
        if base_dir.exists():
            for folder in base_dir.iterdir():
                if folder.is_dir():
                    json_file = folder / 'combined_results.json'
                    if json_file.exists():
                        combined_files.append(json_file)
        
        if not combined_files:
            return {
                'trailers': [],
                'stats': {
                    'total': 0,
                    'parked': 0,
                    'inTransit': 0,
                    'anomalies': 0
                }
            }
        
        # Read and aggregate all combined results
        all_results = []
        for json_file in combined_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    if isinstance(results, list):
                        all_results.extend(results)
            except Exception as e:
                log.info(f"[MetricsServer] Error reading {json_file}: {e}")
                continue
        
        # Group by track_id to get unique trailers
        from collections import defaultdict
        trailers_by_track = defaultdict(list)
        
        for result in all_results:
            track_id = result.get('track_id')
            if track_id:
                trailers_by_track[track_id].append(result)
        
        # Build trailer list (one per unique track_id)
        trailers = []
        for track_id, results_list in trailers_by_track.items():
            # Get the most recent result for this track
            most_recent = max(results_list, key=lambda x: x.get('timestamp', ''))
            
            ocr_text = most_recent.get('ocr_text', '') or most_recent.get('text', '')
            spot = most_recent.get('spot', 'unknown')
            timestamp = most_recent.get('timestamp', '')
            ocr_conf = most_recent.get('ocr_conf', 0.0)
            det_conf = most_recent.get('det_conf', 0.0)
            
            # Extract GPS coordinates - check direct fields first, then world_coords
            lat = most_recent.get('lat')
            lon = most_recent.get('lon')
            world_coords = most_recent.get('world_coords')
            
            # Extract from world_coords if lat/lon not directly available
            # world_coords format: [lat, lon] when GPS coordinates are available
            if (lat is None or lon is None) and world_coords:
                if isinstance(world_coords, (list, tuple)) and len(world_coords) >= 2:
                    try:
                        coord1, coord2 = float(world_coords[0]), float(world_coords[1])
                        # Check if coordinates look like GPS (lat/lon) vs meters
                        # GPS: lat is -90 to 90, lon is -180 to 180
                        # Meters: typically much larger values (hundreds or thousands)
                        if -90.0 <= coord1 <= 90.0 and -180.0 <= coord2 <= 180.0:
                            lat = coord1
                            lon = coord2
                        # Also check if world_coords might be stored as [lon, lat] (less common)
                        elif -90.0 <= coord2 <= 90.0 and -180.0 <= coord1 <= 180.0:
                            lat = coord2
                            lon = coord1
                    except (ValueError, TypeError) as e:
                        # If conversion fails, leave lat/lon as None
                        pass
            
            # Determine status based on spot and confidence
            if spot == 'unknown' or not spot:
                status = 'In Transit'
            else:
                status = 'Parked'
            
            # Use track_id as trailer ID, or generate one
            trailer_id = f"T{track_id}" if track_id else f"T{len(trailers) + 1}"
            
            trailer = {
                'id': trailer_id,
                'plate': ocr_text if ocr_text.strip() else 'N/A',
                'spot': spot if spot != 'unknown' else 'N/A',
                'status': status,
                'detectedAt': timestamp,
                'ocrConfidence': float(ocr_conf) if ocr_conf else 0.0,
                'lat': lat,
                'lon': lon
            }
            
            trailers.append(trailer)
        
        # Calculate stats
        total = len(trailers)
        parked = sum(1 for t in trailers if t['status'] == 'Parked')
        in_transit = sum(1 for t in trailers if t['status'] == 'In Transit')
        anomalies = sum(1 for r in all_results if r.get('det_conf', 1.0) < 0.5)
        
        return {
            'trailers': trailers,
            'stats': {
                'total': total,
                'parked': parked,
                'inTransit': in_transit,
                'anomalies': anomalies
            }
        }
    
    def _get_cameras_with_status(self, force_check: bool = False) -> List[Dict]:
        """
        Get cameras from cameras.yaml and check their status.
        Uses caching to avoid testing cameras on every request.
        
        Args:
            force_check: If True, force a fresh camera status check (bypass cache)
        """
        import yaml
        import time
        
        # Cache camera status for 30 seconds to avoid repeated tests
        if not hasattr(self, '_camera_cache'):
            self._camera_cache = {'data': [], 'timestamp': 0}
        
        cache_ttl = 30  # Cache for 30 seconds
        current_time = time.time()
        
        # Return cached data if still valid and not forcing a check
        if not force_check and (current_time - self._camera_cache['timestamp']) < cache_ttl:
            return self._camera_cache['data']
        
        config_path = Path(__file__).parent.parent / 'config' / 'cameras.yaml'
        if not config_path.exists():
            return []
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            log.info(f"[MetricsServer] Error reading cameras.yaml: {e}")
            return []
        
        cameras = config.get('cameras', [])
        globals_cfg = config.get('globals', {})
        use_gstreamer = globals_cfg.get('use_gstreamer', False)
        
        camera_list = []
        for camera in cameras:
            camera_id = camera.get('id', 'unknown')
            rtsp_url = camera.get('rtsp_url', '')
            width = camera.get('width', 1920)
            height = camera.get('height', 1080)
            fps_cap = camera.get('fps_cap', 30)
            
            # Determine camera type
            if rtsp_url.isdigit() or rtsp_url == '0':
                camera_type = 'USB'
                device_index = int(rtsp_url)
            elif rtsp_url.startswith('rtsp://'):
                camera_type = 'RTSP'
            else:
                camera_type = 'Unknown'
            
            # Test camera connectivity (only if force_check or cache expired)
            status = 'offline'
            fps = 0
            latency = 0
            
            try:
                # Test camera connectivity (warnings are suppressed in test_stream)
                from app.rtsp import test_stream
                is_accessible = test_stream(rtsp_url, width, height, use_gstreamer)
                
                if is_accessible:
                    status = 'online'
                    fps = fps_cap  # Use configured FPS as default
                    latency = 50  # Default latency estimate
                else:
                    status = 'offline'
                    fps = 0
                    latency = 0
            except Exception:
                # Camera is not accessible - mark as offline
                # This is expected when cameras are not connected
                status = 'offline'
                fps = 0
                latency = 0
            
            camera_info = {
                'id': camera_id,
                'name': camera_id,
                'type': camera_type,
                'rtsp_url': rtsp_url,
                'status': status,
                'fps': fps,
                'latency': latency,
                'width': width,
                'height': height,
                'fps_cap': fps_cap
            }
            
            camera_list.append(camera_info)
        
        # Update cache
        self._camera_cache = {'data': camera_list, 'timestamp': current_time}
        
        return camera_list
    
    def _get_metrics(self) -> Dict:
        """Get current metrics snapshot."""
        return {
            'cameras': metrics_registry['cameras'].copy(),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _get_recent_events(self, camera_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get recent events from CSV file."""
        if self.csv_logger is None:
            return []
        
        events = []
        try:
            csv_path = Path(self.csv_logger.get_today_filename())
            if not csv_path.exists():
                return []
            
            import csv
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if camera_id and row.get('camera_id') != camera_id:
                        continue
                    
                    # Normalize row data: ensure all values are JSON-serializable
                    # Convert all values to JSON-safe types (no None to avoid comparison errors)
                    normalized_row = {}
                    for key, value in row.items():
                        # Ensure value is not None and convert to appropriate type
                        # CSV DictReader should return strings, but handle all edge cases
                        if value is None:
                            normalized_row[key] = ''
                        elif value == '':
                            normalized_row[key] = ''
                        else:
                            # Non-empty value - convert to appropriate type
                            try:
                                value_str = str(value).strip()
                                if not value_str:
                                    normalized_row[key] = ''
                                elif key in ['x_world', 'y_world', 'conf']:
                                    # Convert numeric fields to float, use 0.0 for empty/invalid
                                    try:
                                        float_val = float(value_str)
                                        # Check for NaN or infinity
                                        if float_val != float_val or not (-1e308 < float_val < 1e308):
                                            normalized_row[key] = 0.0
                                        else:
                                            normalized_row[key] = float_val
                                    except (ValueError, TypeError, OverflowError):
                                        normalized_row[key] = 0.0
                                elif key == 'track_id':
                                    # Convert track_id to int, use 0 for empty/invalid
                                    try:
                                        normalized_row[key] = int(value_str)
                                    except (ValueError, TypeError):
                                        normalized_row[key] = 0
                                else:
                                    # Keep as string for text fields
                                    normalized_row[key] = value_str
                            except Exception:
                                # Fallback: use empty string for any conversion error
                                normalized_row[key] = ''
                    
                    events.append(normalized_row)
            
            # Final safety check: ensure all events are JSON-serializable
            # Convert any remaining None values to safe defaults
            for event in events:
                for key, value in list(event.items()):
                    if value is None:
                        event[key] = '' if key not in ['x_world', 'y_world', 'conf'] else 0.0
            
            # Return last N events
            return events[-limit:]
        except Exception as e:
            log.info(f"Error reading events: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def update_camera_metrics(self, camera_id: str, fps_ema: float, frames_processed_count: int, 
                             last_publish: Optional[datetime] = None, queue_depth: int = 0):
        """
        Update metrics for a camera.
        
        Args:
            camera_id: Camera identifier
            fps_ema: EMA of frames per second
            frames_processed_count: Total frames processed
            last_publish: Last publish timestamp
            queue_depth: Current queue depth
        """
        if camera_id not in metrics_registry['cameras']:
            metrics_registry['cameras'][camera_id] = {}
        
        metrics_registry['cameras'][camera_id].update({
            'fps_ema': fps_ema,
            'frames_processed': frames_processed_count,
            'last_publish': last_publish.isoformat() if last_publish else None,
            'queue_depth': queue_depth
        })
        
        # Update Prometheus metrics
        fps_gauge.labels(camera_id=camera_id).set(fps_ema)
        frames_processed.labels(camera_id=camera_id)._value._value = frames_processed_count
        if last_publish:
            last_publish_time.labels(camera_id=camera_id).set(last_publish.timestamp())
        queue_depth_gauge.set(queue_depth)
    
    def start(self):
        """Start metrics server in background thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        log.info(f"Metrics server started on port {self.port}")
    
    def _run_server(self):
        """Run Flask server."""
        self.app.run(host='0.0.0.0', port=self.port, debug=False, use_reloader=False)
    
    def _generate_mjpeg(self, camera_id: str):
        """
        Generate MJPEG stream for a camera.
        
        Args:
            camera_id: Camera identifier
            
        Yields:
            JPEG frame data
        """
        no_frame_count = 0
        while True:
            if self.frame_storage is None:
                time.sleep(0.1)
                continue
            
            # Get latest frame (thread-safe)
            frame = None
            try:
                with self.frame_storage.frame_lock:
                    if camera_id in self.frame_storage.latest_frames:
                        frame = self.frame_storage.latest_frames[camera_id].copy()
            except Exception as e:
                log.info(f"[MetricsServer] Error accessing frame storage for {camera_id}: {e}")
                time.sleep(0.1)
                continue
            
            if frame is not None:
                try:
                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                        no_frame_count = 0
                    else:
                        no_frame_count += 1
                except Exception as e:
                    log.info(f"[MetricsServer] Error encoding frame for {camera_id}: {e}")
                    no_frame_count += 1
            else:
                no_frame_count += 1
                # If no frame for a while, wait longer
                if no_frame_count > 10:
                    time.sleep(0.5)
                    continue
            
            # Limit frame rate to ~30 FPS
            time.sleep(1.0 / 30.0)
    
    def _generate_processed_mjpeg(self):
        """Generate MJPEG stream from processed video frames."""
        current_frame_num = 0
        last_successful_frame = -1
        no_frame_count = 0
        consecutive_failures = 0
        
        while True:
            if self.video_processor is None:
                time.sleep(0.1)
                continue
            
            # Get latest frame number and processing status
            results = self.video_processor.get_results()
            frames_processed = results.get('frames_processed', 0)
            is_processing = self.video_processor.is_processing()
            
            if frames_processed > 0:
                # Try to get the current frame we want to display
                # Start from frame 0 and progress through frames sequentially
                frame = None
                
                # Try current frame first
                if current_frame_num < frames_processed:
                    frame = self.video_processor.get_frame(current_frame_num)
                
                # If current frame not available, try a few frames ahead
                if frame is None and current_frame_num < frames_processed:
                    for offset in range(1, min(10, frames_processed - current_frame_num)):
                        try_frame = self.video_processor.get_frame(current_frame_num + offset)
                        if try_frame is not None:
                            frame = try_frame
                            current_frame_num = current_frame_num + offset
                            break
                
                # If still no frame, try previous frames
                if frame is None and current_frame_num > 0:
                    for offset in range(1, min(10, current_frame_num + 1)):
                        try_frame = self.video_processor.get_frame(current_frame_num - offset)
                        if try_frame is not None:
                            frame = try_frame
                            current_frame_num = current_frame_num - offset
                            break
                
                if frame is not None:
                    try:
                        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if ret:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                            last_successful_frame = current_frame_num
                            no_frame_count = 0
                            consecutive_failures = 0
                            
                            # Advance to next frame for next iteration
                            current_frame_num += 1
                            
                            # If we've reached the end and processing is done, loop back to start
                            if not is_processing and current_frame_num >= frames_processed:
                                current_frame_num = 0  # Loop back to start
                        else:
                            consecutive_failures += 1
                    except Exception as e:
                        log.info(f"[MetricsServer] Error encoding frame {current_frame_num}: {e}")
                        consecutive_failures += 1
                else:
                    no_frame_count += 1
                    consecutive_failures += 1
                    # If we can't find a frame, try to advance anyway
                    if current_frame_num < frames_processed:
                        current_frame_num += 1
            else:
                no_frame_count += 1
                consecutive_failures += 1
            
            # If processing is done and we've shown all frames, loop back
            if not is_processing and frames_processed > 0 and current_frame_num >= frames_processed:
                current_frame_num = 0  # Loop back to start for continuous playback
            
            # Limit frame rate - faster when processing, slower when done
            if is_processing:
                time.sleep(1.0 / 15.0)  # 15 FPS when processing
            else:
                # When done, play at normal speed (loop)
                time.sleep(1.0 / 10.0)  # 10 FPS for playback
    
    def stop(self):
        """Stop metrics server."""
        self.running = False
        # Flask doesn't have a clean shutdown, but thread will exit when app stops

