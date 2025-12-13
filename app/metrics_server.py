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
            print(f"[MetricsServer] Unhandled Exception: {error_msg}")
            print(f"[MetricsServer] Traceback: {traceback_str}")
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
            print(f"[MetricsServer] process_video called with video_id: {video_id}")
            try:
                print(f"[MetricsServer] Checking video_processor: {self.video_processor is not None}")
                if self.video_processor is None:
                    print(f"[MetricsServer] ERROR: Video processor not available")
                    return jsonify({'error': 'Video processor not available'}), 500
                
                # Check if already processing
                try:
                    is_processing = self.video_processor.is_processing()
                    print(f"[MetricsServer] Video processor is_processing: {is_processing}")
                    if is_processing:
                        print(f"[MetricsServer] ERROR: Video processing already in progress")
                        return jsonify({'error': 'Video processing already in progress'}), 400
                except Exception as e:
                    print(f"[MetricsServer] ERROR checking is_processing: {e}")
                    import traceback
                    traceback.print_exc()
                    return jsonify({'error': f'Error checking processing status: {str(e)}'}), 500
                
                # Find video file
                try:
                    print(f"[MetricsServer] Looking for video file with pattern: {video_id}_*")
                    print(f"[MetricsServer] Upload directory: {self.upload_dir}")
                    print(f"[MetricsServer] Upload directory exists: {self.upload_dir.exists()}")
                    video_files = list(self.upload_dir.glob(f"{video_id}_*"))
                    print(f"[MetricsServer] Found {len(video_files)} matching files")
                    if not video_files:
                        print(f"[MetricsServer] ERROR: Video file not found for ID: {video_id}")
                        all_files = list(self.upload_dir.glob('*'))
                        print(f"[MetricsServer] All files in directory ({len(all_files)}): {[str(f.name) for f in all_files]}")
                        return jsonify({'error': 'Video file not found'}), 404
                    
                    video_path = str(video_files[0])
                    print(f"[MetricsServer] Found video file: {video_path}")
                except Exception as e:
                    print(f"[MetricsServer] ERROR: Failed to find video file: {e}")
                    import traceback
                    traceback.print_exc()
                    return jsonify({'error': f'Failed to find video file: {str(e)}'}), 500
                
                # Get detect_every_n parameter safely (must be done in request context)
                detect_every_n = 20  # default
                try:
                    if request.is_json:
                        try:
                            json_data = request.get_json(silent=True)
                            if json_data and 'detect_every_n' in json_data:
                                detect_every_n = int(json_data['detect_every_n'])
                        except (ValueError, TypeError) as e:
                            print(f"[MetricsServer] Warning: Invalid detect_every_n value, using default: {e}")
                except Exception as e:
                    print(f"[MetricsServer] Warning: Failed to parse request JSON, using default detect_every_n=5: {e}")
                
                # Capture video_path and detect_every_n for background thread
                # (Flask request context is not available in background threads)
                captured_video_path = video_path
                captured_detect_every_n = detect_every_n
                
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
                        print(f"[MetricsServer] Starting video processing: {captured_video_path}, detect_every_n={captured_detect_every_n}")
                        
                        # Update status: video processing started
                        with self.status_lock:
                            self.processing_status['status'] = 'processing_video'
                            self.processing_status['message'] = 'Processing video...'
                            self.processing_status['video_processing_complete'] = False
                            self.processing_status['ocr_processing_complete'] = False
                        
                        frame_count = 0
                        last_results_check = 0
                        
                        for frame_num, processed_frame, events in self.video_processor.process_video(
                            captured_video_path, camera_id="test-video", detect_every_n=captured_detect_every_n
                        ):
                            frame_count += 1
                            
                            # Check if we should stop
                            if not self.video_processor.is_processing():
                                print(f"[MetricsServer] Processing stopped at frame {frame_num}")
                                break
                            
                            # Log progress every 30 frames
                            if frame_count - last_results_check >= 30:
                                results = self.video_processor.get_results()
                                print(f"[MetricsServer] Progress: {results['frames_processed']} frames, {results['detections']} detections, {results['tracks']} tracks, {results['ocr_results']} OCR results")
                                last_results_check = frame_count
                        
                        # Final results check
                        final_results = self.video_processor.get_results()
                        print(f"[MetricsServer] Video processing completed: {frame_count} frames processed")
                        print(f"[MetricsServer] Final results: {final_results}")
                        
                        # Update status: video processing complete
                        with self.status_lock:
                            self.processing_status['status'] = 'processing_ocr'
                            self.processing_status['message'] = 'Video processing completed. Running OCR on cropped images...'
                            self.processing_status['video_processing_complete'] = True
                        
                        # Stage 2: Run OCR on cropped images
                        if self.video_processor.ocr is not None and self.video_processor.crops_dir is not None:
                            crops_dir = self.video_processor.crops_dir
                            # Ensure crops_dir is a Path object
                            from pathlib import Path
                            if not isinstance(crops_dir, Path):
                                crops_dir = Path(crops_dir)
                            if crops_dir.exists() and (crops_dir / "crops_metadata.json").exists():
                                print(f"[MetricsServer] Starting OCR processing on crops in: {crops_dir}")
                                try:
                                    from app.batch_ocr_processor import BatchOCRProcessor
                                    
                                    # Create batch OCR processor
                                    batch_ocr = BatchOCRProcessor(
                                        self.video_processor.ocr,
                                        self.video_processor.preprocessor
                                    )
                                    
                                    # Process all crops (convert Path to string)
                                    crops_dir_str = str(crops_dir)
                                    ocr_results = batch_ocr.process_crops_directory(crops_dir_str)
                                    
                                    # Match OCR results back to detections
                                    combined_results = batch_ocr.match_ocr_to_detections(crops_dir_str, ocr_results)
                                    
                                    print(f"[MetricsServer] OCR processing completed: {len(ocr_results)} crops processed")
                                    
                                    # Update status: OCR complete
                                    with self.status_lock:
                                        self.processing_status['status'] = 'completed'
                                        self.processing_status['message'] = 'All the processes completed on the video:'
                                        self.processing_status['ocr_processing_complete'] = True
                                    
                                except Exception as ocr_error:
                                    print(f"[MetricsServer] Error during OCR processing: {ocr_error}")
                                    import traceback
                                    traceback.print_exc()
                                    with self.status_lock:
                                        self.processing_status['status'] = 'error'
                                        self.processing_status['message'] = f'Video processing completed, but OCR failed: {str(ocr_error)}'
                                        self.processing_status['ocr_processing_complete'] = False
                            else:
                                print(f"[MetricsServer] No crops directory or metadata found, skipping OCR")
                                with self.status_lock:
                                    self.processing_status['status'] = 'completed'
                                    self.processing_status['message'] = 'Video processing completed (no crops to process)'
                                    self.processing_status['ocr_processing_complete'] = True
                        else:
                            print(f"[MetricsServer] OCR not available or crops not saved, skipping OCR processing")
                            with self.status_lock:
                                self.processing_status['status'] = 'completed'
                                self.processing_status['message'] = 'Video processing completed (OCR not available)'
                                self.processing_status['ocr_processing_complete'] = True
                        
                    except Exception as e:
                        print(f"[MetricsServer] Error processing video: {e}")
                        import traceback
                        traceback.print_exc()
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
                
                print(f"[MetricsServer] Video processing thread started for video_id: {video_id}")
                return jsonify({'status': 'processing_started', 'video_id': video_id})
                
            except Exception as e:
                print(f"[MetricsServer] FATAL ERROR in process_video endpoint: {e}")
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
            """Get dashboard data aggregated from combined_results.json files."""
            try:
                data = self._get_dashboard_data_from_json()
                return jsonify(data)
            except Exception as e:
                print(f"[MetricsServer] Error getting dashboard data: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/dashboard/events', methods=['GET'])
        def get_dashboard_events():
            """Get events from combined_results.json files."""
            try:
                limit = int(request.args.get('limit', 1000))
                events = self._get_events_from_json(limit)
                return jsonify({'events': events, 'count': len(events)})
            except Exception as e:
                print(f"[MetricsServer] Error getting dashboard events: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/cameras', methods=['GET'])
        def get_cameras():
            """Get list of cameras from cameras.yaml with status."""
            try:
                # Check if force refresh is requested
                force_check = request.args.get('force', 'false').lower() == 'true'
                cameras = self._get_cameras_with_status(force_check=force_check)
                return jsonify({'cameras': cameras})
            except Exception as e:
                print(f"[MetricsServer] Error getting cameras: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/inventory', methods=['GET'])
        def get_inventory():
            """Get inventory data from combined_results.json files."""
            try:
                data = self._get_inventory_from_json()
                return jsonify(data)
            except Exception as e:
                print(f"[MetricsServer] Error getting inventory data: {e}")
                import traceback
                traceback.print_exc()
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
    
    def _get_dashboard_data_from_json(self) -> Dict:
        """Get dashboard data aggregated from combined_results.json files."""
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
                print(f"[MetricsServer] Error reading {json_file}: {e}")
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
        
        # Calculate KPIs
        unique_tracks = set()
        unique_spots = set()
        ocr_successful = 0
        total_detections = len(all_results)
        low_conf_detections = 0
        today = datetime.now().date()
        
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
            
            # Check if detection is from today
            timestamp = result.get('timestamp', '')
            if timestamp:
                try:
                    result_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).date()
                    if result_date == today:
                        pass  # Could track today's detections here
                except:
                    pass
        
        # Calculate OCR accuracy
        ocr_accuracy = (ocr_successful / total_detections * 100) if total_detections > 0 else 0
        
        # Get camera health
        cameras = self._get_cameras_with_status()
        online_cameras = sum(1 for c in cameras if c.get('status') == 'online')
        degraded_cameras = sum(1 for c in cameras if c.get('status') == 'degraded')
        
        # Build accuracy chart data (today only)
        accuracy_data = []
        # Group by hour for chart, filter by today's date
        from collections import defaultdict
        hourly_data = defaultdict(lambda: {
            'total': 0, 
            'total_det_conf': 0.0,
            'total_ocr_conf': 0.0,
            'ocr_attempts': 0  # Count of detections where OCR was actually attempted
        })
        
        today = datetime.now().date()
        
        for result in all_results:
            timestamp = result.get('timestamp', '')
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    result_date = dt.date()
                    
                    # Only include results from today
                    if result_date != today:
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
    
    def _get_events_from_json(self, limit: int = 1000) -> List[Dict]:
        """Get events from combined_results.json files."""
        base_dir = Path(__file__).parent.parent / 'out' / 'crops' / 'test-video'
        
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
                            # Transform to event format expected by frontend
                            # Frontend expects 'text' for OCR text, not 'ocr_text'
                            ocr_text = result.get('ocr_text', '') or result.get('text', '')
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
                                'world_coords': result.get('world_coords', {}),
                                # Keep original fields for reference
                                'ocr_text': ocr_text,
                                'det_conf': result.get('det_conf', 0.0)
                            }
                            all_events.append(event)
            except Exception as e:
                print(f"[MetricsServer] Error reading {json_file}: {e}")
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
                print(f"[MetricsServer] Error reading {json_file}: {e}")
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
                'ocrConfidence': float(ocr_conf) if ocr_conf else 0.0
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
            print(f"[MetricsServer] Error reading cameras.yaml: {e}")
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
            print(f"Error reading events: {e}")
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
        print(f"Metrics server started on port {self.port}")
    
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
                print(f"[MetricsServer] Error accessing frame storage for {camera_id}: {e}")
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
                    print(f"[MetricsServer] Error encoding frame for {camera_id}: {e}")
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
                        print(f"[MetricsServer] Error encoding frame {current_frame_num}: {e}")
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

