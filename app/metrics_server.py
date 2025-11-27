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
    
    def _setup_error_handlers(self):
        """Setup Flask error handlers."""
        @self.app.errorhandler(Exception)
        def handle_exception(e):
            import traceback
            error_msg = str(e)
            traceback_str = traceback.format_exc()
            print(f"[MetricsServer] Unhandled Exception: {error_msg}")
            print(f"[MetricsServer] Traceback: {traceback_str}")
            return jsonify({'error': 'Internal server error', 'message': error_msg}), 500
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Serve dashboard index.html."""
            web_dir = Path(__file__).parent.parent / 'web'
            return send_from_directory(str(web_dir), 'index.html')
        
        @self.app.route('/<path:path>')
        def static_files(path):
            """Serve static files (JS, CSS)."""
            web_dir = Path(__file__).parent.parent / 'web'
            return send_from_directory(str(web_dir), path)
        
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
                detect_every_n = 5  # default
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
                
                # Start processing in background thread
                def process_in_background():
                    try:
                        import traceback
                        print(f"[MetricsServer] Starting video processing: {captured_video_path}, detect_every_n={captured_detect_every_n}")
                        
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
                        
                    except Exception as e:
                        print(f"[MetricsServer] Error processing video: {e}")
                        import traceback
                        traceback.print_exc()
                        # Make sure processing flag is cleared on error
                        try:
                            self.video_processor.stop_processing()
                        except:
                            pass
                
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
            return jsonify(results)
        
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
                    events.append(row)
            
            # Return last N events
            return events[-limit:]
        except Exception as e:
            print(f"Error reading events: {e}")
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

