"""
Metrics Server with Web Dashboard

Flask server exposing JSON metrics, Prometheus metrics, health check,
events API, and static web dashboard files.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from flask import Flask, jsonify, send_from_directory, request
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
import threading


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
    
    def __init__(self, port: int = 8080, csv_logger=None):
        """
        Initialize metrics server.
        
        Args:
            port: HTTP server port
            csv_logger: Optional CSV logger for events API
        """
        self.port = port
        self.csv_logger = csv_logger
        self.app = Flask(__name__, static_folder='../web', static_url_path='')
        self._setup_routes()
        self.thread = None
        self.running = False
    
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
    
    def stop(self):
        """Stop metrics server."""
        self.running = False
        # Flask doesn't have a clean shutdown, but thread will exit when app stops

