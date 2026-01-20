"""
Video Recorder with GPS Logging

Records video from camera and logs GPS coordinates every second.
GPS data is stored in a JSON file for later matching during video processing.
"""

import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import threading
import time
from typing import Optional, Dict


class VideoRecorder:
    """
    Video recorder that captures video and GPS data simultaneously.
    """
    
    def __init__(self, output_dir: str = "out/recordings", gps_sensor=None):
        """
        Initialize video recorder.
        
        Args:
            output_dir: Directory to save recorded videos
            gps_sensor: Optional GPSSensor instance
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gps_sensor = gps_sensor
        
        self.recording = False
        self.video_writer = None
        self.video_path = None
        self.gps_log_path = None
        self.gps_log = {}  # timestamp -> {'lat': float, 'lon': float, 'heading': float (optional), 'speed': float (optional)}
        self.lock = threading.Lock()
        
        self.camera_id = None
        self.frame_width = None
        self.frame_height = None
        self.fps = 30
        
        # GPS logging thread
        self.gps_logging_thread = None
        self.gps_logging_running = False
    
    def start_recording(self, camera_id: str, frame_width: int, frame_height: int, fps: int = 30):
        """
        Start recording video and GPS data.
        
        Args:
            camera_id: Camera identifier
            frame_width: Video frame width
            frame_height: Video frame height
            fps: Video frame rate
        """
        if self.recording:
            print(f"[VideoRecorder] Already recording, stop current recording first")
            return False
        
        self.camera_id = camera_id
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_filename = f"{camera_id}_{timestamp}.mp4"
        gps_filename = f"{camera_id}_{timestamp}_gps.json"
        
        self.video_path = self.output_dir / video_filename
        self.gps_log_path = self.output_dir / gps_filename
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(self.video_path),
            fourcc,
            fps,
            (frame_width, frame_height)
        )
        
        if not self.video_writer.isOpened():
            print(f"[VideoRecorder] Failed to open video writer")
            return False
        
        self.recording = True
        self.gps_log = {}
        
        # Start GPS logging thread
        if self.gps_sensor:
            self.gps_logging_running = True
            self.gps_logging_thread = threading.Thread(
                target=self._gps_logging_loop,
                daemon=True
            )
            self.gps_logging_thread.start()
            print(f"[VideoRecorder] Started GPS logging thread")
        
        print(f"[VideoRecorder] Started recording: {self.video_path}")
        return True
    
    def _gps_logging_loop(self):
        """
        Background thread that logs GPS coordinates every second.
        """
        while self.gps_logging_running:
            if self.gps_sensor:
                gps_data = self.gps_sensor.get_current_gps()
                if gps_data:
                    timestamp = datetime.utcnow()
                    timestamp_str = timestamp.isoformat()
                    
                    with self.lock:
                        log_entry = {
                            'lat': gps_data['lat'],
                            'lon': gps_data['lon'],
                            'timestamp': timestamp_str
                        }
                        # Include heading and speed if available
                        if 'heading' in gps_data:
                            log_entry['heading'] = gps_data['heading']
                        if 'speed' in gps_data:
                            log_entry['speed'] = gps_data['speed']
                        self.gps_log[timestamp_str] = log_entry
                    
                    # Log every 10 seconds for debugging
                    if len(self.gps_log) % 10 == 0:
                        print(f"[VideoRecorder] GPS logged: ({gps_data['lat']:.6f}, {gps_data['lon']:.6f}) - Total entries: {len(self.gps_log)}")
            
            # Sleep for 1 second
            time.sleep(0.1)
    
    def write_frame(self, frame: np.ndarray):
        """
        Write a frame to the video file.
        
        Args:
            frame: BGR image frame
        """
        if not self.recording or self.video_writer is None:
            return False
        
        # Resize frame if needed
        if frame.shape[1] != self.frame_width or frame.shape[0] != self.frame_height:
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        
        self.video_writer.write(frame)
        return True
    
    def stop_recording(self):
        """
        Stop recording and save GPS log file.
        
        Returns:
            Tuple of (video_path, gps_log_path) or (None, None) if not recording
        """
        if not self.recording:
            return None, None
        
        self.recording = False
        self.gps_logging_running = False
        
        # Wait for GPS logging thread to finish
        if self.gps_logging_thread:
            self.gps_logging_thread.join(timeout=2.0)
        
        # Release video writer
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        # Save GPS log to JSON file
        if self.gps_log and self.gps_log_path:
            try:
                with open(self.gps_log_path, 'w') as f:
                    json.dump(self.gps_log, f, indent=2)
                print(f"[VideoRecorder] Saved GPS log: {self.gps_log_path} ({len(self.gps_log)} entries)")
            except Exception as e:
                print(f"[VideoRecorder] Error saving GPS log: {e}")
        
        video_path = str(self.video_path) if self.video_path else None
        gps_log_path = str(self.gps_log_path) if self.gps_log_path else None
        
        print(f"[VideoRecorder] Stopped recording: {video_path}")
        
        # Reset paths
        self.video_path = None
        self.gps_log_path = None
        
        return video_path, gps_log_path
    
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self.recording
    
    def get_gps_log(self) -> Dict:
        """Get current GPS log (for testing/debugging)."""
        with self.lock:
            return self.gps_log.copy()
