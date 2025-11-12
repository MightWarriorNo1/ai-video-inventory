"""
ByteTrack Multi-Object Tracker Wrapper

This module provides a wrapper for ByteTrack tracking with Kalman filtering.
"""

from typing import List, Dict, Tuple
import numpy as np


class KalmanFilter:
    """Simple Kalman filter for bounding box tracking."""
    
    def __init__(self):
        # State: [cx, cy, s, r, vx, vy, vs, vr] where (cx,cy) is center, s is area, r is aspect ratio
        self.state = np.zeros(8, dtype=np.float32)
        self.covariance = np.eye(8, dtype=np.float32) * 10.0
        
        # Motion model (constant velocity)
        self.F = np.eye(8, dtype=np.float32)
        self.F[0, 4] = 1.0  # cx += vx
        self.F[1, 5] = 1.0  # cy += vy
        self.F[2, 6] = 1.0  # s += vs
        self.F[3, 7] = 1.0  # r += vr
        
        # Measurement model (observe center, area, aspect ratio)
        self.H = np.zeros((4, 8), dtype=np.float32)
        self.H[0, 0] = 1.0  # observe cx
        self.H[1, 1] = 1.0  # observe cy
        self.H[2, 2] = 1.0  # observe s
        self.H[3, 3] = 1.0  # observe r
        
        # Process noise
        self.Q = np.eye(8, dtype=np.float32) * 0.1
        
        # Measurement noise
        self.R = np.eye(4, dtype=np.float32) * 1.0
    
    def init(self, bbox: List[float]):
        """Initialize filter with bounding box [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        s = w * h
        r = w / (h + 1e-6)
        
        self.state = np.array([cx, cy, s, r, 0, 0, 0, 0], dtype=np.float32)
        self.covariance = np.eye(8, dtype=np.float32) * 10.0
    
    def predict(self):
        """Predict next state."""
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
    
    def update(self, bbox: List[float]):
        """Update with measurement [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        s = w * h
        r = w / (h + 1e-6)
        
        z = np.array([cx, cy, s, r], dtype=np.float32)
        
        # Kalman update
        y = z - self.H @ self.state  # Innovation
        S = self.H @ self.covariance @ self.H.T + self.R  # Innovation covariance
        K = self.covariance @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.state = self.state + K @ y
        self.covariance = (np.eye(8) - K @ self.H) @ self.covariance
    
    def get_bbox(self) -> List[float]:
        """Get predicted bounding box [x1, y1, x2, y2]."""
        cx, cy, s, r = self.state[0], self.state[1], self.state[2], self.state[3]
        h = np.sqrt(s / (r + 1e-6))
        w = s / (h + 1e-6)
        
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        return [float(x1), float(y1), float(x2), float(y2)]


class ByteTrackWrapper:
    """
    ByteTrack multi-object tracker with Kalman filtering.
    
    Maintains track identities across frames using IoU matching and Kalman filtering.
    """
    
    def __init__(self, frame_rate: int = 30, track_thresh: float = 0.5, track_buffer: int = 30, match_thresh: float = 0.8):
        """
        Initialize ByteTrack tracker.
        
        Args:
            frame_rate: Video frame rate
            track_thresh: Detection confidence threshold for tracking
            track_buffer: Number of frames to keep lost tracks
            match_thresh: IoU threshold for matching
        """
        self.frame_rate = frame_rate
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        
        self.next_track_id = 1
        self.tracks = {}  # track_id -> {bbox, conf, cls, kf, frame_count, state}
        self.lost_tracks = {}  # Lost tracks that might be recovered
        self.frame_count = 0
    
    def _bbox_to_xyxy(self, bbox: List[float]) -> List[float]:
        """Ensure bbox is in [x1, y1, x2, y2] format."""
        if len(bbox) == 4:
            return bbox
        return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
    
    def update(self, detections: List[Dict], frame: np.ndarray = None) -> List[Dict]:
        """
        Update tracker with new detections (ByteTrack algorithm).
        
        Args:
            detections: List of detection dicts with keys: bbox, cls, conf
            frame: Optional frame for visualization
            
        Returns:
            List of track dicts with keys: track_id, bbox, cls, conf
        """
        self.frame_count += 1
        
        # Predict all active tracks
        for track_id, track in list(self.tracks.items()):
            if track['state'] == 'tracked':
                track['kf'].predict()
                track['bbox'] = track['kf'].get_bbox()
        
        # Separate detections into high and low confidence
        high_conf_dets = [d for d in detections if d['conf'] >= self.track_thresh]
        low_conf_dets = [d for d in detections if d['conf'] < self.track_thresh]
        
        # First association: match high confidence detections with tracked objects
        tracked_tracks = {tid: t for tid, t in self.tracks.items() if t['state'] == 'tracked'}
        matched_pairs, unmatched_tracks, unmatched_dets = self._associate_detections_to_tracks(
            high_conf_dets, tracked_tracks
        )
        
        # Update matched tracks
        for det_idx, track_id in matched_pairs:
            det = high_conf_dets[det_idx]
            track = self.tracks[track_id]
            track['kf'].update(det['bbox'])
            track['bbox'] = det['bbox']  # Use detection bbox
            track['conf'] = det['conf']
            track['cls'] = det['cls']
            track['frame_count'] = self.frame_count
            track['state'] = 'tracked'
        
        # Second association: match remaining detections with lost tracks
        lost_tracks = {tid: t for tid, t in self.tracks.items() if t['state'] == 'lost'}
        matched_pairs_lost, unmatched_lost, unmatched_dets_rem = self._associate_detections_to_tracks(
            [high_conf_dets[i] for i in unmatched_dets], lost_tracks
        )
        
        # Reactivate lost tracks
        for det_idx_rem, track_id in matched_pairs_lost:
            det = [high_conf_dets[i] for i in unmatched_dets][det_idx_rem]
            track = self.tracks[track_id]
            track['kf'].update(det['bbox'])
            track['bbox'] = det['bbox']
            track['conf'] = det['conf']
            track['cls'] = det['cls']
            track['frame_count'] = self.frame_count
            track['state'] = 'tracked'
        
        # Create new tracks for unmatched high confidence detections
        final_unmatched_dets = [unmatched_dets[i] for i in range(len(unmatched_dets)) 
                               if i not in [p[0] for p in matched_pairs_lost]]
        for det_idx in final_unmatched_dets:
            det = high_conf_dets[det_idx]
            track_id = self.next_track_id
            self.next_track_id += 1
            
            kf = KalmanFilter()
            kf.init(det['bbox'])
            
            self.tracks[track_id] = {
                'bbox': det['bbox'],
                'conf': det['conf'],
                'cls': det['cls'],
                'kf': kf,
                'frame_count': self.frame_count,
                'state': 'tracked'
            }
        
        # Mark unmatched tracks as lost
        for track_id in unmatched_tracks:
            if track_id in self.tracks:
                self.tracks[track_id]['state'] = 'lost'
        
        # Remove tracks that have been lost for too long
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track['state'] == 'lost' and (self.frame_count - track['frame_count']) > self.track_buffer:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        # Return active tracks
        active_tracks = []
        for track_id, track in self.tracks.items():
            if track['state'] == 'tracked':
                active_tracks.append({
                    'track_id': track_id,
                    'bbox': track['bbox'],
                    'cls': track['cls'],
                    'conf': track['conf']
                })
        
        return active_tracks
    
    def _associate_detections_to_tracks(self, detections: List[Dict], tracks: Dict) -> Tuple[List, List, List]:
        """
        Associate detections to tracks using IoU matching.
        
        Returns:
            (matched_pairs, unmatched_tracks, unmatched_dets)
        """
        if len(tracks) == 0:
            return [], [], list(range(len(detections)))
        
        if len(detections) == 0:
            return [], list(tracks.keys()), []
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(detections), len(tracks)), dtype=np.float32)
        track_ids = list(tracks.keys())
        
        for i, det in enumerate(detections):
            for j, track_id in enumerate(track_ids):
                track = tracks[track_id]
                iou_matrix[i, j] = self._compute_iou(det['bbox'], track['bbox'])
        
        # Greedy matching
        matched_pairs = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(tracks.keys())
        
        # Sort by IoU (highest first)
        matches = []
        for i in range(len(detections)):
            for j in range(len(tracks)):
                if iou_matrix[i, j] > self.match_thresh:
                    matches.append((iou_matrix[i, j], i, j))
        
        matches.sort(reverse=True, key=lambda x: x[0])
        
        used_dets = set()
        used_tracks = set()
        
        for _, i, j in matches:
            if i not in used_dets and j not in used_tracks:
                matched_pairs.append((i, track_ids[j]))
                used_dets.add(i)
                used_tracks.add(j)
        
        unmatched_dets = [i for i in range(len(detections)) if i not in used_dets]
        unmatched_tracks = [track_ids[j] for j in range(len(tracks)) if j not in used_tracks]
        
        return matched_pairs, unmatched_tracks, unmatched_dets
    
    def _compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Compute IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        inter_area = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area

