"""
Database Manager for Video Frame Records

Stores video processing results (OCR, GPS, images) in local database.
Matches diagram requirements: Licence Plate/Trailer, latitude, longitude, speed, barrier, Confidence, Image Path
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import threading


class VideoFrameDB:
    """
    Manages database operations for video frame records.
    Uses SQLite for local storage on NVIDIA device.
    """
    
    def __init__(self, db_path: str = "data/video_frames.db"):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        
        # Initialize database schema
        self._initialize_database()
    
    def _initialize_database(self):
        """Create database tables if they don't exist."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS video_frame_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            licence_plate_trailer TEXT,
            latitude REAL,
            longitude REAL,
            speed REAL,
            barrier REAL,
            confidence REAL,
            image_path TEXT,
            camera_id TEXT,
            video_path TEXT,
            frame_number INTEGER,
            track_id INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_processed BOOLEAN DEFAULT 0,
            created_on DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_on DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_video_frame_records_unprocessed 
            ON video_frame_records(is_processed, created_on) 
            WHERE is_processed = 0;
        
        CREATE INDEX IF NOT EXISTS idx_video_frame_records_camera 
            ON video_frame_records(camera_id, created_on);
        
        CREATE INDEX IF NOT EXISTS idx_video_frame_records_licence 
            ON video_frame_records(licence_plate_trailer, is_processed);
        """
        
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.executescript(create_table_sql)
                conn.commit()
                self._migrate_add_assigned_spot_columns(conn)
                conn.commit()
                print(f"[VideoFrameDB] Database initialized: {self.db_path}")
            except Exception as e:
                print(f"[VideoFrameDB] Error initializing database: {e}")
                raise
            finally:
                conn.close()
    
    def _migrate_add_assigned_spot_columns(self, conn: sqlite3.Connection):
        """Add assigned_spot_id, assigned_spot_name, assigned_distance_ft, processed_comment if missing."""
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(video_frame_records)")
        existing = {row[1] for row in cursor.fetchall()}
        for col, typ in [
            ('assigned_spot_id', 'TEXT'),
            ('assigned_spot_name', 'TEXT'),
            ('assigned_distance_ft', 'REAL'),
            ('processed_comment', 'TEXT'),
        ]:
            if col not in existing:
                cursor.execute(f"ALTER TABLE video_frame_records ADD COLUMN {col} {typ}")
    
    def insert_frame_record(
        self,
        licence_plate_trailer: str,
        latitude: float,
        longitude: float,
        speed: Optional[float] = None,
        barrier: Optional[float] = None,
        confidence: float = 0.0,
        image_path: str = "",
        camera_id: str = "",
        video_path: str = "",
        frame_number: int = 0,
        track_id: Optional[int] = None,
        timestamp: Optional[datetime] = None
    ) -> int:
        """
        Insert a video frame record into the database.
        
        Args:
            licence_plate_trailer: OCR result (license plate or trailer ID)
            latitude: GPS latitude
            longitude: GPS longitude
            speed: Vehicle speed (optional)
            barrier: Heading/barrier angle (optional)
            confidence: OCR confidence score
            image_path: Path to cropped image
            camera_id: Camera identifier
            video_path: Path to source video
            frame_number: Frame number in video
            track_id: Tracking ID
            timestamp: Timestamp (defaults to now)
            
        Returns:
            Inserted record ID
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        insert_sql = """
        INSERT INTO video_frame_records (
            licence_plate_trailer, latitude, longitude, speed, barrier,
            confidence, image_path, camera_id, video_path, frame_number,
            track_id, timestamp, is_processed
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """
        
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(insert_sql, (
                    licence_plate_trailer,
                    latitude,
                    longitude,
                    speed,
                    barrier,
                    confidence,
                    image_path,
                    camera_id,
                    video_path,
                    frame_number,
                    track_id,
                    timestamp.isoformat()
                ))
                record_id = cursor.lastrowid
                conn.commit()
                return record_id
            except Exception as e:
                print(f"[VideoFrameDB] Error inserting record: {e}")
                conn.rollback()
                raise
            finally:
                conn.close()
    
    def get_unprocessed_records(
        self,
        limit: int = 50,
        cutoff_seconds: int = 10,
        camera_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Get latest unprocessed records (similar to GetLatestUnprocessedScans).
        
        Args:
            limit: Maximum number of records to return
            cutoff_seconds: Only return records older than this many seconds
            camera_id: Optional camera filter
            
        Returns:
            List of unprocessed records
        """
        if limit <= 0:
            return []
        
        cutoff_time = datetime.utcnow().timestamp() - cutoff_seconds
        
        query = """
        SELECT id, licence_plate_trailer, latitude, longitude, speed, barrier,
               confidence, image_path, camera_id, video_path, frame_number,
               track_id, timestamp, created_on
        FROM video_frame_records
        WHERE is_processed = 0
          AND datetime(created_on) < datetime('now', '-' || ? || ' seconds')
        """
        
        params = [cutoff_seconds]
        
        if camera_id:
            query += " AND camera_id = ?"
            params.append(camera_id)
        
        query += """
        ORDER BY created_on DESC
        LIMIT ?
        """
        params.append(limit)
        
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert to list of dicts
                records = []
                for row in rows:
                    records.append({
                        'id': row['id'],
                        'licence_plate_trailer': row['licence_plate_trailer'],
                        'latitude': row['latitude'],
                        'longitude': row['longitude'],
                        'speed': row['speed'],
                        'barrier': row['barrier'],
                        'confidence': row['confidence'],
                        'image_path': row['image_path'],
                        'camera_id': row['camera_id'],
                        'video_path': row['video_path'],
                        'frame_number': row['frame_number'],
                        'track_id': row['track_id'],
                        'timestamp': row['timestamp'],
                        'created_on': row['created_on']
                    })
                
                return records
            except Exception as e:
                print(f"[VideoFrameDB] Error fetching unprocessed records: {e}")
                return []
            finally:
                conn.close()
    
    def mark_as_processed(
        self,
        record_id: int,
        comment: str = "",
        assigned_spot_id: Optional[str] = None,
        assigned_spot_name: Optional[str] = None,
        assigned_distance_ft: Optional[float] = None,
    ):
        """
        Mark a record as processed and optionally store assigned spot and distance.
        
        Args:
            record_id: Record ID to mark as processed
            comment: Optional comment (e.g. "Location Updated: G 94" or "Ignore - ...")
            assigned_spot_id: Spot id from reference (e.g. CSV/GeoJSON)
            assigned_spot_name: Spot display name
            assigned_distance_ft: Distance in feet from record to spot centroid
        """
        update_sql = """
        UPDATE video_frame_records
        SET is_processed = 1,
            updated_on = CURRENT_TIMESTAMP,
            assigned_spot_id = ?,
            assigned_spot_name = ?,
            assigned_distance_ft = ?,
            processed_comment = ?
        WHERE id = ?
        """
        
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(update_sql, (
                    assigned_spot_id,
                    assigned_spot_name,
                    assigned_distance_ft,
                    comment,
                    record_id,
                ))
                conn.commit()
            except Exception as e:
                print(f"[VideoFrameDB] Error marking record as processed: {e}")
                conn.rollback()
            finally:
                conn.close()
    
    def get_all_records(
        self,
        limit: int = 50,
        offset: int = 0,
        is_processed: Optional[bool] = None,
        camera_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Get all records with optional filtering.
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            is_processed: Filter by processed status (None = all)
            camera_id: Filter by camera ID (None = all)
            
        Returns:
            List of records
        """
        query = """
        SELECT id, licence_plate_trailer, latitude, longitude, speed, barrier,
               confidence, image_path, camera_id, video_path, frame_number,
               track_id, timestamp, created_on, is_processed,
               assigned_spot_id, assigned_spot_name, assigned_distance_ft, processed_comment
        FROM video_frame_records
        WHERE 1=1
        """
        
        params = []
        
        if is_processed is not None:
            query += " AND is_processed = ?"
            params.append(1 if is_processed else 0)
        
        if camera_id:
            query += " AND camera_id = ?"
            params.append(camera_id)
        
        query += " ORDER BY created_on DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert to list of dicts (new columns may not exist on old DBs)
                records = []
                for row in rows:
                    rec = {
                        'id': row['id'],
                        'licence_plate_trailer': row['licence_plate_trailer'],
                        'latitude': row['latitude'],
                        'longitude': row['longitude'],
                        'speed': row['speed'],
                        'barrier': row['barrier'],
                        'confidence': row['confidence'],
                        'image_path': row['image_path'],
                        'camera_id': row['camera_id'],
                        'video_path': row['video_path'],
                        'frame_number': row['frame_number'],
                        'track_id': row['track_id'],
                        'timestamp': row['timestamp'],
                        'created_on': row['created_on'],
                        'is_processed': bool(row['is_processed'])
                    }
                    try:
                        rec['assigned_spot_id'] = row['assigned_spot_id']
                        rec['assigned_spot_name'] = row['assigned_spot_name']
                        rec['assigned_distance_ft'] = row['assigned_distance_ft']
                        rec['processed_comment'] = row['processed_comment']
                    except (IndexError, KeyError):
                        rec['assigned_spot_id'] = rec['assigned_spot_name'] = rec['processed_comment'] = None
                        rec['assigned_distance_ft'] = None
                    records.append(rec)
                
                return records
            except Exception as e:
                print(f"[VideoFrameDB] Error fetching records: {e}")
                return []
            finally:
                conn.close()
    
    def get_statistics(self) -> Dict:
        """Get database statistics."""
        stats_sql = """
        SELECT 
            COUNT(*) as total_records,
            SUM(CASE WHEN is_processed = 0 THEN 1 ELSE 0 END) as unprocessed_count,
            SUM(CASE WHEN is_processed = 1 THEN 1 ELSE 0 END) as processed_count
        FROM video_frame_records
        """
        
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(stats_sql)
                row = cursor.fetchone()
                
                return {
                    'total': row['total_records'] if row else 0,
                    'unprocessed': row['unprocessed_count'] if row else 0,
                    'processed': row['processed_count'] if row else 0
                }
            except Exception as e:
                print(f"[VideoFrameDB] Error getting statistics: {e}")
                return {'total': 0, 'unprocessed': 0, 'processed': 0}
            finally:
                conn.close()

    def delete_records_by_ids(self, ids: List[int]) -> int:
        """
        Delete records by their IDs. Used after successful upload to AWS.
        Returns the number of rows deleted.
        """
        if not ids:
            return 0
        placeholders = ",".join("?" * len(ids))
        delete_sql = f"DELETE FROM video_frame_records WHERE id IN ({placeholders})"
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(delete_sql, ids)
                deleted = cursor.rowcount
                conn.commit()
                return deleted
            except Exception as e:
                print(f"[VideoFrameDB] Error deleting records: {e}")
                conn.rollback()
                return 0
            finally:
                conn.close()
