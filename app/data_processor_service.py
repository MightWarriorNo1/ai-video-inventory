"""
Data Processor Service

Runs periodically to process unprocessed video frame records.
Python equivalent of the Azure Function timer trigger in SuperAmazingAlgo.cs
"""

import time
import threading
import logging
import csv
from typing import List, Dict, Optional
from pathlib import Path
import json


def _get_csv_val(row_lower: Dict[str, str], _fieldnames: List[str], keys: List[str]) -> Optional[str]:
    """Get value from CSV row by trying multiple possible column names."""
    for key in keys:
        val = row_lower.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    # Match any header that contains the key (e.g. "Latitude" -> "latitude")
    for k, v in row_lower.items():
        if v is None or not str(v).strip():
            continue
        for key in keys:
            if key in k or k in key:
                return str(v).strip()
    return None


class DataProcessorService:
    """
    Service that runs periodically to process unprocessed video frame records.
    Similar to the Azure Function timer trigger that runs every 5 seconds.
    """
    
    def __init__(self, video_frame_db, parking_spots: List[Dict] = None, logger=None):
        """
        Initialize data processor service.
        
        Args:
            video_frame_db: VideoFrameDB instance
            parking_spots: List of parking spots (can be loaded from GeoJSON)
            logger: Optional logger instance
        """
        from app.data_processor import DataProcessor
        
        self.db = video_frame_db
        self.processor = DataProcessor(video_frame_db, logger)
        self.parking_spots = parking_spots or []
        self.logger = logger or logging.getLogger(__name__)
        
        self.running = False
        self.worker_thread = None
        self.interval_seconds = 5  # Run every 5 seconds (like Azure Function)
        self.limit = 50  # Process up to 50 records per cycle
        
    def load_parking_spots_from_geojson(self, geojson_path: str):
        """
        Load parking spots from GeoJSON file.
        
        Args:
            geojson_path: Path to GeoJSON file (e.g., config/spots.geojson)
        """
        try:
            with open(geojson_path, 'r') as f:
                geojson_data = json.load(f)
            
            spots = []
            for feature in geojson_data.get('features', []):
                props = feature.get('properties', {})
                geometry = feature.get('geometry', {})
                
                if geometry.get('type') == 'Point':
                    coords = geometry.get('coordinates', [])
                    if len(coords) >= 2:
                        spots.append({
                            'id': props.get('id', ''),
                            'name': props.get('name', ''),
                            'latitude': coords[1],  # GeoJSON is [lon, lat]
                            'longitude': coords[0]
                        })
            
            self.parking_spots = spots
            self.logger.info("Data processor: loaded %d parking spots from GeoJSON %s", len(spots), geojson_path)
            
        except Exception as e:
            self.logger.error("Data processor: error loading GeoJSON %s: %s", geojson_path, e, exc_info=True)
            self.parking_spots = []
    
    def load_parking_spots_from_csv(self, csv_path: str):
        """
        Load reference parking spots from a CSV file.

        Supports two formats:
        1) Geofences format: Name, Latitude 1-5, Longitude 1-5 (5-point polygon).
           Uses Latitude 5 / Longitude 5 as the spot centroid (reference point).
        2) Simple format: id, name, latitude, longitude (or lat, lon/lng).

        Header row required.

        Args:
            csv_path: Path to CSV file
        """
        self.logger.info("Data processor: loading parking spots from CSV: %s", csv_path)
        try:
            spots = []
            with open(csv_path, 'r', newline='', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    self.logger.error("CSV has no header row")
                    self.parking_spots = []
                    return
                fieldnames = [fn.strip().lower() for fn in reader.fieldnames]
                for row in reader:
                    # Normalize keys (strip, lower) for lookup
                    row_lower = {k.strip().lower(): v for k, v in row.items() if k}
                    # Try Geofences format first: centroid = 5th point (Latitude 5, Longitude 5)
                    lat = _get_csv_val(row_lower, fieldnames, ['latitude 5', 'lat 5', 'latitude5', 'lat5'])
                    lon = _get_csv_val(row_lower, fieldnames, ['longitude 5', 'lon 5', 'lng 5', 'longitude5', 'lon5', 'lng5'])
                    if lat is None or lon is None:
                        # Fallback: single latitude/longitude columns
                        lat = _get_csv_val(row_lower, fieldnames, ['latitude', 'lat'])
                        lon = _get_csv_val(row_lower, fieldnames, ['longitude', 'lon', 'lng'])
                    name = _get_csv_val(row_lower, fieldnames, ['name', 'spot', 'spot_name']) or ''
                    id_val = _get_csv_val(row_lower, fieldnames, ['id', 'spot_id']) or name
                    if lat is not None and lon is not None:
                        try:
                            spots.append({
                                'id': str(id_val).strip(),
                                'name': str(name).strip() or str(id_val).strip(),
                                'latitude': float(lat),
                                'longitude': float(lon)
                            })
                        except (ValueError, TypeError):
                            continue
            self.parking_spots = spots
            self.logger.info("Data processor: loaded %d parking spots from CSV %s", len(spots), csv_path)
        except Exception as e:
            self.logger.error("Data processor: error loading CSV %s: %s", csv_path, e, exc_info=True)
            self.parking_spots = []
    
    def run_once(self) -> Dict:
        """
        Run one processing cycle (e.g. background periodic run).
        Processes up to self.limit records.
        Returns stats dict.
        """
        if not self.parking_spots:
            self.logger.warning("Data processor run_once: no parking spots loaded, skipping")
            return {'success': False, 'message': 'No parking spots loaded', 'processed': 0}
        self.logger.info("Data processor: run_once starting (limit=%d spots)", self.limit)
        try:
            results = self.processor.process_all_unprocessed(
                parking_spots=self.parking_spots,
                limit=self.limit
            )
            processed = len(results) if results else 0
            self.logger.info("Data processor: run_once finished, processed %d record(s)", processed)
            return {
                'success': True,
                'processed': processed,
                'results': results or []
            }
        except Exception as e:
            self.logger.error("Data processor run_once error: %s", e, exc_info=True)
            return {'success': False, 'message': str(e), 'processed': 0}

    def run_all(self, limit: int = 1_000_000) -> Dict:
        """
        Process all unprocessed records (e.g. after loading CSV via Run Processor).
        No effective limit by default; processes every unprocessed record in one run.
        Only records with a valid spot assignment are marked as processed (proper values in DB).
        Returns stats dict.
        """
        if not self.parking_spots:
            self.logger.warning("Data processor run_all: no parking spots loaded, skipping")
            return {'success': False, 'message': 'No parking spots loaded', 'processed': 0}
        self.logger.info("Data processor: run_all starting (limit=%d, spots=%d)", limit, len(self.parking_spots))
        try:
            results = self.processor.process_all_unprocessed(
                parking_spots=self.parking_spots,
                limit=limit
            )
            processed = len(results) if results else 0
            self.logger.info("Data processor: run_all finished, processed %d record(s)", processed)
            return {
                'success': True,
                'processed': processed,
                'results': results or []
            }
        except Exception as e:
            self.logger.error("Data processor run_all error: %s", e, exc_info=True)
            return {'success': False, 'message': str(e), 'processed': 0}

    def start(self):
        """Start the data processor service."""
        # Background job disabled (5-second periodic processor)
        # if self.running:
        #     return
        # self.running = True
        # self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        # self.worker_thread.start()
        # self.logger.info(f"Data processor service started (interval: {self.interval_seconds}s)")
        self.logger.info("Data processor background job is disabled")
        return

    def stop(self):
        """Stop the data processor service."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        self.logger.info("Data processor service stopped")

    def _worker_loop(self):
        """Main worker loop that runs periodically. (Not used while start() is disabled.)"""
        while self.running:
            try:
                self._process_cycle()
            except Exception as e:
                self.logger.error("Data processor: error in cycle: %s", e, exc_info=True)
            time.sleep(self.interval_seconds)

    def _process_cycle(self):
        """Process one cycle of unprocessed records."""
        if not self.parking_spots:
            self.logger.warning("Data processor: no parking spots loaded, skipping cycle")
            return
        records = self.processor.get_latest_unprocessed_scans(limit=self.limit)
        if not records:
            return
        self.logger.info("Data processor: cycle processing %d unprocessed record(s)", len(records))
        results = self.processor.process_all_unprocessed(
            parking_spots=self.parking_spots,
            limit=self.limit
        )
        if results:
            self.logger.info("Data processor: cycle done, processed %d record(s) successfully", len(results))
            for result in results:
                self.logger.debug(
                    "Data processor: record %s plate %s -> spot %s (%.2f ft)",
                    result['record_id'], result['plate'],
                    result['nearest_spot'].get('name', 'Unknown'), result['distance']
                )
    
    def get_statistics(self) -> Dict:
        """Get processing statistics."""
        db_stats = self.db.get_statistics()
        return {
            **db_stats,
            'parking_spots_loaded': len(self.parking_spots),
            'service_running': self.running,
            'interval_seconds': self.interval_seconds
        }


def create_data_processor_service(
    db_path: str = "data/video_frames.db",
    geojson_path: str = "config/spots.geojson",
    interval_seconds: int = 5
) -> DataProcessorService:
    """
    Factory function to create and configure data processor service.
    
    Args:
        db_path: Path to video frame database
        geojson_path: Path to parking spots GeoJSON file
        interval_seconds: Processing interval in seconds
        
    Returns:
        Configured DataProcessorService instance
    """
    from app.video_frame_db import VideoFrameDB
    
    # Initialize database
    db = VideoFrameDB(db_path=db_path)
    
    # Create service
    service = DataProcessorService(db, logger=logging.getLogger(__name__))
    service.interval_seconds = interval_seconds
    
    # Load parking spots
    if Path(geojson_path).exists():
        service.load_parking_spots_from_geojson(geojson_path)
    else:
        logging.warning(f"GeoJSON file not found: {geojson_path}")
    
    return service
