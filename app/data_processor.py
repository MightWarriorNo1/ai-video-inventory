"""
Data Processor - Python Conversion of SuperAmazingAlgo.cs

Fetches unprocessed video frame records from database and calculates nearest location.
This is the Python equivalent of the .NET scan processor application.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path
import math


class DataProcessor:
    """
    Python version of SuperAmazingAlgo.cs
    Processes unprocessed video frame records to calculate nearest parking spot locations.
    """
    
    def __init__(self, video_frame_db, logger=None):
        """
        Initialize data processor.
        
        Args:
            video_frame_db: VideoFrameDB instance
            logger: Optional logger instance
        """
        self.db = video_frame_db
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuration (can be loaded from config file)
        self.config = {
            'min_speed': 0.0,
            'max_speed': 100.0,
            'max_slots': 5,
            'max_slot_distance': 100.0,  # feet
            'observation_cutoff_minutes': 30,
            'observations_count': 3
        }
        
        # Processed license plates (to avoid duplicate processing)
        self._processed_plates = set()
    
    def get_latest_unprocessed_scans(
        self,
        limit: int = 50,
        cutoff_seconds: int = 10,
        camera_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Get latest unprocessed video frame records.
        Python equivalent of GetLatestUnprocessedScans from SuperAmazingAlgo.cs
        
        Args:
            limit: Maximum number of records to return
            cutoff_seconds: Only return records older than this many seconds
            camera_id: Optional camera filter
            
        Returns:
            List of unprocessed records
        """
        if limit <= 0:
            return []
        
        try:
            # Get unprocessed records from database
            records = self.db.get_unprocessed_records(
                limit=limit,
                cutoff_seconds=cutoff_seconds,
                camera_id=camera_id
            )
            self.logger.info(
                "Data processor: fetched %d unprocessed record(s) from DB (limit=%d)",
                len(records) if records else 0, limit
            )
            
            # Group by license plate/trailer and select latest per plate
            # Similar to GetUnprocessedScansAsync logic
            unique_latest_per_plate = {}
            
            for record in records:
                plate = record.get('licence_plate_trailer', '')
                if not plate:
                    continue
                
                # Keep only the latest record per plate
                if plate not in unique_latest_per_plate:
                    unique_latest_per_plate[plate] = record
                else:
                    # Compare timestamps to keep the latest
                    current_time = datetime.fromisoformat(record.get('created_on', ''))
                    existing_time = datetime.fromisoformat(
                        unique_latest_per_plate[plate].get('created_on', '')
                    )
                    if current_time > existing_time:
                        unique_latest_per_plate[plate] = record
            
            # Convert to list and sort by ID
            result = sorted(unique_latest_per_plate.values(), key=lambda x: x['id'])
            
            # Limit to requested number
            out = result[:limit]
            self.logger.info(
                "Data processor: %d unique plate(s), returning %d record(s) for processing",
                len(unique_latest_per_plate), len(out)
            )
            return out
            
        except Exception as e:
            self.logger.error("Data processor: error retrieving unprocessed records: %s", e, exc_info=True)
            return []
    
    def is_valid_plate(self, plate: str) -> bool:
        """
        Check if license plate/trailer ID is valid.
        Python equivalent of IsValidRfidTag.
        
        Args:
            plate: License plate or trailer ID string
            
        Returns:
            True if valid
        """
        return plate and len(plate) > 0  # Simplified - adjust based on requirements
    
    def is_valid_speed(self, speed: Optional[float], min_speed: float, max_speed: float) -> bool:
        """
        Check if speed is within valid range.
        Python equivalent of IsValidSpeed.
        
        Args:
            speed: Speed value (can be None)
            min_speed: Minimum valid speed
            max_speed: Maximum valid speed
            
        Returns:
            True if speed is valid
        """
        if speed is None:
            return False
        return min_speed <= speed <= max_speed
    
    def distance_in_feet(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two GPS coordinates in feet.
        Python equivalent of GeofenceUtils.DistanceInFeet.
        
        Args:
            lat1, lon1: First GPS coordinate
            lat2, lon2: Second GPS coordinate
            
        Returns:
            Distance in feet
        """
        # Haversine formula to calculate distance in meters, then convert to feet
        R = 6371000  # Earth radius in meters
        
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_phi / 2) ** 2 +
             math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        distance_meters = R * c
        distance_feet = distance_meters * 3.28084  # Convert meters to feet
        
        return distance_feet
    
    def get_closest_parking_spots(
        self,
        record: Dict,
        parking_spots: List[Dict],
        max_slot_distance: float
    ) -> List[Dict]:
        """
        Get closest parking spots within max distance.
        
        Args:
            record: Video frame record with GPS coordinates
            parking_spots: List of parking spot dictionaries with 'latitude' and 'longitude'
            max_slot_distance: Maximum distance in feet
            
        Returns:
            List of parking spots sorted by distance (closest first)
        """
        record_lat = record.get('latitude')
        record_lon = record.get('longitude')
        
        if record_lat is None or record_lon is None:
            return []
        
        valid_spots = []
        
        for spot in parking_spots:
            spot_lat = spot.get('latitude')
            spot_lon = spot.get('longitude')
            
            if spot_lat is None or spot_lon is None:
                continue
            
            distance = self.distance_in_feet(record_lat, record_lon, spot_lat, spot_lon)
            
            if distance < max_slot_distance:
                valid_spots.append({
                    **spot,
                    'distance': distance
                })
        
        # Sort by distance
        valid_spots.sort(key=lambda x: x['distance'])
        
        return valid_spots
    
    def _nearest_spot_distance_and_name(
        self, record: Dict, parking_spots: List[Dict]
    ) -> Optional[tuple]:
        """Return (distance_ft, spot_name) for the single nearest spot, or None if no coords."""
        record_lat = record.get('latitude')
        record_lon = record.get('longitude')
        if record_lat is None or record_lon is None:
            return None
        nearest_dist = None
        nearest_name = None
        for spot in parking_spots:
            spot_lat = spot.get('latitude')
            spot_lon = spot.get('longitude')
            if spot_lat is None or spot_lon is None:
                continue
            d = self.distance_in_feet(record_lat, record_lon, spot_lat, spot_lon)
            if nearest_dist is None or d < nearest_dist:
                nearest_dist = d
                nearest_name = spot.get('name', 'Unknown')
        if nearest_dist is None:
            return None
        return (nearest_dist, nearest_name)
    
    def process_record(
        self,
        record: Dict,
        parking_spots: List[Dict]
    ) -> Optional[Dict]:
        """
        Process a single video frame record.
        Python equivalent of ProcessIndividualScanAsync.
        
        Args:
            record: Video frame record from database
            parking_spots: List of parking spots to match against
            
        Returns:
            Processing result dict or None if record should be ignored
        """
        plate = record.get('licence_plate_trailer', '')
        record_id = record.get('id')
        lat = record.get('latitude')
        lon = record.get('longitude')
        speed = record.get('speed')
        max_dist = self.config['max_slot_distance']
        
        self.logger.info(
            "Data processor: [record %s] plate=%r lat=%s lon=%s speed=%s (max_slot_distance=%.0f ft)",
            record_id, plate, lat, lon, speed, max_dist
        )
        
        # Validate license plate (do not mark as processed; leave for retry so DB gets only proper values)
        if not self.is_valid_plate(plate):
            self.logger.info(
                "Data processor: [record %s] SKIP - invalid or empty plate (plate=%r)",
                record_id, plate
            )
            return None
        
        # Validate speed
        if not self.is_valid_speed(speed, self.config['min_speed'], self.config['max_speed']):
            self.logger.info(
                "Data processor: [record %s] SKIP - speed out of range (speed=%s, valid range %.1f–%.1f)",
                record_id, speed, self.config['min_speed'], self.config['max_speed']
            )
            return None
        
        # Check if already processed (same plate in this run)
        if plate in self._processed_plates:
            self.logger.info(
                "Data processor: [record %s] SKIP - plate %r already assigned in this run",
                record_id, plate
            )
            return None
        
        # Get closest parking spots
        closest_spots = self.get_closest_parking_spots(
            record,
            parking_spots,
            max_dist
        )
        
        # Limit to max_slots
        closest_spots = closest_spots[:self.config['max_slots']]
        
        # Only write to DB when we can assign a spot; otherwise leave unprocessed for retry
        if not closest_spots:
            nearest = self._nearest_spot_distance_and_name(record, parking_spots)
            if nearest is not None:
                dist_ft, spot_name = nearest
                self.logger.info(
                    "Data processor: [record %s] SKIP - no spot within %.0f ft; nearest spot %r is %.1f ft away",
                    record_id, max_dist, spot_name, dist_ft
                )
            else:
                self.logger.info(
                    "Data processor: [record %s] SKIP - no spot within %.0f ft; record has no valid lat/lon or no spots have coords",
                    record_id, max_dist
                )
            return None
        
        # Get nearest parking spot
        nearest_spot = closest_spots[0]
        distance_ft = nearest_spot['distance']
        
        result = {
            'record_id': record['id'],
            'plate': plate,
            'nearest_spot': nearest_spot,
            'distance': distance_ft,
            'all_valid_spots': closest_spots
        }
        
        # Mark as processed and store assigned spot and distance in DB
        spot_name = nearest_spot.get('name', 'Unknown')
        self.db.mark_as_processed(
            record['id'],
            comment=f"Location Updated: {spot_name}",
            assigned_spot_id=nearest_spot.get('id'),
            assigned_spot_name=nearest_spot.get('name'),
            assigned_distance_ft=distance_ft,
        )
        self._processed_plates.add(plate)
        self.logger.info(
            "Data processor: record %s -> plate %s -> spot %s (%.1f ft)",
            record_id, plate, spot_name, distance_ft
        )
        return result
    
    def process_all_unprocessed(
        self,
        parking_spots: List[Dict],
        limit: int = 50,
        camera_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Process all unprocessed records.
        
        Args:
            parking_spots: List of parking spots
            limit: Maximum number of records to process
            camera_id: Optional camera filter
            
        Returns:
            List of processing results
        """
        self.logger.info(
            "Data processor: process_all_unprocessed starting (limit=%d, parking_spots=%d)",
            limit, len(parking_spots)
        )
        # Get unprocessed records
        records = self.get_latest_unprocessed_scans(
            limit=limit,
            cutoff_seconds=10,
            camera_id=camera_id
        )
        if not records:
            self.logger.info("Data processor: no unprocessed records to process")
            return []
        
        results = []
        
        for record in records:
            try:
                result = self.process_record(record, parking_spots)
                if result:
                    results.append(result)
            except Exception as e:
                self.logger.error("Data processor: error processing record %s: %s", record.get('id'), e, exc_info=True)
                # Do not mark as processed; leave record for retry so DB gets only proper values
        
        self.logger.info("Data processor: process_all_unprocessed done, %d result(s) from %d record(s)", len(results), len(records))
        return results
