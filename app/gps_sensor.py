"""
GPS Sensor Interface Module

Handles real-time GPS data from camera-mounted GPS sensor.
Supports common GPS protocols: NMEA, UART, USB GPS dongles.
"""

import serial
import threading
from typing import Optional, Dict
from datetime import datetime
import time
import json
from pathlib import Path


class GPSSensor:
    """
    GPS sensor interface for camera-mounted GPS devices.
    Supports NMEA 0183 protocol (common standard).
    """
    
    def __init__(self, port: str = None, baudrate: int = 4800):
        """
        Initialize GPS sensor.
        
        Args:
            port: Serial port (e.g., '/dev/ttyUSB0' on Linux, 'COM3' on Windows)
            baudrate: Serial baudrate (typically 4800 for GPS, some use 9600 or 38400)
        """
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.current_gps = None  # {'lat': float, 'lon': float, 'timestamp': datetime}
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        
        # GPS history for interpolation
        self.gps_history = []  # List of (timestamp, lat, lon)
        self.max_history = 10
        
        # Accuracy improvement: Store recent readings for averaging
        self.recent_readings = []  # List of {'lat', 'lon', 'fix_quality', 'hdop', 'timestamp'}
        self.max_recent_readings = 120  # Average over last 120 readings (~2 minutes at 1Hz) - maximum for best accuracy
        self.min_readings_for_average = 20  # Minimum readings needed for averaging - increased for better accuracy
        
        # Calibration offset (can be set to correct systematic bias)
        self.calibration_offset_lat = 0.0
        self.calibration_offset_lon = 0.0
    
    def start(self):
        """Start GPS reading thread."""
        if self.port:
            try:
                self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
                self.running = True
                self.thread = threading.Thread(target=self._read_gps_loop, daemon=True)
                self.thread.start()
                print(f"[GPSSensor] Started GPS sensor on {self.port}")
            except Exception as e:
                print(f"[GPSSensor] Failed to open GPS port {self.port}: {e}")
                print(f"[GPSSensor] GPS sensor will be disabled")
                self.serial_conn = None
        else:
            print(f"[GPSSensor] No GPS port specified, GPS sensor disabled")
    
    def _convert_to_decimal(self, coord, direction, is_latitude=True):
        """
        Convert NMEA format (DDMM.MMMM) to decimal degrees.
        Matches the old implementation's coordinate conversion.
        """
        if not coord or not direction:
            return 0.0
        
        try:
            # NMEA format: DDMM.MMMM (degrees and decimal minutes)
            coord_float = float(coord)
            if is_latitude:
                degrees = int(coord_float / 100)
                minutes = coord_float % 100
            else:
                degrees = int(coord_float / 100)
                minutes = coord_float % 100
            
            decimal = degrees + (minutes / 60.0)
            
            # Apply direction (N/S for lat, E/W for lon)
            if direction.upper() in ['S', 'W']:
                decimal = -decimal
            
            return decimal
        except (ValueError, TypeError):
            return 0.0
    
    def _read_gps_loop(self):
        """Background thread to continuously read GPS data."""
        try:
            import pynmea2
        except ImportError:
            print(f"[GPSSensor] Warning: pynmea2 not installed. Install with: pip install pynmea2")
            print(f"[GPSSensor] GPS sensor will use mock data for testing")
            self._read_gps_loop_mock()
            return
        
        while self.running:
            try:
                if self.serial_conn:
                    # Wait for enough data (like old code - ensures complete sentence)
                    buffer = self.serial_conn.in_waiting
                    if buffer < 80:
                        time.sleep(0.2)
                        continue
                    
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    
                    # Only use RMC sentences (like old code - simpler and more direct)
                    if line.startswith('$GPRMC') or line.startswith('$GNRMC'):
                        try:
                            msg = pynmea2.parse(line)
                            
                            # Extract all fields into dictionary (like old code)
                            data = {}
                            for field in msg.fields:
                                label, attr = field[:2]
                                try:
                                    value = getattr(msg, attr, None)
                                    data[attr] = value
                                except AttributeError:
                                    pass
                            
                            # Convert coordinates using manual conversion (like old code)
                            if 'lat' in data and 'lat_dir' in data and 'lon' in data and 'lon_dir' in data:
                                lat = self._convert_to_decimal(data['lat'], data['lat_dir'], is_latitude=True)
                                lon = self._convert_to_decimal(data['lon'], data['lon_dir'], is_latitude=False)
                                
                                if lat != 0.0 and lon != 0.0:
                                    with self.lock:
                                        # Extract heading and speed from RMC sentence
                                        heading = None
                                        speed = None
                                        
                                        # Get true course (heading) in degrees
                                        if 'true_course' in data and data['true_course'] is not None:
                                            try:
                                                heading = float(data['true_course'])
                                            except (ValueError, TypeError):
                                                heading = None
                                        
                                        # Get speed over ground in knots, convert to mph
                                        if 'spd_over_grnd' in data and data['spd_over_grnd'] is not None:
                                            try:
                                                speed_knots = float(data['spd_over_grnd'])
                                                speed = speed_knots * 1.15078  # Convert knots to mph
                                            except (ValueError, TypeError):
                                                speed = None
                                        
                                        gps_data = {
                                            'lat': lat,
                                            'lon': lon,
                                            'timestamp': datetime.utcnow()
                                        }
                                        
                                        # Add heading and speed if available
                                        if heading is not None:
                                            gps_data['heading'] = heading
                                        if speed is not None:
                                            gps_data['speed'] = speed
                                        
                                        self.current_gps = gps_data
                                        
                                        # Add to history for interpolation
                                        self.gps_history.append(gps_data.copy())
                                        if len(self.gps_history) > self.max_history:
                                            self.gps_history.pop(0)  # Remove oldest
                        except (pynmea2.ParseError, ValueError, AttributeError, KeyError) as e:
                            # Skip invalid sentences
                            pass
                    elif line.startswith('$G'):
                        # Log other GPS sentences for debugging (like old code)
                        pass
            except Exception as e:
                # Silently handle errors (GPS might temporarily lose signal)
                time.sleep(0.1)
        
        if self.serial_conn:
            self.serial_conn.close()
    
    def _read_gps_loop_mock(self):
        """Mock GPS reading for testing (when pynmea2 not available)."""
        # Use a fixed test location
        test_lat = 41.911641
        test_lon = -89.044685
        
        while self.running:
            with self.lock:
                self.current_gps = {
                    'lat': test_lat,
                    'lon': test_lon,
                    'timestamp': datetime.utcnow()
                }
                self.gps_history.append(self.current_gps.copy())
                if len(self.gps_history) > self.max_history:
                    self.gps_history.pop(0)
            time.sleep(1.0)  # Update every second
    
    def _recalculate_averaging(self):
        """Recalculate averaged coordinates if we have enough samples."""
        if len(self.recent_readings) >= self.min_readings_for_average:
            # Filter out poor quality readings
            good_readings = []
            for r in self.recent_readings:
                fix_qual = r.get('fix_quality')
                hdop_val = r.get('hdop')
                
                # Handle type conversion for fix_quality
                if fix_qual is not None:
                    try:
                        fix_qual = float(fix_qual) if isinstance(fix_qual, str) else fix_qual
                    except (ValueError, TypeError):
                        fix_qual = None
                
                # Handle type conversion for HDOP
                if hdop_val is not None:
                    try:
                        hdop_val = float(hdop_val) if isinstance(hdop_val, str) else hdop_val
                    except (ValueError, TypeError):
                        hdop_val = None
                
                # Accept if fix_quality is None/missing OR >= 1
                fix_ok = (fix_qual is None) or (fix_qual is not None and fix_qual >= 1)
                # Accept if HDOP is None/missing OR < 10
                hdop_ok = (hdop_val is None) or (hdop_val is not None and hdop_val < 10)
                
                if fix_ok and hdop_ok:
                    good_readings.append(r)
            
            if len(good_readings) >= 3 and self.current_gps:
                # Remove outliers using IQR (Interquartile Range) method
                if len(good_readings) >= 5:
                    lats = [r['lat'] for r in good_readings]
                    lons = [r['lon'] for r in good_readings]
                    
                    # Calculate quartiles
                    lats_sorted = sorted(lats)
                    lons_sorted = sorted(lons)
                    q1_lat_idx = len(lats_sorted) // 4
                    q3_lat_idx = (3 * len(lats_sorted)) // 4
                    q1_lon_idx = len(lons_sorted) // 4
                    q3_lon_idx = (3 * len(lons_sorted)) // 4
                    
                    q1_lat = lats_sorted[q1_lat_idx]
                    q3_lat = lats_sorted[q3_lat_idx]
                    q1_lon = lons_sorted[q1_lon_idx]
                    q3_lon = lons_sorted[q3_lon_idx]
                    
                    iqr_lat = q3_lat - q1_lat
                    iqr_lon = q3_lon - q1_lon
                    
                    # Filter outliers (keep readings within 1.5 * IQR)
                    filtered_readings = []
                    for r in good_readings:
                        lat_ok = (q1_lat - 1.5 * iqr_lat) <= r['lat'] <= (q3_lat + 1.5 * iqr_lat)
                        lon_ok = (q1_lon - 1.5 * iqr_lon) <= r['lon'] <= (q3_lon + 1.5 * iqr_lon)
                        if lat_ok and lon_ok:
                            filtered_readings.append(r)
                    
                    if len(filtered_readings) >= 3:
                        good_readings = filtered_readings
                
                # Calculate weighted average based on HDOP (lower HDOP = higher weight)
                # HDOP < 1.0 = weight 3, < 2.0 = weight 2, < 5.0 = weight 1, else weight 0.5
                total_weight = 0.0
                weighted_lat = 0.0
                weighted_lon = 0.0
                
                for r in good_readings:
                    hdop = r.get('hdop')
                    if hdop is not None:
                        try:
                            hdop_val = float(hdop) if isinstance(hdop, str) else hdop
                            if hdop_val < 1.0:
                                weight = 3.0
                            elif hdop_val < 2.0:
                                weight = 2.0
                            elif hdop_val < 5.0:
                                weight = 1.0
                            else:
                                weight = 0.5
                        except (ValueError, TypeError):
                            weight = 1.0
                    else:
                        weight = 1.0
                    
                    weighted_lat += r['lat'] * weight
                    weighted_lon += r['lon'] * weight
                    total_weight += weight
                
                if total_weight > 0:
                    avg_lat = weighted_lat / total_weight
                    avg_lon = weighted_lon / total_weight
                else:
                    # Fallback to simple mean if no weights
                    avg_lat = sum(r['lat'] for r in good_readings) / len(good_readings)
                    avg_lon = sum(r['lon'] for r in good_readings) / len(good_readings)
                
                # Apply calibration offset if set
                avg_lat += self.calibration_offset_lat
                avg_lon += self.calibration_offset_lon
                
                # Update current GPS with averaged coordinates
                self.current_gps['lat'] = avg_lat
                self.current_gps['lon'] = avg_lon
                self.current_gps['averaged'] = True
                self.current_gps['samples_used'] = len(good_readings)
                return True
        return False
    
    def set_calibration_offset(self, lat_offset: float, lon_offset: float):
        """
        Set calibration offset to correct systematic GPS bias.
        
        This can be used to compensate for known GPS device errors.
        Calculate offset by comparing GPS reading to known accurate location.
        
        Args:
            lat_offset: Latitude offset in degrees (positive = north)
            lon_offset: Longitude offset in degrees (positive = east)
        """
        with self.lock:
            self.calibration_offset_lat = lat_offset
            self.calibration_offset_lon = lon_offset
    
    def get_current_gps(self) -> Optional[Dict[str, float]]:
        """
        Get current camera GPS coordinates.
        Matches old implementation - returns raw coordinates immediately.
        
        Returns:
            Dict with 'lat', 'lon', 'heading' (optional), and 'speed' (optional) keys, or None if not available
        """
        with self.lock:
            if self.current_gps:
                result = {
                    'lat': self.current_gps['lat'],
                    'lon': self.current_gps['lon'],
                    'timestamp': self.current_gps['timestamp']
                }
                # Include heading and speed if available
                if 'heading' in self.current_gps:
                    result['heading'] = self.current_gps['heading']
                if 'speed' in self.current_gps:
                    result['speed'] = self.current_gps['speed']
                return result
            return None
    
    def get_gps_at_timestamp(self, target_timestamp: datetime) -> Optional[Dict[str, float]]:
        """
        Get GPS coordinate at a specific timestamp (for matching with video frames).
        Uses interpolation if GPS history is available.
        
        Args:
            target_timestamp: Target timestamp to get GPS for
            
        Returns:
            Dict with 'lat', 'lon', 'heading' (optional), and 'speed' (optional) keys, or None if not available
        """
        with self.lock:
            if not self.gps_history:
                # No history, return current GPS
                if self.current_gps:
                    result = {
                        'lat': self.current_gps['lat'],
                        'lon': self.current_gps['lon'],
                        'timestamp': target_timestamp
                    }
                    if 'heading' in self.current_gps:
                        result['heading'] = self.current_gps['heading']
                    if 'speed' in self.current_gps:
                        result['speed'] = self.current_gps['speed']
                    return result
                return None
            
            # Find closest GPS reading
            closest = min(self.gps_history, 
                         key=lambda x: abs((x['timestamp'] - target_timestamp).total_seconds()))
            
            result = {
                'lat': closest['lat'],
                'lon': closest['lon'],
                'timestamp': target_timestamp
            }
            if 'heading' in closest:
                result['heading'] = closest['heading']
            if 'speed' in closest:
                result['speed'] = closest['speed']
            return result
    
    def stop(self):
        """Stop GPS reading."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.serial_conn:
            self.serial_conn.close()
            self.serial_conn = None
        print(f"[GPSSensor] GPS sensor stopped")


def load_gps_log(gps_log_path: str) -> Dict[str, Dict]:
    """
    Load GPS log file created during video recording.
    
    Args:
        gps_log_path: Path to GPS log JSON file
        
    Returns:
        Dict mapping timestamp strings to GPS coordinates
    """
    try:
        with open(gps_log_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[GPSSensor] GPS log file not found: {gps_log_path}")
        return {}
    except Exception as e:
        print(f"[GPSSensor] Error loading GPS log: {e}")
        return {}


def get_gps_for_timestamp(gps_log: Dict[str, Dict], target_timestamp: datetime, 
                         tolerance_seconds: float = 0.1) -> Optional[Dict[str, float]]:
    """
    Get GPS coordinate for a specific timestamp from GPS log.
    Finds the closest GPS reading within tolerance.
    
    Args:
        gps_log: GPS log dict from load_gps_log()
        target_timestamp: Target timestamp
        tolerance_seconds: Maximum time difference in seconds (default: 2.0)
        
    Returns:
        Dict with 'lat', 'lon', 'heading' (optional), and 'speed' (optional) keys, or None if not found
    """
    if not gps_log or len(gps_log) == 0:
        return None
    
    target_ts = target_timestamp
    
    # Find closest timestamp
    closest_key = None
    min_diff = float('inf')
    
    for ts_str in gps_log.keys():
        try:
            # Handle both with and without 'Z' suffix
            ts_str_clean = ts_str.replace('Z', '+00:00')
            ts = datetime.fromisoformat(ts_str_clean)
            diff = abs((ts - target_ts).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest_key = ts_str
        except (ValueError, TypeError) as e:
            # Skip invalid timestamps
            continue
    
    if closest_key and min_diff <= tolerance_seconds:
        gps_data = gps_log[closest_key]
        if gps_data and 'lat' in gps_data and 'lon' in gps_data:
            result = {
                'lat': gps_data.get('lat'),
                'lon': gps_data.get('lon'),
                'timestamp': datetime.fromisoformat(closest_key.replace('Z', '+00:00'))
            }
            # Include heading and speed if available in log
            if 'heading' in gps_data:
                result['heading'] = gps_data.get('heading')
            if 'speed' in gps_data:
                result['speed'] = gps_data.get('speed')
            return result
    
    return None
