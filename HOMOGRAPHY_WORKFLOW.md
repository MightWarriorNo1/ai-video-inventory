# Homography Workflow - Complete End-to-End Guide

## Executive Summary

Homography is a mathematical transformation that maps image pixel coordinates to real-world GPS coordinates. This enables the system to determine the exact geographic location of detected trailers, allowing integration with mapping systems, yard management, and location-based services.

**Key Value Proposition:**
- Convert camera image coordinates → GPS coordinates (lat/lon)
- Works directly with Google Maps and other mapping systems
- Enables precise trailer location tracking in real-world coordinates
- Supports yard management and spatial analytics
- One-time calibration per camera, reusable indefinitely

---

## Table of Contents

1. [What is Homography?](#1-what-is-homography)
2. [Calibration Process](#2-calibration-process)
3. [How to Get GPS Coordinates](#3-how-to-get-gps-coordinates)
4. [System Integration](#4-system-integration)
5. [Coordinate Transformation Process](#5-coordinate-transformation-process)
6. [Output Format](#6-output-format)
7. [Using GPS Coordinates](#7-using-gps-coordinates)
8. [Accuracy and Limitations](#8-accuracy-and-limitations)
9. [Testing and Verification](#9-testing-and-verification)
10. [Maintenance and Updates](#10-maintenance-and-updates)
11. [Technical Architecture](#11-technical-architecture)
12. [Business Value](#12-business-value)

---

## 1. What is Homography?

### 1.1 Definition

Homography is a 3×3 transformation matrix that projects points from a 2D image plane to a 2D world plane (ground plane). In our system, this world plane is mapped to GPS coordinates.

### 1.2 Mathematical Overview

```
Image Point (x_img, y_img) → Homography Matrix (H) → World Point (x_world, y_world) → GPS (lat, lon)
```

**Transformation Flow:**
1. **Image Coordinates:** Pixel position in camera image (e.g., 1150, 783)
2. **Homography Matrix:** 3×3 transformation matrix (calculated during calibration)
3. **World Coordinates:** Local meters relative to reference point (e.g., 0.0, 0.0)
4. **GPS Coordinates:** Geographic coordinates (e.g., 41.91164053, -89.04468542)

### 1.3 Why We Use It

- **Camera Perspective Correction:** Accounts for camera angle, tilt, and distortion
- **Coordinate Transformation:** Converts pixel positions to real-world locations
- **GPS Integration:** Enables direct mapping to geographic coordinates
- **Yard Management:** Supports parking spot resolution and spatial queries
- **Automation:** Eliminates manual location tracking

### 1.4 How It Works

The homography matrix (H) transforms a point from image space to world space:

```
[x']     [h11  h12  h13]   [x]
[y']  =  [h21  h22  h23] × [y]
[w']     [h31  h32  h33]   [1]
```

Where:
- `(x, y)` = image coordinates
- `(x', y')` = world coordinates (normalized)
- `H` = 3×3 homography matrix

---

## 2. Calibration Process

### 2.1 Prerequisites

**Required:**
- Clear calibration image from camera showing yard area
- At least 4 distinct, identifiable landmark points
- GPS coordinates for each landmark point
- Access to Google Maps or GPS device

**Recommended:**
- 6-8 calibration points for better accuracy
- Points well-distributed across the image
- Points on the ground plane (same elevation)
- Good lighting and visibility

### 2.2 Step-by-Step Calibration

#### Step 1: Capture Calibration Image

1. **Capture a clear image** from the camera showing the yard
2. **Ensure good lighting** and visibility of landmarks
3. **Image should show** the area where trailers will be detected
4. **Save the image** in a known location

**Best Practices:**
- Use a frame when yard is relatively clear (fewer trailers blocking landmarks)
- Ensure image is sharp and in focus
- Capture during daylight for best visibility
- Include the full area of interest

#### Step 2: Identify Landmark Points

Choose 4+ distinct points such as:

**Ground-Level Points (Best):**
- Corners of parking spaces
- Intersections of painted lines (yellow lines)
- Fixed building features where they meet ground (dock corners, door edges)
- Permanent ground markings

**Building Features (Secondary):**
- Dock door corners (where they meet ground)
- Building corners
- Fixed signage locations

**Points to Avoid:**
- Trailer corners or features (trailers move)
- Shadows (can shift)
- Temporary objects
- Elevated platforms (unless necessary)

**Best Practices:**
- Use points on the ground plane (same elevation)
- Spread points across the image (not clustered)
- Choose points that are clearly visible and permanent
- Mix horizontal and vertical distribution

#### Step 3: Get GPS Coordinates

See [Section 3: How to Get GPS Coordinates](#3-how-to-get-gps-coordinates) for detailed methods.

**Quick Summary:**
1. Identify physical location of each landmark point
2. Use Google Maps to get GPS coordinates
3. Record coordinates for each point
4. Verify coordinates are accurate

#### Step 4: Run Calibration Tool

**Command:**
```bash
python tools/calibrate_h.py --image path/to/calibration_image.jpg --save config/calib/camera-01_h.json
```

**Replace:**
- `path/to/calibration_image.jpg` with your actual image path
- `camera-01` with your camera ID (must match `config/cameras.yaml`)

**Interactive Process:**

1. **Image window opens** showing calibration image
2. **Click on first landmark point** in the image
3. **Enter GPS latitude** when prompted (e.g., `41.91164053`)
4. **Enter GPS longitude** when prompted (e.g., `-89.04468542`)
5. **First point becomes reference origin** (0,0 in local meters)
6. **Repeat for remaining points** (minimum 4 total)
7. **Press 'q'** to complete calibration

**Example Terminal Output:**
```
=== Homography Calibration ===
Instructions:
1. Click 4+ landmark points on the image
2. Enter corresponding GPS coordinates (latitude, longitude)
   Example: Latitude: 40.7128, Longitude: -74.0060
   Tip: Use Google Maps to get GPS coordinates for each point
3. Press 'q' when done (minimum 4 points required)

NOTE: The first GPS point will be used as the reference origin (0,0)
      All other points will be converted relative to this reference.

Point 1: Image (1150, 783)
  Enter GPS Latitude (degrees, e.g., 40.7128): 41.91164053
  Enter GPS Longitude (degrees, e.g., -74.0060): -89.04468542
  Reference point set: (41.91164053, -89.04468542)
  GPS (41.911641, -89.044685) -> Local (0.00m, 0.00m)

Point 2: Image (1256, 919)
  Enter GPS Latitude (degrees, e.g., 40.7128): 41.91182291
  Enter GPS Longitude (degrees, e.g., -74.0060): -89.04469387
  GPS (41.911823, -89.044694) -> Local (-0.70m, 20.30m)

Point 3: Image (1595, 915)
  Enter GPS Latitude (degrees, e.g., 40.7128): 41.91182363
  Enter GPS Longitude (degrees, e.g., -74.0060): -89.04464264
  GPS (41.911824, -89.044643) -> Local (3.54m, 20.38m)

Point 4: Image (1385, 797)
  Enter GPS Latitude (degrees, e.g., 40.7128): 41.91164119
  Enter GPS Longitude (degrees, e.g., -74.0060): -89.04463401
  GPS (41.911641, -89.044634) -> Local (4.26m, 0.07m)

[q pressed]

Calibration saved to: config/calib/camera-01_h.json
RMSE: 2.56e-15 meters
GPS Reference: (41.911641, -89.044685)
```

**Important Notes:**
- The **first GPS point** becomes the reference origin (0,0 in local meters)
- All other points are converted relative to this reference
- Make sure your GPS coordinates are accurate (within 1-2 meters)
- More points = better accuracy (aim for 6-8 points)

#### Step 5: Verify Calibration Quality

**Check RMSE (Root Mean Square Error):**
- **Excellent:** RMSE < 1.0 meters
- **Good:** RMSE 1.0 - 1.5 meters
- **Acceptable:** RMSE 1.5 - 2.5 meters
- **Poor:** RMSE > 2.5 meters (recalibrate with more points)

**Calibration File Generated:**
The tool creates a JSON file at `config/calib/{camera_id}_h.json`:

```json
{
  "H": [
    [-0.00747, 0.00952, 1.13949],
    [0.00657, -0.11225, 80.33627],
    [0.00037, -0.00237, 1.0]
  ],
  "rmse": 2.56e-15,
  "image_points": [
    [1150, 783],
    [1256, 919],
    [1595, 915],
    [1385, 797]
  ],
  "world_points": [
    [0.0, 0.0],
    [-0.70, 20.30],
    [3.54, 20.38],
    [4.26, 0.07]
  ],
  "gps_points": [
    [41.91164053, -89.04468542],
    [41.91182291, -89.04469387],
    [41.91182363, -89.04464264],
    [41.91164119, -89.04463401]
  ],
  "gps_reference": {
    "lat": 41.91164053,
    "lon": -89.04468542,
    "description": "Reference GPS point corresponding to homography origin (0,0)"
  },
  "use_gps": true,
  "image_path": "img_00389.jpg"
}
```

**File Contents Explained:**
- `H`: 3×3 homography matrix (core transformation)
- `rmse`: Calibration error in meters
- `image_points`: Pixel coordinates clicked in image
- `world_points`: Local meter coordinates (relative to reference)
- `gps_points`: Original GPS coordinates entered
- `gps_reference`: Reference point for GPS conversion
- `use_gps`: Flag indicating GPS mode enabled

---

## 3. How to Get GPS Coordinates

This section provides detailed methods for obtaining GPS coordinates for calibration points.

### 3.1 Method 1: Google Maps Web (Recommended)

**Best for:** Desktop/laptop users, precise point selection

**Steps:**

1. **Open Google Maps** in your web browser
   - Go to: https://www.google.com/maps

2. **Navigate to your yard location**
   - Search for your facility address
   - Or drag/zoom to your location

3. **Switch to Satellite View** (for better accuracy)
   - Click "Satellite" button in bottom-left corner
   - This shows actual ground features

4. **Right-click on landmark point**
   - Right-click exactly on the point you want coordinates for
   - A popup menu appears

5. **Click the coordinates** at the top of the popup
   - Coordinates appear as: `41.91164053, -89.04468542`
   - Clicking them copies to clipboard

6. **Copy the coordinates**
   - Format: `latitude, longitude` (decimal degrees)
   - Example: `41.91164053, -89.04468542`

**Tips:**
- Use satellite view for precise point selection
- Zoom in for better accuracy
- Verify you're clicking the exact physical location
- Double-check coordinates match the landmark

**Visual Guide:**
```
Google Maps → Satellite View → Right-click point → Click coordinates → Copy
```

### 3.2 Method 2: Google Maps Mobile App

**Best for:** On-site measurement, using phone GPS

**Steps:**

1. **Open Google Maps** on your phone
   - iOS: App Store
   - Android: Google Play Store

2. **Navigate to your yard location**
   - Search for address or use current location

3. **Enable Satellite View**
   - Tap layers icon (usually bottom-right)
   - Select "Satellite"

4. **Long-press on landmark point**
   - Press and hold on the exact point
   - A red pin appears

5. **Tap the coordinates** at the bottom
   - Coordinates appear: `41.91164053, -89.04468542`
   - Tap to copy or view details

6. **Copy coordinates**
   - Share or copy the coordinates
   - Format: Decimal degrees

**Advantages:**
- Can be done on-site at physical location
- Uses phone GPS for current location
- Easy to verify physical point matches map point

**Tips:**
- Ensure GPS is enabled on phone
- Wait for GPS lock (location accuracy indicator)
- Use satellite view for better visual matching

### 3.3 Method 3: GPS Device or Survey Tool

**Best for:** High accuracy requirements, professional surveys

**Steps:**

1. **Go to physical location** of each landmark point
2. **Use GPS device** to record coordinates
3. **Wait for GPS lock** (accuracy indicator)
4. **Record coordinates** for each point
5. **Verify coordinate system** is WGS84 (decimal degrees)

**GPS Devices:**
- Handheld GPS units (Garmin, Magellan)
- Survey-grade GPS (for high accuracy)
- Smartphone GPS apps (GPS Status, GPS Essentials)

**Coordinate Format:**
- Must be decimal degrees: `41.91164053, -89.04468542`
- NOT degrees/minutes/seconds: `41°54'41.9"N 89°02'40.9"W`

**Conversion if needed:**
- Degrees/minutes/seconds → Decimal degrees:
  - `DD = D + M/60 + S/3600`
  - Example: `41°54'41.9"N` = `41 + 54/60 + 41.9/3600` = `41.911639`

### 3.4 Method 4: Online Coordinate Tools

**Best for:** Quick lookups, verification

**Tools:**
- **Google Maps:** Right-click → coordinates
- **OpenStreetMap:** Right-click → "Show address"
- **LatLong.net:** Interactive coordinate finder
- **GPS Coordinates:** https://www.gps-coordinates.net/

**Steps (LatLong.net example):**
1. Go to https://www.latlong.net/
2. Search for your location
3. Click on map to set point
4. Copy coordinates displayed

### 3.5 Method 5: Existing Survey Data

**Best for:** Facilities with existing surveys

**Steps:**
1. **Obtain survey data** from facility records
2. **Verify coordinate system** is WGS84
3. **Convert if needed** to decimal degrees
4. **Match survey points** to calibration landmarks
5. **Use coordinates** directly in calibration

**Coordinate Systems:**
- **WGS84:** Standard GPS system (what we use)
- **UTM:** May need conversion
- **State Plane:** May need conversion

### 3.6 GPS Coordinate Format

**Required Format: Decimal Degrees**
```
Latitude:  41.91164053  (range: -90 to 90)
Longitude: -89.04468542 (range: -180 to 180)
```

**Format Examples:**
- ✅ Correct: `41.91164053, -89.04468542`
- ✅ Correct: `41.9116, -89.0447` (fewer decimals OK)
- ❌ Wrong: `41°54'41.9"N, 89°02'40.9"W` (degrees/minutes/seconds)
- ❌ Wrong: `N41.9116, W89.0447` (with N/S/E/W prefixes)

**Precision:**
- **6 decimal places:** ~0.1 meter accuracy (recommended)
- **4 decimal places:** ~10 meter accuracy (acceptable)
- **More decimals:** Better precision, but GPS accuracy limits benefit

### 3.7 Verifying GPS Coordinates

**Before Using in Calibration:**

1. **Check format:** Decimal degrees (not DMS)
2. **Verify range:**
   - Latitude: -90 to 90
   - Longitude: -180 to 180
3. **Test in Google Maps:**
   - Paste coordinates into search bar
   - Verify location matches physical point
4. **Compare distances:**
   - If you know distance between points, verify GPS coordinates match
   - Example: Two points 20m apart should have GPS coordinates ~20m apart

**Distance Check:**
- Use online distance calculator
- Or calculate: `distance = sqrt((lat2-lat1)² + (lon2-lon1)²) × 111320`
- Should match known physical distance

### 3.8 Common Issues and Solutions

**Problem: Coordinates don't match physical location**
- **Solution:** Verify you're using correct coordinate system (WGS84)
- **Solution:** Check for typos in coordinates
- **Solution:** Ensure you clicked the correct point on map

**Problem: Coordinates format error**
- **Solution:** Convert from degrees/minutes/seconds to decimal degrees
- **Solution:** Remove N/S/E/W prefixes, use negative for S/W

**Problem: Low accuracy coordinates**
- **Solution:** Use satellite view for better precision
- **Solution:** Zoom in closer on map
- **Solution:** Use GPS device at physical location

**Problem: Can't find exact point on map**
- **Solution:** Use satellite view to see physical features
- **Solution:** Use street view to verify location
- **Solution:** Go to physical location and use phone GPS

---

## 4. System Integration

### 4.1 Calibration File Location

**File Structure:**
```
config/
  calib/
    {camera_id}_h.json    # Homography calibration for each camera
```

**Examples:**
- `config/calib/camera-01_h.json`
- `config/calib/lifecam-hd6000-01_h.json`
- `config/calib/dock-camera-west_h.json`

**Naming Convention:**
- File name must match camera ID from `config/cameras.yaml`
- Format: `{camera_id}_h.json`
- Case-sensitive (must match exactly)

**Example cameras.yaml:**
```yaml
cameras:
  - id: camera-01
    rtsp_url: "rtsp://..."
```

**Corresponding calibration file:**
```
config/calib/camera-01_h.json
```

### 4.2 Automatic Loading

The system automatically loads homography when application starts:

**Loading Process:**
```python
# Application startup (app/main_trt_demo.py)
for camera in cameras:
    camera_id = camera['id']
    calib_path = f"config/calib/{camera_id}_h.json"
    
    if os.path.exists(calib_path):
        # Load homography matrix
        with open(calib_path, 'r') as f:
            calib_data = json.load(f)
            homography = np.array(calib_data['H'])
        
        # Load GPS reference if available
        if 'gps_reference' in calib_data:
            gps_ref = calib_data['gps_reference']
            gps_references[camera_id] = {
                'lat': gps_ref['lat'],
                'lon': gps_ref['lon']
            }
```

**Startup Output:**
```
[TrailerVisionApp] Loading configuration...
Loaded homography for camera-01 with GPS reference: (41.911641, -89.044685)
Loaded homography for camera-02 with GPS reference: (41.912000, -89.045000)
Warning: Homography not found for camera-03: config/calib/camera-03_h.json
```

### 4.3 Real-Time Processing Flow

**Complete Pipeline:**
```
Camera Frame (1920x1080)
    ↓
Trailer Detection (YOLO)
    ↓
Bounding Box: (x1, y1, x2, y2)
    ↓
Calculate Center: (x_img, y_img)
    ↓
Homography Projection
    ↓
World Coordinates: (x_meters, y_meters)
    ↓
GPS Conversion (if reference available)
    ↓
GPS Coordinates: (lat, lon)
    ↓
Spot Resolution (GeoJSON)
    ↓
Output Event with GPS
```

**Processing Code:**
```python
# Get bounding box center
center_x = (x1 + x2) / 2.0
center_y = (y1 + y2) / 2.0

# Project to world coordinates
point_img = np.array([[center_x, center_y]], dtype=np.float32)
point_world = cv2.perspectiveTransform(point_img, H)
x_world, y_world = point_world[0][0]

# Convert to GPS
if gps_reference:
    lat, lon = meters_to_gps(x_world, y_world, 
                             gps_ref['lat'], gps_ref['lon'])
```

---

## 5. Coordinate Transformation Process

### 5.1 Image to World Coordinates

**Input:** Image pixel coordinates (x_img, y_img)
**Process:** Apply homography matrix
**Output:** Local world coordinates in meters (x_world, y_world)

**Mathematical Operation:**
```python
# Prepare point
point_img = np.array([[x_img, y_img]], dtype=np.float32)
point_img = np.array([point_img])  # Shape: (1, 1, 2)

# Apply homography
point_world = cv2.perspectiveTransform(point_img, H)

# Extract coordinates
x_world, y_world = point_world[0][0]
```

**Example:**
```
Input:  Image point (1150, 783) pixels
Output: World point (0.0, 0.0) meters
```

### 5.2 World to GPS Coordinates

**Input:** Local meters (x_world, y_world)
**Process:** Convert using GPS reference point
**Output:** GPS coordinates (latitude, longitude)

**Conversion Formula:**
```python
# Constants
METERS_PER_DEGREE_LAT = 111320.0  # Approximately constant
ref_lat_rad = np.radians(ref_lat)

# Convert meters to degrees
lat = ref_lat + (y_meters / METERS_PER_DEGREE_LAT)
lon = ref_lon + (x_meters / (METERS_PER_DEGREE_LAT * np.cos(ref_lat_rad)))
```

**Where:**
- `ref_lat`, `ref_lon`: GPS reference point (first calibration point)
- `111320`: Meters per degree of latitude (constant)
- `cos(ref_lat)`: Accounts for longitude convergence at different latitudes

**Example:**
```
Input:  World point (0.0, 0.0) meters
        Reference: (41.91164053, -89.04468542)
Output: GPS point (41.91164053, -89.04468542)
```

### 5.3 Complete Transformation Example

**Full Pipeline:**
```
Step 1: Image Detection
  Bounding box: (1100, 750, 1200, 850)
  Center: (1150, 800)

Step 2: Homography Projection
  Image point: (1150, 800)
  ↓ [Apply H matrix]
  World point: (2.5, 5.0) meters

Step 3: GPS Conversion
  World point: (2.5, 5.0) meters
  Reference: (41.91164053, -89.04468542)
  ↓ [Convert]
  GPS point: (41.911645, -89.044663)
```

**Code Implementation:**
```python
def project_to_gps(camera_id, x_img, y_img):
    # Get homography and GPS reference
    H = homographies[camera_id]
    gps_ref = gps_references[camera_id]
    
    # Step 1: Image to world
    point_img = np.array([[x_img, y_img]], dtype=np.float32)
    point_img = np.array([point_img])
    point_world = cv2.perspectiveTransform(point_img, H)
    x_world, y_world = point_world[0][0]
    
    # Step 2: World to GPS
    lat, lon = meters_to_gps(x_world, y_world, 
                             gps_ref['lat'], gps_ref['lon'])
    
    return (lat, lon)
```

---

## 6. Output Format

### 6.1 Detection Event Structure

Each detected trailer generates an event with coordinates:

**Complete Event Example:**
```json
{
  "ts_iso": "2024-01-15T10:30:45.123Z",
  "camera_id": "camera-01",
  "track_id": 42,
  "bbox": "1150,783,1250,900",
  "text": "71GG59",
  "conf": 0.95,
  "ocr_method": "original",
  
  // Local coordinates (meters) - for spot resolution
  "x_world": 0.0,
  "y_world": 0.0,
  
  // GPS coordinates (degrees) - for Google Maps, APIs
  "lat": 41.91164053,
  "lon": -89.04468542,
  
  // Parking spot resolution
  "spot": "Dock-44",
  "method": "point-in-polygon"
}
```

### 6.2 Coordinate Fields Explained

| Field | Type | Description | Use Case |
|-------|------|-------------|----------|
| `x_world` | float | Eastward distance in meters from reference | Spot resolution, GeoJSON polygons |
| `y_world` | float | Northward distance in meters from reference | Spot resolution, GeoJSON polygons |
| `lat` | float | GPS latitude in decimal degrees | Google Maps, external APIs, databases |
| `lon` | float | GPS longitude in decimal degrees | Google Maps, external APIs, databases |

**Notes:**
- If GPS reference is not available, `lat` and `lon` will be `None`
- `x_world` and `y_world` are always available if homography exists
- Coordinates are relative to the first calibration point (reference origin)

### 6.3 Output Locations

**Real-Time Events:**
- Published to event bus (Kafka, MQTT, etc.)
- Logged to CSV files
- Sent to REST API endpoints
- Stored in database (TimescaleDB, PostgreSQL)

**Batch Processing:**
- Saved in `combined_results.json`
- Includes all detections with GPS coordinates
- Can be exported for analysis

---

## 7. Using GPS Coordinates

### 7.1 Google Maps Integration

**Method 1: Direct Paste**
1. Copy `lat` and `lon` from detection event
2. Paste into Google Maps search bar: `41.91164053, -89.04468542`
3. Press Enter
4. Google Maps navigates to exact location

**Method 2: URL Format**
```
https://www.google.com/maps?q=41.91164053,-89.04468542
```

**Method 3: Embed in Applications**
```html
<a href="https://www.google.com/maps?q=41.91164053,-89.04468542" target="_blank">
  View Location on Google Maps
</a>
```

### 7.2 API Integration

**REST API Example:**
```python
import requests

# Detection event with GPS
event = {
    "lat": 41.91164053,
    "lon": -89.04468542,
    "text": "71GG59",
    "spot": "Dock-44",
    "timestamp": "2024-01-15T10:30:45Z"
}

# Send to external system
response = requests.post(
    "https://api.example.com/trailers",
    json=event,
    headers={"Content-Type": "application/json"}
)
```

**Webhook Integration:**
```python
# Configure webhook URL
webhook_url = "https://your-system.com/webhook/trailer-detection"

# Send event
requests.post(webhook_url, json=event)
```

### 7.3 Database Storage

**PostgreSQL/TimescaleDB:**
```sql
-- Create table
CREATE TABLE trailer_events (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    camera_id VARCHAR(50),
    track_id INTEGER,
    text VARCHAR(50),
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    spot VARCHAR(50)
);

-- Insert event
INSERT INTO trailer_events (timestamp, camera_id, lat, lon, text, spot)
VALUES ('2024-01-15 10:30:45', 'camera-01', 41.91164053, -89.04468542, '71GG59', 'Dock-44');

-- Spatial queries
SELECT * FROM trailer_events 
WHERE ST_DWithin(
    ST_MakePoint(lon, lat)::geography,
    ST_MakePoint(-89.04468542, 41.91164053)::geography,
    100  -- 100 meters radius
);
```

**MongoDB:**
```javascript
// Insert event
db.trailer_events.insertOne({
    timestamp: ISODate("2024-01-15T10:30:45Z"),
    camera_id: "camera-01",
    lat: 41.91164053,
    lon: -89.04468542,
    text: "71GG59",
    spot: "Dock-44"
});

// Spatial query
db.trailer_events.find({
    location: {
        $near: {
            $geometry: {
                type: "Point",
                coordinates: [-89.04468542, 41.91164053]
            },
            $maxDistance: 100  // 100 meters
        }
    }
});
```

### 7.4 Mapping Libraries

**Leaflet.js:**
```javascript
// Add marker with GPS coordinates
var marker = L.marker([41.91164053, -89.04468542])
    .addTo(map)
    .bindPopup('Trailer: 71GG59<br>Spot: Dock-44');
```

**Google Maps JavaScript API:**
```javascript
// Create marker
var marker = new google.maps.Marker({
    position: {lat: 41.91164053, lng: -89.04468542},
    map: map,
    title: 'Trailer: 71GG59'
});
```

---

## 8. Accuracy and Limitations

### 8.1 Accuracy Factors

**GPS Accuracy:**
- **Consumer GPS:** ±3-5 meters
- **Survey-grade GPS:** ±1-2 meters
- **Google Maps coordinates:** ±3-5 meters
- **Phone GPS:** ±5-10 meters (depending on conditions)

**Homography Accuracy:**
- **Depends on calibration quality** (RMSE)
- **Better with more calibration points** (6-8 vs 4)
- **Affected by camera stability** (movement degrades accuracy)
- **Point distribution matters** (well-spread vs clustered)

**Combined Accuracy:**
- **Expected:** ±5-10 meters total error
- **Best case:** ±3-5 meters (with excellent calibration)
- **Worst case:** ±15 meters (poor calibration or GPS)

### 8.2 Limitations

**Coordinate System:**
- Simple approximation suitable for **< 10 km radius**
- Assumes **flat terrain** (minimal elevation changes)
- Best for **single facility/yard**
- For larger areas, consider UTM or proper map projections

**Elevation:**
- Homography assumes **2D plane** (ground level)
- **Elevated points** (dock platforms) may have reduced accuracy
- Use **ground-level calibration points** when possible
- Elevation differences > 1 meter can cause noticeable errors

**Camera Movement:**
- Calibration **invalid if camera moves**
- **Recalibrate after camera repositioning**
- Consider **camera stabilization/mounting**
- Monitor for camera drift over time

**Perspective:**
- Works best for **overhead or angled views**
- **Extreme angles** may reduce accuracy
- **Wide-angle lenses** may introduce distortion
- Consider lens calibration for high accuracy

### 8.3 Improving Accuracy

**Best Practices:**

1. **Use 8-12 calibration points** (not just 4)
   - More points = better accuracy
   - Allows RANSAC to filter outliers

2. **Choose well-distributed points**
   - Spread across image (not clustered)
   - Mix horizontal and vertical distribution
   - Cover full area of interest

3. **Use accurate GPS measurements**
   - Survey-grade GPS if available
   - Verify coordinates in Google Maps
   - Check distances between points

4. **Calibrate with ground-level points**
   - Avoid elevated platforms
   - Use same elevation for all points
   - Prefer ground markings/intersections

5. **Verify RMSE < 1.5 meters**
   - Lower is better
   - Recalibrate if RMSE > 2.5 meters

6. **Recalibrate if camera moves**
   - Any camera movement invalidates calibration
   - Check periodically for accuracy degradation

7. **Use stable camera mounting**
   - Prevent camera movement/vibration
   - Use fixed mounts, not temporary setups

---

## 9. Testing and Verification

### 9.1 Verification Steps

**Step 1: Check Calibration Loaded**

Run application and check startup logs:
```bash
python app/main_trt_demo.py
```

**Expected Output:**
```
[TrailerVisionApp] Loading configuration...
Loaded homography for camera-01 with GPS reference: (41.911641, -89.044685)
✓ Homography loaded successfully
```

**If Not Loaded:**
- Check file path: `config/calib/{camera_id}_h.json`
- Verify camera ID matches `config/cameras.yaml`
- Check file exists and is valid JSON

**Step 2: Verify GPS Output**

Process a video or live feed and check detection events:

```python
# Check event structure
event = {
    "lat": 41.91164053,  # Should not be None
    "lon": -89.04468542,  # Should not be None
    "x_world": 0.0,
    "y_world": 0.0
}
```

**Verification:**
- `lat` and `lon` should not be `None`
- Coordinates should be reasonable (within yard bounds)
- Values should change as trailers move

**Step 3: Test in Google Maps**

1. **Copy GPS coordinates** from detection event
2. **Paste into Google Maps** search bar
3. **Verify location** matches physical trailer position
4. **Check accuracy** (should be within 5-10 meters)

**Example:**
```
Event: lat=41.91164053, lon=-89.04468542
Google Maps: https://www.google.com/maps?q=41.91164053,-89.04468542
Expected: Location should match trailer position in yard
```

**Step 4: Accuracy Testing**

**Known Point Test:**
1. Place trailer at **known GPS location**
2. Run detection system
3. Compare detected GPS vs. actual GPS
4. Measure error distance

**Distance Test:**
1. Place trailers at **two known locations**
2. Get GPS coordinates for both
3. Calculate distance between detections
4. Compare to known physical distance

**Expected Results:**
- Error < 10 meters (good)
- Error < 5 meters (excellent)
- Error > 15 meters (recalibrate)

### 9.2 Troubleshooting

**Problem: GPS coordinates are None**

**Causes:**
- Calibration file missing GPS reference
- Calibration done in meters mode (not GPS mode)
- GPS reference not loaded properly

**Solutions:**
1. Check calibration file has `gps_reference` field
2. Recalibrate with GPS coordinates (not `--use-meters`)
3. Verify file loads correctly at startup

**Problem: Coordinates don't match Google Maps**

**Causes:**
- Wrong coordinate system (not WGS84)
- Incorrect GPS input during calibration
- Typo in GPS coordinates

**Solutions:**
1. Verify GPS coordinates are WGS84 decimal degrees
2. Recheck GPS coordinates used in calibration
3. Test each calibration point in Google Maps

**Problem: Poor accuracy (>10m error)**

**Causes:**
- Poor calibration (high RMSE)
- Inaccurate GPS input
- Camera movement
- Too few calibration points

**Solutions:**
1. Recalibrate with more points (8-12)
2. Verify GPS coordinates are accurate
3. Check camera hasn't moved
4. Use ground-level calibration points

**Problem: Homography not loading**

**Causes:**
- File path mismatch
- Missing calibration file
- Invalid JSON format

**Solutions:**
1. Verify file name: `config/calib/{camera_id}_h.json`
2. Check camera ID matches `config/cameras.yaml`
3. Validate JSON format
4. Check file permissions

**Problem: Coordinates jump around**

**Causes:**
- Unstable calibration
- Camera vibration
- Poor calibration points

**Solutions:**
1. Recalibrate with more stable points
2. Stabilize camera mounting
3. Use more calibration points

---

## 10. Maintenance and Updates

### 10.1 When to Recalibrate

**Required:**
- Camera is **moved or repositioned**
- Camera **lens is adjusted**
- Significant **accuracy degradation** observed
- Camera **mounting changes**

**Recommended:**
- **Periodic verification** (quarterly)
- After **yard layout changes**
- If **detection accuracy issues** arise
- After **camera maintenance**

**Signs You Need Recalibration:**
- GPS coordinates consistently off by >15 meters
- Coordinates don't match physical locations
- RMSE was high during calibration (>2.5m)
- Camera was moved/adjusted

### 10.2 Calibration Management

**Multiple Cameras:**
- Each camera requires **separate calibration**
- Store calibrations in `config/calib/{camera_id}_h.json`
- Maintain **calibration log/documentation**

**Version Control:**
- Track **calibration dates** and RMSE values
- Document **GPS reference points** used
- Keep **backup of calibration files**
- Version calibration files if needed

**Documentation:**
- Record which image was used
- Note GPS coordinates for each point
- Save calibration image for reference
- Track RMSE and accuracy metrics

### 10.3 Backup and Recovery

**Backup Calibration Files:**
```bash
# Backup all calibrations
cp -r config/calib/ backups/calib_$(date +%Y%m%d)/
```

**Restore:**
```bash
# Restore from backup
cp backups/calib_20240115/* config/calib/
```

---

## 11. Technical Architecture

### 11.1 Components

**Calibration Tool** (`tools/calibrate_h.py`):
- Interactive point selection GUI
- GPS coordinate input and validation
- Homography matrix calculation (OpenCV)
- Calibration file generation (JSON)
- RMSE calculation and reporting

**GPS Utilities** (`app/gps_utils.py`):
- `meters_to_gps()`: Converts local meters to GPS (lat/lon)
- `gps_to_meters()`: Converts GPS to local meters
- `validate_gps_coordinate()`: Validates GPS coordinates

**Main Application** (`app/main_trt_demo.py`):
- Calibration file loading at startup
- Real-time coordinate transformation
- GPS coordinate output in events
- Integration with detection pipeline

**Video Processor** (`app/video_processor.py`):
- Batch processing support
- Coordinate transformation for saved crops
- Event generation with GPS coordinates
- Support for video file processing

### 11.2 Data Flow

```
┌─────────────────┐
│ Calibration     │
│ Image + GPS     │
│ Points          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Homography      │
│ Calculation     │
│ (OpenCV)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Calibration     │
│ File (JSON)     │
│ config/calib/   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Application     │
│ Startup         │
│ Load H Matrix   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Real-Time       │
│ Processing      │
│                 │
│ Image → World   │
│ World → GPS     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Detection       │
│ Events with     │
│ GPS Coordinates │
└─────────────────┘
```

### 11.3 Key Files and Locations

**Calibration:**
- `tools/calibrate_h.py` - Calibration tool
- `config/calib/{camera_id}_h.json` - Calibration files

**GPS Conversion:**
- `app/gps_utils.py` - GPS utilities

**Integration:**
- `app/main_trt_demo.py` - Main application
- `app/video_processor.py` - Video processing
- `config/cameras.yaml` - Camera configuration

---

## 12. Business Value

### 12.1 Use Cases

**Yard Management:**
- **Real-time trailer location tracking**
- **Parking spot assignment** and verification
- **Yard utilization analytics**
- **Trailer movement patterns**

**Integration:**
- **Google Maps integration** for visualization
- **External system APIs** (WMS, TMS)
- **Database spatial queries**
- **Mobile app integration**

**Analytics:**
- **Trailer movement patterns**
- **Dwell time analysis**
- **Yard efficiency metrics**
- **Historical location data**

**Automation:**
- **Automated spot assignment**
- **Location-based alerts**
- **Integration with yard management systems**
- **Reduced manual tracking**

### 12.2 ROI

**Benefits:**
- **Precise location tracking** without manual input
- **Automated spot assignment** reduces errors
- **Integration with existing mapping systems**
- **Reduced manual yard management** overhead
- **Better yard utilization** through analytics

**Cost Savings:**
- **Reduced manual location tracking** time
- **Faster trailer identification** and retrieval
- **Improved yard utilization** efficiency
- **Better integration** with logistics systems
- **Reduced errors** in spot assignment

**Time Savings:**
- **No manual GPS entry** required
- **Automated location updates**
- **Instant location queries**
- **Real-time tracking** capabilities

### 12.3 Competitive Advantages

- **GPS Integration:** Direct output to mapping systems
- **Accuracy:** ±5-10 meter precision
- **Automation:** No manual intervention required
- **Scalability:** Works with multiple cameras
- **Integration Ready:** Standard GPS format (lat/lon)

---

## Summary

### Key Points

1. **One-Time Calibration:** Set up once per camera, reusable indefinitely
2. **GPS Integration:** Direct output to GPS coordinates (lat/lon)
3. **Google Maps Compatible:** Copy-paste coordinates directly
4. **Accurate:** ±5-10 meters typical accuracy
5. **Automated:** No manual intervention required after calibration

### Workflow Summary

```
1. Capture calibration image
2. Identify 4+ landmark points
3. Get GPS coordinates for each point (Google Maps)
4. Run calibration tool (interactive)
5. Verify RMSE < 1.5 meters
6. System automatically loads calibration
7. All detections output GPS coordinates
8. Use coordinates in Google Maps/external systems
```

### Next Steps

1. **For New Cameras:** Follow calibration process (Section 2)
2. **For Existing System:** Verify calibration files are in place
3. **For Testing:** Use verification steps (Section 9)
4. **For Integration:** Use output format (Section 6) for API/database integration

---

## Appendix: Quick Reference

### Calibration Command
```bash
python tools/calibrate_h.py --image <image_path> --save config/calib/<camera_id>_h.json
```

### File Locations
- **Calibration files:** `config/calib/{camera_id}_h.json`
- **Camera config:** `config/cameras.yaml`
- **GPS utilities:** `app/gps_utils.py`
- **Calibration tool:** `tools/calibrate_h.py`

### Coordinate Format
- **Input:** Decimal degrees (e.g., `41.91164053, -89.04468542`)
- **Output:** JSON with `lat` and `lon` fields
- **Range:** Lat: -90 to 90, Lon: -180 to 180

### Accuracy Targets
- **RMSE:** < 1.5 meters (calibration quality)
- **Total Error:** ±5-10 meters (GPS + homography)
- **Best Case:** ±3-5 meters

### Google Maps Links
- **Format:** `https://www.google.com/maps?q=lat,lon`
- **Example:** `https://www.google.com/maps?q=41.91164053,-89.04468542`

---

**Document Version:** 1.0  
**Last Updated:** 2024  
**For Questions:** Refer to troubleshooting section (9.2) or contact support

