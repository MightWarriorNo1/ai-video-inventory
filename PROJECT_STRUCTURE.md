# Trailer Vision Edge - Project Structure

This document provides an overview of the complete project structure for the AI EdgeBox Orion Trailer Vision system.

## Directory Structure

```
EdgeOrion/
├── app/                          # Main application package
│   ├── __init__.py
│   ├── ai/                       # AI modules
│   │   ├── __init__.py
│   │   ├── detector_trt.py       # TensorRT YOLO detector wrapper
│   │   └── tracker_bytetrack.py  # ByteTrack multi-object tracker
│   ├── ocr/                      # OCR modules
│   │   ├── __init__.py
│   │   ├── alphabet.txt          # OCR alphabet (0-9, A-Z, space, hyphen, slash)
│   │   └── recognize.py          # TensorRT OCR recognizer (CRNN/PP-OCR)
│   ├── csv_logger.py             # Daily CSV rotation logger
│   ├── event_bus.py              # Multi-publisher (Azure SB, Kafka, MQTT)
│   ├── main_trt_demo.py          # Main application loop
│   ├── media_rotator.py          # Screenshot rotation manager
│   ├── metrics_server.py         # Flask metrics server + web dashboard
│   ├── rtsp.py                   # Camera stream capture (RTSP/USB)
│   ├── spot_resolver.py          # Parking spot resolver (GeoJSON)
│   └── uploader.py               # S3/Azure Blob upload manager
│
├── config/                       # Configuration files
│   ├── calib/                    # Homography calibration files
│   ├── cameras.yaml              # Camera configuration
│   └── spots.geojson             # Parking spot polygons
│
├── models/                       # TensorRT engines (user-supplied)
│   ├── trailer_detector.engine   # YOLO detector engine
│   └── ocr_crnn.engine          # OCR engine
│
├── tools/                        # Utilities
│   ├── __init__.py
│   └── calibrate_h.py            # Homography calibration tool
│
├── web/                          # Web dashboard
│   ├── index.html                # Dashboard HTML
│   ├── app.js                    # Dashboard JavaScript
│   └── style.css                 # Dashboard styles
│
├── services/                     # Backend services
│   └── ingest/                   # REST ingest API
│       ├── Dockerfile
│       ├── main.py               # FastAPI ingest service
│       ├── requirements.txt
│       └── sql/
│           └── 001_init.sql       # TimescaleDB schema
│
├── observability/                # Observability stack
│   ├── grafana/
│   │   ├── dashboards/
│   │   │   └── trailer_dashboard.json
│   │   └── provisioning/
│   │       ├── dashboards/
│   │       │   └── dashboards.yaml
│   │       └── datasources/
│   │           └── datasources.yaml
│   ├── mosquitto.conf            # MQTT broker config
│   └── prometheus.yml            # Prometheus config
│
├── out/                          # Output directory
│   ├── events_YYYYMMDD.csv      # Daily rotated CSV logs
│   └── screenshots/              # Screenshot storage
│       └── <camera-id>/
│
├── docker-compose.yml            # Docker Compose for services
├── requirements.txt              # Python dependencies
├── README.md                     # Quick start guide
└── .gitignore                    # Git ignore rules
```

## Key Components

### 1. Main Application (`app/main_trt_demo.py`)
- End-to-end processing pipeline
- Camera ingestion → Detection → Tracking → OCR → Homography → Spot Resolution
- Event logging, publishing, and metrics

### 2. AI Modules
- **Detector** (`app/ai/detector_trt.py`): TensorRT YOLO wrapper (placeholder for real TRT implementation)
- **Tracker** (`app/ai/tracker_bytetrack.py`): ByteTrack multi-object tracking (placeholder)

### 3. OCR Module
- **Recognizer** (`app/ocr/recognize.py`): TensorRT CRNN/PP-OCR wrapper (placeholder)
- **Alphabet** (`app/ocr/alphabet.txt`): Character set for OCR

### 4. Utilities
- **RTSP Capture** (`app/rtsp.py`): Camera stream handling with GStreamer support
- **Spot Resolver** (`app/spot_resolver.py`): GeoJSON polygon-based spot resolution
- **CSV Logger** (`app/csv_logger.py`): Daily rotating CSV logs
- **Media Rotator** (`app/media_rotator.py`): Screenshot management

### 5. Integrations
- **Event Bus** (`app/event_bus.py`): Azure Service Bus, Kafka, MQTT publishers
- **Uploader** (`app/uploader.py`): S3 and Azure Blob uploads
- **Metrics Server** (`app/metrics_server.py`): Flask server with Prometheus metrics and web dashboard

### 6. Services
- **Ingest API** (`services/ingest/main.py`): FastAPI service for TimescaleDB ingestion
- **TimescaleDB** (`services/ingest/sql/001_init.sql`): Database schema with hypertables

### 7. Observability
- **Prometheus**: Metrics collection
- **Grafana**: Pre-provisioned dashboards
- **MQTT**: Mosquitto broker

## Configuration

### Environment Variables
See `.env.example` for all available environment variables:
- Metrics port, Azure Service Bus, Kafka, MQTT
- S3/Azure Blob upload credentials
- REST ingest API settings

### Camera Configuration (`config/cameras.yaml`)
- Global settings (detector confidence, frame rate, etc.)
- Per-camera settings (RTSP URL, dimensions, FPS)

### Parking Spots (`config/spots.geojson`)
- GeoJSON polygons defining parking spots in world coordinates (meters)
- Each feature has a `name` property for the spot label

### Homography Calibration (`config/calib/<camera-id>_h.json`)
- Per-camera homography matrices
- Generated using `tools/calibrate_h.py`

## Data Flow

1. **Camera** → RTSP/USB stream
2. **Detection** → YOLO TensorRT (every N frames)
3. **Tracking** → ByteTrack (every frame)
4. **OCR** → CRNN/PP-OCR TensorRT (on tracked regions)
5. **Homography** → Image → World coordinates
6. **Spot Resolution** → GeoJSON polygon matching
7. **Outputs**:
   - CSV logs (daily rotation)
   - Screenshots (rotated)
   - Event buses (Azure SB, Kafka, MQTT)
   - REST API → TimescaleDB
   - Cloud uploads (S3, Azure Blob)

## Next Steps

1. **Place TensorRT engines** in `models/`:
   - `trailer_detector.engine`
   - `ocr_crnn.engine`

2. **Replace placeholder implementations**:
   - `app/ai/detector_trt.py`: Add real TensorRT inference
   - `app/ai/tracker_bytetrack.py`: Integrate actual ByteTrack
   - `app/ocr/recognize.py`: Add real TensorRT OCR inference

3. **Configure cameras** in `config/cameras.yaml`

4. **Calibrate cameras** using `tools/calibrate_h.py`

5. **Define parking spots** in `config/spots.geojson`

6. **Set environment variables** for integrations

7. **Run the application**: `python3 -m app.main_trt_demo`

## Docker Services

Run `docker compose up` to start:
- Prometheus (port 9090)
- Grafana (port 3000)
- Kafka (port 29092)
- MQTT/Mosquitto (port 1883)
- TimescaleDB (port 5432)
- Ingest API (port 8000)

## Web Dashboard

Access at `http://<device-ip>:8080/`:
- Per-camera FPS and metrics
- Recent events table
- Real-time updates

## Metrics Endpoints

- `GET /metrics.json` - JSON metrics
- `GET /metrics` - Prometheus format
- `GET /healthz` - Health check
- `GET /events` - Recent events from CSV


