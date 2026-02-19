# Trailer Vision Edge - AI EdgeBox Orion

Production-ready, Jetson-optimized edge application for video-based trailer inventory tracking.

## Quick Start

### Prerequisites
- NVIDIA Jetson Orin (JetPack)
- Python 3.8+
- TensorRT engines: `models/trailer_detector.engine`, `models/ocr_crnn.engine`

### Installation

```bash
sudo apt-get update && sudo apt-get install -y python3-pip python3-opencv git
pip3 install -r requirements.txt
clear
```

### Configuration

1. Edit `config/cameras.yaml` with your camera RTSP URLs
2. Calibrate cameras: `python3 tools/calibrate_h.py --image frame.jpg --save config/calib/<camera-id>_h.json`
3. Define parking spots in `config/spots.geojson`
4. Set environment variables (see `.env.example`)

### Running

```bash
python3 -m app.main_trt_demo
```

Or with Docker:
```bash
docker compose build
docker compose up
```

### Dashboard

Access web UI at: `http://<device-ip>:8080/`

## Architecture

- **Camera Ingestion**: RTSP/USB via OpenCV/GStreamer
- **Detection**: YOLO (TensorRT) for vehicle/trailer detection
- **Tracking**: ByteTrack for multi-object tracking
- **OCR**: CRNN/PP-OCR (TensorRT) for trailer ID recognition
- **Geometry**: Homography projection (image → world coordinates)
- **Spot Resolution**: GeoJSON polygon matching
- **Outputs**: CSV logs, screenshots, metrics, event bus publishing
- **Integrations**: Azure Service Bus, Kafka, MQTT, S3, Azure Blob, TimescaleDB

## Automated Video Processing Workflow

The system now supports fully automated video processing with GPU-aware sequential processing:

### Features

1. **45-Second Auto-Chunking**: Video recordings are automatically split into 45-second chunks
2. **Automatic Processing**: Each chunk is automatically queued for processing when saved
3. **Sequential GPU Operations**: Video processing and OCR run sequentially to prevent GPU memory conflicts
4. **Extensible Server Upload**: Ready-to-implement hooks for server data upload

### Workflow

```
Record 45s chunk → Save video + GPS log → Queue video processing
  ↓
Video Processing (GPU): Detection + Tracking + Save Crops (NO OCR)
  ↓
Video Processing Complete → Queue OCR job
  ↓
OCR Processing (GPU): Process saved crops sequentially
  ↓
OCR Complete → Ready for server upload (extensible hook)
```

### Implementation Details

- **Processing Queue Manager** (`app/processing_queue.py`): Manages sequential processing with GPU lock
- **Auto-Chunking Video Recorder** (`app/video_recorder.py`): Automatically splits recordings into 45-second chunks
- **Extensibility Hooks**: 
  - `on_video_complete(video_path, crops_dir, results)`: Called after video processing
  - `on_ocr_complete(video_path, crops_dir, ocr_results)`: Called after OCR processing
  - `upload_to_server(video_path, crops_dir, data)`: Implement your server upload logic here

### Configuration

The automation is enabled by default when the application starts. To customize:

- **Chunk Duration**: Modify `chunk_duration_seconds` in `VideoRecorder` initialization (default: 45.0)
- **Processing Parameters**: Adjust `detect_every_n` and `detection_mode` in queue manager
- **Server Upload**: Implement `upload_to_server()` method in `TrailerVisionApp` class

## Project Structure

- `app/` - Core application code
- `config/` - Camera configs, homography, parking spots
- `models/` - TensorRT engines
- `tools/` - Calibration utilities
- `web/` - Web dashboard
- `services/ingest/` - REST API for TimescaleDB ingestion
- `observability/` - Prometheus & Grafana configs

See the full runbook for detailed documentation.

## AWS deployment (device + cloud dashboard)

To run recording, processing, and OCR on the NVIDIA device and host the backend, database, and YardVision dashboard on an AWS server, use the **separate Flask Dashboard API** in `services/dashboard-api/` and follow **docs/AWS_DEPLOYMENT_GUIDE.md** for a single step-by-step guide.


