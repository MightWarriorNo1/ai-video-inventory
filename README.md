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
sudo nvpmodel -m 0 && sudo jetson_clocks
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

## Project Structure

- `app/` - Core application code
- `config/` - Camera configs, homography, parking spots
- `models/` - TensorRT engines
- `tools/` - Calibration utilities
- `web/` - Web dashboard
- `services/ingest/` - REST API for TimescaleDB ingestion
- `observability/` - Prometheus & Grafana configs

See the full runbook for detailed documentation.

