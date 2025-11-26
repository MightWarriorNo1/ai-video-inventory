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

## Project Structure

- `app/` - Core application code
- `config/` - Camera configs, homography, parking spots, dataset configs
- `models/` - TensorRT engines
- `tools/` - Calibration utilities, training scripts
- `web/` - Web dashboard
- `services/ingest/` - REST API for TimescaleDB ingestion
- `observability/` - Prometheus & Grafana configs

## Training Custom YOLOv8 Model

To improve trailer back detection accuracy, you can train a custom YOLOv8 model:

1. **Prepare dataset**: Annotate trailer back images
2. **Train model**: `python tools/train_yolo_trailer.py --data config/trailer_dataset.yaml`
3. **Export to ONNX**: `python tools/export_trained_model.py --weights runs/detect/trailer_back_detector/weights/best.pt`
4. **Build TensorRT engine**: `python build_engines.py --detector-onnx models/best.onnx --fp16`

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed instructions.

## Documentation

- **Training Guide**: [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Train custom YOLOv8 models
- **Engine Building**: [BUILD_ENGINES.md](BUILD_ENGINES.md) - Rebuild TensorRT engines
- **OCR Training**: [tools/EXTRACT_TRAINING_DATA_README.md](tools/EXTRACT_TRAINING_DATA_README.md) - OCR data extraction


