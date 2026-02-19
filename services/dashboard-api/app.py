"""
AWS Dashboard API – Flask app for YardVision dashboard and device ingest.

- Receives video frame records from the edge device (POST /api/ingest/video-frame-records).
- Serves dashboard APIs (dashboard/data, events, inventory, yard-view, reports, video-frame-records, cameras).
- Optionally serves the built React YardVision static files.

Set DATABASE_URL to your PostgreSQL connection string.
"""

import os
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from db import get_all_records, get_statistics, insert_records

app = Flask(__name__)

# Optional: serve React dashboard static files (set DASHBOARD_STATIC_DIR to yardvision-dashboard/dist)
DASHBOARD_STATIC_DIR = os.getenv("DASHBOARD_STATIC_DIR")
API_KEY = os.getenv("DASHBOARD_API_KEY")  # optional; if set, device must send X-API-Key header
# Upload directory for cropped images from device
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads")).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "webp"}


def _save_uploaded_image(file) -> str | None:
    """Save an uploaded image file to UPLOAD_DIR; return URL path or None."""
    if not file or file.filename == "":
        return None
    ext = (Path(secure_filename(file.filename)).suffix or ".jpg").lower().lstrip(".")
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        ext = "jpg"
    name = f"{uuid.uuid4().hex}.{ext}"
    dest = UPLOAD_DIR / name
    try:
        file.save(str(dest))
        return f"/api/images/{name}"
    except Exception:
        return None


def _filter_records_by_date(records: list, date_str: str | None) -> list:
    if not date_str:
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
    try:
        filter_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return records
    out = []
    for r in records:
        created = r.get("created_on") or r.get("timestamp") or ""
        if not created:
            continue
        try:
            s = str(created)
            if "T" in s:
                rec_date = datetime.fromisoformat(s.replace("Z", "+00:00")).date()
            else:
                rec_date = datetime.strptime(s[:10], "%Y-%m-%d").date()
            if rec_date == filter_date:
                out.append(r)
        except (ValueError, TypeError):
            continue
    return out


def _get_dashboard_data(date_str: str | None = None) -> dict:
    stats = get_statistics()
    if stats.get("total", 0) == 0:
        return {
            "kpis": {
                "trailersOnYard": {"value": 0, "change": "+0", "icon": "🚛"},
                "newDetections24h": {"value": 0, "ocrAccuracy": "0%", "icon": "📈"},
                "anomalies": {"value": 0, "description": "No anomalies", "icon": "⚠️"},
                "camerasOnline": {"value": 0, "degraded": 0, "icon": "📷"},
            },
            "queueStatus": {"ingestQ": 0, "ocrQ": 0, "pubQ": 0},
            "accuracyChart": [],
            "yardUtilization": [],
            "cameraHealth": [],
        }
    records = get_all_records(limit=2000, offset=0, is_processed=None, camera_id=None)
    records = _filter_records_by_date(records, date_str)
    if not records and date_str:
        records = get_all_records(limit=500, offset=0, is_processed=None, camera_id=None)
    unique_tracks = set(r.get("track_id") for r in records if r.get("track_id") is not None)
    unique_spots = set()
    for r in records:
        s = (r.get("assigned_spot_name") or r.get("processed_comment") or "").strip()
        if s and s.lower() != "unknown":
            unique_spots.add(s)
    total_detections = len(records)
    ocr_ok = sum(1 for r in records if (r.get("licence_plate_trailer") or "").strip())
    ocr_accuracy = (ocr_ok / total_detections * 100) if total_detections else 0
    low_conf = sum(1 for r in records if (r.get("confidence") or 0) < 0.5)
    hourly = defaultdict(lambda: {"total": 0, "total_det": 0.0, "total_ocr": 0.0, "ocr_n": 0})
    for r in records:
        created = r.get("created_on") or r.get("timestamp") or ""
        if created:
            try:
                dt = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
                key = dt.strftime("%H:00")
                hourly[key]["total"] += 1
                conf = float(r.get("confidence") or 0)
                hourly[key]["total_det"] += conf
                if (r.get("licence_plate_trailer") or "").strip():
                    hourly[key]["total_ocr"] += conf
                    hourly[key]["ocr_n"] += 1
            except (ValueError, TypeError):
                pass
    accuracy_chart = []
    for h in range(24):
        key = f"{h:02d}:00"
        d = hourly.get(key, {"total": 0, "total_det": 0.0, "total_ocr": 0.0, "ocr_n": 0})
        det_pct = (d["total_det"] / d["total"] * 100) if d["total"] else 0
        ocr_pct = (d["total_ocr"] / d["ocr_n"] * 100) if d["ocr_n"] else 0
        accuracy_chart.append({"time": key, "detection": round(det_pct, 1), "ocr": round(ocr_pct, 1)})
    spot_counts = defaultdict(int)
    for r in records:
        s = (r.get("assigned_spot_name") or "").strip() or (r.get("processed_comment") or "").strip()
        if s and s.lower() != "unknown":
            spot_counts[s] += 1
    yard_util = [{"lane": k, "utilization": v} for k, v in sorted(spot_counts.items())]
    return {
        "kpis": {
            "trailersOnYard": {"value": len(unique_tracks), "change": f"+{len(unique_tracks)}", "icon": "🚛"},
            "newDetections24h": {"value": total_detections, "ocrAccuracy": f"{ocr_accuracy:.1f}%", "icon": "📈"},
            "anomalies": {"value": low_conf, "description": f"{low_conf} low confidence", "icon": "⚠️"},
            "camerasOnline": {"value": 0, "degraded": 0, "icon": "📷"},
        },
        "queueStatus": {"ingestQ": 0, "ocrQ": 0, "pubQ": 0},
        "accuracyChart": accuracy_chart,
        "yardUtilization": yard_util,
        "cameraHealth": [],
    }


def _get_events(limit: int, date_str: str | None = None) -> list:
    stats = get_statistics()
    if stats.get("total", 0) == 0:
        return []
    records = get_all_records(limit=min(limit, 2000), offset=0, is_processed=None, camera_id=None)
    records = _filter_records_by_date(records, date_str)
    events = []
    for r in records:
        ts = r.get("timestamp") or r.get("created_on") or ""
        events.append({
            "ts_iso": ts,
            "camera_id": r.get("camera_id") or "N/A",
            "track_id": r.get("track_id") if r.get("track_id") is not None else "N/A",
            "text": (r.get("licence_plate_trailer") or "").strip() or "",
            "conf": float(r.get("confidence") or 0),
            "spot": (r.get("assigned_spot_name") or "").strip() or "unknown",
            "ocr_conf": float(r.get("confidence") or 0),
            "lat": r.get("latitude"),
            "lon": r.get("longitude"),
        })
    events.sort(key=lambda x: x.get("ts_iso", ""), reverse=True)
    return events[:limit]


def _get_inventory() -> dict:
    stats = get_statistics()
    if stats.get("total", 0) == 0:
        return {"trailers": [], "stats": {"total": 0, "parked": 0, "inTransit": 0, "anomalies": 0}}
    records = get_all_records(limit=1000, offset=0, is_processed=None, camera_id=None)
    by_track = defaultdict(list)
    for r in records:
        tid = r.get("track_id")
        if tid is not None:
            by_track[tid].append(r)
    trailers = []
    for track_id, rlist in by_track.items():
        r = max(rlist, key=lambda x: (x.get("created_on") or x.get("timestamp") or ""))
        spot = (r.get("assigned_spot_name") or "").strip() or "N/A"
        status = "Parked" if r.get("is_processed") or (spot and spot != "N/A") else "In Transit"
        trailers.append({
            "id": f"T{track_id}" if track_id is not None else f"R{r.get('id', 0)}",
            "plate": (r.get("licence_plate_trailer") or "").strip() or "N/A",
            "spot": spot if spot != "N/A" else "N/A",
            "status": status,
            "detectedAt": r.get("created_on") or r.get("timestamp") or "",
            "ocrConfidence": float(r.get("confidence") or 0),
            "lat": r.get("latitude"),
            "lon": r.get("longitude"),
            "imageUrl": (r.get("image_url") or "").strip() or None,
        })
    total = len(trailers)
    parked = sum(1 for t in trailers if t["status"] == "Parked")
    anomalies = sum(1 for rec in records if (rec.get("confidence") or 0) < 0.5)
    return {
        "trailers": trailers,
        "stats": {"total": total, "parked": parked, "inTransit": total - parked, "anomalies": anomalies},
    }


def _get_yard_view() -> dict:
    try:
        records = get_all_records(limit=500, offset=0, is_processed=None, camera_id=None)
        spot_to_latest = {}
        for r in records:
            sid = (r.get("assigned_spot_id") or "").strip() or (r.get("assigned_spot_name") or "").strip()
            if not sid:
                continue
            created = r.get("created_on") or r.get("timestamp") or ""
            if sid not in spot_to_latest or (spot_to_latest[sid].get("created_on") or "") < created:
                spot_to_latest[sid] = r
        spots = []
        for sid, r in spot_to_latest.items():
            name = (r.get("assigned_spot_name") or sid).strip()
            lane = name.split("-")[0] if name else "A"
            spots.append({
                "id": sid,
                "lane": lane,
                "row": 1,
                "occupied": True,
                "trailerId": f"T{r.get('track_id')}" if r.get("track_id") is not None else None,
                "plate": (r.get("licence_plate_trailer") or "").strip() or None,
            })
        lanes = sorted(set(s["lane"] for s in spots)) if spots else ["A", "B", "C", "D", "Dock"]
        return {"spots": spots, "lanes": lanes}
    except Exception:
        return {"spots": [], "lanes": ["A", "B", "C", "D", "Dock"]}


def _get_reports() -> dict:
    try:
        stats = get_statistics()
        all_records = get_all_records(limit=5000, offset=0, is_processed=None, camera_id=None)
        total = len(all_records)
        ocr_ok = sum(1 for r in all_records if (r.get("licence_plate_trailer") or "").strip())
        ocr_pct = (ocr_ok / total * 100) if total else 0
        anomalies = sum(1 for r in all_records if (r.get("confidence") or 0) < 0.5)
        today = datetime.utcnow().strftime("%Y-%m-%d")
        week_start = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
        daily_count = len(_filter_records_by_date(all_records, today))
        weekly_records = [r for r in all_records if (str(r.get("created_on") or "")[:10] >= week_start)]
        monthly_records = [r for r in all_records if (str(r.get("created_on") or "")[:10] >= (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")]
        return {
            "daily": {"date": today, "totalDetections": daily_count, "ocrAccuracy": round(ocr_pct, 1), "anomalies": anomalies, "avgProcessingTime": 86},
            "weekly": {"week": "Last 7 days", "totalDetections": len(weekly_records), "ocrAccuracy": round(ocr_pct, 1), "anomalies": anomalies, "avgProcessingTime": 92},
            "monthly": {"month": datetime.utcnow().strftime("%B %Y"), "totalDetections": len(monthly_records), "ocrAccuracy": round(ocr_pct, 1), "anomalies": anomalies, "avgProcessingTime": 88},
        }
    except Exception:
        return {
            "daily": {"date": "", "totalDetections": 0, "ocrAccuracy": 0, "anomalies": 0, "avgProcessingTime": 0},
            "weekly": {"week": "", "totalDetections": 0, "ocrAccuracy": 0, "anomalies": 0, "avgProcessingTime": 0},
            "monthly": {"month": "", "totalDetections": 0, "ocrAccuracy": 0, "anomalies": 0, "avgProcessingTime": 0},
        }


# ----- Ingest (device upload) -----

def _require_api_key():
    if not API_KEY:
        return None
    key = request.headers.get("X-API-Key") or request.args.get("api_key")
    if key != API_KEY:
        return jsonify({"error": "Unauthorized", "message": "Invalid or missing API key"}), 401
    return None


@app.route("/healthz")
def healthz():
    return jsonify({"status": "healthy", "timestamp": datetime.utcnow().isoformat()})


@app.route("/api/ingest/video-frame-records", methods=["POST"])
def ingest_video_frame_records():
    err = _require_api_key()
    if err:
        return err
    records = None
    device_id = request.headers.get("X-Device-ID")

    if request.content_type and "multipart/form-data" in request.content_type:
        # Single combined request: form part "records" (JSON) + optional "image_0", "image_1", ... (files)
        import json as json_module
        records_part = request.form.get("records")
        if not records_part:
            return jsonify({"error": "Multipart body must include 'records' form field (JSON)"}), 400
        try:
            data = json_module.loads(records_part)
        except (ValueError, TypeError) as e:
            return jsonify({"error": f"Invalid 'records' JSON: {e}"}), 400
        records = data.get("records")
        if not isinstance(records, list):
            return jsonify({"error": "Body must contain 'records' array"}), 400
        device_id = data.get("device_id") or device_id
        # Attach uploaded images by index: image_0 -> records[0].image_url, etc.
        for key in request.files:
            if key.startswith("image_"):
                try:
                    idx = int(key[6:])
                except ValueError:
                    continue
                if 0 <= idx < len(records):
                    url = _save_uploaded_image(request.files[key])
                    if url:
                        records[idx]["image_url"] = url
    else:
        # JSON body (records only; images already uploaded or none)
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400
        records = data.get("records")
        if not isinstance(records, list):
            return jsonify({"error": "Body must contain 'records' array"}), 400
        device_id = data.get("device_id") or device_id

    try:
        count = insert_records(records, device_id=device_id)
        return jsonify({"status": "success", "count": count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ingest/upload-image", methods=["POST"])
def ingest_upload_image():
    """Accept a cropped image file from the device; save and return URL for the record."""
    err = _require_api_key()
    if err:
        return err
    if "file" not in request.files and "image" not in request.files:
        return jsonify({"error": "No file part; use 'file' or 'image' form key"}), 400
    file = request.files.get("file") or request.files.get("image")
    if not file or file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    ext = (Path(secure_filename(file.filename)).suffix or ".jpg").lower().lstrip(".")
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        ext = "jpg"
    name = f"{uuid.uuid4().hex}.{ext}"
    dest = UPLOAD_DIR / name
    try:
        file.save(str(dest))
        url = f"/api/images/{name}"
        return jsonify({"status": "success", "url": url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/images/<path:filename>")
def serve_image(filename):
    """Serve an uploaded cropped image by filename (safe path, no traversal)."""
    filename = secure_filename(filename)
    if not filename or ".." in filename:
        return jsonify({"error": "Invalid filename"}), 400
    path = UPLOAD_DIR / filename
    if not path.is_file():
        return jsonify({"error": "Not found"}), 404
    return send_from_directory(str(UPLOAD_DIR), filename)


# ----- Dashboard API (same contract as device metrics_server) -----

@app.route("/api/dashboard/data", methods=["GET"])
def get_dashboard_data():
    date_str = request.args.get("date")
    return jsonify(_get_dashboard_data(date_str=date_str))


@app.route("/api/dashboard/events", methods=["GET"])
def get_dashboard_events():
    limit = int(request.args.get("limit", 1000))
    date_str = request.args.get("date")
    events = _get_events(limit, date_str=date_str)
    return jsonify({"events": events, "count": len(events)})


@app.route("/api/cameras", methods=["GET"])
def get_cameras():
    return jsonify({"cameras": []})


@app.route("/api/inventory", methods=["GET"])
def get_inventory():
    return jsonify(_get_inventory())


@app.route("/api/yard-view", methods=["GET"])
def get_yard_view():
    return jsonify(_get_yard_view())


@app.route("/api/reports", methods=["GET"])
def get_reports():
    return jsonify(_get_reports())


@app.route("/api/video-frame-records", methods=["GET"])
def get_video_frame_records():
    limit = request.args.get("limit", default=50, type=int)
    offset = request.args.get("offset", default=0, type=int)
    is_processed = request.args.get("is_processed", default=None, type=str)
    camera_id = request.args.get("camera_id", default=None, type=str)
    is_processed_bool = None
    if is_processed is not None:
        is_processed_bool = is_processed.lower() == "true"
    records = get_all_records(limit=limit, offset=offset, is_processed=is_processed_bool, camera_id=camera_id)
    stats = get_statistics()
    return jsonify({
        "records": records,
        "stats": stats,
        "limit": limit,
        "offset": offset,
        "total": stats.get("total", 0),
    })


# ----- Optional: serve React dashboard static files -----

if DASHBOARD_STATIC_DIR:
    static_dir = Path(DASHBOARD_STATIC_DIR)

    @app.route("/")
    def index():
        return send_from_directory(str(static_dir), "index.html")

    @app.route("/<path:path>")
    def static_files(path):
        if path.startswith("api/"):
            return jsonify({"error": "Not found"}), 404
        fp = static_dir / path
        if fp.is_file():
            return send_from_directory(str(static_dir), path)
        return send_from_directory(str(static_dir), "index.html")
else:
    @app.route("/")
    def index():
        return jsonify({"service": "dashboard-api", "docs": "Use /api/dashboard/data, /api/inventory, etc."})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
