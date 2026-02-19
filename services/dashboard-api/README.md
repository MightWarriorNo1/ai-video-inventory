# Dashboard API (AWS)

Flask app for the cloud side of the Edge + AWS setup:

- **Ingest:** `POST /api/ingest/video-frame-records` – accepts batches of video frame records from the edge device.
- **Dashboard API:** Same contract as the device metrics server: `/api/dashboard/data`, `/api/dashboard/events`, `/api/inventory`, `/api/yard-view`, `/api/reports`, `/api/video-frame-records`, `/api/cameras`.

Data is read from PostgreSQL (table `video_frame_records`). Run the schema in `sql/001_video_frame_records.sql` before first use.

**Quick start:** Set `DATABASE_URL`, then `pip install -r requirements.txt` and `python app.py`. See the project root **docs/AWS_DEPLOYMENT_GUIDE.md** for the full step-by-step deployment guide.
