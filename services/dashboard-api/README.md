# Dashboard API (AWS)

Flask app for the cloud side of the Edge + AWS setup:

- **Ingest (combined):** `POST /api/ingest/video-frame-records` – accepts either (1) JSON body `{ "records": [...], "device_id": "..." }`, or (2) multipart with a `records` form field (same JSON) plus optional `image_0`, `image_1`, … (file parts). Image index matches the record index; the server saves each image and sets `records[i].image_url` before inserting. One request sends both data and cropped images.  
  **Legacy:** `POST /api/ingest/upload-image` – single image upload (multipart), returns `{ "url": "/api/images/..." }`.  
  **Serve:** `GET /api/images/<filename>` – serves uploaded images.
- **Dashboard API:** Same contract as the device metrics server: `/api/dashboard/data`, `/api/dashboard/events`, `/api/inventory`, `/api/yard-view`, `/api/reports`, `/api/video-frame-records`, `/api/cameras`.

Data is read from PostgreSQL (table `video_frame_records`). Run the schema in `sql/001_video_frame_records.sql` and `sql/002_add_image_url.sql` before first use.

**Quick start:** Set `DATABASE_URL`, then `pip install -r requirements.txt` and `python app.py`. See the project root **docs/AWS_DEPLOYMENT_GUIDE.md** for the full step-by-step deployment guide.
