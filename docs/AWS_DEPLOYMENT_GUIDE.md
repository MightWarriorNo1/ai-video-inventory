# AWS Deployment Guide – Edge Device + Cloud Dashboard

This guide walks you through running **recording, video processing, OCR, and data processor on the NVIDIA device** and **backend, database, and YardVision dashboard on an AWS server**. The device uploads results to the server; the dashboard reads from the server database.

---

## Architecture Overview

| Where | What runs |
|-------|-----------|
| **NVIDIA device** | Recording, data processor service, video processing, OCR. Results stored in local SQLite and **uploaded to AWS** after OCR. |
| **AWS server** | PostgreSQL (video_frame_records), **Dashboard API** (Flask), **YardVision React dashboard** (built and served). |

- Device uses env vars `EDGE_UPLOAD_URL` and optionally `DASHBOARD_API_KEY` to POST to the AWS Dashboard API.
- The Dashboard API is a **separate Flask app** in `services/dashboard-api/`: it accepts ingest and serves the same API contract the React dashboard expects.

---

## Step 1 – Prepare the AWS Server

1. **Access the server**  
   Use Remote Desktop (RDP) with the server’s IP, username, and password.

2. **Install prerequisites**
   - **Python 3.10+** (e.g. from python.org or Windows Store).
   - **PostgreSQL 14+** (with or without TimescaleDB). Options:
     - Install PostgreSQL locally on the Windows server, or
     - Use Amazon RDS for PostgreSQL (then use the RDS endpoint as the DB host).
   - **Node.js 18+** (for building the React dashboard): [nodejs.org](https://nodejs.org).

3. **Open firewall** (if applicable)  
   Allow inbound TCP for the port the Dashboard API will use (e.g. **8080**). If you use a reverse proxy (e.g. port 80/443), allow those instead.

---

## Step 2 – Create the Database and Schema on AWS

1. **Create a database** (if you don’t have one):
   - Example: database name `trailer_vision`, user `postgres`, password of your choice.

2. **Run the schema** for the dashboard API:
   - On the AWS server, copy the file:
     - From the repo: `services/dashboard-api/sql/001_video_frame_records.sql`
   - Execute it against your PostgreSQL instance, e.g.:
     - **psql:** `psql -U postgres -d trailer_vision -f 001_video_frame_records.sql`
     - Or use pgAdmin / any PostgreSQL client to run the script.

3. **Note the connection string**  
   Format:  
   `postgresql://USER:PASSWORD@HOST:5432/trailer_vision`  
   Example (local): `postgresql://postgres:YourPassword@localhost:5432/trailer_vision`  
   Example (RDS): `postgresql://postgres:YourPassword@your-rds-endpoint.region.rds.amazonaws.com:5432/trailer_vision`

---

## Step 3 – Deploy the Dashboard API (Flask) on AWS

1. **Copy the dashboard API to the server**  
   Copy the project (or at least) the folder `services/dashboard-api/` to the AWS server (e.g. `C:\edgeorion\services\dashboard-api\`).

2. **Create a virtual environment and install dependencies** (in a shell on the server):
   ```bash
   cd C:\edgeorion\services\dashboard-api
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set environment variables**  
   Set at least:
   - `DATABASE_URL` = your PostgreSQL connection string from Step 2.  
   Optional:
   - `PORT` = port the Flask app listens on (default 8080).  
   - `DASHBOARD_API_KEY` = shared secret; if set, the device must send it in `X-API-Key` header.  
   - `DASHBOARD_STATIC_DIR` = full path to the built React app (e.g. `C:\edgeorion\yardvision-dashboard\dist`) so the same process serves the dashboard; set this **after** building the React app in Step 4.

   Example (PowerShell, current session):
   ```powershell
   $env:DATABASE_URL = "postgresql://postgres:YourPassword@localhost:5432/trailer_vision"
   $env:PORT = "8080"
   ```

4. **Run the Flask app**
   ```bash
   python app.py
   ```
   Or with gunicorn (Linux/WSL): `gunicorn -w 2 -b 0.0.0.0:8080 app:app`

5. **Verify**  
   In a browser or with curl:  
   `http://<AWS_SERVER_IP>:8080/healthz`  
   Should return `{"status":"healthy",...}`.  
   Also try: `http://<AWS_SERVER_IP>:8080/api/dashboard/data` – should return JSON (possibly with zeros if DB is empty).

---

## Step 4 – Build and Serve the YardVision React Dashboard on AWS

1. **Copy the React app to the server**  
   Copy the whole project (or at least `yardvision-dashboard/`) to the AWS server.

2. **Point the dashboard at the AWS API**  
   The dashboard uses relative `/api` by default. If you serve the dashboard from the **same origin** as the Dashboard API (e.g. same host and port via `DASHBOARD_STATIC_DIR`), no change is needed.  
   If you serve the dashboard from another port or domain, set the API base when building, e.g. in `yardvision-dashboard/src/services/api.js` set:
   `const API_BASE = 'http://<AWS_SERVER_IP>:8080/api'`  
   (or use a Vite env variable and build with that).

3. **Build the dashboard** (on the server):
   ```bash
   cd C:\edgeorion\yardvision-dashboard
   npm ci
   npm run build
   ```
   This creates the `dist/` folder.

4. **Serve the dashboard**
   - **Option A – Same process as API:**  
     Set `DASHBOARD_STATIC_DIR=C:\edgeorion\yardvision-dashboard\dist` (or the actual path), restart the Dashboard API (Step 3). Then open `http://<AWS_SERVER_IP>:8080/` – you should see the dashboard, and it will call `/api` on the same host.
   - **Option B – Separate web server:**  
     Serve the contents of `dist/` with IIS or nginx, and proxy `/api` to `http://127.0.0.1:8080` (or wherever the Dashboard API runs).

5. **Verify**  
   Open `http://<AWS_SERVER_IP>:8080/` (if using Option A). Dashboard should load; data may be empty until the device uploads.

---

## Step 5 – Configure the NVIDIA Device to Upload to AWS

1. **On the device**, set environment variables (e.g. in `.env` in the project root or in the shell that starts the app):
   - **Required:**  
     `EDGE_UPLOAD_URL=http://<AWS_SERVER_IP>:8080`  
     (Use the real AWS server IP or hostname; use `https://` if you put the API behind TLS.)
   - **Optional:**  
     `DASHBOARD_API_KEY=<same secret as on server>`  
     (Only if you set `DASHBOARD_API_KEY` on the server.)  
     `EDGE_DEVICE_ID=yard1`  
     (Identifies this device in the server DB.)  
     `EDGE_UPLOAD_INTERVAL_SECONDS=60`  
     (How often to upload processed records; default 60.)  
     `EDGE_UPLOAD_BATCH_SIZE=100`  
     (Max processed records per upload; default 100.)

   Example `.env` on the device:
   ```
   EDGE_UPLOAD_URL=http://54.123.45.67:8080
   DASHBOARD_API_KEY=your-secret-key
   EDGE_DEVICE_ID=yard1
   ```

2. **Upload behaviour**
   - **Only processed data is uploaded.** Records are written to local SQLite after OCR; the **data processor** (device UI: start data processor / run processor) assigns parking spots and sets `is_processed = 1`. A **periodic upload thread** (every 60 s by default) uploads only records with `is_processed = 1` to AWS.
   - **After a successful upload, those records are deleted from the device SQLite** so they are not sent again.

3. **Run the app on the device as usual:**
   ```bash
   python3 -m app.main_trt_demo
   ```
   Ensure the **data processor** runs on the device (e.g. from the device web UI or API) so that records get spot assignments and become eligible for upload.

4. **Verify upload**  
   Process at least one video through OCR, then run the data processor so records are marked processed. After the next upload interval:
   - On AWS: open `http://<AWS_SERVER_IP>:8080/api/video-frame-records?limit=10` – you should see records (with `assigned_spot_id` / `assigned_spot_name` when applicable).
   - On the device, those uploaded records will no longer appear in local SQLite (they were deleted after upload).
   - Open the YardVision dashboard on the server – new data should appear.

---

## Step 6 – Optional: Run Dashboard API as a Service (Windows)

- Use **NSSM** or **Windows Task Scheduler** to run `python app.py` (or `gunicorn` if on WSL) at startup so the Dashboard API stays up after reboot.
- Or run it in a terminal that stays open (e.g. via RDP).

---

## Summary Checklist

| Step | Action |
|------|--------|
| 1 | Install Python, PostgreSQL, Node.js on AWS server; open firewall for API port. |
| 2 | Create DB, run `services/dashboard-api/sql/001_video_frame_records.sql`, note `DATABASE_URL`. |
| 3 | Copy `services/dashboard-api/`, install deps, set `DATABASE_URL` (and optional `PORT`, `DASHBOARD_API_KEY`), run `python app.py`. |
| 4 | Copy `yardvision-dashboard/`, build with `npm run build`, set `DASHBOARD_STATIC_DIR` to `dist` and restart API (or serve `dist/` with IIS/nginx and proxy `/api`). |
| 5 | On device set `EDGE_UPLOAD_URL` (and optionally `DASHBOARD_API_KEY`, `EDGE_DEVICE_ID`), run `python3 -m app.main_trt_demo`. |

---

## Troubleshooting

- **Device: “Server upload skipped”**  
  `EDGE_UPLOAD_URL` (or `AWS_DASHBOARD_API_URL`) is not set. Set it and restart the app.

- **Device: “Server upload failed: 401”**  
  API key mismatch. Set `DASHBOARD_API_KEY` on the server and the same value on the device (or leave both unset to disable key check).

- **Dashboard shows no data**  
  Only **processed** records are uploaded (after the data processor assigns spots). Ensure the data processor has run on the device, then wait for the next upload interval (default 60 s). Check device logs for "Uploaded N processed records to AWS and deleted N from SQLite". Then check `http://<AWS_IP>:8080/api/dashboard/data` and `api/video-frame-records` in the browser.

- **Database connection errors on AWS**  
  Check `DATABASE_URL`, that PostgreSQL is running, and that the firewall allows connections to port 5432 if the DB is on another host.

- **CORS**  
  The dashboard is expected to be served from the same origin as the API (or use a reverse proxy). If you serve the dashboard from a different origin, you may need to enable CORS in the Flask app for that origin.
