"""
PostgreSQL access for video_frame_records (AWS dashboard API).
"""

import os
from contextlib import contextmanager
from datetime import datetime
from typing import List, Dict, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/trailer_vision",
)


@contextmanager
def get_connection():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _row_to_record(row: dict) -> dict:
    """Convert DB row to API record shape (match device SQLite keys)."""
    ts = row.get("timestamp") or row.get("created_on")
    if hasattr(ts, "isoformat"):
        ts = ts.isoformat()
    created = row.get("created_on")
    if hasattr(created, "isoformat"):
        created = created.isoformat()
    return {
        "id": row.get("id"),
        "licence_plate_trailer": row.get("licence_plate_trailer"),
        "latitude": float(row["latitude"]) if row.get("latitude") is not None else None,
        "longitude": float(row["longitude"]) if row.get("longitude") is not None else None,
        "speed": float(row["speed"]) if row.get("speed") is not None else None,
        "barrier": float(row["barrier"]) if row.get("barrier") is not None else None,
        "confidence": float(row["confidence"]) if row.get("confidence") is not None else 0,
        "image_path": row.get("image_path"),
        "camera_id": row.get("camera_id"),
        "video_path": row.get("video_path"),
        "frame_number": row.get("frame_number") or 0,
        "track_id": row.get("track_id"),
        "timestamp": ts,
        "created_on": created,
        "is_processed": bool(row.get("is_processed")),
        "assigned_spot_id": row.get("assigned_spot_id"),
        "assigned_spot_name": row.get("assigned_spot_name"),
        "assigned_distance_ft": float(row["assigned_distance_ft"]) if row.get("assigned_distance_ft") is not None else None,
        "processed_comment": row.get("processed_comment"),
    }


def get_statistics() -> Dict:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    COUNT(*) AS total_records,
                    SUM(CASE WHEN is_processed = FALSE THEN 1 ELSE 0 END) AS unprocessed_count,
                    SUM(CASE WHEN is_processed = TRUE THEN 1 ELSE 0 END) AS processed_count
                FROM video_frame_records
            """)
            row = cur.fetchone()
    if not row:
        return {"total": 0, "unprocessed": 0, "processed": 0}
    return {
        "total": row["total_records"] or 0,
        "unprocessed": row["unprocessed_count"] or 0,
        "processed": row["processed_count"] or 0,
    }


def get_all_records(
    limit: int = 50,
    offset: int = 0,
    is_processed: Optional[bool] = None,
    camera_id: Optional[str] = None,
) -> List[Dict]:
    query = """
        SELECT id, licence_plate_trailer, latitude, longitude, speed, barrier,
               confidence, image_path, camera_id, video_path, frame_number,
               track_id, timestamp, created_on, is_processed,
               assigned_spot_id, assigned_spot_name, assigned_distance_ft, processed_comment
        FROM video_frame_records
        WHERE 1=1
    """
    params = []
    if is_processed is not None:
        query += " AND is_processed = %s"
        params.append(is_processed)
    if camera_id:
        query += " AND camera_id = %s"
        params.append(camera_id)
    query += " ORDER BY created_on DESC NULLS LAST LIMIT %s OFFSET %s"
    params.extend([limit, offset])

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
    return [_row_to_record(dict(r)) for r in rows]


def insert_records(records: List[Dict], device_id: Optional[str] = None) -> int:
    """Insert a batch of video frame records (processed data from device). Returns count inserted."""
    if not records:
        return 0
    insert_sql = """
        INSERT INTO video_frame_records (
            licence_plate_trailer, latitude, longitude, speed, barrier,
            confidence, image_path, camera_id, video_path, frame_number,
            track_id, timestamp, is_processed, device_id,
            assigned_spot_id, assigned_spot_name, assigned_distance_ft, processed_comment
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    rows = []
    for r in records:
        ts = r.get("timestamp") or r.get("created_on")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                ts = datetime.utcnow()
        elif ts is None:
            ts = datetime.utcnow()
        is_processed = r.get("is_processed")
        if is_processed is None:
            is_processed = True  # Device sends only processed records
        rows.append((
            (r.get("licence_plate_trailer") or "").strip() or None,
            r.get("latitude") if r.get("latitude") is not None else None,
            r.get("longitude") if r.get("longitude") is not None else None,
            r.get("speed"),
            r.get("barrier"),
            float(r.get("confidence") or 0),
            (r.get("image_path") or "").strip() or None,
            (r.get("camera_id") or "").strip() or None,
            (r.get("video_path") or "").strip() or None,
            int(r.get("frame_number") or 0),
            r.get("track_id") if r.get("track_id") is not None else None,
            ts,
            bool(is_processed),
            device_id,
            (r.get("assigned_spot_id") or "").strip() or None,
            (r.get("assigned_spot_name") or "").strip() or None,
            float(r["assigned_distance_ft"]) if r.get("assigned_distance_ft") is not None else None,
            (r.get("processed_comment") or "").strip() or None,
        ))
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(insert_sql, rows)
            return len(rows)
