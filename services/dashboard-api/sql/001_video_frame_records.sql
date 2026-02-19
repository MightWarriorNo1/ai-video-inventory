-- Video frame records table for AWS dashboard (mirrors device SQLite schema)
-- Run this on your PostgreSQL/TimescaleDB instance.

CREATE TABLE IF NOT EXISTS video_frame_records (
    id BIGSERIAL PRIMARY KEY,
    licence_plate_trailer TEXT,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    speed DOUBLE PRECISION,
    barrier DOUBLE PRECISION,
    confidence DOUBLE PRECISION DEFAULT 0,
    image_path TEXT,
    camera_id TEXT,
    video_path TEXT,
    frame_number INTEGER DEFAULT 0,
    track_id INTEGER,
    timestamp TIMESTAMPTZ,
    is_processed BOOLEAN DEFAULT FALSE,
    created_on TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_on TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    assigned_spot_id TEXT,
    assigned_spot_name TEXT,
    assigned_distance_ft DOUBLE PRECISION,
    processed_comment TEXT,
    device_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_video_frame_records_created ON video_frame_records(created_on DESC);
CREATE INDEX IF NOT EXISTS idx_video_frame_records_camera ON video_frame_records(camera_id, created_on DESC);
CREATE INDEX IF NOT EXISTS idx_video_frame_records_processed ON video_frame_records(is_processed) WHERE is_processed = FALSE;
CREATE INDEX IF NOT EXISTS idx_video_frame_records_licence ON video_frame_records(licence_plate_trailer);
