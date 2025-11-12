-- TimescaleDB initialization for trailer events

-- Create hypertable for time-series data
CREATE TABLE IF NOT EXISTS trailer_events (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    camera_id VARCHAR(100) NOT NULL,
    track_id INTEGER,
    bbox TEXT,
    text VARCHAR(100),
    conf REAL,
    x_world REAL,
    y_world REAL,
    spot VARCHAR(50),
    method VARCHAR(50)
);

-- Create hypertable (TimescaleDB extension)
SELECT create_hypertable('trailer_events', 'ts', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_trailer_events_camera_id ON trailer_events(camera_id);
CREATE INDEX IF NOT EXISTS idx_trailer_events_ts ON trailer_events(ts DESC);
CREATE INDEX IF NOT EXISTS idx_trailer_events_spot ON trailer_events(spot);
CREATE INDEX IF NOT EXISTS idx_trailer_events_track_id ON trailer_events(track_id);

-- Create continuous aggregate for hourly statistics (optional)
CREATE MATERIALIZED VIEW IF NOT EXISTS trailer_events_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', ts) AS hour,
    camera_id,
    spot,
    COUNT(*) AS event_count,
    AVG(conf) AS avg_confidence
FROM trailer_events
GROUP BY hour, camera_id, spot;

-- Add refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy('trailer_events_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);


