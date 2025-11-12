"""
REST Ingest API for Trailer Events

FastAPI service that receives trailer detection events and writes them to TimescaleDB.
"""

import os
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import psycopg2
from psycopg2 import pool
import uvicorn


# Database connection
DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'postgresql://postgres:postgres@localhost:5432/trailer_vision'
)


# Pydantic models
class TrailerEvent(BaseModel):
    """Trailer detection event model."""
    ts_iso: str = Field(..., description="ISO timestamp")
    camera_id: str = Field(..., description="Camera identifier")
    track_id: Optional[int] = Field(None, description="Track ID")
    bbox: Optional[str] = Field(None, description="Bounding box (x1,y1,x2,y2)")
    text: Optional[str] = Field(None, description="Recognized text")
    conf: Optional[float] = Field(None, description="Confidence score")
    x_world: Optional[float] = Field(None, description="World X coordinate")
    y_world: Optional[float] = Field(None, description="World Y coordinate")
    spot: Optional[str] = Field(None, description="Parking spot name")
    method: Optional[str] = Field(None, description="Spot resolution method")


class EventBatch(BaseModel):
    """Batch of events."""
    events: List[TrailerEvent]


# FastAPI app
app = FastAPI(
    title="Trailer Vision Ingest API",
    description="REST API for ingesting trailer detection events into TimescaleDB",
    version="1.0.0"
)


# Database connection pool
db_pool: Optional[pool.ThreadedConnectionPool] = None


@app.on_event("startup")
def startup():
    """Initialize database connection pool."""
    global db_pool
    try:
        db_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            dsn=DATABASE_URL
        )
        print(f"Connected to database: {DATABASE_URL}")
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        raise


@app.on_event("shutdown")
def shutdown():
    """Close database connection pool."""
    global db_pool
    if db_pool:
        db_pool.closeall()
        print("Database connection pool closed")


@app.get("/healthz")
async def healthz():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.post("/events")
async def ingest_event(event: TrailerEvent):
    """
    Ingest a single trailer event.
    
    Args:
        event: Trailer event data
        
    Returns:
        Success response
    """
    if db_pool is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        conn = db_pool.getconn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO trailer_events (
                    ts, camera_id, track_id, bbox, text, conf,
                    x_world, y_world, spot, method
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    datetime.fromisoformat(event.ts_iso.replace('Z', '+00:00')),
                    event.camera_id,
                    event.track_id,
                    event.bbox,
                    event.text,
                    event.conf,
                    event.x_world,
                    event.y_world,
                    event.spot,
                    event.method
                )
            )
            conn.commit()
            cur.close()
        finally:
            db_pool.putconn(conn)
        
        return {"status": "success", "message": "Event ingested"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting event: {str(e)}")


@app.post("/events/batch")
async def ingest_events_batch(batch: EventBatch):
    """
    Ingest a batch of trailer events.
    
    Args:
        batch: Batch of events
        
    Returns:
        Success response with count
    """
    if db_pool is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        conn = db_pool.getconn()
        try:
            cur = conn.cursor()
            for event in batch.events:
                cur.execute(
                    """
                    INSERT INTO trailer_events (
                        ts, camera_id, track_id, bbox, text, conf,
                        x_world, y_world, spot, method
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        datetime.fromisoformat(event.ts_iso.replace('Z', '+00:00')),
                        event.camera_id,
                        event.track_id,
                        event.bbox,
                        event.text,
                        event.conf,
                        event.x_world,
                        event.y_world,
                        event.spot,
                        event.method
                    )
                )
            conn.commit()
            cur.close()
        finally:
            db_pool.putconn(conn)
        
        return {"status": "success", "count": len(batch.events)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting batch: {str(e)}")


@app.get("/events")
async def get_events(
    camera_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    Query events from database.
    
    Args:
        camera_id: Filter by camera ID
        limit: Maximum number of events
        offset: Offset for pagination
        
    Returns:
        List of events
    """
    if db_pool is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        conn = db_pool.getconn()
        try:
            cur = conn.cursor()
            if camera_id:
                cur.execute(
                    """
                    SELECT ts, camera_id, track_id, bbox, text, conf,
                           x_world, y_world, spot, method
                    FROM trailer_events
                    WHERE camera_id = %s
                    ORDER BY ts DESC
                    LIMIT %s OFFSET %s
                    """,
                    (camera_id, limit, offset)
                )
            else:
                cur.execute(
                    """
                    SELECT ts, camera_id, track_id, bbox, text, conf,
                           x_world, y_world, spot, method
                    FROM trailer_events
                    ORDER BY ts DESC
                    LIMIT %s OFFSET %s
                    """,
                    (limit, offset)
                )
            
            rows = cur.fetchall()
            cur.close()
        finally:
            db_pool.putconn(conn)
        
        events = []
        for row in rows:
            events.append({
                'ts_iso': row[0].isoformat() if row[0] else None,
                'camera_id': row[1],
                'track_id': row[2],
                'bbox': row[3],
                'text': row[4],
                'conf': row[5],
                'x_world': row[6],
                'y_world': row[7],
                'spot': row[8],
                'method': row[9]
            })
        
        return {"events": events, "count": len(events)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying events: {str(e)}")


def main():
    """Run the ingest API server."""
    port = int(os.getenv('INGEST_PORT', '8000'))
    uvicorn.run(app, host='0.0.0.0', port=port)


if __name__ == '__main__':
    main()

