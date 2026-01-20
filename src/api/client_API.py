from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, Response, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import asyncio
from time import sleep
from contextlib import asynccontextmanager
import subprocess
import threading

from DatabaseManagerPostgre import PostgreDatabaseManager
from ServoControl import ContinuousServo
try:
    from ..pipeline.DetectionWithGStreamer_pipeline import HailoTrackerPipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    print("[WARNING] Pipeline module not found - AI endpoints will be limited")

## MODELS
# Servo control request
class ServoRequest(BaseModel):
    angle: int = Field(..., ge=-180, le=180, description="Servo position (-180 to 180 degrees)")
    speed: int = Field(50, ge=0, le=100, description="Speed percentage (0 to 100%)")
# Camera settings for capture
class CameraSettings(BaseModel):
    width: int = Field(1536, ge=1536, le=4608)
    height: int = Field(864, ge=864, le=2592)
# Pipeline configuration model
class PipelineConfig(BaseModel):
    """Configuration for starting AI pipeline"""
    rpicamera: bool = Field(False, description="Use Raspberry Pi camera")
    video_file: Optional[str] = Field(None, description="Path to video file")
    enable_rtsp: bool = Field(False, description="Enable RTSP streaming")
    enable_websocket: bool = Field(False, description="Enable WebSocket")
    enable_webrtc: bool = Field(False, description="Enable WebRTC")
    enable_mqtt: bool = Field(True, description="Enable MQTT")
    enable_recording: bool = Field(False, description="Enable video recording")
    output_path: Optional[str] = Field(None, description="Output video path")
    enable_debug: bool = Field(False, description="Enable debug mode")
# Pipeline status response model
class PipelineStatus(BaseModel):
    """Pipeline status response"""
    running: bool
    mode: Optional[str]
    fps: Optional[float]
    total_tracked: Optional[int]
    config: Optional[Dict[str, Any]]
# Detection response model
class DetectionResponse(BaseModel):
    """Response model for detection data via REST API."""
    timestamp: str
    frame_id: int
    label: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    track_id: Optional[int]
# Statistics response model
class StatisticsResponse(BaseModel):
    """Response model for traffic statistics."""
    class_name: str
    total_detections: int
    unique_tracks: int
    avg_confidence: float
    min_confidence: float
    max_confidence: float
# Health check response model for database
class HealthResponseDB(BaseModel):
    """Health check response."""
    status: str
    total_detections: int
    latest_detection: Optional[str]
    database_size: str

# Global database instance
db: Optional[PostgreDatabaseManager] = None
servo_L: Optional[ContinuousServo] = None
pipeline: Optional['HailoTrackerPipeline'] = None
pipeline_thread: Optional[threading.Thread] = None
pipeline_config: Optional[Dict[str, Any]] = None

# database configuration
DB_CONFIG = {'host': '192.168.37.31', 'port': 5432, 'database': 'hailo_db', 'user': 'hailo_user', 'password': 'hailo_pass', 'min_connections': 2, 'max_connections': 10 }

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager. Handles startup and shutdown services.
    """
    global servo_L, db
    # Startup
    print("[FastAPI] Starting up...")
    # Initialize database
    try:
        # Create database manager instance
        db = PostgreDatabaseManager(DB_CONFIG)
        print("[FastAPI] Database connected successfully")
    except Exception as e:
        print(f"[FastAPI] Database connection failed: {e}")
        db = None
    # Initialize servo
    try:
        servo_L = ContinuousServo(chip=0, pin=18) # FIRST SERVO ON PIN 18
        print("[FastAPI] Servo initialized")
    except Exception as e:
        print(f"[FastAPI] Servo initialization failed: {e}")
        servo_L = None
    yield
    # Shutdown
    print("[FastAPI] Shutting down...")
    if servo_L: servo_L.cleanup()
    if db: db.close()

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)

## CONFIGURATION FOR CAPTURE STORAGE
CAPTURE_DIR = Path("captures")
CAPTURE_DIR.mkdir(exist_ok=True)

# Allow frontend access
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],)

## HELPER FUNCTIONS
def check_pipeline_instance():
    """Check if pipeline instance exists and is properly initialized"""
    global pipeline
    if pipeline is None:
        return False, "Pipeline not initialized"
    return True, "Pipeline instance available"
def check_db_instance():
    """Check if database instance exists and is connected"""
    global db
    if db is None:
        return False, "Database not initialized"
    try:
        if db.verify_pool():
            return True, "Database connected"
        else:
            return False, "Database connection pool verification failed"
    except Exception as e:
        return False, f"Database error: {str(e)}"

## SERVO MOVE ENDPOINT
@app.post("/servo/move")
async def move_servo(data: ServoRequest):
    """Move servo from -180 to 180 degrees)"""
    if not (-180 <= data.angle <= 180):
        raise HTTPException(400, "Angle must be between -180–180°")
    try:
	    # set angle
        pulse = await servo_L.rotate_degrees(data.angle,data.speed)
        return {"status": "ok", "angle_set": data.angle, "speed_value": pulse}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Servo error: {str(e)}")

## DATABASE MANAGEMENT ENDPOINTS
@app.get("/database/status")
async def database_status():
    """Check database connection status"""
    is_connected, message = check_db_instance()
    if is_connected and db:
        try:
            health = db.get_health_status()
            return { "connected": True, "status": health['status'], "message": message, "health": health }
        except Exception as e:
            return { "connected": False, "status": "error", "message": f"Health check failed: {str(e)}" }    
    return { "connected": False, "status": "disconnected", "message": message }

@app.post("/database/start")
async def database_start():
    """Start/initialize database connection"""
    global db    
    if db is not None:
        is_connected, message = check_db_instance()
        if is_connected:
            return { "status": "already_running", "message": "Database is already connected" }
    try:
        db = PostgreDatabaseManager(DB_CONFIG)
        return { "status": "started", "message": "Database connection established successfully" }
    except Exception as e:
        db = None
        raise HTTPException( status_code=500, detail=f"Failed to start database connection: {str(e)}" )

@app.post("/database/stop")
async def database_stop():
    """Stop/close database connection"""
    global db
    if db is None:
        return { "status": "already_stopped", "message": "Database is not running" }
    try:
        db.close()
        db = None
        return { "status": "stopped", "message": "Database connection closed successfully"
        }
    except Exception as e:
        raise HTTPException( status_code=500, detail=f"Failed to stop database: {str(e)}" )

@app.post("/database/restart")
async def database_restart():
    """Restart database connection"""
    global db
    # Stop if running
    if db is not None:
        try:
            db.close()
        except Exception as e:
            print(f"[FastAPI] Error closing database: {e}")
        db = None
    # Start fresh connection
    try:
        db = PostgreDatabaseManager(DB_CONFIG)
        return { "status": "restarted", "message": "Database connection restarted successfully" }
    except Exception as e:
        db = None
        raise HTTPException( status_code=500, detail=f"Failed to restart database: {str(e)}" )

app.get("/health/ai", response_model=HealthResponseDB)
async def health_check_ai():
    """Health check endpoint for database."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    try:
        health = db.get_health_status()
        return HealthResponseDB(
            status=health['status'],
            total_detections=health['total_detections'],
            latest_detection=health['latest_detection'].isoformat() if health['latest_detection'] else None,
            database_size=health['database_size']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

## DATABASE QUERIES ENDPOINTS
# STATISTICS FROM DB ENDPOINT
@app.get("/api/detections", response_model=List[DetectionResponse])
async def get_detections(minutes: int = 5, limit: int = 100):
    """
    Get recent detections from database.
    Query parameters:
    - minutes: Look back N minutes (default: 5)
    - limit: Maximum records to return (default: 100)
    Example: GET /api/detections?minutes=10&limit=50
    """
    if db is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    try:
        detections = db.get_recent_detections(minutes=minutes, limit=limit)
        return [
            DetectionResponse(
                id=d['id'],
                timestamp=d['timestamp_'].isoformat() if hasattr(d['timestamp_'], 'isoformat') else str(d['timestamp_']),
                frame_id=d['frame_id'],
                class_name=d['class_name'],
                confidence=d['confidence'],
                bbox=[d['x1'], d['y1'], d['x2'], d['y2']],
                track_id=d['track_id']
            )
            for d in detections
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching detections: {str(e)}")

# REAL-TIME CURRENT STATE ENDPOINT
@app.get("/api/statistics", response_model=Dict[str, StatisticsResponse])
async def get_statistics(hours: int = 1):
    """
    Get aggregated traffic statistics.
    Returns count and average confidence for each object type
    within the specified time window.
    Args:
        hours: Time window in hours (default: 1)
    Example: GET /api/statistics?hours=24
    """
    if db is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    try:
        stats = db.get_statistics(hours=hours)
        return {
            class_name: StatisticsResponse(
                class_name=class_name,
                total_detections=data['total_detections'],
                unique_tracks=data['unique_tracks'],
                avg_confidence=data['avg_confidence'],
                min_confidence=data['min_confidence'],
                max_confidence=data['max_confidence']
            )
            for class_name, data in stats.items()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching statistics: {str(e)}")

# BULK INSERT MULTIPLE DETECTIONS ENDPOINT
@app.post("/api/detections/bulk")
async def create_bulk_detections(detections: List[Dict[str, Any]]):
    """Manually insert bulk detections into database."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not initialized")    
    try:
        db.insert_batch_detections(detections)
        return { "message": "Detections inserted successfully", "count": len(detections) }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inserting detections: {str(e)}")

## AI PIPELINE DETECTION ENDPOINTS
@app.get("/pipeline/status", response_model=PipelineStatus)
async def pipeline_status():
    """Get current pipeline status"""
    if not PIPELINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Pipeline module not available" ) 
    if pipeline is None:
        return PipelineStatus(running=False, mode=None, fps=None, total_tracked=None, config=None )
    try:
        # Determine mode
        mode = "unknown"
        if pipeline_config:
            if pipeline_config.get('video_file'):
                mode = "video_file"
            elif pipeline_config.get('rpicamera'):
                mode = "rpicamera"
            else:
                mode = "rtsp_stream"
        return PipelineStatus(
            running=getattr(pipeline, 'running', False),
            mode=mode,
            fps=getattr(pipeline, 'fps_actual', 0.0),
            total_tracked=pipeline.counter.get_total_count() if hasattr(pipeline, 'counter') else 0,
            config=pipeline_config
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting pipeline status: {str(e)}" )


@app.post("/pipeline/start")
async def pipeline_start(config: PipelineConfig):
    """Start AI detection pipeline with specified configuration"""
    global pipeline, pipeline_thread, pipeline_config
    if not PIPELINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Pipeline module not available" )
    # Check if already running
    if pipeline is not None and getattr(pipeline, 'running', False):
        raise HTTPException(status_code=409, detail="Pipeline is already running. Stop it first." )
    # Check database connection
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected. Start database first." )
    try:
        # Generate output path if recording enabled
        output_path = config.output_path
        if config.enable_recording and not output_path:
            if config.video_file:
                import os
                base_name = os.path.splitext(config.video_file)[0]
                output_path = f"{base_name}_processed.mp4"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"hailo_output_{timestamp}.mp4"
        # Store configuration
        pipeline_config = config.dict()
        pipeline_config['output_path'] = output_path
        # Create pipeline instance
        pipeline = HailoTrackerPipeline(
            rpicamera=config.rpicamera,
            video_file=config.video_file,
            enable_rtsp=config.enable_rtsp,
            enable_websocket=config.enable_websocket,
            enable_webrtc=config.enable_webrtc,
            enable_mqtt=config.enable_mqtt,
            enable_recording=config.enable_recording,
            output_video_path=output_path,
            enable_debug=config.enable_debug,
        )
        # Set database instance if pipeline expects it
        if hasattr(pipeline, 'db') and pipeline.db is None:
            pipeline.db = db
        # Create and start pipeline
        pipeline.create_pipeline()
        # Start in separate thread
        def run_pipeline():
            try:
                pipeline.start()
            except Exception as e:
                print(f"[Pipeline Thread] Error: {e}")
        pipeline_thread = threading.Thread(target=run_pipeline, daemon=True)
        pipeline_thread.start()
        # Wait a moment to ensure it starts
        await asyncio.sleep(1)
        return { "status": "started", "message": "Pipeline started successfully", "config": pipeline_config }
    except Exception as e:
        pipeline = None
        pipeline_config = None
        raise HTTPException( status_code=500, detail=f"Failed to start pipeline: {str(e)}" )

@app.post("/pipeline/stop")
async def pipeline_stop():
    """Stop AI detection pipeline"""
    global pipeline, pipeline_config
    if pipeline is None:
        return { "status": "already_stopped", "message": "Pipeline is not running" }
    try:
        # Stop the pipeline
        if hasattr(pipeline, 'stop'):
            pipeline.stop()
        # Clear references
        pipeline = None
        pipeline_config = None
        return { "status": "stopped", "message": "Pipeline stopped successfully" }
    except Exception as e:
        raise HTTPException( status_code=500, detail=f"Failed to stop pipeline: {str(e)}" )

@app.post("/pipeline/restart")
async def pipeline_restart(config: Optional[PipelineConfig] = None):
    """Restart pipeline with same or new configuration"""
    global pipeline, pipeline_config
    # Stop if running
    if pipeline is not None:
        try:
            if hasattr(pipeline, 'stop'):
                pipeline.stop()
            await asyncio.sleep(1)
        except Exception as e:
            print(f"[FastAPI] Error stopping pipeline: {e}")
        pipeline = None
    # Use previous config if not provided
    if config is None:
        if pipeline_config is None:
            raise HTTPException( status_code=400, detail="No previous configuration found. Provide new configuration." )
        config = PipelineConfig(**pipeline_config)
    # Start with configuration
    return await pipeline_start(config)

@app.get("/pipeline/modes")
async def pipeline_modes():
    """Get available pipeline modes and their descriptions"""
    return {
        "modes": {
            "rpicamera": {
                "description": "Use Raspberry Pi Camera (IMX708)",
                "config": {"rpicamera": True}
            },
            "rtsp_stream": {
                "description": "Use RTSP IP Camera Stream",
                "config": {"rpicamera": False, "video_file": None}
            },
            "video_file": {
                "description": "Analyze video file",
                "config": {"video_file": "/path/to/video.mp4"}
            }
        },
        "features": {
            "enable_rtsp": "Enable RTSP streaming output",
            "enable_websocket": "Enable WebSocket data streaming",
            "enable_webrtc": "Enable WebRTC streaming",
            "enable_mqtt": "Enable MQTT message publishing",
            "enable_recording": "Record processed video output",
            "enable_debug": "Enable debug logging"
        }
    }

@app.get("/api/current/ai")
async def get_current_state():
    """Get current real-time state from running pipeline"""
    if pipeline is None:
        raise HTTPException( status_code=503, detail="Pipeline not running" )
    try:
        current_detections = []
        if hasattr(pipeline, 'current_detections'):
            current_detections = [
                det.to_dict() if hasattr(det, 'to_dict') else det
                for det in pipeline.current_detections
            ]
        return {
            "frame_id": getattr(pipeline, 'frame_id', 0),
            "timestamp": datetime.now().isoformat(),
            "detections": current_detections,
            "statistics": {
                "total_count": pipeline.counter.get_total_count() if hasattr(pipeline, 'counter') else 0,
                "counts_by_type": pipeline.counter.get_counts() if hasattr(pipeline, 'counter') else {},
                "fps": round(getattr(pipeline, 'fps_actual', 0.0), 1)
            }
        }
    except Exception as e:
        raise HTTPException( status_code=500, detail=f"Error getting current state: {str(e)}")

## CAMERA ENDPOINTS
# CAMERA CAPTURE (PHOTO)
@app.post("/camera/stream/capture")
async def camera_capture(settings: Optional[CameraSettings] = None):
    """Capture a single photo from the camera"""
    if settings is None:
        settings = CameraSettings()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = CAPTURE_DIR / f"capture_{timestamp}.jpg"
    try:
        result = subprocess.run([
            "rpicam-still",
            "-n",
            "--width", str(settings.width),
            "--height", str(settings.height),
            "-o", str(output)
        ], capture_output=True, timeout=10)
        
        if result.returncode != 0:
            raise HTTPException(
                status_code=500, 
                detail=f"Camera capture failed: {result.stderr.decode()}"
            )
        if not output.exists():
            raise HTTPException(status_code=500, detail="Output file not created")
        return FileResponse(output, media_type="image/jpeg", filename=output.name)
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Camera capture timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# CAMERA STREAM (SNAPSHOT STREAM)
@app.get("/camera/stream/snapshots")
def stream_snapshots(width: int = 1536, height: int = 864, framerate: int = 30):
    """
    Stream jednotlivých JPEG snímků
    Spolehlivější než video stream
    """
    def generate_snapshots():
        frame_delay = 1.0 / framerate
        try:
            while True:
                # Zachytíme jeden frame
                temp_img = "/tmp/stream_frame.jpg"               
                result = subprocess.run([
                    "rpicam-still",
                    "-n",
                    "--width", str(width),
                    "--height", str(height),
                    "-o", temp_img,
                    "-t", "1"  # Rychlé zachycení
                ], capture_output=True, timeout=2)
                if result.returncode != 0:
                    sleep(frame_delay)
                    continue
                # Načteme obrázek
                with open(temp_img, 'rb') as f:
                    frame_bytes = f.read()
                # Pošleme jako MJPEG frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + 
                       frame_bytes + b'\r\n')
                sleep(frame_delay)
        except Exception as e:
            print(f"Stream error: {e}")
    return StreamingResponse(
        generate_snapshots(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

#CAMERA VIEWER (IN HTML)
@app.get("/camera/stream/hls")
async def camera_viewer():
    """
    Jednoduchá HTML stránka pro zobrazení streamu
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Raspberry Pi Camera Stream</title>
        <style> body { font-family: Arial, sans-serif; max-width: 1200px; margin: 50px auto; padding: 20px; background: #1a1a1a; color: white;} .stream-container { margin: 20px 0; border: 2px solid #333; border-radius: 8px; overflow: hidden;} img { width: 100%; height: auto; display: block; } h2 { color: #4CAF50; } .info { background: #333; padding: 15px; border-radius: 5px; margin: 10px 0; } button { background: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px;} button:hover { background: #45a049; }</style>
    </head>
    <body>
        <h1>Raspberry Pi Camera Stream</h1>
        <div class="stream-container">
            <img id="stream" src="/camera/stream/snapshots" alt="Camera Stream">
        </div> 
        <script>
            function switchStream(type) {
                const streamImg = document.getElementById('stream');
                const currentStream = document.getElementById('current-stream');
                const streamUrl = `/camera/stream/${type}`;
                streamImg.src = streamUrl;
                currentStream.textContent = streamUrl;
                console.log('Switching to:', streamUrl);
            } 
            // Kontrola stavu streamu
            const streamImg = document.getElementById('stream');
            const status = document.getElementById('status');
            streamImg.onload = () => { status.textContent = 'Running'; status.style.color = '#4CAF50'; };
            streamImg.onerror = () => { status.textContent = 'Error'; status.style.color = '#f44336'; };
        </script>
    </body>
    </html>
    """    
    return Response(content=html_content, media_type="text/html")

# UTILITY ENDPOINTS FOR CAPTURES STORAGE MANAGEMENT
@app.get("/captures")
async def list_captures():
    """List all captured images"""
    captures = sorted(CAPTURE_DIR.glob("*.jpg"), reverse=True)
    return {
        "total": len(captures),
        "captures": [f.name for f in captures[:20]]  # Last 20
    }

@app.delete("/captures/{filename}")
async def delete_capture(filename: str):
    """Delete a specific capture"""
    filepath = CAPTURE_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")
    filepath.unlink()
    return {"status": "deleted", "filename": filename}

# HEALTH CHECK ENDPOINT
@app.get("/health/camera")
async def health_check():
    """Check if camera is accessible"""
    try:
        result = subprocess.run(
            ["rpicam-hello", "--list-cameras"],
            capture_output=True,
            timeout=5
        )
        return {
            "status": "healthy",
            "camera_available": result.returncode == 0
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
    
# ROOT ENDPOINT
@app.get("/")
def root():
    """API root with endpoint documentation"""
    return {
        "message": "Raspberry Pi Control API",
        "version": "2.0.0",
        "endpoints": {
            "servo": {
                "move": "POST /servo/move"
            },
            "database": {
                "status": "GET /database/status",
                "start": "POST /database/start",
                "stop": "POST /database/stop",
                "restart": "POST /database/restart",
                "health": "GET /health/ai",
            },
            "pipeline": {
                "status": "GET /pipeline/status",
                "start": "POST /pipeline/start",
                "stop": "POST /pipeline/stop",
                "restart": "POST /pipeline/restart",
                "modes": "GET /pipeline/modes",
                "current_state": "GET /api/current/ai",
            },
            "ai_detection": {
                "detections": "GET /api/detections?minutes=5&limit=100",
                "statistics": "GET /api/statistics?hours=1",
                "bulk_insert": "POST /api/detections/bulk"
            },
            "camera": {
                "capture": "POST /camera/stream/capture",
                "stream": "GET /camera/stream/snapshots",
                "viewer": "GET /camera/stream/hls",
                "list_captures": "GET /captures",
                "delete_capture": "DELETE /captures/{filename}",
                "health": "GET /health/camera",
            },
        }
    }

if __name__ == "__main__":
    """
    Start FastAPI server with Uvicorn.
    Configuration:
    - host: 0.0.0.0 (accessible from other devices on network)
    - port: 8000
    - log_level: info
    Run:
        python script.py
    Or with custom settings:
        uvicorn script:app --host 0.0.0.0 --port 8000 --reload
    Access:
    - API Docs: http://localhost:8000/docs
    - WebSocket: ws://localhost:8000/ws
    - Video Stream: http://localhost:8000/stream
    """
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",  # Listen on all interfaces
        port=8000,
        log_level="info",
        access_log=True
    )