import os
import uuid
import time
import threading
import subprocess
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import cv2
import numpy as np
from ultralytics import YOLO
import uvicorn

app = FastAPI(title="RTSP Object Detection Service")

# Store active processing jobs
active_jobs = {}

class RTSPRequest(BaseModel):
    input_rtsp: str
    task_type: str = "object_detection"
    detection_confidence: float = 0.5
    frame_skip: int = 0  # Process every nth frame (0 means process all)

class RTSPResponse(BaseModel):
    job_id: str
    status: str
    output_rtsp: Optional[str] = None
    message: Optional[str] = None

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Using the nano model; can be changed to s/m/l/x for better performance

# MediaMTX RTSP server settings
RTSP_SERVER_PORT = 8554
RTSP_SERVER_PATH = f"rtsp://localhost:{RTSP_SERVER_PORT}/"

def check_mediamtx_status():
    """Check if MediaMTX is running"""
    try:
        # Check if the server is already running
        result = subprocess.run(
            ["pgrep", "-f", "mediamtx"], 
            capture_output=True, 
            text=True
        )
        
        if result.stdout:
            print("MediaMTX is running")
            return True
        else:
            print("Warning: MediaMTX is not running. Please start it before using this service.")
            return False
    except Exception as e:
        print(f"Error checking MediaMTX status: {e}")
        return False

def process_rtsp_stream(job_id: str, input_rtsp: str, confidence: float, frame_skip: int):
    """Process RTSP stream with YOLOv8 and output to another RTSP stream"""
    try:
        # Update job status
        active_jobs[job_id]["status"] = "processing"
        
        # Setup output path for temporary files if needed
        output_path = f"output_{job_id}"
        os.makedirs(output_path, exist_ok=True)
        
        # Generate unique stream name
        stream_name = f"detection_{job_id}"
        output_rtsp = RTSP_SERVER_PATH + stream_name
        active_jobs[job_id]["output_rtsp"] = output_rtsp
        
        # Setup FFmpeg command for RTSP output to MediaMTX
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", "640x480",  # Adjust based on your camera resolution
            "-r", "30",  # Frame rate
            "-i", "-",  # Input from pipe
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "ultrafast",
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            output_rtsp
        ]
        
        # Start FFmpeg process
        ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd, 
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Open RTSP input stream
        cap = cv2.VideoCapture(input_rtsp)
        if not cap.isOpened():
            raise Exception(f"Failed to open RTSP stream: {input_rtsp}")
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Set resolution for processed output
        processed_width = 640
        processed_height = 480
        
        frame_count = 0
        start_time = time.time()
        
        # Process frames
        while cap.isOpened() and active_jobs[job_id]["status"] == "processing":
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every nth frame if frame_skip > 0
            if frame_skip > 0 and frame_count % (frame_skip + 1) != 0:
                frame_count += 1
                continue
                
            # Resize frame for consistent output
            frame = cv2.resize(frame, (processed_width, processed_height))
            
            # Perform object detection with YOLOv8
            results = model(frame, conf=confidence)
            
            # Draw detection results on the frame
            annotated_frame = results[0].plot()
            
            # Send the processed frame to FFmpeg
            ffmpeg_process.stdin.write(annotated_frame.tobytes())
            
            frame_count += 1
            
            # Print stats every 100 frames
            if frame_count % 100 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                print(f"Job {job_id}: Processed {frame_count} frames at {fps:.2f} FPS")
        
        # Clean up
        cap.release()
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
        
        print(f"Job {job_id} completed")
        active_jobs[job_id]["status"] = "completed"
        
    except Exception as e:
        print(f"Error processing RTSP stream: {e}")
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["message"] = str(e)

def start_processing_job(request: RTSPRequest) -> str:
    """Start a new processing job and return job ID"""
    job_id = str(uuid.uuid4())
    
    # Create job record
    active_jobs[job_id] = {
        "status": "starting",
        "input_rtsp": request.input_rtsp,
        "task_type": request.task_type,
        "output_rtsp": None,
        "message": None
    }
    
    # Start processing in a separate thread
    thread = threading.Thread(
        target=process_rtsp_stream, 
        args=(job_id, request.input_rtsp, request.detection_confidence, request.frame_skip)
    )
    thread.daemon = True
    thread.start()
    
    return job_id

@app.on_event("startup")
def startup_event():
    """Run when the application starts"""
    check_mediamtx_status()

@app.post("/process", response_model=RTSPResponse)
async def process_stream(request: RTSPRequest, background_tasks: BackgroundTasks):
    """API endpoint to start processing an RTSP stream"""
    if request.task_type != "object_detection":
        raise HTTPException(status_code=400, detail="Only object_detection task is supported")
    
    # Check if MediaMTX is running
    if not check_mediamtx_status():
        raise HTTPException(status_code=503, detail="MediaMTX server is not running. Please start it first.")
    
    try:
        # Start the processing job
        job_id = start_processing_job(request)
        
        # Return initial response
        return RTSPResponse(
            job_id=job_id,
            status="starting",
            message="Processing job started. Check status endpoint for updates."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{job_id}", response_model=RTSPResponse)
async def get_job_status(job_id: str):
    """Get the status of a processing job"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = active_jobs[job_id]
    
    return RTSPResponse(
        job_id=job_id,
        status=job["status"],
        output_rtsp=job["output_rtsp"],
        message=job["message"]
    )

@app.delete("/stop/{job_id}", response_model=RTSPResponse)
async def stop_job(job_id: str):
    """Stop a processing job"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    active_jobs[job_id]["status"] = "stopping"
    
    # Give the processing loop time to clean up
    time.sleep(1)
    
    return RTSPResponse(
        job_id=job_id,
        status="stopped",
        message="Processing job has been stopped"
    )

@app.get("/jobs", response_model=dict)
async def list_jobs():
    """List all active jobs"""
    return {
        "total_jobs": len(active_jobs),
        "jobs": active_jobs
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    mediamtx_running = check_mediamtx_status()
    return {
        "status": "healthy" if mediamtx_running else "degraded",
        "mediamtx_running": mediamtx_running,
        "active_jobs": len(active_jobs)
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)