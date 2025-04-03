import os
import uuid
import time
import threading
import subprocess
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import cv2
import numpy as np
from ultralytics import YOLO
import uvicorn
import torch
import asyncio

app = FastAPI(title="RTSP Object Detection Service")

# Configuration
MAX_CONCURRENT_JOBS = 4  # Limit number of concurrent processing jobs
PROCESSED_WIDTH = 640
PROCESSED_HEIGHT = 480
FRAME_RATE = 15  # Reduced from 30 for better performance

# MediaMTX RTSP server settings
RTSP_SERVER_PORT = 8554
RTSP_SERVER_PATH = f"rtsp://localhost:{RTSP_SERVER_PORT}/"

# Thread pool for limiting concurrent jobs
job_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS)

# Store active processing jobs
active_jobs = {}

# Central inference queue and results dictionary
inference_queue = queue.Queue(maxsize=100)
inference_results = {}

class RTSPRequest(BaseModel):
    input_rtsp: str
    task_type: str = "object_detection"
    detection_confidence: float = 0.5
    frame_skip: int = 0  # Process every nth frame (0 means process all)
    adaptive_frame_skip: bool = True  # Dynamically adjust frame skip based on performance
    resolution: str = "640x480"  # Format: WIDTHxHEIGHT
    use_motion_detection: bool = False  # Only process frames with motion

class RTSPResponse(BaseModel):
    job_id: str
    status: str
    output_rtsp: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

# Load YOLOv8 model - shared across all jobs
print("Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")  # Using the nano model; can be changed to s/m/l/x
model.to('cuda') if torch.cuda.is_available() else model.to('cpu')  # Move to GPU if available
print(f"Model loaded on {'GPU' if torch.cuda.is_available() else 'CPU'}")

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

def central_inference_worker():
    """Worker function for central inference processing"""
    print("Starting central inference worker...")
    batch_size = 4  # Process frames in batches for better GPU utilization
    batch_frames = []
    batch_ids = []
    
    while True:
        try:
            # Try to collect a batch of frames
            while len(batch_frames) < batch_size:
                try:
                    # Non-blocking get with timeout
                    frame_id, frame = inference_queue.get(timeout=0.1)
                    batch_frames.append(frame)
                    batch_ids.append(frame_id)
                except queue.Empty:
                    # If queue is empty and we have at least one frame, process what we have
                    if batch_frames:
                        break
                    continue
            
            if not batch_frames:
                continue
                
            # Convert frames to batch for YOLO model
            batch_tensor = np.stack(batch_frames)
            
            # Run inference on batch
            results = model(batch_tensor, conf=0.5, verbose=False)
            
            # Process results and store them
            for i, (frame_id, result) in enumerate(zip(batch_ids, results)):
                # Store result in the shared dictionary
                inference_results[frame_id] = result
                
            # Clear batch data
            batch_frames = []
            batch_ids = []
            
        except Exception as e:
            print(f"Error in central inference worker: {e}")
            # Clear batch data on error
            batch_frames = []
            batch_ids = []

def detect_motion(previous_frame, current_frame, threshold=25):
    """Detect motion between frames"""
    if previous_frame is None:
        return True
    
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference between frames
    frame_diff = cv2.absdiff(prev_gray, curr_gray)
    
    # Apply threshold to difference
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
    
    # Calculate percentage of pixels that changed
    motion_pixels = np.sum(thresh > 0)
    total_pixels = thresh.shape[0] * thresh.shape[1]
    motion_percentage = (motion_pixels / total_pixels) * 100
    
    # Return True if motion percentage exceeds threshold
    return motion_percentage > threshold

def process_rtsp_stream(job_id: str, input_rtsp: str, confidence: float, 
                       frame_skip: int, adaptive_frame_skip: bool, 
                       resolution: str, use_motion_detection: bool):
    """Process RTSP stream with YOLOv8 and output to another RTSP stream"""
    try:
        # Update job status
        active_jobs[job_id]["status"] = "processing"
        active_jobs[job_id]["metrics"] = {
            "processed_frames": 0,
            "skipped_frames": 0,
            "avg_processing_time": 0,
            "current_fps": 0,
            "start_time": time.time()
        }
        
        # Parse resolution
        try:
            width, height = map(int, resolution.split('x'))
        except:
            width, height = PROCESSED_WIDTH, PROCESSED_HEIGHT
            print(f"Invalid resolution format. Using default: {width}x{height}")
        
        # Generate unique stream name
        stream_name = f"detection_{job_id}"
        output_rtsp = RTSP_SERVER_PATH + stream_name
        active_jobs[job_id]["output_rtsp"] = output_rtsp
        
        # Use hardware acceleration if available
        hw_accel = []
        # Try to detect available hardware encoders
        try:
            encoders = subprocess.check_output(["ffmpeg", "-encoders"], universal_newlines=True)
            if "h264_nvenc" in encoders:
                hw_accel = ["-c:v", "h264_nvenc", "-preset", "p1"]
                print("Using NVIDIA hardware encoding")
            elif "h264_vaapi" in encoders:
                hw_accel = ["-vaapi_device", "/dev/dri/renderD128", "-vf", "format=nv12,hwupload", "-c:v", "h264_vaapi"]
                print("Using VAAPI hardware encoding")
            elif "h264_qsv" in encoders:
                hw_accel = ["-c:v", "h264_qsv", "-preset", "veryfast"]
                print("Using QuickSync hardware encoding")
            else:
                hw_accel = ["-c:v", "libx264", "-preset", "ultrafast"]
                print("Using software encoding")
        except:
            hw_accel = ["-c:v", "libx264", "-preset", "ultrafast"]
            print("Failed to detect hardware encoders, using software encoding")
        
        # Setup FFmpeg command for RTSP output to MediaMTX
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(FRAME_RATE),  # Reduced frame rate
            "-i", "-",  # Input from pipe
        ] + hw_accel + [
            "-pix_fmt", "yuv420p",
            "-g", "30",  # GOP size for better streaming
            "-bufsize", "2M",  # Buffer size for smoother streaming
            "-maxrate", "2M",  # Max bitrate
            "-r", str(FRAME_RATE),
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            output_rtsp
        ]
        
        # Start FFmpeg process
        ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd, 
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE  # Capture errors for debugging
        )
        
        # Open RTSP input stream
        cap = cv2.VideoCapture(input_rtsp)
        if not cap.isOpened():
            raise Exception(f"Failed to open RTSP stream: {input_rtsp}")
        
        # Initialize variables for adaptive frame skipping
        current_skip = frame_skip
        processing_times = []
        previous_frame = None
        frame_count = 0
        start_time = time.time()
        
        # Process frames
        while cap.isOpened() and active_jobs[job_id]["status"] == "processing":
            # Start frame processing timer
            frame_start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                # Try to reconnect if stream is lost
                print(f"Lost connection to {input_rtsp}, attempting to reconnect...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(input_rtsp)
                if not cap.isOpened():
                    print(f"Failed to reconnect to {input_rtsp}")
                    break
                continue
            
            # Apply frame skipping
            if current_skip > 0 and frame_count % (current_skip + 1) != 0:
                frame_count += 1
                active_jobs[job_id]["metrics"]["skipped_frames"] += 1
                continue
            
            # Resize frame for consistent output
            frame = cv2.resize(frame, (width, height))
            
            # Apply motion detection if enabled
            if use_motion_detection and previous_frame is not None:
                motion_detected = detect_motion(previous_frame, frame)
                if not motion_detected:
                    previous_frame = frame.copy()
                    frame_count += 1
                    active_jobs[job_id]["metrics"]["skipped_frames"] += 1
                    continue
            
            if use_motion_detection:
                previous_frame = frame.copy()
            
            # Generate frame ID for inference tracking
            frame_id = f"{job_id}_{frame_count}"
            
            # Put frame in inference queue
            try:
                inference_queue.put((frame_id, frame), block=False)
            except queue.Full:
                # Skip this frame if queue is full
                print(f"Inference queue full, skipping frame {frame_count}")
                frame_count += 1
                active_jobs[job_id]["metrics"]["skipped_frames"] += 1
                continue
            
            # Wait for inference result
            result = None
            wait_start = time.time()
            while time.time() - wait_start < 1.0:  # Wait up to 1 second for results
                if frame_id in inference_results:
                    result = inference_results.pop(frame_id)
                    break
                time.sleep(0.01)
            
            if result is None:
                # Timeout waiting for inference, use placeholder
                print(f"Timeout waiting for inference result for frame {frame_count}")
                # Draw a placeholder text instead
                cv2.putText(frame, "Processing...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 0, 255), 2, cv2.LINE_AA)
                annotated_frame = frame
            else:
                # Draw detection results on the frame
                annotated_frame = result.plot()
            
            # Send the processed frame to FFmpeg
            try:
                ffmpeg_process.stdin.write(annotated_frame.tobytes())
            except BrokenPipeError:
                print("FFmpeg pipe broken, restarting...")
                ffmpeg_process.kill()
                ffmpeg_process = subprocess.Popen(
                    ffmpeg_cmd, 
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE
                )
                continue
            
            # Calculate processing time for this frame
            processing_time = time.time() - frame_start_time
            processing_times.append(processing_time)
            if len(processing_times) > 30:
                processing_times.pop(0)
            
            # Update metrics
            frame_count += 1
            active_jobs[job_id]["metrics"]["processed_frames"] = frame_count
            active_jobs[job_id]["metrics"]["avg_processing_time"] = sum(processing_times) / len(processing_times)
            elapsed_time = time.time() - start_time
            active_jobs[job_id]["metrics"]["current_fps"] = frame_count / elapsed_time
            
            # Implement adaptive frame skipping
            if adaptive_frame_skip and len(processing_times) >= 10:
                avg_time = sum(processing_times[-10:]) / 10
                target_time = 1.0 / FRAME_RATE  # Target time per frame
                
                if avg_time > target_time * 1.2:  # Too slow
                    current_skip = min(10, current_skip + 1)
                elif avg_time < target_time * 0.8 and current_skip > 0:  # Too fast
                    current_skip = max(0, current_skip - 1)
            
            # Print stats every 100 frames
            if frame_count % 100 == 0:
                print(f"Job {job_id}: Processed {frame_count} frames at {active_jobs[job_id]['metrics']['current_fps']:.2f} FPS, "
                      f"Skip rate: {current_skip}, Avg processing time: {active_jobs[job_id]['metrics']['avg_processing_time']:.3f}s")
        
        # Clean up
        cap.release()
        ffmpeg_process.stdin.close()
        ffmpeg_process.terminate()
        ffmpeg_process.wait(timeout=5)
        
        print(f"Job {job_id} completed")
        active_jobs[job_id]["status"] = "completed"
        
    except Exception as e:
        import traceback
        print(f"Error processing RTSP stream: {e}")
        print(traceback.format_exc())
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["message"] = str(e)

async def start_processing_job(request: RTSPRequest) -> str:
    """Start a new processing job and return job ID"""
    job_id = str(uuid.uuid4())
    
    # Create job record
    active_jobs[job_id] = {
        "status": "starting",
        "input_rtsp": request.input_rtsp,
        "task_type": request.task_type,
        "output_rtsp": None,
        "message": None,
        "metrics": None
    }
    
    # Start processing in the thread pool
    job_executor.submit(
        process_rtsp_stream, 
        job_id, 
        request.input_rtsp, 
        request.detection_confidence, 
        request.frame_skip,
        request.adaptive_frame_skip,
        request.resolution,
        request.use_motion_detection
    )
    
    return job_id

@app.on_event("startup")
async def startup_event():
    """Run when the application starts"""
    check_mediamtx_status()
    
    # Start the central inference worker thread
    inference_thread = threading.Thread(target=central_inference_worker)
    inference_thread.daemon = True
    inference_thread.start()

@app.post("/process", response_model=RTSPResponse)
async def process_stream(request: RTSPRequest, background_tasks: BackgroundTasks):
    """API endpoint to start processing an RTSP stream"""
    if request.task_type != "object_detection":
        raise HTTPException(status_code=400, detail="Only object_detection task is supported")
    
    # Check if MediaMTX is running
    if not check_mediamtx_status():
        raise HTTPException(status_code=503, detail="MediaMTX server is not running. Please start it first.")
    
    # Check if we've reached the job limit
    active_job_count = sum(1 for job in active_jobs.values() if job["status"] in ["starting", "processing"])
    if active_job_count >= MAX_CONCURRENT_JOBS:
        raise HTTPException(
            status_code=429, 
            detail=f"Maximum number of concurrent jobs ({MAX_CONCURRENT_JOBS}) reached. Please try again later."
        )
    
    try:
        # Start the processing job
        job_id = await start_processing_job(request)
        
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
        performance_metrics=job["metrics"],
        message=job["message"]
    )

@app.delete("/stop/{job_id}", response_model=RTSPResponse)
async def stop_job(job_id: str):
    """Stop a processing job"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    active_jobs[job_id]["status"] = "stopping"
    
    # Give the processing loop time to clean up
    await asyncio.sleep(1)
    
    return RTSPResponse(
        job_id=job_id,
        status="stopped",
        message="Processing job has been stopped"
    )

@app.get("/jobs")
async def list_jobs():
    """List all active jobs"""
    return {
        "total_jobs": len(active_jobs),
        "active_jobs_count": sum(1 for job in active_jobs.values() if job["status"] in ["starting", "processing"]),
        "jobs": active_jobs
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    mediamtx_running = check_mediamtx_status()
    active_job_count = sum(1 for job in active_jobs.values() if job["status"] in ["starting", "processing"])
    return {
        "status": "healthy" if mediamtx_running else "degraded",
        "mediamtx_running": mediamtx_running,
        "active_jobs": active_job_count,
        "queue_size": inference_queue.qsize(),
        "max_concurrent_jobs": MAX_CONCURRENT_JOBS
    }

if __name__ == "__main__":
    # Add import for torch
    import torch
    # Add import for asyncio
    import asyncio
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)