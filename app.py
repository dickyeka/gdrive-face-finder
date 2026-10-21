import os
import sys
import json
import uuid
import time
import zipfile
import io
import asyncio
import threading
import urllib.request
from pathlib import Path
from typing import Dict, Any, List

from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, StreamingResponse, FileResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles

APP_DIR = Path(__file__).parent.resolve()
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import server
import gdown

APP_DIR = Path(__file__).parent.resolve()
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = APP_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

JOBS: Dict[str, Dict[str, Any]] = {}
engine = server.get_active_engine()


def extract_folder_id(url: str) -> str:
    """Extract folder ID from Google Drive URL or raw ID."""
    clean = url.strip()
    if "/folders/" in clean:
        fid = clean.split("/folders/")[1].split("?")[0].split("/")[0]
        return fid
    if "id=" in clean:
        fid = clean.split("id=")[1].split("&")[0]
        return fid
    return clean


def run_scan_job(job_id: str, folder_url: str, ref_img_bytes: bytes, threshold: float):
    job = JOBS[job_id]
    job_dir = DATA_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save reference image
        ref_path = job_dir / "reference.jpg"
        ref_path.write_bytes(ref_img_bytes)
        
        job["status"] = "detecting_reference"
        job["message"] = "Detecting face in reference photo..."
        
        ref_faces = engine.extract_encodings(str(ref_path))
        if not ref_faces:
            job["status"] = "error"
            job["error"] = "No face could be detected in the uploaded reference photo. Please upload a clear photo with a visible face."
            return
            
        ref_meta, ref_enc = ref_faces[0]
        job["ref_detected"] = True
        
        # Discover Drive files
        job["status"] = "fetching_drive"
        job["message"] = "Connecting to Google Drive and retrieving album contents..."
        folder_id = extract_folder_id(folder_url)
        
        gdrive_files = gdown.download_folder(id=folder_id, skip_download=True)
        img_files = [f for f in gdrive_files if getattr(f, "path", "").lower().endswith((".jpg", ".jpeg", ".png"))]
        
        if not img_files:
            job["status"] = "error"
            job["error"] = "No supported image files (.jpg, .jpeg, .png) found in the Google Drive folder."
            return
            
        job["total_photos"] = len(img_files)
        job["status"] = "scanning"
        job["message"] = f"Found {len(img_files)} photos. Scanning album..."
        
        photos_dir = job_dir / "photos"
        photos_dir.mkdir(parents=True, exist_ok=True)
        
        matches = []
        all_results = []
        
        for idx, gfile in enumerate(img_files, 1):
            fname = gfile.path
            fid = gfile.id
            local_path = photos_dir / fname
            thumb_url = f"https://lh3.googleusercontent.com/d/{fid}=w1000"
            
            job["current_photo"] = fname
            job["processed_count"] = idx
            
            # Download thumbnail
            if not local_path.exists() or local_path.stat().st_size < 1000:
                for attempt in range(3):
                    try:
                        req = urllib.request.Request(thumb_url, headers={"User-Agent": "Mozilla/5.0"})
                        local_path.write_bytes(urllib.request.urlopen(req, timeout=12).read())
                        break
                    except Exception:
                        time.sleep(0.2)
                        
            if not local_path.exists() or local_path.stat().st_size < 1000:
                continue
                
            try:
                faces = engine.extract_encodings(str(local_path))
            except Exception:
                continue
                
            min_dist = float("inf")
            for fmeta, enc in faces:
                d = engine.compute_distance(ref_enc, enc)
                if d < min_dist:
                    min_dist = d
                    
            is_match = (min_dist <= threshold)
            sim_pct = max(0.0, min(100.0, (1.0 - (min_dist / 1.35)) * 100.0))
            
            item = {
                "filename": fname,
                "file_id": fid,
                "faces_found": len(faces),
                "min_distance": round(min_dist, 4) if min_dist != float("inf") else None,
                "similarity_pct": round(sim_pct, 1) if min_dist != float("inf") else 0,
                "is_match": is_match,
                "drive_view_url": f"https://drive.google.com/file/d/{fid}/view",
                "drive_download_url": f"https://drive.usercontent.google.com/download?id={fid}&export=download",
                "preview_url": f"/preview/{job_id}/{fname}"
            }
            all_results.append(item)
            if is_match:
                matches.append(item)
                job["matches_count"] = len(matches)
                job["latest_match"] = fname
                
        matches.sort(key=lambda x: x["min_distance"] if x["min_distance"] is not None else 999)
        job["matches"] = matches
        job["all_results"] = all_results
        job["status"] = "completed"
        job["message"] = f"Finished! Found {len(matches)} matching photo(s) across {len(img_files)} files."
        
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


async def index_page(request):
    html_file = APP_DIR / "templates" / "index.html"
    return HTMLResponse(html_file.read_text(encoding="utf-8"))


async def start_scan_api(request):
    form = await request.form()
    folder_url = form.get("folder_url", "").strip()
    ref_file = form.get("reference_image")
    threshold_str = form.get("threshold", "1.05")
    
    if not folder_url:
        return JSONResponse({"error": "Google Drive Folder URL is required."}, status_code=400)
    if not ref_file or not hasattr(ref_file, "read"):
        return JSONResponse({"error": "Reference photo is required."}, status_code=400)
        
    try:
        threshold = float(threshold_str)
    except ValueError:
        threshold = 1.05
        
    ref_bytes = await ref_file.read()
    if len(ref_bytes) == 0:
        return JSONResponse({"error": "Uploaded file is empty."}, status_code=400)
        
    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "message": "Initializing scan job...",
        "total_photos": 0,
        "processed_count": 0,
        "matches_count": 0,
        "matches": [],
        "all_results": [],
        "created_at": time.time()
    }
    
    # Run scan worker in background thread
    threading.Thread(target=run_scan_job, args=(job_id, folder_url, ref_bytes, threshold), daemon=True).start()
    
    return JSONResponse({"job_id": job_id, "status": "started"})


async def stream_progress_api(request):
    job_id = request.path_params["job_id"]
    if job_id not in JOBS:
        return JSONResponse({"error": "Job not found."}, status_code=404)
        
    async def event_generator():
        last_state = None
        while True:
            job = JOBS.get(job_id)
            if not job:
                break
                
            state_data = {
                "status": job["status"],
                "message": job.get("message", ""),
                "total_photos": job.get("total_photos", 0),
                "processed_count": job.get("processed_count", 0),
                "matches_count": job.get("matches_count", 0),
                "current_photo": job.get("current_photo", ""),
                "latest_match": job.get("latest_match", ""),
                "error": job.get("error", None)
            }
            
            data_str = json.dumps(state_data)
            if data_str != last_state:
                yield f"data: {data_str}\n\n"
                last_state = data_str
                
            if job["status"] in ["completed", "error"]:
                break
                
            await asyncio.sleep(0.5)
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def get_results_api(request):
    job_id = request.path_params["job_id"]
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "Job not found."}, status_code=404)
    return JSONResponse(job)


async def serve_preview(request):
    job_id = request.path_params["job_id"]
    filename = request.path_params["filename"]
    p = DATA_DIR / job_id / "photos" / filename
    if p.is_file():
        return FileResponse(p, media_type="image/jpeg")
    return JSONResponse({"error": "File not found"}, status_code=404)


async def download_zip_api(request):
    job_id = request.path_params["job_id"]
    job = JOBS.get(job_id)
    if not job or "matches" not in job:
        return JSONResponse({"error": "Job or matches not found."}, status_code=404)
        
    matches = job["matches"]
    zip_buffer = io.BytesIO()
    photos_dir = DATA_DIR / job_id / "photos"
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for m in matches:
            fname = m["filename"]
            fpath = photos_dir / fname
            if fpath.is_file():
                zip_file.write(fpath, arcname=fname)
                
    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=matches_{job_id}.zip"}
    )


routes = [
    Route("/", endpoint=index_page, methods=["GET"]),
    Route("/api/scan", endpoint=start_scan_api, methods=["POST"]),
    Route("/api/stream/{job_id}", endpoint=stream_progress_api, methods=["GET"]),
    Route("/api/results/{job_id}", endpoint=get_results_api, methods=["GET"]),
    Route("/api/download-zip/{job_id}", endpoint=download_zip_api, methods=["GET"]),
    Route("/preview/{job_id}/{filename}", endpoint=serve_preview, methods=["GET"]),
    Mount("/static", app=StaticFiles(directory=str(STATIC_DIR)), name="static"),
]

app = Starlette(debug=True, routes=routes)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
