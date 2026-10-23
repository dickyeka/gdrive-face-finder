import os
import sys
import json
import uuid
import time
import zipfile
import io
import math
import asyncio
import threading
import urllib.request
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2

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


def run_scan_job(job_id: str, folder_url: str, ref_images_bytes: List[bytes], threshold: float):
    job = JOBS[job_id]
    job_dir = DATA_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        ref_dir = job_dir / "references"
        ref_dir.mkdir(parents=True, exist_ok=True)

        job["status"] = "detecting_reference"
        job["message"] = f"Extracting deep face representations from {len(ref_images_bytes)} reference photo(s)..."

        ref_encodings: List[List[float]] = []
        for idx, r_bytes in enumerate(ref_images_bytes):
            ref_path = ref_dir / f"reference_{idx}.jpg"
            ref_path.write_bytes(r_bytes)
            try:
                if hasattr(engine, "extract_reference_encodings"):
                    aug_vecs = engine.extract_reference_encodings(str(ref_path))
                    ref_encodings.extend(aug_vecs)
                else:
                    faces = engine.extract_encodings(str(ref_path))
                    if faces:
                        faces_sorted = sorted(
                            faces,
                            key=lambda x: x[0].box.width * x[0].box.height,
                            reverse=True,
                        )
                        ref_encodings.append(faces_sorted[0][1])
            except Exception:
                continue

        if not ref_encodings:
            job["status"] = "error"
            job["error"] = "No face could be detected in any of the uploaded reference photos. Please upload clear photos with visible faces."
            return

        is_arcface = getattr(engine, "is_arcface_enabled", False)
        job["ref_detected"] = True
        job["ref_count"] = len(ref_images_bytes)
        job["ref_faces_count"] = len(ref_encodings)
        job["engine_name"] = engine.name
        
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
        job["message"] = f"Found {len(img_files)} photos. Downloading & scanning album with {engine.name}..."
        
        photos_dir = job_dir / "photos"
        photos_dir.mkdir(parents=True, exist_ok=True)

        # Pre-download thumbnails concurrently for high speed
        def _download_one(gfile):
            fname = Path(gfile.path).name
            fid = gfile.id
            local_path = photos_dir / fname
            if not local_path.exists() or local_path.stat().st_size < 1000:
                thumb_url = f"https://lh3.googleusercontent.com/d/{fid}=w1000"
                for _ in range(3):
                    try:
                        req = urllib.request.Request(thumb_url, headers={"User-Agent": "Mozilla/5.0"})
                        data = urllib.request.urlopen(req, timeout=15).read()
                        if len(data) > 1000:
                            local_path.write_bytes(data)
                            break
                    except Exception:
                        time.sleep(0.2)
            return (fname, fid, local_path)

        with ThreadPoolExecutor(max_workers=8) as pool:
            downloaded_items = list(pool.map(_download_one, img_files))
        
        matches = []
        all_results = []
        
        for idx, (fname, fid, local_path) in enumerate(downloaded_items, 1):
            job["current_photo"] = fname
            job["processed_count"] = idx
                        
            if not local_path.exists() or local_path.stat().st_size < 1000:
                continue
                
            try:
                img_mat = cv2.imread(str(local_path))
                if img_mat is not None:
                    img_h, img_w, _ = img_mat.shape
                else:
                    img_w, img_h = 1000, 1000
                faces = engine.extract_encodings(str(local_path))
            except Exception:
                continue

            best_sim = -1.0
            min_dist = float("inf")
            biometrics_list = []

            primary_thresh = 0.42
            secondary_thresh = min(threshold, 0.335) if threshold <= 0.50 else max(0.30, threshold - 0.12)

            if faces:
                for fmeta, cand_enc in faces:
                    cand_sim = max((engine.compute_similarity(r, cand_enc) for r in ref_encodings), default=0.0)
                    cand_dist = min((engine.compute_distance(r, cand_enc) for r in ref_encodings), default=float("inf"))

                    if cand_sim > best_sim:
                        best_sim = cand_sim
                    if cand_dist < min_dist:
                        min_dist = cand_dist

                    if is_arcface:
                        is_cand_match = (cand_sim >= secondary_thresh)
                        cand_pct = max(0.0, min(100.0, (cand_sim - 0.25) / (0.65 - 0.25) * 100.0))
                    else:
                        is_cand_match = (cand_dist <= threshold)
                        cand_pct = max(0.0, min(100.0, (1.0 - (cand_dist / 1.35)) * 100.0))

                    # Biometric landmarks and geometric measurements
                    b = fmeta.box
                    lm = fmeta.landmarks or {}
                    re = lm.get("right_eye", [0, 0])
                    le = lm.get("left_eye", [0, 0])
                    ipd = round(math.hypot(le[0] - re[0], le[1] - re[1]), 1)
                    roll_deg = round(math.degrees(math.atan2(le[1] - re[1], le[0] - re[0])), 1)

                    biometrics_list.append({
                        "box": b.to_dict(),
                        "confidence": round(fmeta.confidence, 4),
                        "landmarks": lm,
                        "ipd_px": ipd,
                        "roll_deg": roll_deg,
                        "similarity_score": round(cand_sim, 4),
                        "similarity_pct": round(cand_pct, 1),
                        "is_match": is_cand_match,
                    })

            # Match criteria (includes both Primary Solo & Action Shots and Secondary/Background Appearances)
            if is_arcface:
                is_match = (best_sim >= secondary_thresh)
                # Calibrated percentage: [0.25, 0.65] mapped to [0%, 100%]
                sim_pct = max(0.0, min(100.0, (best_sim - 0.25) / (0.65 - 0.25) * 100.0))
            else:
                is_match = (min_dist <= threshold)
                sim_pct = max(0.0, min(100.0, (1.0 - (min_dist / 1.35)) * 100.0))
            
            category = "primary" if best_sim >= primary_thresh else "secondary"
            item = {
                "filename": fname,
                "file_id": fid,
                "category": category,
                "image_width": img_w,
                "image_height": img_h,
                "faces_found": len(faces),
                "similarity_score": round(best_sim, 4) if best_sim >= -0.9 else 0.0,
                "min_distance": round(min_dist, 4) if min_dist != float("inf") else None,
                "similarity_pct": round(sim_pct, 1),
                "is_match": is_match,
                "biometrics": biometrics_list,
                "drive_view_url": f"https://drive.google.com/file/d/{fid}/view",
                "drive_download_url": f"https://drive.usercontent.google.com/download?id={fid}&export=download",
                "preview_url": f"/preview/{job_id}/{fname}"
            }
            all_results.append(item)
            if is_match:
                matches.append(item)
                job["matches_count"] = len(matches)
                job["latest_match"] = fname
                
        # Sort matches by similarity score descending (highest quality matches first)
        matches.sort(key=lambda x: x["similarity_score"], reverse=True)
        primary_matches = [m for m in matches if m.get("category") == "primary"]
        secondary_matches = [m for m in matches if m.get("category") == "secondary"]
        job["matches"] = matches
        job["primary_matches"] = primary_matches
        job["secondary_matches"] = secondary_matches
        job["primary_count"] = len(primary_matches)
        job["secondary_count"] = len(secondary_matches)
        job["all_results"] = all_results
        job["status"] = "completed"
        job["message"] = (
            f"Finished! Found {len(matches)} matching photo(s) "
            f"({len(primary_matches)} Primary Solo & Action + {len(secondary_matches)} Secondary/Background) "
            f"across {len(img_files)} files."
        )
        try:
            (job_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")
        except Exception:
            pass
        
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


async def index_page(request):
    html_file = APP_DIR / "templates" / "index.html"
    return HTMLResponse(html_file.read_text(encoding="utf-8"))


async def engine_info_api(request):
    is_arcface = getattr(engine, "is_arcface_enabled", False)
    return JSONResponse({
        "engine": engine.name,
        "is_arcface": is_arcface,
        "embedding_size": 512 if is_arcface else 128,
        "default_threshold": 0.33 if is_arcface else 1.05,
        "min_threshold": 0.30 if is_arcface else 0.75,
        "max_threshold": 0.60 if is_arcface else 1.25,
    })


async def start_scan_api(request):
    form = await request.form()
    folder_url = form.get("folder_url", "").strip()
    
    # Support multiple reference images
    ref_files = form.getlist("reference_images")
    if not ref_files:
        single_ref = form.get("reference_image")
        if single_ref and hasattr(single_ref, "read"):
            ref_files = [single_ref]
            
    is_arcface = getattr(engine, "is_arcface_enabled", False)
    default_thresh_str = "0.33" if is_arcface else "1.05"
    threshold_str = form.get("threshold", default_thresh_str)
    
    if not folder_url:
        return JSONResponse({"error": "Google Drive Folder URL is required."}, status_code=400)
    if not ref_files:
        return JSONResponse({"error": "At least one reference photo is required."}, status_code=400)
        
    try:
        threshold = float(threshold_str)
        # Handle conversion if old threshold (> 1.0) sent to ArcFace
        if is_arcface and threshold > 1.0:
            threshold = 0.33
    except ValueError:
        threshold = 0.33 if is_arcface else 1.05
        
    ref_bytes_list: List[bytes] = []
    for rf in ref_files:
        if hasattr(rf, "read"):
            b = await rf.read()
            if len(b) > 0:
                ref_bytes_list.append(b)
                
    if not ref_bytes_list:
        return JSONResponse({"error": "Uploaded reference photo(s) are empty."}, status_code=400)
        
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
    threading.Thread(
        target=run_scan_job,
        args=(job_id, folder_url, ref_bytes_list, threshold),
        daemon=True
    ).start()
    
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


def _get_or_load_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        jfile = DATA_DIR / job_id / "job.json"
        if jfile.is_file():
            try:
                job = json.loads(jfile.read_text(encoding="utf-8"))
                JOBS[job_id] = job
            except Exception:
                pass
    return job


async def get_results_api(request):
    job_id = request.path_params["job_id"]
    job = _get_or_load_job(job_id)
    if not job:
        return JSONResponse({"error": "Job not found."}, status_code=404)
    return JSONResponse(job)


async def get_latest_results_api(request):
    # Find most recent completed job in JOBS or DATA_DIR/*/job.json
    candidates = []
    for jfile in DATA_DIR.glob("*/job.json"):
        try:
            jdata = json.loads(jfile.read_text(encoding="utf-8"))
            if jdata.get("status") == "completed":
                candidates.append((jfile.stat().st_mtime, jdata))
        except Exception:
            continue
    for jid, jdata in JOBS.items():
        if jdata.get("status") == "completed":
            candidates.append((jdata.get("created_at", 0), jdata))
    if not candidates:
        return JSONResponse({"has_latest": False})
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_job = candidates[0][1]
    JOBS[best_job["job_id"]] = best_job
    return JSONResponse({"has_latest": True, "job": best_job})


async def serve_preview(request):
    job_id = request.path_params["job_id"]
    filename = request.path_params["filename"]
    p = DATA_DIR / job_id / "photos" / filename
    if p.is_file():
        return FileResponse(p, media_type="image/jpeg")
    return JSONResponse({"error": "File not found"}, status_code=404)


async def download_zip_api(request):
    job_id = request.path_params["job_id"]
    job = _get_or_load_job(job_id)
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
    Route("/api/info", endpoint=engine_info_api, methods=["GET"]),
    Route("/api/latest-results", endpoint=get_latest_results_api, methods=["GET"]),
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
