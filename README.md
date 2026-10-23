---
title: GDrive Face Finder
emoji: ⚽
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Google Drive Face Finder & Face Recognition MCP Server

An AI-powered computer vision system and web application designed to automatically scan Google Drive photo albums, detect and identify faces, and pull all action shots of specific individuals.

It operates both as a **Modern Web Application** (with live progress tracking, interactive gallery, and 1-click ZIP downloading) and as a **Model Context Protocol (MCP) Server** (compatible with Claude Desktop, Cursor, and Antigravity).

---

## Key Features

* **Instant Web Interface & Live Webcam Capture:** Upload 1 to 5 reference photos or use the integrated **Live Biometric Webcam Scanner** with an oval targeting guide to register your face instantly.
* **Google Photos-Grade Accuracy (ArcFace 512-d):** Powered by InsightFace ArcFace 512-dimensional deep normalized embeddings (`w600k_mbf.onnx`) and OpenCV YuNet 5-point landmark detection, separating true matches from impostors with high cosine discrimination.
* **Primary-Subject Reference Isolation & Multi-Scale Pyramid:** Automatically isolates the largest foreground face in each reference photo (ignoring background bystanders) and builds a **4-vector multi-scale + horizontal-flip reference pyramid** (`112×112` full-res, horizontally flipped, and `40×40 → 112×112` simulated wide-angle downsampled crops) to match both close-up portraits and distant midfield action shots.
* **Two-Tier Match Categorization:**
  * **★ Primary Solo & Action Shots (`Sim >= 0.42`):** High-confidence foreground portraits and close action shots.
  * **Secondary / Background Appearances (`0.335 <= Sim < 0.42`):** Wide-angle midfield, side-profile, and background appearances displayed in their own dedicated gallery section.
  * **Direct Google Drive View & Full-Resolution Download Links:** Complete biometric match table sorted by ArcFace 512-d cosine similarity with 1-click Google Drive view and RAW download links.
* **Biometric HUD & Landmark Overlay:** Displays real-time biometric bounding boxes and 5-point facial landmark reticles (left/right eyes, nose tip, mouth corners) with Interpupillary Distance (IPD in px), roll angle tilt, and confidence metrics directly on gallery cards and in the full-screen lightbox.
* **100% Local & Private:** Runs natively on your CPU/Apple Silicon using OpenCV deep learning ONNX models. No biometric data or images are ever uploaded to cloud LLMs or third parties.
* **Dual Operation Modes:**
  * **Interactive Web App:** Real-time Server-Sent Events (SSE) progress bar, concurrent high-speed Drive thumbnail pre-fetching, lightbox biometric inspector, persistent latest scan results (`/api/latest-results`), and 1-click batch ZIP downloads.
  * **Model Context Protocol (MCP) Server:** Allows AI coding assistants (Claude, Cursor, Antigravity) to autonomously detect, register, match, and cluster faces across file systems.

---

## Project Structure

```
gdrive-face-finder/
├── app.py                  # Fullstack Starlette web application & REST/SSE API
├── server.py               # Model Context Protocol (MCP) stdio server
├── start.sh                # Single-command launcher for the web app
├── config.py               # Configuration & path management
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project build metadata
├── engine/                 # Computer vision backend adapter
│   ├── base.py             # Abstract face engine interface & dataclasses
│   ├── opencv_engine.py    # YuNet (detection) + ArcFace 512-d / SFace engine
│   └── dlib_engine.py      # Optional dlib/face_recognition adapter
├── storage/                # SQLite persistent facial vector database
│   └── face_db.py
├── models/                 # Pretrained deep learning ONNX models
│   ├── face_detection_yunet_2023mar.onnx
│   ├── w600k_mbf.onnx      # InsightFace ArcFace 512-d MobileFaceNet
│   └── face_recognition_sface_2021dec.onnx
├── templates/              # Web application frontend
│   └── index.html          # Biometric HUD Single-Page Application (Tailwind CSS)
└── test_server.py          # Unit test suite
```

---

## Quick Start (Web App)

### 1. Installation

```bash
git clone https://github.com/dickyeka/gdrive-face-finder.git
cd gdrive-face-finder

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch the Web Interface

```bash
./start.sh
```

Open **`http://localhost:8000`** in your browser.

---

## Web API Endpoints

* `GET /` — Biometric Face Finder Single-Page Web Application.
* `GET /api/info` — Active biometric engine metadata (`YuNet + ArcFace 512-d`, embedding dimensions, default thresholds).
* `GET /api/latest-results` — Returns the most recent completed scan job from disk (`data/<job_id>/job.json`).
* `POST /api/scan` — Starts a background Google Drive folder scan with uploaded reference photo(s).
* `GET /api/stream/{job_id}` — Server-Sent Events (SSE) live progress stream.
* `GET /api/results/{job_id}` — Full scan results separated into `primary_matches` and `secondary_matches`.
* `GET /api/download-zip/{job_id}` — Streams a `.zip` archive of all matched photos.

---

## MCP Server Integration

To connect this face recognition pipeline to **Claude Desktop**, **Cursor**, or **Antigravity**:

### Configuration (`claude_desktop_config.json` or `.cursor/mcp.json`)

```json
{
  "mcpServers": {
    "face-recognition": {
      "command": "/path/to/gdrive-face-finder/.venv/bin/python",
      "args": [
        "/path/to/gdrive-face-finder/server.py"
      ]
    }
  }
}
```

### Exposed MCP Tools

* `detect_faces(image_path)` — Detects faces and extracts bounding box coordinates and 5-point landmarks.
* `register_face(name, image_path, notes)` — Saves a person's facial vector into the local SQLite database.
* `match_faces(image_path, tolerance)` — Scans an image and matches detected faces against registered individuals.
* `compare_two_faces(img1, img2)` — Compares two faces to verify identity.
* `cluster_faces(image_paths)` — Automatically groups photos by facial identity.
* `list_known_people()` — Lists registered individuals and sample counts.

---

## Computer Vision Pipeline

```
[ Reference Photos (1–5) ]                     [ Google Drive Album Photo ]
           │                                                │
           ▼                                                ▼
1. Primary Subject Isolation                   1. OpenCV YuNet (ONNX)
   • Selects largest foreground face              • Multi-scale 5-point landmark detection
   • Ignores background bystanders                • Extracts eyes, nose tip & mouth corners
           │                                                │
           ▼                                                ▼
2. 4-Vector Multi-Scale Reference Pyramid      2. Similarity Transformation Alignment
   • 112×112 Canonical Umeyama Alignment          • Warps candidate face to canonical 112×112
   • Horizontal Flip Augmentation                 • ArcFace 512-d L2-normalized embedding
   • 40×40 → 112×112 Wide-Angle Simulation                  │
           │                                                │
           └──────────────────────┬─────────────────────────┘
                                  ▼
                   3. ArcFace 512-d Cosine Verification
                      • Sim >= 0.42        ➔ ★ Primary Solo & Action Shot
                      • 0.335 <= Sim < 0.42 ➔ Secondary / Background Appearance
                      • Sim < 0.335        ➔ Rejected (Non-Match)
```

---

## License

MIT License. Free for personal and commercial use.
