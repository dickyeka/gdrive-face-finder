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

* **Instant Web Interface:** Upload 1 to 5 reference photos (different angles, lighting, action shots), paste a Google Drive folder link, and hit Scan.
* **Google Photos-Grade Accuracy (ArcFace 512-d):** Upgraded to InsightFace ArcFace 512-dimensional deep normalized embeddings (`w600k_mbf.onnx`) and YuNet landmark detection, separating true matches from impostors with massive cosine discrimination.
* **Multi-Photo Reference Pooling:** Upload portrait, side profile, and sports/action photos. The engine computes maximum cosine similarity across all reference vectors, dramatically improving recall on tough action poses.
* **100% Local & Private:** Runs natively on your CPU/Apple Silicon using OpenCV deep learning ONNX models. No biometric data or images are ever uploaded to cloud LLMs or third parties.
* **Dual Operation Modes:**
  * **Interactive Web App:** Real-time Server-Sent Events (SSE) progress bar, lightbox photo viewer, similarity scores, and 1-click batch ZIP downloads.
  * **Model Context Protocol (MCP) Server:** Allows AI coding assistants (Claude, Cursor, Antigravity) to autonomously detect, register, match, and cluster faces across file systems.

---

## Project Structure

```
gdrive-face-finder/
├── app.py                  # Fullstack Starlette web application & API
├── server.py               # Model Context Protocol (MCP) stdio server
├── start.sh                # Single-command launcher for the web app
├── config.py               # Configuration & path management
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project build metadata
├── engine/                 # Computer vision backend adapter
│   ├── base.py             # Abstract face engine interface & dataclasses
│   ├── opencv_engine.py    # Native YuNet (detection) + SFace (recognition) engine
│   └── dlib_engine.py      # Optional dlib/face_recognition adapter
├── storage/                # SQLite persistent facial vector database
│   └── face_db.py
├── models/                 # Pretrained deep learning ONNX models
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
├── templates/              # Web application frontend
│   └── index.html          # Modern Tailwind CSS Single-Page Application
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

* `detect_faces(image_path)` — Detects faces and extracts bounding box coordinates.
* `register_face(name, image_path, notes)` — Saves a person's facial vector into the local SQLite database.
* `match_faces(image_path, tolerance)` — Scans an image and matches detected faces against registered individuals.
* `compare_two_faces(img1, img2)` — Compares two faces to verify identity.
* `cluster_faces(image_paths)` — Automatically groups photos by facial identity.
* `list_known_people()` — Lists registered individuals and sample counts.

---

## Computer Vision Pipeline

```
[ Input Photo ]
       │
       ▼
1. OpenCV YuNet (ONNX)
   • 227 KB ultra-lightweight deep neural network
   • Extracts bounding box + 5 facial landmarks (eyes, nose, mouth corners)
       │
       ▼
2. OpenCV SFace (ONNX)
   • Generates normalized 128-dimensional facial embedding vector
       │
       ▼
3. Vector Distance Matching
   • Computes Euclidean L2 distance: ||v1 - v2||
   • Distance < 0.85 ➔ Very High Confidence Match
   • Distance 0.85 - 1.05 ➔ Confirmed Match
   • Distance > 1.10 ➔ Rejected (Different Person)
```

---

## License

MIT License. Free for personal and commercial use.
