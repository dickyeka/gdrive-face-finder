import os
from pathlib import Path

# Base data directory: check env var, then fallback to project-local data or home directory
def _resolve_data_dir() -> Path:
    if "FACE_MCP_DATA_DIR" in os.environ:
        p = Path(os.environ["FACE_MCP_DATA_DIR"]).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    # Try project-local data directory first
    local_data = Path(__file__).parent.resolve() / "data"
    try:
        local_data.mkdir(parents=True, exist_ok=True)
        return local_data
    except (PermissionError, OSError):
        home_data = Path.home() / ".face_recognition_mcp"
        home_data.mkdir(parents=True, exist_ok=True)
        return home_data

DATA_DIR = _resolve_data_dir()

# Database path
DB_PATH = DATA_DIR / "faces.sqlite3"

# Reference photos storage folder (if local copies are preserved)
KNOWN_FACES_DIR = DATA_DIR / "registered_faces"
KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)

# Default tolerance (distance threshold) for face matching
# For dlib (128-d Euclidean): 0.6 is standard, < 0.5 is strict, > 0.65 is loose
DEFAULT_TOLERANCE = float(os.environ.get("FACE_MCP_TOLERANCE", 0.6))

# Preferred backend engine: "auto", "dlib", or "opencv"
PREFERRED_ENGINE = os.environ.get("FACE_MCP_ENGINE", "auto")

# Supported image file extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
