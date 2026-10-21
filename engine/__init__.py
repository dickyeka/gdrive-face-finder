from typing import Optional
from .base import (
    BaseFaceEngine,
    DetectedFace,
    FaceBoundingBox,
    MatchCandidate,
    RecognizedFace,
)
from .dlib_engine import DlibEngine
from .opencv_engine import OpenCVEngine


def get_engine(preferred: str = "auto") -> BaseFaceEngine:
    """
    Factory function to select and instantiate the best available face recognition engine.
    """
    pref = preferred.lower().strip()

    if pref == "dlib":
        engine = DlibEngine()
        if engine.is_available():
            return engine
        raise RuntimeError("Requested 'dlib' engine is not installed or available.")

    if pref == "opencv":
        engine = OpenCVEngine()
        if engine.is_available():
            return engine
        raise RuntimeError("Requested 'opencv' engine is not installed or available.")

    # Auto mode: prioritize dlib if present, then opencv
    dlib_eng = DlibEngine()
    if dlib_eng.is_available():
        return dlib_eng

    opencv_eng = OpenCVEngine()
    if opencv_eng.is_available():
        return opencv_eng

    # Neither available - return placeholder or informative error
    raise RuntimeError(
        "No face recognition backend is available. Please install either:\n"
        "  - `pip install face-recognition` (standard dlib model)\n"
        "  - `pip install opencv-python numpy pillow` (lightweight OpenCV fallback)"
    )


__all__ = [
    "BaseFaceEngine",
    "DetectedFace",
    "FaceBoundingBox",
    "MatchCandidate",
    "RecognizedFace",
    "DlibEngine",
    "OpenCVEngine",
    "get_engine",
]
