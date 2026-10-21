from pathlib import Path
from typing import List, Tuple

from .base import BaseFaceEngine, DetectedFace, FaceBoundingBox


class DlibEngine(BaseFaceEngine):
    """Face recognition backend powered by the popular 'face_recognition' (dlib) library."""

    def __init__(self):
        self._fr = None
        self._np = None
        self._loaded = False
        self._check_library()

    def _check_library(self):
        try:
            import face_recognition
            import numpy as np
            self._fr = face_recognition
            self._np = np
            self._loaded = True
        except ImportError:
            self._fr = None
            self._np = None
            self._loaded = False

    @property
    def name(self) -> str:
        return "dlib (face_recognition)"

    def is_available(self) -> bool:
        return self._loaded

    def detect_faces(self, image_path: str) -> List[DetectedFace]:
        if not self._loaded:
            raise RuntimeError(
                "face_recognition is not installed. Install via `pip install face-recognition`."
            )

        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found at: {image_path}")

        image = self._fr.load_image_file(str(path))
        # Returns list of tuples: (top, right, bottom, left)
        locations = self._fr.face_locations(image, model="hog")
        landmarks_list = self._fr.face_landmarks(image, locations)

        results = []
        for i, (top, right, bottom, left) in enumerate(locations):
            landmarks = landmarks_list[i] if i < len(landmarks_list) else None
            results.append(
                DetectedFace(
                    box=FaceBoundingBox(top=top, right=right, bottom=bottom, left=left),
                    confidence=1.0,
                    landmarks=landmarks,
                )
            )
        return results

    def extract_encodings(
        self, image_path: str
    ) -> List[Tuple[DetectedFace, List[float]]]:
        if not self._loaded:
            raise RuntimeError(
                "face_recognition is not installed. Install via `pip install face-recognition`."
            )

        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found at: {image_path}")

        image = self._fr.load_image_file(str(path))
        locations = self._fr.face_locations(image, model="hog")
        encodings = self._fr.face_encodings(image, known_face_locations=locations)

        results = []
        for (top, right, bottom, left), enc in zip(locations, encodings):
            face = DetectedFace(
                box=FaceBoundingBox(top=top, right=right, bottom=bottom, left=left),
                confidence=1.0,
            )
            results.append((face, enc.tolist()))
        return results

    def compute_distance(
        self, encoding1: List[float], encoding2: List[float]
    ) -> float:
        """Euclidean distance between two 128-d vectors."""
        if self._np is not None:
            vec1 = self._np.array(encoding1, dtype=self._np.float64)
            vec2 = self._np.array(encoding2, dtype=self._np.float64)
            return float(self._np.linalg.norm(vec1 - vec2))
        import math
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(encoding1, encoding2)))
