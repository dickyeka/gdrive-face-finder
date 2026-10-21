from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class FaceBoundingBox:
    """Bounding box coordinates for a face (standard top, right, bottom, left)."""
    top: int
    right: int
    bottom: int
    left: int

    @property
    def x(self) -> int:
        return self.left

    @property
    def y(self) -> int:
        return self.top

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    def to_dict(self) -> Dict[str, int]:
        return {
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
            "left": self.left,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class DetectedFace:
    """Represents a detected face in an image."""
    box: FaceBoundingBox
    confidence: float = 1.0
    landmarks: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "box": self.box.to_dict(),
            "confidence": round(self.confidence, 4),
            "landmarks": self.landmarks,
        }


@dataclass
class MatchCandidate:
    """Candidate match when comparing against known faces."""
    name: str
    distance: float
    similarity_percentage: float
    is_match: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "distance": round(self.distance, 4),
            "similarity_percentage": round(self.similarity_percentage, 2),
            "is_match": self.is_match,
        }


@dataclass
class RecognizedFace:
    """A face identified in an image."""
    box: FaceBoundingBox
    identified_name: str
    confidence: float
    distance: Optional[float] = None
    similarity_percentage: Optional[float] = None
    top_candidates: Optional[List[MatchCandidate]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "box": self.box.to_dict(),
            "identified_name": self.identified_name,
            "confidence": round(self.confidence, 4),
            "distance": round(self.distance, 4) if self.distance is not None else None,
            "similarity_percentage": round(self.similarity_percentage, 2)
            if self.similarity_percentage is not None
            else None,
            "top_candidates": [c.to_dict() for c in (self.top_candidates or [])],
        }


class BaseFaceEngine(ABC):
    """Abstract interface for Face Recognition backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the engine (e.g. 'dlib', 'opencv')."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if all required binaries and libraries are present."""
        pass

    @abstractmethod
    def detect_faces(self, image_path: str) -> List[DetectedFace]:
        """Detect all face locations in the specified image."""
        pass

    @abstractmethod
    def extract_encodings(
        self, image_path: str
    ) -> List[Tuple[DetectedFace, List[float]]]:
        """Extract face bounding boxes along with 128-d or N-d feature vectors."""
        pass

    @abstractmethod
    def compute_distance(
        self, encoding1: List[float], encoding2: List[float]
    ) -> float:
        """Calculate mathematical distance (lower means more similar)."""
        pass

    def compute_similarity(
        self, encoding1: List[float], encoding2: List[float]
    ) -> float:
        """Calculate similarity score (0.0 to 1.0, higher means more similar)."""
        d = self.compute_distance(encoding1, encoding2)
        return max(0.0, 1.0 - d)
