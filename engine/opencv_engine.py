from pathlib import Path
from typing import List, Tuple
import numpy as np

from .base import BaseFaceEngine, DetectedFace, FaceBoundingBox


class OpenCVEngine(BaseFaceEngine):
    """
    Modern face detection and recognition engine using OpenCV's built-in
    YuNet (detection) and SFace (128-d deep facial embeddings).
    Runs with full hardware acceleration on Apple Silicon / CPU without CMake.
    """

    def __init__(self):
        self._cv2 = None
        self._np = None
        self._loaded = False
        self._detector = None
        self._recognizer = None
        self._check_library()

    def _check_library(self):
        try:
            import cv2
            import numpy as np
            self._cv2 = cv2
            self._np = np

            models_dir = Path(__file__).parent.parent / "models"
            yunet_model = models_dir / "face_detection_yunet_2023mar.onnx"
            sface_model = models_dir / "face_recognition_sface_2021dec.onnx"

            if yunet_model.is_file() and sface_model.is_file():
                # Initialize YuNet detector with initial placeholder input size
                self._detector = cv2.FaceDetectorYN.create(
                    str(yunet_model),
                    "",
                    (320, 320),
                    score_threshold=0.6,
                    nms_threshold=0.3,
                    top_k=5000,
                )
                self._recognizer = cv2.FaceRecognizerSF.create(
                    str(sface_model),
                    "",
                )
                self._loaded = True
        except Exception:
            self._cv2 = None
            self._np = None
            self._loaded = False

    @property
    def name(self) -> str:
        return "opencv (YuNet + SFace ONNX)"

    def is_available(self) -> bool:
        return self._loaded and (self._detector is not None) and (self._recognizer is not None)

    def _load_and_detect(self, image_path: str):
        if not self.is_available():
            raise RuntimeError(
                "OpenCV YuNet/SFace models are missing. Check `models/` directory."
            )

        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found at: {image_path}")

        img = self._cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Unable to decode image file at: {image_path}")

        h, w, _ = img.shape
        self._detector.setInputSize((w, h))
        retval, raw_faces = self._detector.detect(img)

        return img, raw_faces if (raw_faces is not None) else []

    def detect_faces(self, image_path: str) -> List[DetectedFace]:
        img, raw_faces = self._load_and_detect(image_path)
        results = []

        for face in raw_faces:
            x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            confidence = float(face[14])
            landmarks = {
                "right_eye": [float(face[4]), float(face[5])],
                "left_eye": [float(face[6]), float(face[7])],
                "nose_tip": [float(face[8]), float(face[9])],
                "right_mouth": [float(face[10]), float(face[11])],
                "left_mouth": [float(face[12]), float(face[13])],
            }
            results.append(
                DetectedFace(
                    box=FaceBoundingBox(
                        top=max(0, y),
                        right=max(0, x + w),
                        bottom=max(0, y + h),
                        left=max(0, x),
                    ),
                    confidence=confidence,
                    landmarks=landmarks,
                )
            )
        return results

    def extract_encodings(
        self, image_path: str
    ) -> List[Tuple[DetectedFace, List[float]]]:
        img, raw_faces = self._load_and_detect(image_path)
        results = []

        for face in raw_faces:
            x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            confidence = float(face[14])
            landmarks = {
                "right_eye": [float(face[4]), float(face[5])],
                "left_eye": [float(face[6]), float(face[7])],
                "nose_tip": [float(face[8]), float(face[9])],
                "right_mouth": [float(face[10]), float(face[11])],
                "left_mouth": [float(face[12]), float(face[13])],
            }
            face_meta = DetectedFace(
                box=FaceBoundingBox(
                    top=max(0, y),
                    right=max(0, x + w),
                    bottom=max(0, y + h),
                    left=max(0, x),
                ),
                confidence=confidence,
                landmarks=landmarks,
            )

            # Align face crop and compute 128-d feature vector
            aligned_face = self._recognizer.alignCrop(img, face)
            feature_vector = self._recognizer.feature(aligned_face)[0].tolist()

            results.append((face_meta, feature_vector))

        return results

    def compute_distance(
        self, encoding1: List[float], encoding2: List[float]
    ) -> float:
        """Calculate L2 distance between SFace 128-d embeddings (lower means more similar)."""
        v1 = np.array(encoding1, dtype=np.float32).reshape(1, -1)
        v2 = np.array(encoding2, dtype=np.float32).reshape(1, -1)
        dist = self._recognizer.match(v1, v2, self._cv2.FaceRecognizerSF_FR_NORM_L2)
        return float(dist)
