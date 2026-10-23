from pathlib import Path
from typing import List, Tuple
import numpy as np

from .base import BaseFaceEngine, DetectedFace, FaceBoundingBox


class OpenCVEngine(BaseFaceEngine):
    """
    Modern high-precision face detection and recognition engine using:
    - YuNet: Ultra-fast ONNX face detection with 5-point facial landmark alignment
    - ArcFace 512-d (w600k_mbf.onnx): InsightFace deep representation network
      yielding state-of-the-art discrimination comparable to Google Photos People.
    - SFace 128-d: Built-in fallback if ArcFace weights are unavailable.
    """

    def __init__(self):
        self._cv2 = None
        self._np = None
        self._loaded = False
        self._detector = None
        self._recognizer = None
        self._arcface_net = None
        self._has_arcface = False
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
            arcface_model = models_dir / "w600k_mbf.onnx"

            if yunet_model.is_file() and sface_model.is_file():
                # Score threshold 0.5 allows detection of smaller action/sports faces
                self._detector = cv2.FaceDetectorYN.create(
                    str(yunet_model),
                    "",
                    (320, 320),
                    score_threshold=0.5,
                    nms_threshold=0.3,
                    top_k=5000,
                )
                self._recognizer = cv2.FaceRecognizerSF.create(
                    str(sface_model),
                    "",
                )

                # Check and load ArcFace 512-d ONNX
                if arcface_model.is_file():
                    try:
                        self._arcface_net = cv2.dnn.readNetFromONNX(str(arcface_model))
                        self._has_arcface = True
                    except Exception:
                        self._arcface_net = None
                        self._has_arcface = False

                self._loaded = True
        except Exception:
            self._cv2 = None
            self._np = None
            self._loaded = False

    @property
    def is_arcface_enabled(self) -> bool:
        return self._has_arcface and (self._arcface_net is not None)

    @property
    def name(self) -> str:
        if self.is_arcface_enabled:
            return "opencv (YuNet + ArcFace 512-d)"
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

            # Canonical 112x112 affine alignment using YuNet 5 landmarks
            aligned_face = self._recognizer.alignCrop(img, face)

            if self.is_arcface_enabled:
                # ArcFace 512-d normalized embedding
                blob = self._cv2.dnn.blobFromImage(
                    aligned_face,
                    1.0 / 127.5,
                    (112, 112),
                    (127.5, 127.5, 127.5),
                    swapRB=True,
                )
                self._arcface_net.setInput(blob)
                feat = self._arcface_net.forward()[0]
                norm = float(np.linalg.norm(feat))
                if norm > 1e-6:
                    feat = feat / norm
                feature_vector = feat.tolist()
            else:
                # SFace 128-d fallback
                feature_vector = self._recognizer.feature(aligned_face)[0].tolist()

            results.append((face_meta, feature_vector))

        return results

    def extract_reference_encodings(self, image_path: str) -> List[List[float]]:
        """
        Extract primary foreground face from a reference photo and generate
        multi-scale + horizontal-flip augmented embeddings for high-recall
        multi-angle and distant-face matching.
        """
        img, raw_faces = self._load_and_detect(image_path)
        if len(raw_faces) == 0:
            return []

        # Select ONLY the primary (largest area) foreground face
        raw_sorted = sorted(raw_faces, key=lambda f: float(f[2]) * float(f[3]), reverse=True)
        primary_face = raw_sorted[0]
        aligned = self._recognizer.alignCrop(img, primary_face)

        if not self.is_arcface_enabled:
            return [self._recognizer.feature(aligned)[0].tolist()]

        flipped = self._cv2.flip(aligned, 1)
        small_orig = self._cv2.resize(
            self._cv2.resize(aligned, (40, 40), interpolation=self._cv2.INTER_AREA),
            (112, 112),
            interpolation=self._cv2.INTER_LINEAR,
        )
        small_flip = self._cv2.resize(
            self._cv2.resize(flipped, (40, 40), interpolation=self._cv2.INTER_AREA),
            (112, 112),
            interpolation=self._cv2.INTER_LINEAR,
        )

        vecs: List[List[float]] = []
        for variant in (aligned, flipped, small_orig, small_flip):
            blob = self._cv2.dnn.blobFromImage(
                variant,
                1.0 / 127.5,
                (112, 112),
                (127.5, 127.5, 127.5),
                swapRB=True,
            )
            self._arcface_net.setInput(blob)
            feat = self._arcface_net.forward()[0]
            norm = float(np.linalg.norm(feat))
            if norm > 1e-6:
                feat = feat / norm
            vecs.append(feat.tolist())

        return vecs

    def compute_distance(
        self, encoding1: List[float], encoding2: List[float]
    ) -> float:
        """
        Calculate mathematical distance:
        - For 512-d ArcFace: Cosine distance = max(0.0, 1.0 - CosineSimilarity)
        - For 128-d SFace: L2 Norm distance
        """
        if len(encoding1) == 512 and len(encoding2) == 512:
            v1 = np.array(encoding1, dtype=np.float32)
            v2 = np.array(encoding2, dtype=np.float32)
            sim = float(np.dot(v1, v2))
            return max(0.0, 1.0 - sim)

        v1 = np.array(encoding1, dtype=np.float32).reshape(1, -1)
        v2 = np.array(encoding2, dtype=np.float32).reshape(1, -1)
        dist = self._recognizer.match(v1, v2, self._cv2.FaceRecognizerSF_FR_NORM_L2)
        return float(dist)

    def compute_similarity(
        self, encoding1: List[float], encoding2: List[float]
    ) -> float:
        """
        Calculate cosine or normalized similarity score (0.0 to 1.0).
        """
        if len(encoding1) == 512 and len(encoding2) == 512:
            v1 = np.array(encoding1, dtype=np.float32)
            v2 = np.array(encoding2, dtype=np.float32)
            return float(np.dot(v1, v2))

        d = self.compute_distance(encoding1, encoding2)
        return max(0.0, 1.0 - (d / 1.35))
