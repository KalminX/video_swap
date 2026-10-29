"""Face detection using OpenCV's built-in Haar Feature-based Cascade Classifier."""

from pathlib import Path
from typing import List, Optional, Union
import cv2
import numpy as np

from .base import BaseFaceDetector, BoundingBox, DetectionResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LOCAL_WEIGHTS_CASCADE = PROJECT_ROOT / "models" / "weights" / "haarcascade_frontalface_default.xml"


class HaarFaceDetector(BaseFaceDetector):
    """Haar Cascade face detector.
    
    Fast, lightweight, classical detector that requires no external weight downloads.
    """

    def __init__(
        self,
        cascade_path: Optional[Union[str, Path]] = None,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (30, 30),
    ) -> None:
        resolved_path: Optional[Path] = None

        if cascade_path is not None:
            resolved_path = Path(cascade_path)
        elif LOCAL_WEIGHTS_CASCADE.is_file():
            resolved_path = LOCAL_WEIGHTS_CASCADE
        elif hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            cv2_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            if cv2_path.is_file():
                resolved_path = cv2_path

        if resolved_path is None or not resolved_path.is_file():
            raise FileNotFoundError(
                f"Haar cascade XML not found. Checked: {cascade_path}, {LOCAL_WEIGHTS_CASCADE}"
            )

        self._classifier = cv2.CascadeClassifier(str(resolved_path))
        if self._classifier.empty():
            raise RuntimeError(f"Failed to load Haar Cascade from: {resolved_path}")

        self.cascade_path = resolved_path
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    @property
    def name(self) -> str:
        return "OpenCV-Haar-Cascade"

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        if image is None or image.size == 0:
            return []

        # Convert to grayscale for Haar detection if color image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Equalize histogram for improved contrast robustness
        gray_eq = cv2.equalizeHist(gray)

        faces = self._classifier.detectMultiScale(
            gray_eq,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        results: List[DetectionResult] = []
        if len(faces) == 0:
            return results

        for (x, y, w, h) in faces:
            # Haar doesn't provide probability score, default confidence to 1.0
            results.append(
                DetectionResult(
                    bbox=BoundingBox(x=int(x), y=int(y), width=int(w), height=int(h)),
                    confidence=1.0,
                    landmarks=None,
                )
            )

        return results
