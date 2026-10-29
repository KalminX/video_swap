"""Face detection using OpenCV's DNN YuNet ONNX model."""

from pathlib import Path
from typing import List, Optional, Union
import cv2
import numpy as np

from .base import BaseFaceDetector, BoundingBox, DetectionResult


class YuNetFaceDetector(BaseFaceDetector):
    """Modern deep-learning based face detector using YuNet via OpenCV DNN."""

    def __init__(
        self,
        model_path: Union[str, Path],
        conf_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"YuNet model ONNX file not found at: {path}")

        self.model_path = path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self._detector: Optional[cv2.FaceDetectorYN] = None

    @property
    def name(self) -> str:
        return "OpenCV-DNN-YuNet"

    def _get_detector(self, width: int, height: int) -> cv2.FaceDetectorYN:
        if self._detector is None:
            self._detector = cv2.FaceDetectorYN.create(
                model=str(self.model_path),
                config="",
                input_size=(width, height),
                score_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold,
                top_k=self.top_k,
            )
        else:
            self._detector.setInputSize((width, height))
        return self._detector

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        detector = self._get_detector(w, h)
        _, faces = detector.detect(image)

        results: List[DetectionResult] = []
        if faces is None:
            return results

        for face in faces:
            # Face format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rc, y_rc, x_lc, y_lc, score]
            x, y, width, height = face[0:4].astype(int)
            score = float(face[-1])

            # Landmarks (right eye, left eye, nose tip, right mouth corner, left mouth corner)
            landmarks_raw = face[4:14].reshape((5, 2))
            landmarks = [(float(pt[0]), float(pt[1])) for pt in landmarks_raw]

            # Clamp coordinates to image boundaries
            x = max(0, x)
            y = max(0, y)
            width = min(w - x, width)
            height = min(h - y, height)

            results.append(
                DetectionResult(
                    bbox=BoundingBox(x=x, y=y, width=width, height=height),
                    confidence=score,
                    landmarks=landmarks,
                )
            )

        return results
