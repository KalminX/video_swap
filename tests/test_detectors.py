"""Tests for face detectors (Haar Cascade and YuNet DNN)."""

from pathlib import Path
import numpy as np
import pytest

from identity_studio.core.image_ops import load_image
from identity_studio.detectors import HaarFaceDetector, YuNetFaceDetector

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "inputs"
YUNET_WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "weights" / "face_detection_yunet_2023mar.onnx"


def test_haar_detector_detects_face():
    img_path = DATA_DIR / "lena.jpg"
    assert img_path.is_file(), "Test image lena.jpg missing."
    img = load_image(img_path)

    detector = HaarFaceDetector()
    results = detector.detect(img)

    assert len(results) >= 1
    face = results[0]
    assert face.bbox.width > 50
    assert face.bbox.height > 50
    assert face.confidence == 1.0


def test_haar_detector_no_faces():
    img_path = DATA_DIR / "fruits_no_face.jpg"
    assert img_path.is_file(), "Test image fruits_no_face.jpg missing."
    img = load_image(img_path)

    detector = HaarFaceDetector()
    results = detector.detect(img)
    assert len(results) == 0


def test_haar_detector_empty_image():
    detector = HaarFaceDetector()
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert detector.detect(empty) == []


@pytest.mark.skipif(not YUNET_WEIGHTS.is_file(), reason="YuNet model weights not present")
def test_yunet_detector_detects_face():
    img_path = DATA_DIR / "lena.jpg"
    img = load_image(img_path)

    detector = YuNetFaceDetector(model_path=YUNET_WEIGHTS)
    results = detector.detect(img)

    assert len(results) >= 1
    face = results[0]
    assert face.confidence > 0.8
    assert face.landmarks is not None
    assert len(face.landmarks) == 5  # 5 facial landmarks


@pytest.mark.skipif(not YUNET_WEIGHTS.is_file(), reason="YuNet model weights not present")
def test_yunet_detector_no_faces():
    img_path = DATA_DIR / "fruits_no_face.jpg"
    img = load_image(img_path)

    detector = YuNetFaceDetector(model_path=YUNET_WEIGHTS)
    results = detector.detect(img)
    assert len(results) == 0
