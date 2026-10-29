"""Tests for the end-to-end FaceProcessingPipeline and annotation."""

from pathlib import Path
import pytest

from identity_studio.core.image_ops import load_image
from identity_studio.detectors import HaarFaceDetector
from identity_studio.pipeline import FaceProcessingPipeline, FaceAnnotator

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "inputs"


def test_pipeline_success_with_face(tmp_path):
    img_path = DATA_DIR / "lena.jpg"
    out_path = tmp_path / "annotated_lena.jpg"

    pipeline = FaceProcessingPipeline(detector=HaarFaceDetector())
    res = pipeline.process_image(img_path, output_path=out_path)

    assert res.success is True
    assert res.num_faces_detected >= 1
    assert res.output_path is not None
    assert Path(res.output_path).is_file()
    assert res.metadata.width == 512
    assert res.metadata.height == 512

    # Verify timing measurements were recorded
    assert "load_image" in res.timing.measurements
    assert "face_detection" in res.timing.measurements
    assert "annotation" in res.timing.measurements
    assert "save_image" in res.timing.measurements
    assert res.timing.total_ms > 0.0


def test_pipeline_image_without_faces(tmp_path):
    img_path = DATA_DIR / "fruits_no_face.jpg"
    out_path = tmp_path / "annotated_fruits.jpg"

    pipeline = FaceProcessingPipeline(detector=HaarFaceDetector())
    res = pipeline.process_image(img_path, output_path=out_path, save_even_if_no_faces=True)

    assert res.success is True
    assert res.num_faces_detected == 0
    assert res.output_path is not None
    assert Path(res.output_path).is_file()
    assert res.timing.get("face_detection") > 0.0


def test_pipeline_nonexistent_file():
    pipeline = FaceProcessingPipeline(detector=HaarFaceDetector())
    res = pipeline.process_image("file_that_does_not_exist_404.jpg")

    assert res.success is False
    assert res.num_faces_detected == 0
    assert res.error_message is not None
    assert "not found" in res.error_message.lower()


def test_annotator_preserves_dimensions():
    img_path = DATA_DIR / "lena.jpg"
    img = load_image(img_path)
    detector = HaarFaceDetector()
    detections = detector.detect(img)

    annotator = FaceAnnotator(draw_hud=True)
    annotated = annotator.annotate(img, detections, detector_name="Haar", processing_time_ms=12.5)

    assert annotated.shape == img.shape
