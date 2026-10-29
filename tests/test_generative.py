"""Unit and integration tests for Phase 4: Generative AI Face Transformation."""

from pathlib import Path
import numpy as np
import pytest

from identity_studio.core.image_ops import load_image
from identity_studio.generative import (
    FacialMaskGenerator,
    GenerativeFaceTransformer,
    InpaintingMaskConfig,
    STYLE_PRESETS,
    evaluate_transformation_fidelity,
    get_preset,
    list_available_presets,
)
from identity_studio.geometry.landmarks import MediaPipeFaceMeshDetector

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "inputs"


@pytest.fixture(scope="module")
def mesh_detector():
    detector = MediaPipeFaceMeshDetector(max_num_faces=1)
    yield detector
    detector.close()


def test_presets_loading():
    presets = list_available_presets()
    assert len(presets) >= 4
    assert "cyborg_robot" in presets
    assert "extraterrestrial_alien" in presets

    cyborg = get_preset("cyborg_robot")
    assert cyborg.guidance_scale > 1.0
    assert 0.0 < cyborg.strength <= 1.0
    assert "chrome" in cyborg.prompt.lower() or "cybernetic" in cyborg.prompt.lower()


def test_mask_generation(mesh_detector):
    img = load_image(DATA_DIR / "lena.jpg")
    landmarks_list = mesh_detector.extract_landmarks(img)
    assert len(landmarks_list) == 1

    generator = FacialMaskGenerator()
    mask = generator.generate_mask(img.shape, landmarks_list[0])

    assert mask.shape == (img.shape[0], img.shape[1])
    assert mask.dtype == np.uint8

    # Nose tip must be within masked region (white)
    nose = landmarks_list[0].nose_tip
    assert mask[nose[1], nose[0]] > 200

    # Top-left corner outside face should be black (0)
    assert mask[10, 10] == 0


def test_mask_generation_preserve_eyes(mesh_detector):
    img = load_image(DATA_DIR / "lena.jpg")
    landmarks = mesh_detector.extract_landmarks(img)[0]

    cfg = InpaintingMaskConfig(preserve_eyes=True, feather_blur_kernel=0)
    generator = FacialMaskGenerator(config=cfg)
    mask = generator.generate_mask(img.shape, landmarks)

    # Eye center should be excluded from mask (0)
    lx, ly = int(landmarks.left_eye_center[0]), int(landmarks.left_eye_center[1])
    assert mask[ly, lx] == 0


def test_composition_fidelity_metrics():
    original = np.full((100, 100, 3), 150, dtype=np.uint8)
    transformed = original.copy()
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255  # 40x40 face region

    # Modify face region only
    transformed[30:70, 30:70] = 50

    metrics = evaluate_transformation_fidelity(original, transformed, mask)
    # Background must be 100% preserved
    assert metrics.background_preservation_ratio == 100.0
    assert metrics.background_psnr_db == 100.0
    assert metrics.face_transformation_delta == 100.0


def test_transformer_mock_pipeline(mesh_detector):
    img = load_image(DATA_DIR / "lena.jpg")
    landmarks = mesh_detector.extract_landmarks(img)[0]
    mask = FacialMaskGenerator().generate_mask(img.shape, landmarks)

    transformer = GenerativeFaceTransformer(use_mock_backend=True)
    res = transformer.transform_face(img, mask, preset_name="cyborg_robot")

    assert res.transformed_image.shape == img.shape
    assert res.preset_used == "Cybernetic Android"
    assert res.metrics.background_preservation_ratio > 95.0
    assert res.timing.total_ms > 0.0
