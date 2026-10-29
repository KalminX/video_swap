"""Unit tests for image loading, manipulation, and transformations."""

from pathlib import Path
import numpy as np
import pytest

from identity_studio.core.image_ops import (
    load_image,
    save_image,
    get_image_metadata,
    convert_color_space,
    resize_image,
    crop_center,
    rotate_image,
    ColorSpace,
)


@pytest.fixture
def sample_image():
    """Generates a synthetic 100x120 RGB test image."""
    img = np.zeros((120, 100, 3), dtype=np.uint8)
    img[:, :, 0] = 50   # Blue
    img[:, :, 1] = 100  # Green
    img[:, :, 2] = 200  # Red
    return img


def test_metadata_extraction(sample_image):
    meta = get_image_metadata(sample_image)
    assert meta.width == 100
    assert meta.height == 120
    assert meta.channels == 3
    assert meta.total_pixels == 12000
    assert pytest.approx(meta.aspect_ratio, 0.01) == 100 / 120


def test_save_and_load(tmp_path, sample_image):
    save_path = tmp_path / "subfolder" / "test.png"
    saved = save_image(sample_image, save_path)
    assert saved.is_file()

    loaded = load_image(save_path)
    assert loaded.shape == sample_image.shape
    assert np.array_equal(loaded, sample_image)


def test_load_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load_image("non_existent_file_xyz_123.jpg")


def test_color_conversions(sample_image):
    gray = convert_color_space(sample_image, ColorSpace.BGR, ColorSpace.GRAY)
    assert len(gray.shape) == 2
    assert gray.shape == (120, 100)

    rgb = convert_color_space(sample_image, ColorSpace.BGR, ColorSpace.RGB)
    assert rgb.shape == sample_image.shape
    # Channels should be swapped
    assert np.array_equal(rgb[:, :, 0], sample_image[:, :, 2])


def test_resize_aspect_ratio(sample_image):
    # Original is 100w x 120h (aspect ratio ~0.833)
    resized = resize_image(sample_image, target_width=50, keep_aspect_ratio=True)
    assert resized.shape[1] == 50
    assert resized.shape[0] == 60


def test_resize_exact_dimensions(sample_image):
    resized = resize_image(sample_image, target_width=64, target_height=64, keep_aspect_ratio=False)
    assert resized.shape[1] == 64
    assert resized.shape[0] == 64


def test_crop_center(sample_image):
    cropped = crop_center(sample_image, crop_width=40, crop_height=30)
    assert cropped.shape == (30, 40, 3)


def test_rotate_image(sample_image):
    # 90 degree rotation with expanded bounds should swap width and height
    rotated = rotate_image(sample_image, angle_degrees=90, expand_bounds=True)
    assert rotated.shape[0] == 100
    assert rotated.shape[1] == 120
