"""Unit and integration tests for Phase 3: Classical Face Swapping."""

from pathlib import Path
import numpy as np
import pytest

from identity_studio.core.image_ops import load_image
from identity_studio.faceswap import (
    BlendMode,
    ClassicalFaceSwapper,
    compute_delaunay_triangles,
    create_face_mask,
    match_color_reinhard,
    warp_face_mesh,
    warp_triangle,
)
from identity_studio.geometry.landmarks import MediaPipeFaceMeshDetector

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "inputs"


@pytest.fixture(scope="module")
def mesh_detector():
    detector = MediaPipeFaceMeshDetector(max_num_faces=2)
    yield detector
    detector.close()


def test_delaunay_triangulation():
    points = np.array([
        [10, 10],
        [100, 10],
        [50, 100],
        [50, 50],
    ], dtype=np.int32)

    triangles = compute_delaunay_triangles((150, 150), points)
    assert len(triangles) >= 1
    for tri in triangles:
        assert len(tri) == 3
        assert len(set(tri)) == 3  # All distinct indices


def test_warp_triangle_execution():
    src = np.full((100, 100, 3), 120, dtype=np.uint8)
    dst = np.zeros((100, 100, 3), dtype=np.uint8)

    src_tri = np.array([[10, 10], [50, 10], [30, 50]], dtype=np.float32)
    dst_tri = np.array([[20, 20], [60, 20], [40, 60]], dtype=np.float32)

    warp_triangle(src, dst, src_tri, dst_tri)
    # The center of destination triangle should now contain source color (120)
    assert dst[30, 40, 0] > 0


def test_color_transfer_reinhard():
    # Source is dark blue, target is bright red
    src = np.zeros((50, 50, 3), dtype=np.uint8)
    src[:, :, 0] = 50  # Blue

    tgt = np.zeros((50, 50, 3), dtype=np.uint8)
    tgt[:, :, 2] = 200  # Red

    matched = match_color_reinhard(src, tgt)
    assert matched.shape == src.shape
    # Matched image red channel should increase significantly
    assert np.mean(matched[:, :, 2]) > np.mean(src[:, :, 2])


def test_create_face_mask():
    hull_pts = np.array([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=np.int32)
    mask = create_face_mask((100, 100), hull_pts, erode_iterations=1)

    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8
    # Center must be white (255)
    assert mask[50, 50] == 255
    # Corners outside hull must be black (0)
    assert mask[5, 5] == 0


def test_end_to_end_face_swap(mesh_detector):
    lena_path = DATA_DIR / "lena.jpg"
    david_path = DATA_DIR / "david1.jpg"

    assert lena_path.is_file() and david_path.is_file()
    lena = load_image(lena_path)
    david = load_image(david_path)

    swapper = ClassicalFaceSwapper(mesh_detector=mesh_detector, blend_mode=BlendMode.NORMAL)
    res = swapper.swap_faces(source_image=lena, target_image=david)

    assert res.swapped_image.shape == david.shape
    assert res.delaunay_triangles_count > 10
    assert res.face_mask.shape == (david.shape[0], david.shape[1])
    assert res.timing.total_ms > 0.0
