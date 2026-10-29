"""Unit and integration tests for Phase 2: Facial Geometry, Alignment, and Head Pose."""

from pathlib import Path
import numpy as np
import pytest

from identity_studio.core.image_ops import load_image
from identity_studio.geometry import (
    FacialGeometryVisualizer,
    HeadPoseEstimator,
    MediaPipeFaceMeshDetector,
    FaceGeometryTracker,
    align_face_2d,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "inputs"


@pytest.fixture(scope="module")
def mesh_detector():
    detector = MediaPipeFaceMeshDetector(max_num_faces=2)
    yield detector
    detector.close()


def test_mediapipe_landmark_extraction(mesh_detector):
    img_path = DATA_DIR / "lena.jpg"
    img = load_image(img_path)

    landmarks_list = mesh_detector.extract_landmarks(img)
    assert len(landmarks_list) == 1

    face = landmarks_list[0]
    # MediaPipe Face Mesh extracts at least 468 points
    assert len(face.points) >= 468

    # Verify eye centers and key landmarks are non-zero and within frame
    lx, ly = face.left_eye_center
    rx, ry = face.right_eye_center
    assert 0 < lx < face.image_width
    assert 0 < ly < face.image_height
    assert 0 < rx < face.image_width
    assert 0 < ry < face.image_height

    # In pixel space, viewer's left eye (rx) has smaller X coordinate than viewer's right eye (lx)
    assert rx < lx


def test_mediapipe_no_faces(mesh_detector):
    img_path = DATA_DIR / "fruits_no_face.jpg"
    img = load_image(img_path)

    landmarks_list = mesh_detector.extract_landmarks(img)
    assert len(landmarks_list) == 0


def test_align_face_2d(mesh_detector):
    img_path = DATA_DIR / "lena.jpg"
    img = load_image(img_path)
    landmarks_list = mesh_detector.extract_landmarks(img)
    assert len(landmarks_list) >= 1

    target_size = (256, 256)
    res = align_face_2d(img, landmarks_list[0], target_size=target_size)

    assert res.aligned_image.shape == (256, 256, 3)
    assert res.transform_matrix.shape == (2, 3)
    assert res.inverse_matrix.shape == (2, 3)
    assert res.scale_factor > 0.0

    # Invertibility check: M followed by inv_M on a test point returns original point
    pt = np.array([100.0, 100.0, 1.0])
    forward_pt = res.transform_matrix @ pt
    forward_homo = np.array([forward_pt[0], forward_pt[1], 1.0])
    recon_pt = res.inverse_matrix @ forward_homo
    assert pytest.approx(recon_pt[0], abs=1e-2) == 100.0
    assert pytest.approx(recon_pt[1], abs=1e-2) == 100.0


def test_head_pose_estimation(mesh_detector):
    img_path = DATA_DIR / "lena.jpg"
    img = load_image(img_path)
    landmarks_list = mesh_detector.extract_landmarks(img)
    assert len(landmarks_list) >= 1

    estimator = HeadPoseEstimator()
    pose = estimator.estimate_pose(landmarks_list[0])

    # Lena has a slight tilt and turn, Euler angles should be reasonable
    assert -90.0 < pose.pitch_deg < 90.0
    assert -90.0 < pose.yaw_deg < 90.0
    assert -90.0 < pose.roll_deg < 90.0

    assert pose.axis_points_2d is not None
    assert pose.axis_points_2d.shape == (4, 2)


def test_visualizer_rendering(mesh_detector):
    img_path = DATA_DIR / "lena.jpg"
    img = load_image(img_path)
    landmarks_list = mesh_detector.extract_landmarks(img)
    assert len(landmarks_list) >= 1

    visualizer = FacialGeometryVisualizer()
    mesh_img = visualizer.draw_landmarks(img, landmarks_list[0], draw_contours=True)
    assert mesh_img.shape == img.shape

    estimator = HeadPoseEstimator()
    pose = estimator.estimate_pose(landmarks_list[0])
    pose_img = visualizer.draw_head_pose(mesh_img, pose)
    assert pose_img.shape == img.shape


def test_face_tracker():
    tracker = FaceGeometryTracker(max_num_faces=1)
    img_path = DATA_DIR / "lena.jpg"
    img = load_image(img_path)

    # Frame 1
    state1 = tracker.process_frame(img)
    assert len(state1) == 1
    assert state1[0].aligned_face is not None

    # Frame 2 (checks smoothing)
    state2 = tracker.process_frame(img)
    assert len(state2) == 1
    assert abs(state2[0].smoothed_pitch - state1[0].smoothed_pitch) < 1.0

    tracker.close()
