"""3D Head Pose and Orientation Estimation using Perspective-n-Point (solvePnP)."""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

from .landmarks import FacialLandmarks, FacialRegionIndices


@dataclass
class HeadPoseResult:
    pitch_deg: float  # Up / Down (+: Down, -: Up)
    yaw_deg: float    # Left / Right (+: Right, -: Left)
    roll_deg: float   # Tilt (+: Clockwise, -: Counter-clockwise)
    rotation_vector: np.ndarray
    translation_vector: np.ndarray
    axis_points_2d: Optional[np.ndarray] = None  # [Nose, X-tip, Y-tip, Z-tip]


class HeadPoseEstimator:
    """Estimates 3D head orientation and Euler angles from 2D facial landmarks."""

    # 3D canonical anthropometric face model coordinates (in millimeters)
    # Aligned with image coordinate system: +X right, +Y down, +Z forward
    MODEL_POINTS_3D = np.array(
        [
            (0.0, 0.0, 0.0),          # 1: Nose tip
            (0.0, 330.0, -65.0),      # 152: Chin
            (225.0, -170.0, -135.0),  # 263: Left eye outer corner (viewer right)
            (-225.0, -170.0, -135.0), # 33: Right eye outer corner (viewer left)
            (150.0, 150.0, -125.0),   # 287: Left mouth corner (viewer right)
            (-150.0, 150.0, -125.0),  # 57: Right mouth corner (viewer left)
        ],
        dtype=np.float64,
    )

    def estimate_pose(self, landmarks: FacialLandmarks, axis_length: float = 120.0) -> HeadPoseResult:
        """Calculates 3D head rotation and translation vectors using solvePnP.
        
        Args:
            landmarks: Extracted FacialLandmarks.
            axis_length: Length of the projected 3D coordinate axes in millimeters.
            
        Returns:
            HeadPoseResult with Euler angles (pitch, yaw, roll) and projected 2D axis points.
        """
        w = landmarks.image_width
        h = landmarks.image_height

        # Extract corresponding 2D landmark coordinates
        image_points_2d = []
        for idx in FacialRegionIndices.HEAD_POSE_LANDMARKS:
            pt = landmarks.points[idx]
            image_points_2d.append([float(pt.pixel_x), float(pt.pixel_y)])
        image_points_2d = np.array(image_points_2d, dtype=np.float64)

        # Approximate camera intrinsic matrix
        focal_length = float(w)
        center = (float(w) / 2.0, float(h) / 2.0)
        camera_matrix = np.array(
            [
                [focal_length, 0.0, center[0]],
                [0.0, focal_length, center[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # Assume zero lens distortion

        # Solve Perspective-n-Point
        success, rvec, tvec = cv2.solvePnP(
            self.MODEL_POINTS_3D,
            image_points_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return HeadPoseResult(
                pitch_deg=0.0,
                yaw_deg=0.0,
                roll_deg=0.0,
                rotation_vector=np.zeros(3),
                translation_vector=np.zeros(3),
            )

        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)

        # Compute Euler angles from rotation matrix
        # Pitch (X), Yaw (Y), Roll (Z)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        pitch = float(angles[0])
        yaw = float(angles[1])
        roll = float(angles[2])

        # Project 3D coordinate axes onto 2D image plane:
        # Origin = nose tip (0, 0, 0)
        # X-axis = Red (Right)
        # Y-axis = Green (Down)
        # Z-axis = Blue (Forward / out of face)
        axis_3d = np.array(
            [
                (0.0, 0.0, 0.0),
                (axis_length, 0.0, 0.0),
                (0.0, axis_length, 0.0),
                (0.0, 0.0, axis_length),
            ],
            dtype=np.float64,
        )

        projected_axes, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
        axis_pts_2d = projected_axes.reshape(-1, 2)

        return HeadPoseResult(
            pitch_deg=pitch,
            yaw_deg=yaw,
            roll_deg=roll,
            rotation_vector=rvec,
            translation_vector=tvec,
            axis_points_2d=axis_pts_2d,
        )
