"""Facial geometry tracking and temporal smoothing for video and webcam streams."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from .alignment import AlignedFaceResult, align_face_2d
from .head_pose import HeadPoseEstimator, HeadPoseResult
from .landmarks import FacialLandmarks, MediaPipeFaceMeshDetector


@dataclass
class TrackedFaceState:
    landmarks: FacialLandmarks
    pose: HeadPoseResult
    smoothed_pitch: float
    smoothed_yaw: float
    smoothed_roll: float
    aligned_face: Optional[AlignedFaceResult] = None


class FaceGeometryTracker:
    """Tracks facial geometry, computes 3D head pose, and applies exponential smoothing."""

    def __init__(
        self,
        smoothing_alpha: float = 0.65,  # Weight of new frame vs history (0 < alpha <= 1)
        max_num_faces: int = 2,
        align_target_size: Tuple[int, int] = (256, 256),
    ) -> None:
        self.detector = MediaPipeFaceMeshDetector(max_num_faces=max_num_faces)
        self.pose_estimator = HeadPoseEstimator()
        self.smoothing_alpha = smoothing_alpha
        self.align_target_size = align_target_size

        # Historic smoothed pose state for stability
        self._prev_pitch: Optional[float] = None
        self._prev_yaw: Optional[float] = None
        self._prev_roll: Optional[float] = None

    def process_frame(self, frame: np.ndarray, align: bool = True) -> List[TrackedFaceState]:
        """Processes a single video or webcam frame and returns tracked face states."""
        landmarks_list = self.detector.extract_landmarks(frame)
        tracked_faces: List[TrackedFaceState] = []

        for idx, landmarks in enumerate(landmarks_list):
            raw_pose = self.pose_estimator.estimate_pose(landmarks)

            # Apply temporal smoothing (EMA)
            if self._prev_pitch is None:
                s_pitch = raw_pose.pitch_deg
                s_yaw = raw_pose.yaw_deg
                s_roll = raw_pose.roll_deg
            else:
                a = self.smoothing_alpha
                s_pitch = a * raw_pose.pitch_deg + (1.0 - a) * self._prev_pitch
                s_yaw = a * raw_pose.yaw_deg + (1.0 - a) * self._prev_yaw
                s_roll = a * raw_pose.roll_deg + (1.0 - a) * self._prev_roll

            if idx == 0:  # Primary face state tracking
                self._prev_pitch = s_pitch
                self._prev_yaw = s_yaw
                self._prev_roll = s_roll

            aligned: Optional[AlignedFaceResult] = None
            if align:
                aligned = align_face_2d(frame, landmarks, target_size=self.align_target_size)

            smoothed_pose = HeadPoseResult(
                pitch_deg=s_pitch,
                yaw_deg=s_yaw,
                roll_deg=s_roll,
                rotation_vector=raw_pose.rotation_vector,
                translation_vector=raw_pose.translation_vector,
                axis_points_2d=raw_pose.axis_points_2d,
            )

            tracked_faces.append(
                TrackedFaceState(
                    landmarks=landmarks,
                    pose=smoothed_pose,
                    smoothed_pitch=s_pitch,
                    smoothed_yaw=s_yaw,
                    smoothed_roll=s_roll,
                    aligned_face=aligned,
                )
            )

        return tracked_faces

    def reset(self) -> None:
        """Reset temporal state filters."""
        self._prev_pitch = None
        self._prev_yaw = None
        self._prev_roll = None

    def close(self) -> None:
        self.detector.close()
