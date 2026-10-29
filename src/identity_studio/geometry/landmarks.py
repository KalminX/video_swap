"""MediaPipe 468-point 3D facial landmark extraction and anatomical indexing."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class LandmarkPoint:
    x: float  # Normalized [0, 1]
    y: float  # Normalized [0, 1]
    z: float  # Normalized depth relative to face center
    pixel_x: int = 0
    pixel_y: int = 0


@dataclass
class FacialLandmarks:
    """468/478-point 3D facial landmarks with semantic region accessors."""
    points: List[LandmarkPoint]
    image_width: int
    image_height: int

    def get_pixel_coords(self, indices: List[int]) -> np.ndarray:
        """Extract (N, 2) pixel coordinates for a subset of landmark indices."""
        coords = []
        for idx in indices:
            if idx < len(self.points):
                pt = self.points[idx]
                coords.append([pt.pixel_x, pt.pixel_y])
        return np.array(coords, dtype=np.int32)

    def get_3d_coords(self, indices: List[int]) -> np.ndarray:
        """Extract (N, 3) normalized coordinates for a subset of landmark indices."""
        coords = []
        for idx in indices:
            if idx < len(self.points):
                pt = self.points[idx]
                coords.append([pt.x, pt.y, pt.z])
        return np.array(coords, dtype=np.float32)

    @property
    def left_eye_center(self) -> Tuple[float, float]:
        """Estimated center of the subject's left eye (viewer's right)."""
        pts = self.get_pixel_coords(FacialRegionIndices.LEFT_EYE)
        if len(pts) == 0:
            return (0.0, 0.0)
        return (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))

    @property
    def right_eye_center(self) -> Tuple[float, float]:
        """Estimated center of the subject's right eye (viewer's left)."""
        pts = self.get_pixel_coords(FacialRegionIndices.RIGHT_EYE)
        if len(pts) == 0:
            return (0.0, 0.0)
        return (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))

    @property
    def nose_tip(self) -> Tuple[int, int]:
        pt = self.points[1]  # Standard landmark 1 is nose tip
        return (pt.pixel_x, pt.pixel_y)

    @property
    def chin(self) -> Tuple[int, int]:
        pt = self.points[152]  # Standard landmark 152 is lowest chin point
        return (pt.pixel_x, pt.pixel_y)


class FacialRegionIndices:
    """Canonical MediaPipe 468 Face Mesh landmark index groupings."""
    # Contours & Oval
    FACE_OVAL = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10
    ]

    # Eyes
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

    # Eyebrows
    LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
    RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

    # Nose
    NOSE_BRIDGE = [168, 6, 197, 195, 5]
    NOSE_TIP = [1, 2, 98, 327]

    # Lips (Outer & Inner)
    LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]

    # Key anthropometric landmarks for 3D Head Pose solvePnP
    # [Nose Tip, Chin, Left Eye Corner, Right Eye Corner, Left Mouth Corner, Right Mouth Corner]
    HEAD_POSE_LANDMARKS = [1, 152, 263, 33, 287, 57]


class MediaPipeFaceMeshDetector:
    """Wrapper around MediaPipe Face Mesh for multi-face 3D landmark extraction."""

    def __init__(
        self,
        max_num_faces: int = 5,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.3,
        min_tracking_confidence: float = 0.3,
    ) -> None:
        import mediapipe as mp

        self.mp_face_mesh = mp.solutions.face_mesh
        self._detector = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract_landmarks(self, image: np.ndarray) -> List[FacialLandmarks]:
        """Detect and extract 3D facial landmarks from a BGR image array.
        
        Args:
            image: BGR numpy image array.
            
        Returns:
            List[FacialLandmarks]: Landmark structures for each detected face.
        """
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        # MediaPipe expects RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb_image)

        face_landmarks_list: List[FacialLandmarks] = []
        if not results.multi_face_landmarks:
            return face_landmarks_list

        for face_lms in results.multi_face_landmarks:
            pts: List[LandmarkPoint] = []
            for lm in face_lms.landmark:
                px = int(min(np.floor(lm.x * w), w - 1))
                py = int(min(np.floor(lm.y * h), h - 1))
                pts.append(
                    LandmarkPoint(
                        x=lm.x,
                        y=lm.y,
                        z=lm.z,
                        pixel_x=max(0, px),
                        pixel_y=max(0, py),
                    )
                )

            face_landmarks_list.append(
                FacialLandmarks(
                    points=pts,
                    image_width=w,
                    image_height=h,
                )
            )

        return face_landmarks_list

    def close(self) -> None:
        self._detector.close()
