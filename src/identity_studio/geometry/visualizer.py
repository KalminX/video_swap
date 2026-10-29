"""Visualization utilities for facial landmarks, contours, and 3D head pose orientation."""

from typing import List, Optional, Tuple
import cv2
import numpy as np

from .head_pose import HeadPoseResult
from .landmarks import FacialLandmarks, FacialRegionIndices


class FacialGeometryVisualizer:
    """Renders 468-point landmarks, anatomical facial loops, and 3D orientation axes."""

    def __init__(
        self,
        contour_color: Tuple[int, int, int] = (255, 200, 0),    # Cyan/Gold in BGR
        eye_color: Tuple[int, int, int] = (0, 255, 255),        # Yellow
        lip_color: Tuple[int, int, int] = (0, 100, 255),        # Orange
        eyebrow_color: Tuple[int, int, int] = (255, 0, 200),    # Pink
        nose_color: Tuple[int, int, int] = (0, 255, 0),         # Green
        landmark_dot_radius: int = 1,
    ) -> None:
        self.contour_color = contour_color
        self.eye_color = eye_color
        self.lip_color = lip_color
        self.eyebrow_color = eyebrow_color
        self.nose_color = nose_color
        self.landmark_dot_radius = landmark_dot_radius

    def draw_landmarks(
        self,
        image: np.ndarray,
        landmarks: FacialLandmarks,
        draw_contours: bool = True,
        draw_all_points: bool = False,
    ) -> np.ndarray:
        """Annotates facial landmarks and anatomical contours onto an image copy."""
        canvas = image.copy()

        if draw_all_points:
            for pt in landmarks.points:
                cv2.circle(canvas, (pt.pixel_x, pt.pixel_y), self.landmark_dot_radius, (180, 180, 180), -1)

        if draw_contours:
            self._draw_polyline(canvas, landmarks, FacialRegionIndices.FACE_OVAL, self.contour_color, closed=False)
            self._draw_polyline(canvas, landmarks, FacialRegionIndices.LEFT_EYE, self.eye_color, closed=True)
            self._draw_polyline(canvas, landmarks, FacialRegionIndices.RIGHT_EYE, self.eye_color, closed=True)
            self._draw_polyline(canvas, landmarks, FacialRegionIndices.LEFT_EYEBROW, self.eyebrow_color, closed=False)
            self._draw_polyline(canvas, landmarks, FacialRegionIndices.RIGHT_EYEBROW, self.eyebrow_color, closed=False)
            self._draw_polyline(canvas, landmarks, FacialRegionIndices.NOSE_BRIDGE, self.nose_color, closed=False)
            self._draw_polyline(canvas, landmarks, FacialRegionIndices.NOSE_TIP, self.nose_color, closed=False)
            self._draw_polyline(canvas, landmarks, FacialRegionIndices.LIPS_OUTER, self.lip_color, closed=True)
            self._draw_polyline(canvas, landmarks, FacialRegionIndices.LIPS_INNER, self.lip_color, closed=True)

        return canvas

    def draw_head_pose(
        self,
        image: np.ndarray,
        pose: HeadPoseResult,
        origin_pt: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Draws 3D orientation axis vectors and Euler angle metrics on the image."""
        canvas = image.copy()

        if pose.axis_points_2d is not None and len(pose.axis_points_2d) >= 4:
            pts = pose.axis_points_2d
            origin = (int(pts[0][0]), int(pts[0][1])) if origin_pt is None else origin_pt
            x_tip = (int(pts[1][0]), int(pts[1][1]))
            y_tip = (int(pts[2][0]), int(pts[2][1]))
            z_tip = (int(pts[3][0]), int(pts[3][1]))

            # Draw 3D axis arrows
            # X-axis (Pitch/Right) -> Red
            cv2.arrowedLine(canvas, origin, x_tip, (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.2)
            # Y-axis (Yaw/Down) -> Green
            cv2.arrowedLine(canvas, origin, y_tip, (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.2)
            # Z-axis (Roll/Forward out of face) -> Blue
            cv2.arrowedLine(canvas, origin, z_tip, (255, 0, 0), 3, cv2.LINE_AA, tipLength=0.2)

        # Draw Orientation telemetry overlay
        pose_text = f"Pitch: {pose.pitch_deg:+.1f} deg | Yaw: {pose.yaw_deg:+.1f} deg | Roll: {pose.roll_deg:+.1f} deg"
        (tw, th), _ = cv2.getTextSize(pose_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        h, w = canvas.shape[:2]

        badge_y = h - 16
        cv2.rectangle(canvas, (10, badge_y - th - 6), (18 + tw, badge_y + 4), (20, 20, 20), cv2.FILLED)
        cv2.putText(
            canvas,
            pose_text,
            (14, badge_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return canvas

    def _draw_polyline(
        self,
        image: np.ndarray,
        landmarks: FacialLandmarks,
        indices: List[int],
        color: Tuple[int, int, int],
        closed: bool = False,
    ) -> None:
        pts = landmarks.get_pixel_coords(indices)
        if len(pts) > 1:
            cv2.polylines(image, [pts], isClosed=closed, color=color, thickness=1, lineType=cv2.LINE_AA)
