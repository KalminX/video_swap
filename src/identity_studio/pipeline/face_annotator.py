"""Visual annotation for face detections with bounding boxes, badges, and latency overlays."""

from typing import List, Optional, Tuple
import cv2
import numpy as np

from ..detectors.base import DetectionResult


class FaceAnnotator:
    """Renders detection bounding boxes, confidence tags, and runtime metrics on images."""

    def __init__(
        self,
        box_color: Tuple[int, int, int] = (0, 220, 100),  # Emerald green (BGR)
        box_thickness: int = 2,
        font_scale: float = 0.5,
        draw_landmarks: bool = True,
        draw_hud: bool = True,
    ) -> None:
        self.box_color = box_color
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.draw_landmarks = draw_landmarks
        self.draw_hud = draw_hud

    def annotate(
        self,
        image: np.ndarray,
        detections: List[DetectionResult],
        detector_name: str = "",
        processing_time_ms: Optional[float] = None,
    ) -> np.ndarray:
        """Draw bounding boxes and optional info overlay onto an image copy.
        
        Args:
            image: Original BGR image.
            detections: List of detection results.
            detector_name: Optional name of the detector for HUD.
            processing_time_ms: Optional latency measurement for HUD.
            
        Returns:
            np.ndarray: Annotated BGR image copy.
        """
        output = image.copy()
        h, w = output.shape[:2]

        # Draw each detected face
        for idx, det in enumerate(detections, start=1):
            bbox = det.bbox
            x1, y1 = bbox.x, bbox.y
            x2, y2 = bbox.x2, bbox.y2

            # Bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), self.box_color, self.box_thickness)

            # Corner accents
            corner_len = min(int(min(bbox.width, bbox.height) * 0.2), 20)
            accent_color = (0, 255, 255)  # Yellow accent
            accent_thick = max(1, self.box_thickness + 1)
            # Top-left
            cv2.line(output, (x1, y1), (x1 + corner_len, y1), accent_color, accent_thick)
            cv2.line(output, (x1, y1), (x1, y1 + corner_len), accent_color, accent_thick)
            # Top-right
            cv2.line(output, (x2, y1), (x2 - corner_len, y1), accent_color, accent_thick)
            cv2.line(output, (x2, y1), (x2, y1 + corner_len), accent_color, accent_thick)
            # Bottom-left
            cv2.line(output, (x1, y2), (x1 + corner_len, y2), accent_color, accent_thick)
            cv2.line(output, (x1, y2), (x1, y2 - corner_len), accent_color, accent_thick)
            # Bottom-right
            cv2.line(output, (x2, y2), (x2 - corner_len, y2), accent_color, accent_thick)
            cv2.line(output, (x2, y2), (x2, y2 - corner_len), accent_color, accent_thick)

            # Label badge
            label = f"Face #{idx}"
            if det.confidence < 1.0:
                label += f": {det.confidence:.1%}"

            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1)
            badge_y1 = max(0, y1 - th - 8)
            badge_y2 = y1
            badge_x2 = min(w, x1 + tw + 10)

            # Semi-transparent background for label
            cv2.rectangle(output, (x1, badge_y1), (badge_x2, badge_y2), (30, 30, 30), cv2.FILLED)
            cv2.putText(
                output,
                label,
                (x1 + 4, badge_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Landmarks (if available)
            if self.draw_landmarks and det.landmarks:
                for pt in det.landmarks:
                    cv2.circle(output, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)

        # Draw HUD overlay if requested
        if self.draw_hud:
            hud_texts = [f"Faces: {len(detections)}"]
            if detector_name:
                hud_texts.append(f"Model: {detector_name}")
            if processing_time_ms is not None:
                hud_texts.append(f"Time: {processing_time_ms:.1f}ms")

            hud_str = " | ".join(hud_texts)
            (tw, th), _ = cv2.getTextSize(hud_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            hud_h = th + 14
            overlay = output.copy()
            cv2.rectangle(overlay, (0, 0), (w, hud_h), (20, 20, 20), cv2.FILLED)
            cv2.addWeighted(overlay, 0.65, output, 0.35, 0, output)
            cv2.putText(
                output,
                hud_str,
                (12, hud_h - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )

        return output
