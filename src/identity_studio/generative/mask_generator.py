"""Facial inpainting mask generation with feature preservation controls."""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

from ..geometry.landmarks import FacialLandmarks, FacialRegionIndices


@dataclass
class InpaintingMaskConfig:
    dilate_pixels: int = 15
    feather_blur_kernel: int = 17
    preserve_eyes: bool = False
    preserve_lips: bool = False


class FacialMaskGenerator:
    """Creates soft inpainting masks from 468-point facial geometry."""

    def __init__(self, config: Optional[InpaintingMaskConfig] = None) -> None:
        self.config = config or InpaintingMaskConfig()

    def generate_mask(
        self,
        image_shape: Tuple[int, int],
        landmarks: FacialLandmarks,
    ) -> np.ndarray:
        """Generates an 8-bit inpainting mask (255 = inpaint face, 0 = keep original scene).
        
        Args:
            image_shape: (height, width) of the target image.
            landmarks: FacialLandmarks instance.
            
        Returns:
            np.ndarray: uint8 single-channel mask array.
        """
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # 1. Fill base face oval polygon
        oval_indices = list(dict.fromkeys(FacialRegionIndices.FACE_OVAL))
        oval_pts = landmarks.get_pixel_coords(oval_indices)
        hull = cv2.convexHull(oval_pts)
        cv2.fillConvexPoly(mask, hull, 255)

        # 2. Expand mask outward slightly to encompass eyebrows and hairline
        if self.config.dilate_pixels > 0:
            k_size = self.config.dilate_pixels * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            mask = cv2.dilate(mask, kernel)

        # 3. Optional: Mask out eyes to preserve original gaze and eye identity
        if self.config.preserve_eyes:
            left_eye_pts = landmarks.get_pixel_coords(FacialRegionIndices.LEFT_EYE)
            right_eye_pts = landmarks.get_pixel_coords(FacialRegionIndices.RIGHT_EYE)
            cv2.fillConvexPoly(mask, cv2.convexHull(left_eye_pts), 0)
            cv2.fillConvexPoly(mask, cv2.convexHull(right_eye_pts), 0)

        # 4. Optional: Mask out lips to preserve natural mouth expression
        if self.config.preserve_lips:
            lips_pts = landmarks.get_pixel_coords(FacialRegionIndices.LIPS_OUTER)
            cv2.fillConvexPoly(mask, cv2.convexHull(lips_pts), 0)

        # 5. Soft feathering for seamless edge blending
        if self.config.feather_blur_kernel > 1:
            k = self.config.feather_blur_kernel
            if k % 2 == 0:
                k += 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)

        return mask
