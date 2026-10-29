"""Seamless facial blending using Poisson image editing and feathered masks."""

from enum import Enum
from typing import Tuple
import cv2
import numpy as np


class BlendMode(Enum):
    NORMAL = "normal"
    MIXED = "mixed"
    FEATHERED_ALPHA = "feathered_alpha"


def create_face_mask(
    target_shape: Tuple[int, int],
    hull_points: np.ndarray,
    erode_iterations: int = 4,
    blur_kernel_size: int = 15,
) -> np.ndarray:
    """Generates a soft, feathered binary mask from face convex hull points.
    
    Args:
        target_shape: (height, width) of the target image.
        hull_points: (N, 2) array of coordinates outlining the face.
        erode_iterations: Morphological erosion iterations to retract borders slightly.
        blur_kernel_size: Gaussian blur kernel size for smooth edge transition.
        
    Returns:
        np.ndarray: uint8 mask (0-255).
    """
    h, w = target_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Fill convex hull polygon
    cv2.fillConvexPoly(mask, np.int32(hull_points), 255)

    # Erode slightly to prevent hair and background from bleeding into the clone boundary
    if erode_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.erode(mask, kernel, iterations=erode_iterations)

    return mask


def seamless_blend(
    warped_face: np.ndarray,
    target_image: np.ndarray,
    mask: np.ndarray,
    mode: BlendMode = BlendMode.NORMAL,
) -> np.ndarray:
    """Blends the warped face into the target image using Poisson editing or feathered alpha.
    
    Args:
        warped_face: Color-corrected warped face on full canvas.
        target_image: Original target image.
        mask: Binary mask of face ROI.
        mode: BlendMode.NORMAL, BlendMode.MIXED, or BlendMode.FEATHERED_ALPHA.
        
    Returns:
        np.ndarray: Seamlessly blended BGR result image.
    """
    h, w = target_image.shape[:2]

    # Calculate center of mass of the mask
    moments = cv2.moments(mask)
    if moments["m00"] < 1.0:
        return target_image.copy()

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    # Ensure center point is within bounds with a safety margin for seamlessClone
    margin = 5
    cx = max(margin, min(w - margin, cx))
    cy = max(margin, min(h - margin, cy))

    if mode == BlendMode.FEATHERED_ALPHA:
        # Gaussian blurred alpha blending
        feathered = cv2.GaussianBlur(mask, (21, 21), 11).astype(np.float32) / 255.0
        feathered_3ch = cv2.merge([feathered, feathered, feathered])
        blended = (
            warped_face.astype(np.float32) * feathered_3ch +
            target_image.astype(np.float32) * (1.0 - feathered_3ch)
        )
        return np.clip(blended, 0, 255).astype(np.uint8)

    clone_flag = cv2.NORMAL_CLONE if mode == BlendMode.NORMAL else cv2.MIXED_CLONE

    try:
        return cv2.seamlessClone(
            warped_face,
            target_image,
            mask,
            (cx, cy),
            clone_flag,
        )
    except cv2.error:
        # Fallback to feathered alpha blending if seamlessClone fails on edge cases
        return seamless_blend(warped_face, target_image, mask, mode=BlendMode.FEATHERED_ALPHA)
