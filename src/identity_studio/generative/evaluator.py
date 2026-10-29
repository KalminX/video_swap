"""Evaluation metrics for measuring composition and background preservation."""

from dataclasses import dataclass
from typing import Tuple
import cv2
import numpy as np


@dataclass
class CompositionMetrics:
    background_preservation_ratio: float  # Percentage of exact background similarity [0 - 100%]
    background_psnr_db: float             # Peak Signal-to-Noise Ratio on unmasked background
    face_transformation_delta: float      # Mean pixel change within transformed face region


def evaluate_transformation_fidelity(
    original_image: np.ndarray,
    transformed_image: np.ndarray,
    mask: np.ndarray,
    mask_threshold: int = 20,
) -> CompositionMetrics:
    """Computes quantitative fidelity scores comparing original and transformed images.
    
    Args:
        original_image: Input BGR image.
        transformed_image: Output transformed BGR image.
        mask: uint8 inpainting mask (values > threshold represent the modified face).
        mask_threshold: Cut-off threshold separating face from background.
        
    Returns:
        CompositionMetrics dataclass.
    """
    if original_image.shape != transformed_image.shape:
        raise ValueError("Image shapes must match for evaluation.")

    bg_region = mask <= mask_threshold
    face_region = mask > mask_threshold

    # 1. Background Preservation (MSE on background pixels)
    if np.any(bg_region):
        diff_bg = original_image[bg_region].astype(np.float64) - transformed_image[bg_region].astype(np.float64)
        mse_bg = np.mean(diff_bg ** 2)
        psnr_bg = 100.0 if mse_bg < 1e-5 else 10.0 * np.log10((255.0 ** 2) / mse_bg)
        # Ratio where difference is below perceptual threshold (delta < 5 intensity levels)
        close_pixels = np.all(np.abs(diff_bg) < 5.0, axis=1)
        bg_preservation_ratio = float(np.mean(close_pixels) * 100.0)
    else:
        bg_preservation_ratio = 100.0
        psnr_bg = 100.0

    # 2. Face Transformation Delta (mean absolute change in face pixels)
    if np.any(face_region):
        diff_face = np.abs(
            original_image[face_region].astype(np.float64) - transformed_image[face_region].astype(np.float64)
        )
        face_delta = float(np.mean(diff_face))
    else:
        face_delta = 0.0

    return CompositionMetrics(
        background_preservation_ratio=bg_preservation_ratio,
        background_psnr_db=float(psnr_bg),
        face_transformation_delta=face_delta,
    )
