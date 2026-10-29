"""Color and lighting correction using Reinhard CIELAB color transfer."""

from typing import Optional
import cv2
import numpy as np


def match_color_reinhard(
    source_image: np.ndarray,
    target_image: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Matches the color and lighting distribution of source_image to target_image in CIELAB space.
    
    Args:
        source_image: BGR image to be color-corrected (e.g. warped face).
        target_image: BGR reference image providing target lighting and skin tone.
        mask: Optional binary mask (0-255 or boolean) defining the facial ROI.
        
    Returns:
        np.ndarray: Color-adjusted BGR image.
    """
    # Convert BGR to CIELAB (L: lightness, a: green-red, b: blue-yellow)
    src_lab = cv2.cvtColor(source_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_image, cv2.COLOR_BGR2LAB).astype(np.float32)

    if mask is not None:
        roi = mask > 0
        if not np.any(roi):
            return source_image.copy()

        src_pixels = src_lab[roi]
        tgt_pixels = tgt_lab[roi]
    else:
        src_pixels = src_lab.reshape(-1, 3)
        tgt_pixels = tgt_lab.reshape(-1, 3)

    # Compute mean and standard deviation for each channel
    src_mean = np.mean(src_pixels, axis=0)
    src_std = np.std(src_pixels, axis=0)
    src_std = np.where(src_std < 1e-4, 1.0, src_std)

    tgt_mean = np.mean(tgt_pixels, axis=0)
    tgt_std = np.std(tgt_pixels, axis=0)

    # Scale and shift channels
    matched_lab = src_lab.copy()
    for ch in range(3):
        matched_lab[:, :, ch] = (
            (src_lab[:, :, ch] - src_mean[ch]) * (tgt_std[ch] / src_std[ch]) + tgt_mean[ch]
        )

    matched_lab = np.clip(matched_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)
