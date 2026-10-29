"""2D Affine Facial Alignment based on inter-ocular geometry."""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

from .landmarks import FacialLandmarks


@dataclass
class AlignedFaceResult:
    aligned_image: np.ndarray
    transform_matrix: np.ndarray
    inverse_matrix: np.ndarray
    target_size: Tuple[int, int]
    rotation_angle_deg: float
    scale_factor: float


def align_face_2d(
    image: np.ndarray,
    landmarks: FacialLandmarks,
    target_size: Tuple[int, int] = (256, 256),
    desired_eye_y: float = 0.38,
    desired_eye_dist_ratio: float = 0.35,
) -> AlignedFaceResult:
    """Aligns a face into a canonical upright orientation based on eye centers.
    
    Args:
        image: Original BGR image.
        landmarks: FacialLandmarks instance for the target face.
        target_size: Output (width, height) in pixels.
        desired_eye_y: Vertical position ratio for eyes in output frame (e.g., 0.38).
        desired_eye_dist_ratio: Desired distance between eyes as ratio of output width.
        
    Returns:
        AlignedFaceResult with aligned image array and bidirectional affine matrices.
    """
    target_w, target_h = target_size
    left_eye = landmarks.left_eye_center    # Subject's left (viewer's right)
    right_eye = landmarks.right_eye_center  # Subject's right (viewer's left)

    # Compute delta and angle between eyes
    # Note: right_eye is to the left in pixel coordinates, left_eye is to the right
    dx = left_eye[0] - right_eye[0]
    dy = left_eye[1] - right_eye[1]
    dist = np.sqrt(dx ** 2 + dy ** 2)

    if dist < 1e-4:
        # Fallback if eyes cannot be differentiated
        return AlignedFaceResult(
            aligned_image=cv2.resize(image, target_size),
            transform_matrix=np.eye(2, 3, dtype=np.float32),
            inverse_matrix=np.eye(2, 3, dtype=np.float32),
            target_size=target_size,
            rotation_angle_deg=0.0,
            scale_factor=1.0,
        )

    # Rotation angle in degrees (clockwise)
    angle = np.degrees(np.arctan2(dy, dx))

    # Scale to match desired inter-ocular distance
    desired_dist = target_w * desired_eye_dist_ratio
    scale = desired_dist / dist

    # Midpoint between eyes in source image
    eye_center = ((right_eye[0] + left_eye[0]) / 2.0, (right_eye[1] + left_eye[1]) / 2.0)

    # Target eye midpoint in aligned frame
    target_eye_x = target_w * 0.5
    target_eye_y = target_h * desired_eye_y

    # Calculate 2x3 affine transformation matrix
    M = cv2.getRotationMatrix2D(eye_center, angle, scale)

    # Adjust translation offset so eye center lands at (target_eye_x, target_eye_y)
    t_x = target_eye_x - (M[0, 0] * eye_center[0] + M[0, 1] * eye_center[1])
    t_y = target_eye_y - (M[1, 0] * eye_center[0] + M[1, 1] * eye_center[1])
    M[0, 2] = t_x
    M[1, 2] = t_y

    # Invert affine matrix for un-warping later
    inv_M = cv2.invertAffineTransform(M)

    # Warp image to aligned canonical frame
    aligned = cv2.warpAffine(
        image,
        M,
        (target_w, target_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT,
    )

    return AlignedFaceResult(
        aligned_image=aligned,
        transform_matrix=M,
        inverse_matrix=inv_M,
        target_size=target_size,
        rotation_angle_deg=float(angle),
        scale_factor=float(scale),
    )
