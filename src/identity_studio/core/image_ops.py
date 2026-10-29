"""Foundational image processing operations using OpenCV and NumPy."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np


class ColorSpace(Enum):
    BGR = "BGR"
    RGB = "RGB"
    GRAY = "GRAY"
    HSV = "HSV"


@dataclass
class ImageMetadata:
    width: int
    height: int
    channels: int
    dtype: str
    aspect_ratio: float
    total_pixels: int


def load_image(file_path: Union[str, Path], color_space: ColorSpace = ColorSpace.BGR) -> np.ndarray:
    """Load an image from disk with validation.
    
    Args:
        file_path: Path to the image file.
        color_space: Target color space (default: BGR, as standard in OpenCV).
        
    Returns:
        np.ndarray: The loaded image array.
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file cannot be decoded as an image.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path.resolve()}")

    # Read image using OpenCV
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to decode image file (corrupted or unsupported format): {path}")

    # Standardize alpha channel if present (convert 4-channel BGRA to BGR unless requested otherwise)
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif len(image.shape) == 2 and color_space != ColorSpace.GRAY:
        # 1-channel grayscale image loaded, convert to 3-channel BGR first
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if color_space != ColorSpace.BGR:
        return convert_color_space(image, ColorSpace.BGR, color_space)

    return image


def save_image(image: np.ndarray, file_path: Union[str, Path]) -> Path:
    """Save an image array to disk, auto-creating parent directories.
    
    Args:
        image: Numpy array image (assumed BGR format for OpenCV writer).
        file_path: Destination path.
        
    Returns:
        Path: The absolute path of the saved file.
        
    Raises:
        ValueError: If image is empty or invalid.
        IOError: If writing to disk fails.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot save an empty or None image array.")

    path = Path(file_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(path), image)
    if not success:
        raise IOError(f"Failed to write image to: {path}")

    return path


def get_image_metadata(image: np.ndarray) -> ImageMetadata:
    """Extract dimension, channel, and pixel information from an image array."""
    if image is None or image.size == 0:
        raise ValueError("Cannot extract metadata from an empty or None image.")

    shape = image.shape
    height = shape[0]
    width = shape[1]
    channels = shape[2] if len(shape) > 2 else 1
    total_pixels = width * height
    aspect_ratio = width / height if height > 0 else 0.0

    return ImageMetadata(
        width=width,
        height=height,
        channels=channels,
        dtype=str(image.dtype),
        aspect_ratio=aspect_ratio,
        total_pixels=total_pixels,
    )


def convert_color_space(
    image: np.ndarray,
    from_space: ColorSpace,
    to_space: ColorSpace,
) -> np.ndarray:
    """Convert an image between common color spaces."""
    if from_space == to_space:
        return image.copy()

    conversion_code = None

    if from_space == ColorSpace.BGR:
        if to_space == ColorSpace.RGB:
            conversion_code = cv2.COLOR_BGR2RGB
        elif to_space == ColorSpace.GRAY:
            conversion_code = cv2.COLOR_BGR2GRAY
        elif to_space == ColorSpace.HSV:
            conversion_code = cv2.COLOR_BGR2HSV
    elif from_space == ColorSpace.RGB:
        if to_space == ColorSpace.BGR:
            conversion_code = cv2.COLOR_RGB2BGR
        elif to_space == ColorSpace.GRAY:
            conversion_code = cv2.COLOR_RGB2GRAY
        elif to_space == ColorSpace.HSV:
            conversion_code = cv2.COLOR_RGB2HSV
    elif from_space == ColorSpace.GRAY:
        if to_space == ColorSpace.BGR:
            conversion_code = cv2.COLOR_GRAY2BGR
        elif to_space == ColorSpace.RGB:
            conversion_code = cv2.COLOR_GRAY2RGB

    if conversion_code is None:
        raise NotImplementedError(f"Conversion from {from_space} to {to_space} is not directly supported.")

    return cv2.cvtColor(image, conversion_code)


def resize_image(
    image: np.ndarray,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    keep_aspect_ratio: bool = True,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resize image, optionally maintaining the aspect ratio."""
    h, w = image.shape[:2]

    if target_width is None and target_height is None:
        return image.copy()

    if keep_aspect_ratio:
        if target_width is not None and target_height is not None:
            # Scale down to fit inside bounding box
            scale = min(target_width / w, target_height / h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
        elif target_width is not None:
            scale = target_width / w
            new_w = target_width
            new_h = max(1, int(h * scale))
        else:
            scale = target_height / h
            new_w = max(1, int(w * scale))
            new_h = target_height
    else:
        new_w = target_width if target_width is not None else w
        new_h = target_height if target_height is not None else h

    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def crop_center(image: np.ndarray, crop_width: int, crop_height: int) -> np.ndarray:
    """Crop the center portion of an image with boundary protection."""
    h, w = image.shape[:2]
    crop_w = min(w, crop_width)
    crop_h = min(h, crop_height)

    start_x = (w - crop_w) // 2
    start_y = (h - crop_h) // 2

    return image[start_y : start_y + crop_h, start_x : start_x + crop_w].copy()


def rotate_image(
    image: np.ndarray,
    angle_degrees: float,
    expand_bounds: bool = True,
    border_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Rotate image by an arbitrary angle in degrees.
    
    Args:
        image: Input image array.
        angle_degrees: Positive is counter-clockwise.
        expand_bounds: Whether to expand image canvas so corners aren't clipped.
        border_color: Fill color for newly exposed borders.
    """
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)

    # Rotation matrix
    rot_mat = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    if expand_bounds:
        cos = np.abs(rot_mat[0, 0])
        sin = np.abs(rot_mat[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust transformation matrix to account for new center
        rot_mat[0, 2] += (new_w / 2.0) - center[0]
        rot_mat[1, 2] += (new_h / 2.0) - center[1]
    else:
        new_w, new_h = w, h

    return cv2.warpAffine(
        image,
        rot_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_color,
    )
