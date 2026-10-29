"""Piecewise affine warping of Delaunay triangles between source and target facial meshes."""

from typing import List, Tuple
import cv2
import numpy as np


def warp_triangle(
    src_image: np.ndarray,
    dst_canvas: np.ndarray,
    src_tri: np.ndarray,
    dst_tri: np.ndarray,
) -> None:
    """Warps an individual triangle patch from src_image into dst_canvas in-place.
    
    Args:
        src_image: Source BGR image.
        dst_canvas: Target canvas to composite into (modified in-place).
        src_tri: (3, 2) array of triangle vertices in source image coordinates.
        dst_tri: (3, 2) array of triangle vertices in target canvas coordinates.
    """
    # 1. Find bounding rectangle for each triangle
    r_src = cv2.boundingRect(np.float32([src_tri]))
    r_dst = cv2.boundingRect(np.float32([dst_tri]))

    # Ensure valid bounding box dimensions
    if r_src[2] <= 0 or r_src[3] <= 0 or r_dst[2] <= 0 or r_dst[3] <= 0:
        return

    # 2. Offset vertices by top-left coordinate of bounding rectangle
    src_tri_cropped = []
    dst_tri_cropped = []

    for i in range(3):
        src_tri_cropped.append(((src_tri[i][0] - r_src[0]), (src_tri[i][1] - r_src[1])))
        dst_tri_cropped.append(((dst_tri[i][0] - r_dst[0]), (dst_tri[i][1] - r_dst[1])))

    # 3. Extract source image patch
    src_h, src_w = src_image.shape[:2]
    x_s, y_s, w_s, h_s = r_src
    # Boundary guard
    if x_s < 0 or y_s < 0 or x_s + w_s > src_w or y_s + h_s > src_h:
        # Safe crop with padding
        x1 = max(0, x_s)
        y1 = max(0, y_s)
        x2 = min(src_w, x_s + w_s)
        y2 = min(src_h, y_s + h_s)
        src_cropped = np.zeros((h_s, w_s, 3), dtype=src_image.dtype)
        if x2 > x1 and y2 > y1:
            src_cropped[y1 - y_s : y2 - y_s, x1 - x_s : x2 - x_s] = src_image[y1:y2, x1:x2]
    else:
        src_cropped = src_image[y_s : y_s + h_s, x_s : x_s + w_s]

    # 4. Compute affine transform between triangular patches
    warp_mat = cv2.getAffineTransform(
        np.float32(src_tri_cropped),
        np.float32(dst_tri_cropped),
    )

    # 5. Apply affine transform to source patch
    w_d, h_d = r_dst[2], r_dst[3]
    dst_cropped = cv2.warpAffine(
        src_cropped,
        warp_mat,
        (w_d, h_d),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # 6. Create triangular mask for target patch
    mask = np.zeros((h_d, w_d, 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(dst_tri_cropped), (1.0, 1.0, 1.0), 16, 0)

    # 7. Composite warped triangle into destination canvas
    x_d, y_d = r_dst[0], r_dst[1]
    dst_h, dst_w = dst_canvas.shape[:2]

    # Boundary check for destination placement
    x1_d = max(0, x_d)
    y1_d = max(0, y_d)
    x2_d = min(dst_w, x_d + w_d)
    y2_d = min(dst_h, y_d + h_d)

    if x2_d > x1_d and y2_d > y1_d:
        crop_mask = mask[y1_d - y_d : y2_d - y_d, x1_d - x_d : x2_d - x_d]
        crop_patch = dst_cropped[y1_d - y_d : y2_d - y_d, x1_d - x_d : x2_d - x_d]

        dst_region = dst_canvas[y1_d:y2_d, x1_d:x2_d].astype(np.float32)
        blended = dst_region * (1.0 - crop_mask) + crop_patch.astype(np.float32) * crop_mask
        dst_canvas[y1_d:y2_d, x1_d:x2_d] = np.clip(blended, 0, 255).astype(dst_canvas.dtype)


def warp_face_mesh(
    src_image: np.ndarray,
    target_shape: Tuple[int, int],
    src_points: np.ndarray,
    target_points: np.ndarray,
    triangles: List[Tuple[int, int, int]],
) -> np.ndarray:
    """Warps an entire source face onto the target geometry via Delaunay triangles.
    
    Args:
        src_image: Source BGR image.
        target_shape: (height, width) of the target image canvas.
        src_points: (N, 2) source face landmark coordinates.
        target_points: (N, 2) target face landmark coordinates.
        triangles: List of (i1, i2, i3) triangle indices.
        
    Returns:
        np.ndarray: Full-sized canvas containing warped source face.
    """
    h_t, w_t = target_shape[:2]
    warped_face = np.zeros((h_t, w_t, 3), dtype=np.uint8)

    for tri in triangles:
        idx1, idx2, idx3 = tri
        t_src = np.array([src_points[idx1], src_points[idx2], src_points[idx3]], dtype=np.float32)
        t_dst = np.array([target_points[idx1], target_points[idx2], target_points[idx3]], dtype=np.float32)

        warp_triangle(src_image, warped_face, t_src, t_dst)

    return warped_face
