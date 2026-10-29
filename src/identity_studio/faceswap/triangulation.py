"""Delaunay Triangulation for facial landmark geometry."""

from typing import List, Tuple
import cv2
import numpy as np

from ..geometry.landmarks import FacialLandmarks, FacialRegionIndices


def get_face_hull_points(landmarks: FacialLandmarks) -> Tuple[np.ndarray, List[int]]:
    """Extracts convex hull points and their corresponding landmark indices for a face.
    
    Returns:
        hull_points: (N, 2) int32 array of coordinates.
        hull_indices: List of landmark indices corresponding to each hull point.
    """
    # Use facial oval and outer contour points
    contour_indices = list(dict.fromkeys(FacialRegionIndices.FACE_OVAL))
    contour_pts = landmarks.get_pixel_coords(contour_indices)

    hull_idx_list = cv2.convexHull(contour_pts, returnPoints=False)
    hull_indices = [contour_indices[i[0]] for i in hull_idx_list]
    hull_points = landmarks.get_pixel_coords(hull_indices)

    return hull_points, hull_indices


def compute_delaunay_triangles(
    image_shape: Tuple[int, int],
    points: np.ndarray,
) -> List[Tuple[int, int, int]]:
    """Computes Delaunay triangulation over a set of points and returns point index triplets.
    
    Args:
        image_shape: (height, width) of the bounding canvas.
        points: (N, 2) array of coordinates.
        
    Returns:
        List of (idx1, idx2, idx3) integer triplets indexing into `points`.
    """
    h, w = image_shape[:2]
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)

    # Insert points into subdiv
    for pt in points:
        # Clamp strictly within rect
        px = float(max(0, min(w - 1, pt[0])))
        py = float(max(0, min(h - 1, pt[1])))
        subdiv.insert((px, py))

    triangle_list = subdiv.getTriangleList()
    triangle_indices: List[Tuple[int, int, int]] = []

    for t in triangle_list:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        # Check if triangle vertices are within the image rectangle
        if (
            0 <= pt1[0] < w and 0 <= pt1[1] < h and
            0 <= pt2[0] < w and 0 <= pt2[1] < h and
            0 <= pt3[0] < w and 0 <= pt3[1] < h
        ):
            # Find the closest matching point indices in our input points array
            idx1 = _find_closest_point_index(pt1, points)
            idx2 = _find_closest_point_index(pt2, points)
            idx3 = _find_closest_point_index(pt3, points)

            if idx1 is not None and idx2 is not None and idx3 is not None:
                # Ensure unique vertices (not collinear degenerate triangles)
                if len({idx1, idx2, idx3}) == 3:
                    tri = tuple(sorted([idx1, idx2, idx3]))
                    if tri not in triangle_indices:
                        triangle_indices.append(tri)

    return triangle_indices


def _find_closest_point_index(pt: Tuple[float, float], points: np.ndarray, eps: float = 1.5) -> int:
    dists = np.hypot(points[:, 0] - pt[0], points[:, 1] - pt[1])
    min_idx = int(np.argmin(dists))
    if dists[min_idx] <= eps:
        return min_idx
    return None
