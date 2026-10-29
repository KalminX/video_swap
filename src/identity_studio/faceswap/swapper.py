"""Classical Face Swapper using Delaunay Triangulation and Poisson Blending."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from ..core.timer import ExecutionTimer, TimingReport
from ..geometry.landmarks import FacialLandmarks, MediaPipeFaceMeshDetector
from .blending import BlendMode, create_face_mask, seamless_blend
from .color_correction import match_color_reinhard
from .triangulation import compute_delaunay_triangles, get_face_hull_points
from .warping import warp_face_mesh


@dataclass
class FaceSwapResult:
    swapped_image: np.ndarray
    warped_face: np.ndarray
    face_mask: np.ndarray
    delaunay_triangles_count: int
    timing: TimingReport
    source_landmarks: FacialLandmarks
    target_landmarks: FacialLandmarks


class ClassicalFaceSwapper:
    """Orchestrates landmark extraction, Delaunay triangulation, warping, and seamless cloning."""

    def __init__(
        self,
        mesh_detector: Optional[MediaPipeFaceMeshDetector] = None,
        blend_mode: BlendMode = BlendMode.NORMAL,
        enable_color_transfer: bool = True,
    ) -> None:
        self.detector = mesh_detector or MediaPipeFaceMeshDetector(max_num_faces=2)
        self.blend_mode = blend_mode
        self.enable_color_transfer = enable_color_transfer

    def swap_faces(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        source_landmarks: Optional[FacialLandmarks] = None,
        target_landmarks: Optional[FacialLandmarks] = None,
    ) -> FaceSwapResult:
        """Replaces the face in target_image with the identity features from source_image.
        
        Args:
            source_image: BGR image providing identity facial features.
            target_image: BGR image providing scene, body, and pose.
            source_landmarks: Optional precomputed source landmarks.
            target_landmarks: Optional precomputed target landmarks.
            
        Returns:
            FaceSwapResult with final swapped image and diagnostics.
        """
        timer = ExecutionTimer()

        # 1. Landmark Extraction
        with timer.track("landmarks"):
            if source_landmarks is None:
                src_lms_list = self.detector.extract_landmarks(source_image)
                if not src_lms_list:
                    raise ValueError("No face detected in source image.")
                source_landmarks = src_lms_list[0]

            if target_landmarks is None:
                tgt_lms_list = self.detector.extract_landmarks(target_image)
                if not tgt_lms_list:
                    raise ValueError("No face detected in target image.")
                target_landmarks = tgt_lms_list[0]

        # 2. Extract Convex Hull & Triangulation Points
        with timer.track("triangulation"):
            tgt_hull_pts, tgt_hull_indices = get_face_hull_points(target_landmarks)

            # Subset of landmark points covering face oval, eyes, nose, mouth
            # Using all 468 points or a dense subset (e.g. 100+ key landmarks)
            # To preserve speed and avoid sliver triangles, we use hull + key internal feature landmarks
            feature_indices = list(dict.fromkeys(
                tgt_hull_indices +
                list(range(0, min(len(target_landmarks.points), len(source_landmarks.points)), 4))
            ))

            src_points = source_landmarks.get_pixel_coords(feature_indices)
            tgt_points = target_landmarks.get_pixel_coords(feature_indices)

            triangles = compute_delaunay_triangles(target_image.shape, tgt_points)

        # 3. Piecewise Affine Warping
        with timer.track("warping"):
            warped_face = warp_face_mesh(
                src_image=source_image,
                target_shape=target_image.shape,
                src_points=src_points,
                target_points=tgt_points,
                triangles=triangles,
            )

        # 4. Color and Lighting Transfer
        with timer.track("color_transfer"):
            mask = create_face_mask(target_image.shape, tgt_hull_pts)
            if self.enable_color_transfer:
                warped_face = match_color_reinhard(warped_face, target_image, mask=mask)

        # 5. Seamless Blending
        with timer.track("blending"):
            swapped = seamless_blend(
                warped_face=warped_face,
                target_image=target_image,
                mask=mask,
                mode=self.blend_mode,
            )

        return FaceSwapResult(
            swapped_image=swapped,
            warped_face=warped_face,
            face_mask=mask,
            delaunay_triangles_count=len(triangles),
            timing=timer.summary(),
            source_landmarks=source_landmarks,
            target_landmarks=target_landmarks,
        )
