"""Orchestration pipeline for image loading, face detection, annotation, and saving."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
import numpy as np

from ..core.image_ops import load_image, save_image, get_image_metadata, ImageMetadata
from ..core.timer import ExecutionTimer, TimingReport
from ..detectors.base import BaseFaceDetector, DetectionResult
from .face_annotator import FaceAnnotator


@dataclass
class PipelineResult:
    input_path: Optional[str]
    output_path: Optional[str]
    success: bool
    num_faces_detected: int
    detections: List[DetectionResult]
    metadata: Optional[ImageMetadata]
    timing: TimingReport
    error_message: Optional[str] = None


class FaceProcessingPipeline:
    """End-to-end face processing and benchmarking pipeline."""

    def __init__(
        self,
        detector: BaseFaceDetector,
        annotator: Optional[FaceAnnotator] = None,
    ) -> None:
        self.detector = detector
        self.annotator = annotator or FaceAnnotator()

    def process_image(
        self,
        image_input: Union[str, Path, np.ndarray],
        output_path: Optional[Union[str, Path]] = None,
        save_even_if_no_faces: bool = True,
    ) -> PipelineResult:
        """Process an image from file path or numpy array.
        
        Args:
            image_input: File path or already-loaded BGR numpy array.
            output_path: Optional destination path to save annotated image.
            save_even_if_no_faces: Whether to write annotated output when 0 faces found.
            
        Returns:
            PipelineResult containing timings, detections, and status.
        """
        timer = ExecutionTimer()
        input_str = str(image_input) if isinstance(image_input, (str, Path)) else "<in-memory array>"
        saved_path_str: Optional[str] = None

        try:
            # 1. Load image
            if isinstance(image_input, (str, Path)):
                with timer.track("load_image"):
                    image = load_image(image_input)
            else:
                image = image_input

            metadata = get_image_metadata(image)

            # 2. Detect faces
            with timer.track("face_detection"):
                detections = self.detector.detect(image)

            # 3. Annotate
            with timer.track("annotation"):
                # Total latency so far for the HUD
                elapsed_so_far = timer.summary().total_ms
                annotated = self.annotator.annotate(
                    image=image,
                    detections=detections,
                    detector_name=self.detector.name,
                    processing_time_ms=elapsed_so_far,
                )

            # 4. Save output
            if output_path is not None:
                if len(detections) > 0 or save_even_if_no_faces:
                    with timer.track("save_image"):
                        out_path = save_image(annotated, output_path)
                        saved_path_str = str(out_path)

            return PipelineResult(
                input_path=input_str,
                output_path=saved_path_str,
                success=True,
                num_faces_detected=len(detections),
                detections=detections,
                metadata=metadata,
                timing=timer.summary(),
                error_message=None,
            )

        except Exception as exc:
            return PipelineResult(
                input_path=input_str,
                output_path=None,
                success=False,
                num_faces_detected=0,
                detections=[],
                metadata=None,
                timing=timer.summary(),
                error_message=str(exc),
            )
