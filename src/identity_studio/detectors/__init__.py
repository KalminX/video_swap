"""Detectors package exposing detection models and result data structures."""

from .base import BaseFaceDetector, BoundingBox, DetectionResult
from .haar_detector import HaarFaceDetector
from .yunet_detector import YuNetFaceDetector

__all__ = [
    "BaseFaceDetector",
    "BoundingBox",
    "DetectionResult",
    "HaarFaceDetector",
    "YuNetFaceDetector",
]
