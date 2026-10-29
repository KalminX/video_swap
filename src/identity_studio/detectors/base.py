"""Base interface and data models for face detectors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


@dataclass
class DetectionResult:
    bbox: BoundingBox
    confidence: float
    landmarks: Optional[List[Tuple[float, float]]] = None


class BaseFaceDetector(ABC):
    """Abstract base class for all face detection implementations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name or identifier of the detector model."""
        pass

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect faces in a BGR image array.
        
        Args:
            image: BGR numpy image array.
            
        Returns:
            List of DetectionResult instances.
        """
        pass
