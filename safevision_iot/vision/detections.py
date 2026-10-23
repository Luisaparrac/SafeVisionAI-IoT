"""What the detectors see in one frame."""
from dataclasses import dataclass

import numpy as np

# COCO keypoint indexes used by the rules
LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP = 5, 6, 11, 12


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(self.x2 - self.x1, 1.0)

    @property
    def height(self) -> float:
        return max(self.y2 - self.y1, 1.0)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Where the person or animal touches the floor."""
        return (self.x1 + self.x2) / 2, self.y2


@dataclass
class Detection:
    """Something the model found: a person or an animal."""
    track_id: int | None          # None while the tracker has not assigned an id
    detected_class: str           # person, dog, cat, other_animal
    box: Box
    confidence: float


@dataclass
class PersonDetection(Detection):
    keypoints: np.ndarray | None = None   # (17, 2) pixels, COCO order
    keypoint_conf: np.ndarray | None = None

    def has_keypoints(self, *indexes: int, min_conf: float = 0.3) -> bool:
        if self.keypoints is None or self.keypoint_conf is None:
            return False
        return bool(min(self.keypoint_conf[list(indexes)]) >= min_conf)


@dataclass
class FrameDetections:
    """Everything detected in one frame, plus the frame size."""
    timestamp: float
    frame_width: int
    frame_height: int
    items: list[Detection]
