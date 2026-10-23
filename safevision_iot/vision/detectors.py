"""YOLO models: people with body keypoints, and pets."""
from abc import ABC, abstractmethod

import numpy as np

from safevision_iot.vision.detections import Box, Detection, PersonDetection

# COCO classes that count as pets
ANIMAL_CLASSES = {14: "other_animal", 15: "cat", 16: "dog"}  # 14 = bird


class Detector(ABC):
    """Common interface: receives a frame, returns what it found."""

    @abstractmethod
    def detect(self, frame) -> list[Detection]: ...


class PoseDetector(Detector):
    """YOLO Pose: finds people and their 17 body keypoints, with tracking."""

    def __init__(self, model_path: str, min_confidence: float = 0.3):
        from ultralytics import YOLO  # heavy import, only when the AI is used

        self._model = YOLO(model_path)
        self._min_confidence = min_confidence

    def detect(self, frame) -> list[Detection]:
        result = self._model.track(frame, persist=True, classes=[0], conf=self._min_confidence,
                                   tracker="bytetrack.yaml", verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        ids = _track_ids(result, len(boxes))
        keypoints = result.keypoints.xy.cpu().numpy()
        kp_conf = (result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None
                   else np.ones(keypoints.shape[:2]))
        return [
            PersonDetection(ids[i], "person", Box(*boxes[i]), float(confs[i]), keypoints[i], kp_conf[i])
            for i in range(len(boxes))
        ]


class AnimalDetector(Detector):
    """YOLO (normal detection model): finds dogs, cats and birds, with tracking."""

    def __init__(self, model_path: str, min_confidence: float = 0.4):
        from ultralytics import YOLO

        self._model = YOLO(model_path)
        self._min_confidence = min_confidence

    def detect(self, frame) -> list[Detection]:
        result = self._model.track(frame, persist=True, classes=list(ANIMAL_CLASSES),
                                   conf=self._min_confidence, tracker="bytetrack.yaml",
                                   verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.int().cpu().numpy()
        ids = _track_ids(result, len(boxes))
        # Animal ids are offset so they never collide with people ids
        return [
            Detection(None if ids[i] is None else 100_000 + ids[i], ANIMAL_CLASSES[int(classes[i])],
                      Box(*boxes[i]), float(confs[i]))
            for i in range(len(boxes))
        ]


class TrainedFallDetector(Detector):
    """Model trained by the team (see training/): boxes labeled 'fall' or 'person'."""

    def __init__(self, model_path: str, min_confidence: float = 0.5, fall_label: str = "fall"):
        from ultralytics import YOLO

        self._model = YOLO(model_path)
        self._min_confidence = min_confidence
        self._fall_class_ids = fall_class_ids(self._model.names, fall_label)
        if not self._fall_class_ids:
            raise ValueError(f"The model has no '{fall_label}' class: {self._model.names}")

    def detect(self, frame) -> list[Detection]:
        result = self._model(frame, classes=self._fall_class_ids, conf=self._min_confidence,
                             verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        return [Detection(None, "person", Box(*boxes[i]), float(confs[i])) for i in range(len(boxes))]


def fall_class_ids(names: dict[int, str], fall_label: str = "fall") -> list[int]:
    """Classes of a trained model that mean "fallen person".

    Public datasets name them differently ("fall", "Fall-Detected", "fallen"),
    so any class containing the label counts, except negative ones ("no_fall").
    """
    label = fall_label.lower()
    return [i for i, name in names.items()
            if label in name.lower() and not name.lower().startswith(("no", "not"))]


def _track_ids(result, count: int) -> list[int | None]:
    if result.boxes.id is None:
        return [None] * count
    return [int(i) for i in result.boxes.id.int().cpu().numpy()]
