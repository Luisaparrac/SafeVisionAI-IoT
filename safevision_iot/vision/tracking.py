"""Follows each person or animal across frames (short history per subject)."""
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from safevision_iot.vision.detections import Box, Detection, PersonDetection
from safevision_iot.vision.posture import Posture, PostureClassifier


@dataclass
class Observation:
    """How a subject looked in one frame."""
    timestamp: float
    box: Box
    box_height: float
    center: tuple[float, float]           # pixels
    floor_point: tuple[float, float]      # normalized 0..1 (bottom center of the box)
    posture: Posture | None               # only for people
    hip_y: float | None                   # only for people
    keypoints: np.ndarray | None
    keypoint_conf: np.ndarray | None
    confidence: float


@dataclass
class TrackedSubject:
    """One person or animal seen over time. The history is a bounded queue."""
    track_id: int
    detected_class: str
    history: deque = field(default_factory=lambda: deque(maxlen=300))
    last_seen: float = 0.0
    last_motion: float = 0.0

    @property
    def is_person(self) -> bool:
        return self.detected_class == "person"

    @property
    def last(self) -> Observation:
        return self.history[-1]

    def previous(self) -> Observation | None:
        return self.history[-2] if len(self.history) > 1 else None

    def observation_before(self, timestamp: float, seconds: float) -> Observation | None:
        """Most recent observation at least `seconds` older than `timestamp`."""
        for obs in reversed(self.history):
            if timestamp - obs.timestamp >= seconds:
                return obs
        return None

    def recent(self, timestamp: float, seconds: float):
        """Observations of the last `seconds`, newest first."""
        for obs in reversed(self.history):
            if timestamp - obs.timestamp > seconds:
                return
            yield obs


class TrackRegistry:
    """Keeps one TrackedSubject per id.

    When a person falls, the tracker often loses them (their shape changes
    suddenly) and gives them a new id. The registry links that new id to the
    nearest subject of the same class seen a moment ago, so the history that
    shows the fall is not lost.
    """

    def __init__(self, posture: PostureClassifier, motion_threshold: float = 0.02,
                 max_gap: float = 3.0, forget_after: float = 10.0):
        self._posture = posture
        self._motion_threshold = motion_threshold
        self._max_gap = max_gap
        self._forget_after = forget_after
        self.subjects: dict[int, TrackedSubject] = {}
        self._aliases: dict[int, int] = {}
        self._next_id = 1_000_000

    def update(self, detections: list[Detection], timestamp: float,
               frame_width: int, frame_height: int) -> list[TrackedSubject]:
        updated: list[TrackedSubject] = []
        used: set[int] = set()
        for detection in detections:
            subject = self._subject_for(detection, timestamp, used)
            used.add(subject.track_id)
            self._record(subject, detection, timestamp, frame_width, frame_height)
            updated.append(subject)
        self._forget(timestamp)
        return updated

    # ---------- identity ----------
    def _subject_for(self, detection: Detection, timestamp: float, used: set[int]) -> TrackedSubject:
        tid = detection.track_id
        if tid is not None and self._aliases.get(tid) in self.subjects:
            return self.subjects[self._aliases[tid]]
        if tid is not None and tid in self.subjects:
            return self.subjects[tid]
        nearest = self._nearest(detection, timestamp, used)
        if nearest is None:
            key = tid if tid is not None else self._new_id()
            nearest = self.subjects[key] = TrackedSubject(key, detection.detected_class,
                                                          last_motion=timestamp)
        if tid is not None:
            self._aliases[tid] = nearest.track_id
        return nearest

    def _nearest(self, detection: Detection, timestamp: float, used: set[int]) -> TrackedSubject | None:
        cx, cy = detection.box.center
        best, best_distance = None, 1.5
        for subject in self.subjects.values():
            if (subject.track_id in used or subject.detected_class != detection.detected_class
                    or timestamp - subject.last_seen > self._max_gap):
                continue
            sx, sy = subject.last.center
            distance = np.hypot(cx - sx, cy - sy) / max(subject.last.box_height, 1)
            if distance < best_distance:
                best, best_distance = subject, distance
        return best

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    # ---------- history ----------
    def _record(self, subject: TrackedSubject, detection: Detection, timestamp: float,
                frame_width: int, frame_height: int) -> None:
        fx, fy = detection.box.bottom_center
        person = detection if isinstance(detection, PersonDetection) else None
        observation = Observation(
            timestamp=timestamp,
            box=detection.box,
            box_height=detection.box.height,
            center=detection.box.center,
            floor_point=(fx / frame_width, fy / frame_height),
            posture=self._posture.classify(person) if person else None,
            hip_y=self._posture.hip_height(person) if person else None,
            keypoints=None if person is None or person.keypoints is None else person.keypoints.copy(),
            keypoint_conf=None if person is None or person.keypoint_conf is None else person.keypoint_conf.copy(),
            confidence=detection.confidence,
        )
        previous = subject.last if subject.history else None
        if previous is None or self._movement(previous, observation) > self._motion_threshold:
            subject.last_motion = timestamp
        subject.history.append(observation)
        subject.last_seen = timestamp

    @staticmethod
    def _movement(a: Observation, b: Observation) -> float:
        """How much the subject moved between two frames, relative to its size."""
        if a.keypoints is not None and b.keypoints is not None:
            mask = (a.keypoint_conf > 0.3) & (b.keypoint_conf > 0.3)
            if mask.sum() >= 4:
                distance = np.linalg.norm(a.keypoints[mask] - b.keypoints[mask], axis=1).mean()
                return float(distance / max(b.box_height, 1))
        distance = np.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1])
        return float(distance / max(b.box_height, 1))

    def _forget(self, timestamp: float) -> None:
        stale = [tid for tid, s in self.subjects.items() if timestamp - s.last_seen > self._forget_after]
        for tid in stale:
            del self.subjects[tid]
        self._aliases = {k: v for k, v in self._aliases.items() if v in self.subjects}
