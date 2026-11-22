"""Rules that turn what the camera sees into risk events.

Every rule implements the same interface (`Rule.evaluate`), so the analyzer
can apply all of them the same way (polymorphism). Each rule keeps its own
state per subject.
"""
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from safevision_iot.vision.detections import Box, FrameDetections, FurnitureDetection
from safevision_iot.vision.posture import Posture
from safevision_iot.vision.tracking import Observation, TrackedSubject
from safevision_iot.vision.zone import Zone

# Names exactly as stored in the event_types table
FALL = "Caída"
IMMOBILITY = "Inmovilidad prolongada"
RESTRICTED_ZONE = "Permanencia en zona restringida"
ABNORMAL_MOVEMENT = "Movimiento anormal"
COLLAPSE = "Colapso"
PET_ON_FURNITURE = "Mascota sobre mueble"
CHILD_ON_FURNITURE = "Niño subido a mueble"
CHILD_RISK_ZONE = "Niño en zona de riesgo"

FURNITURE_NAMES = {"chair": "silla", "couch": "sofá", "bed": "cama", "dining table": "mesa"}
PET_NAMES = {"dog": "perro", "cat": "gato"}


@dataclass
class Finding:
    """A risk event found by a rule, before it is sent to the backend."""
    event_type: str
    track_id: int
    detected_class: str
    confidence: float
    description: str
    zone_id: int | None = None
    is_child: bool = False


class Rule(ABC):
    @abstractmethod
    def evaluate(self, subject: TrackedSubject, now: float) -> Finding | None: ...

    def observe_frame(self, frame: FrameDetections) -> None:
        """Called once per frame, before evaluating the subjects."""

    def forget(self, track_id: int) -> None:
        """Called when a subject leaves the scene."""


class FallRule(Rule):
    """Person goes from standing to lying quickly, the hips drop, and stays down.

    1. Lying now, and standing less than `window` seconds ago.
    2. The hips went down more than `drop_ratio` of the standing body height.
    3. Still lying `confirm` seconds later -> fall.
    If the person stands up before the confirmation, it was not a fall.
    """

    def __init__(self, window: float = 1.5, drop_ratio: float = 0.3, confirm: float = 2.0):
        self._window = window
        self._drop_ratio = drop_ratio
        self._confirm = confirm
        self._started: dict[int, float] = {}    # track_id -> moment the fall began
        self._reported: set[int] = set()

    def evaluate(self, subject: TrackedSubject, now: float) -> Finding | None:
        if not subject.is_person:
            return None
        tid, current = subject.track_id, subject.last
        if current.posture == Posture.STANDING:
            self._started.pop(tid, None)
            self._reported.discard(tid)
            return None
        if current.posture != Posture.LYING or tid in self._reported:
            return None
        if tid not in self._started:
            # Compare with the moment the hips were highest while standing: during
            # a fall the last "standing" frames are already halfway down.
            standing = [o for o in subject.recent(now, self._window) if o.posture == Posture.STANDING]
            if standing:
                upright = min(standing, key=lambda o: o.hip_y)
                if current.hip_y - upright.hip_y > self._drop_ratio * upright.box_height:
                    self._started[tid] = now
            return None
        lying_for = now - self._started[tid]
        if lying_for < self._confirm:
            return None
        self._reported.add(tid)
        return Finding(FALL, tid, "person", current.confidence,
                       f"Caída: pasó de estar de pie al suelo y sigue ahí {lying_for:.0f} s")

    def forget(self, track_id: int) -> None:
        self._started.pop(track_id, None)
        self._reported.discard(track_id)


class ImmobilityRule(Rule):
    """The subject barely moves for longer than the limit of its class.

    People and pets have different limits: a pet sleeping one hour is normal.
    """

    def __init__(self, person_seconds: float = 60, pet_seconds: float = 1800):
        self._limits = {"person": person_seconds}
        self._pet_seconds = pet_seconds

    def evaluate(self, subject: TrackedSubject, now: float) -> Finding | None:
        limit = self._limits.get(subject.detected_class, self._pet_seconds)
        still = now - subject.last_motion
        if still < limit:
            return None
        posture = f", {subject.last.posture.label}" if subject.last.posture else ""
        return Finding(IMMOBILITY, subject.track_id, subject.detected_class, subject.last.confidence,
                       f"Sin movimiento significativo durante {still:.0f} s{posture}")


class RestrictedZoneRule(Rule):
    """The subject stays inside a restricted zone longer than `dwell` seconds.

    The position used is the bottom center of the box (where the feet or paws
    touch the floor), relative to the image size, like the zone coordinates.
    A child generates a different event, so the rule waits up to `age_wait`
    extra seconds for the age votes before deciding which event to report.
    """

    def __init__(self, zones: Callable[[], list[Zone]], dwell: float = 2.0, age_wait: float = 3.0):
        self._zones = zones
        self._dwell = dwell
        self._age_wait = age_wait
        self._entered: dict[tuple[int, int], float] = {}   # (track_id, zone_id) -> since
        self._reported: set[tuple[int, int]] = set()

    def evaluate(self, subject: TrackedSubject, now: float) -> Finding | None:
        x, y = subject.last.floor_point
        for zone in self._zones():
            key = (subject.track_id, zone.zone_id)
            if not zone.contains(x, y):
                self._entered.pop(key, None)
                self._reported.discard(key)
                continue
            since = self._entered.setdefault(key, now)
            if subject.age_pending and now - since < self._dwell + self._age_wait:
                continue
            if now - since >= self._dwell and key not in self._reported:
                self._reported.add(key)
                event = CHILD_RISK_ZONE if subject.is_child else RESTRICTED_ZONE
                who = f"{subject.age_label}: " if subject.age_label else ""
                return Finding(event, subject.track_id, subject.detected_class,
                               subject.last.confidence,
                               f"{who}en la zona '{zone.name}' durante {now - since:.0f} s",
                               zone_id=zone.zone_id)
        return None

    def forget(self, track_id: int) -> None:
        for key in [k for k in self._entered if k[0] == track_id]:
            del self._entered[key]
        self._reported = {k for k in self._reported if k[0] != track_id}


class FurnitureRule(Rule):
    """Template: the subject stays on a piece of furniture for `seconds`."""

    def __init__(self, seconds: float):
        self._seconds = seconds
        self._furniture: list[FurnitureDetection] = []
        self._since: dict[int, float] = {}
        self._reported: set[int] = set()

    def observe_frame(self, frame: FrameDetections) -> None:
        self._furniture = frame.furniture

    def evaluate(self, subject: TrackedSubject, now: float) -> Finding | None:
        tid = subject.track_id
        item = None
        if self._applies(subject):
            item = next((f for f in self._furniture if self._is_on(subject, f)), None)
        if item is None:
            self._since.pop(tid, None)
            self._reported.discard(tid)
            return None
        since = self._since.setdefault(tid, now)
        if now - since < self._seconds or tid in self._reported:
            return None
        self._reported.add(tid)
        return self._finding(subject, item, now - since)

    def forget(self, track_id: int) -> None:
        self._since.pop(track_id, None)
        self._reported.discard(track_id)

    @abstractmethod
    def _applies(self, subject: TrackedSubject) -> bool: ...

    @abstractmethod
    def _is_on(self, subject: TrackedSubject, item: FurnitureDetection) -> bool: ...

    @abstractmethod
    def _finding(self, subject: TrackedSubject, item: FurnitureDetection, seconds: float) -> Finding: ...


class PetOnFurnitureRule(FurnitureRule):
    """A dog or cat overlaps a couch, bed or table and its paws are above the floor line."""

    FURNITURE = ("couch", "bed", "dining table")
    MIN_OVERLAP = 0.5
    HEIGHT_LIMIT = 0.8      # fraction of the furniture height, from its top

    def __init__(self, seconds: float = 3.0):
        super().__init__(seconds)

    def _applies(self, subject: TrackedSubject) -> bool:
        return subject.detected_class in PET_NAMES

    def _is_on(self, subject: TrackedSubject, item: FurnitureDetection) -> bool:
        if item.detected_class not in self.FURNITURE:
            return False
        box = subject.last.box
        limit = item.box.y1 + self.HEIGHT_LIMIT * item.box.height
        return _overlap_ratio(box, item.box) >= self.MIN_OVERLAP and box.y2 <= limit

    def _finding(self, subject: TrackedSubject, item: FurnitureDetection, seconds: float) -> Finding:
        pet, furniture = PET_NAMES[subject.detected_class], FURNITURE_NAMES[item.detected_class]
        return Finding(PET_ON_FURNITURE, subject.track_id, subject.detected_class, subject.last.confidence,
                       f"Mascota sobre mueble: {pet} sobre {furniture} durante {seconds:.0f} s")


class ChildOnFurnitureRule(FurnitureRule):
    """A child's feet are on a piece of furniture. Sitting on a chair is normal."""

    HEIGHT_LIMIT = 0.75     # fraction of the furniture height, from its top

    def __init__(self, seconds: float = 2.0):
        super().__init__(seconds)

    def _applies(self, subject: TrackedSubject) -> bool:
        return subject.is_person and subject.is_child

    def _is_on(self, subject: TrackedSubject, item: FurnitureDetection) -> bool:
        if item.detected_class == "chair" and subject.last.posture == Posture.SITTING:
            return False
        x, y = subject.last.box.bottom_center
        limit = item.box.y1 + self.HEIGHT_LIMIT * item.box.height
        return item.box.x1 <= x <= item.box.x2 and item.box.y1 <= y <= limit

    def _finding(self, subject: TrackedSubject, item: FurnitureDetection, seconds: float) -> Finding:
        furniture = FURNITURE_NAMES[item.detected_class]
        return Finding(CHILD_ON_FURNITURE, subject.track_id, "person", subject.last.confidence,
                       f"Niño subido a mueble: {subject.age_label} sobre {furniture} durante {seconds:.0f} s")


class AbnormalMovementRule(Rule):
    """Repeated sudden arm movements, such as throwing objects or hitting.

    Wrist speed is measured in body heights per second, so it does not depend
    on the distance to the camera. To avoid confusing a fall with this event,
    a movement does not count if the person is lying, if the hips are dropping
    fast, or if the person ends up lying within `settle_seconds`. That is why
    each movement is only counted once `settle_seconds` have passed.
    """

    WRISTS = (9, 10)        # COCO indices
    MIN_KEYPOINT_CONF = 0.3

    def __init__(self, speed_threshold: float = 0.8, min_peaks: int = 5, window_seconds: float = 3.0,
                 max_hip_speed: float = 0.5, settle_seconds: float = 1.0):
        self._speed_threshold = speed_threshold
        self._min_peaks = min_peaks
        self._window = window_seconds
        self._max_hip_speed = max_hip_speed
        self._settle = settle_seconds

    def evaluate(self, subject: TrackedSubject, now: float) -> Finding | None:
        if not subject.is_person or subject.last.keypoints is None:
            return None
        settled_until = now - self._settle
        observations = list(subject.recent(now, self._window + self._settle))    # newest first
        peaks = 0
        next_lying = None
        for newer, older in zip(observations, observations[1:]):
            if newer.posture == Posture.LYING:
                next_lying = newer.timestamp
            if newer.timestamp > settled_until or older.timestamp < settled_until - self._window:
                continue
            ends_lying = next_lying is not None and next_lying - newer.timestamp <= self._settle
            if not ends_lying and self._is_peak(older, newer):
                peaks += 1
        if peaks < self._min_peaks:
            return None
        return Finding(ABNORMAL_MOVEMENT, subject.track_id, "person",
                       min(1.0, peaks / (2 * self._min_peaks)),
                       f"Movimiento anormal: {peaks} movimientos bruscos de brazos en {self._window:g} s "
                       "(posible lanzamiento de objetos o agresión).")

    def _is_peak(self, older: Observation, newer: Observation) -> bool:
        if Posture.LYING in (older.posture, newer.posture):
            return False
        if self._hip_speed(older, newer) > self._max_hip_speed:
            return False
        return self._wrist_speed(older, newer) > self._speed_threshold

    @staticmethod
    def _hip_speed(older: Observation, newer: Observation) -> float:
        """Vertical hip speed between two frames, in body heights per second."""
        dt = newer.timestamp - older.timestamp
        if dt <= 0 or newer.box_height <= 0 or older.hip_y is None or newer.hip_y is None:
            return 0.0
        return abs(newer.hip_y - older.hip_y) / newer.box_height / dt

    def _wrist_speed(self, older: Observation, newer: Observation) -> float:
        """Fastest visible wrist between two frames, in body heights per second."""
        dt = newer.timestamp - older.timestamp
        if dt <= 0 or newer.box_height <= 0:
            return 0.0
        if older.keypoints is None or newer.keypoints is None:
            return 0.0
        if older.keypoint_conf is None or newer.keypoint_conf is None:
            return 0.0
        speeds = [float(np.linalg.norm(newer.keypoints[w] - older.keypoints[w])) / newer.box_height / dt
                  for w in self.WRISTS
                  if older.keypoint_conf[w] > self.MIN_KEYPOINT_CONF
                  and newer.keypoint_conf[w] > self.MIN_KEYPOINT_CONF]
        return max(speeds, default=0.0)


class TrainedFallRule(Rule):
    """Uses the team's trained model: a 'fall' box on a person for `confirm` seconds."""

    def __init__(self, confirm: float = 2.0, min_overlap: float = 0.3):
        self._confirm = confirm
        self._min_overlap = min_overlap
        self._fall_boxes: list[tuple[Box, float]] = []
        self._since: dict[int, float] = {}
        self._reported: set[int] = set()

    def observe(self, boxes: list[tuple[Box, float]]) -> None:
        """Fall boxes found by the trained model in the current frame (box, confidence)."""
        self._fall_boxes = boxes

    def evaluate(self, subject: TrackedSubject, now: float) -> Finding | None:
        if not subject.is_person:
            return None
        tid = subject.track_id
        match = self._best_match(subject)
        if match is None:
            self._since.pop(tid, None)
            if subject.last.posture == Posture.STANDING:
                self._reported.discard(tid)
            return None
        since = self._since.setdefault(tid, now)
        if now - since < self._confirm or tid in self._reported:
            return None
        self._reported.add(tid)
        return Finding(FALL, tid, "person", match,
                       f"Caída detectada por el modelo entrenado ({now - since:.0f} s en el suelo)")

    def forget(self, track_id: int) -> None:
        self._since.pop(track_id, None)
        self._reported.discard(track_id)

    def _best_match(self, subject: TrackedSubject) -> float | None:
        person = subject.last.box
        best = None
        for box, confidence in self._fall_boxes:
            if _iou(person, box) >= self._min_overlap and (best is None or confidence > best):
                best = confidence
        return best


def _overlap_ratio(box: Box, other: Box) -> float:
    """Fraction of `box` that is inside `other`."""
    ix = max(0.0, min(box.x2, other.x2) - max(box.x1, other.x1))
    iy = max(0.0, min(box.y2, other.y2) - max(box.y1, other.y1))
    return ix * iy / (box.width * box.height)


def _iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0