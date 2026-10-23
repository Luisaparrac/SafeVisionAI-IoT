"""Rules that turn what the camera sees into risk events.

Every rule implements the same interface (`Rule.evaluate`), so the analyzer
can apply all of them the same way (polymorphism). Each rule keeps its own
state per subject.
"""
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from safevision_iot.vision.detections import Box
from safevision_iot.vision.posture import Posture
from safevision_iot.vision.tracking import TrackedSubject
from safevision_iot.vision.zone import Zone

# Names exactly as stored in the event_types table
FALL = "Caída"
IMMOBILITY = "Inmovilidad prolongada"
RESTRICTED_ZONE = "Permanencia en zona restringida"


@dataclass
class Finding:
    """A risk event found by a rule, before it is sent to the backend."""
    event_type: str
    track_id: int
    detected_class: str
    confidence: float
    description: str
    zone_id: int | None = None


class Rule(ABC):
    @abstractmethod
    def evaluate(self, subject: TrackedSubject, now: float) -> Finding | None: ...

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
    """

    def __init__(self, zones: Callable[[], list[Zone]], dwell: float = 2.0):
        self._zones = zones
        self._dwell = dwell
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
            if now - since >= self._dwell and key not in self._reported:
                self._reported.add(key)
                return Finding(RESTRICTED_ZONE, subject.track_id, subject.detected_class,
                               subject.last.confidence,
                               f"En la zona '{zone.name}' durante {now - since:.0f} s",
                               zone_id=zone.zone_id)
        return None

    def forget(self, track_id: int) -> None:
        for key in [k for k in self._entered if k[0] == track_id]:
            del self._entered[key]
        self._reported = {k for k in self._reported if k[0] != track_id}


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


def _iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0