"""Applies every rule to every subject and avoids repeating the same alert."""
from safevision_iot.vision.detections import Box, FrameDetections
from safevision_iot.vision.rules import Finding, Rule, TrainedFallRule
from safevision_iot.vision.tracking import TrackRegistry


class Cooldown:
    """Remembers when each (subject, event type) was last reported."""

    def __init__(self, seconds: float):
        self._seconds = seconds
        self._last: dict[tuple[int, str], float] = {}

    def allow(self, track_id: int, event_type: str, now: float) -> bool:
        key = (track_id, event_type)
        if now - self._last.get(key, float("-inf")) < self._seconds:
            return False
        self._last[key] = now
        return True


class SceneAnalyzer:
    """Receives the detections of each frame and returns the risk findings."""

    def __init__(self, registry: TrackRegistry, rules: list[Rule], cooldown: Cooldown):
        self._registry = registry
        self._rules = rules
        self._cooldown = cooldown

    @property
    def subjects(self):
        return self._registry.subjects

    def process(self, frame: FrameDetections, fall_boxes: list[tuple[Box, float]] | None = None) -> list[Finding]:
        before = set(self._registry.subjects)
        for rule in self._rules:
            if isinstance(rule, TrainedFallRule):
                rule.observe(fall_boxes or [])
        subjects = self._registry.update(frame.items, frame.timestamp, frame.frame_width, frame.frame_height)
        for gone in before - set(self._registry.subjects):
            for rule in self._rules:
                rule.forget(gone)

        findings: list[Finding] = []
        for subject in subjects:
            for rule in self._rules:
                finding = rule.evaluate(subject, frame.timestamp)
                if finding and self._cooldown.allow(finding.track_id, finding.event_type, frame.timestamp):
                    findings.append(finding)
        return findings
