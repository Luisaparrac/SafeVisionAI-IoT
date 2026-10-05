"""Event that the IoT module reports to the backend."""
from dataclasses import asdict, dataclass

DETECTED_CLASSES = ("person", "dog", "cat", "other_animal")


@dataclass
class DetectionEvent:
    camera_id: int
    event_type: str                  # name in event_types, e.g. "Fall"
    subject_id: int | None = None
    zone_id: int | None = None
    detected_class: str = "person"
    confidence: float | None = None  # 0..1
    evidence_url: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        if self.detected_class not in DETECTED_CLASSES:
            raise ValueError(f"detected_class must be one of {DETECTED_CLASSES}")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")

    def to_payload(self) -> dict:
        """Body for POST /api/eventos, without empty fields."""
        return {key: value for key, value in asdict(self).items() if value is not None}
