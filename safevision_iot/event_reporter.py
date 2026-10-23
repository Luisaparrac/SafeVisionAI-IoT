"""Sends the findings to the backend without stopping the video analysis."""
import logging
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

from safevision_iot.backend_client import BackendClient, BackendError
from safevision_iot.detection_event import DetectionEvent
from safevision_iot.vision.rules import Finding


class EventReporter:
    """Turns each Finding into a DetectionEvent, saves a photo and sends it.

    Sending happens in a background thread fed by a FIFO queue: if the backend
    is slow (Render waking up), the analysis keeps running and the events are
    sent in order as soon as possible.
    """

    def __init__(self, client: BackendClient | None, camera_id: int, evidence_dir: str,
                 person_subject_id: int | None = None, pet_subject_id: int | None = None,
                 max_pending: int = 100):
        self._client = client
        self._camera_id = camera_id
        self._evidence_dir = Path(evidence_dir)
        self._person_subject_id = person_subject_id
        self._pet_subject_id = pet_subject_id
        self._queue: queue.Queue[DetectionEvent] = queue.Queue(maxsize=max_pending)
        self._log = logging.getLogger("safevision.events")
        self.sent: list[dict] = []
        if client is not None:
            threading.Thread(target=self._worker, name="event-sender", daemon=True).start()

    def report(self, finding: Finding, frame=None) -> DetectionEvent:
        evidence = self._save_evidence(finding, frame)
        event = self.to_event(finding, evidence)
        self._log.warning("EVENT %s | %s #%s | %.0f%% | %s", event.event_type, event.detected_class,
                          finding.track_id, (event.confidence or 0) * 100, event.description)
        if self._client is not None:
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                self._log.error("Too many events waiting; dropping %s", event.event_type)
        return event

    def to_event(self, finding: Finding, evidence: Path | None = None) -> DetectionEvent:
        subject_id = self._person_subject_id if finding.detected_class == "person" else self._pet_subject_id
        description = finding.description + (f". Evidencia: {evidence.name}" if evidence else "")
        return DetectionEvent(
            camera_id=self._camera_id,
            event_type=finding.event_type,
            subject_id=subject_id,
            zone_id=finding.zone_id,
            detected_class=finding.detected_class,
            confidence=round(min(max(finding.confidence, 0.0), 1.0), 3),
            description=description[:2000],
        )

    def wait_until_sent(self, timeout: float = 30) -> None:
        end = time.time() + timeout
        while self._queue.unfinished_tasks and time.time() < end:
            time.sleep(0.2)

    def _save_evidence(self, finding: Finding, frame) -> Path | None:
        if frame is None:
            return None
        self._evidence_dir.mkdir(parents=True, exist_ok=True)
        name = f"{datetime.now():%Y%m%d_%H%M%S}_{finding.event_type.replace(' ', '_')}.jpg"
        path = self._evidence_dir / name
        cv2.imwrite(str(path), frame)
        return path

    def _worker(self) -> None:
        while True:
            event = self._queue.get()
            try:
                self._send_with_retries(event)
            finally:
                self._queue.task_done()

    def _send_with_retries(self, event: DetectionEvent, attempts: int = 5) -> None:
        delay = 2.0
        for attempt in range(1, attempts + 1):
            try:
                created = self._client.send_event(event)
                self._log.info("Sent to backend: event #%s (%s)", created.get("id"), created.get("priority"))
                self.sent.append(created)
                return
            except BackendError as exc:
                if not exc.retryable or attempt == attempts:
                    self._log.error("Could not send %s: %s", event.event_type, exc)
                    return
                self._log.warning("Backend not available (attempt %d/%d), retrying in %.0f s",
                                  attempt, attempts, delay)
                time.sleep(delay)
                delay = min(delay * 2, 60)
