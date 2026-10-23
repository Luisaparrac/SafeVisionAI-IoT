"""Connects the camera frames with the detectors, the rules and the reporter."""
import logging
import threading
import time

import cv2

from safevision_iot.event_reporter import EventReporter
from safevision_iot.vision.analyzer import SceneAnalyzer
from safevision_iot.vision.detections import FrameDetections
from safevision_iot.vision.detectors import Detector
from safevision_iot.vision.rules import Finding


class FrameBuffer:
    """Holds only the newest frame. The camera writes, the AI reads.

    The AI is slower than the camera; keeping only the newest frame means it
    always analyzes what is happening now instead of falling behind.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._timestamp = 0.0

    def put(self, frame, timestamp: float | None = None) -> None:
        with self._lock:
            self._frame, self._timestamp = frame, timestamp if timestamp is not None else time.time()

    def latest(self):
        with self._lock:
            return self._frame, self._timestamp


class DetectionPipeline:
    """One analysis step: detectors -> analyzer -> reporter (and optional preview)."""

    def __init__(self, detectors: list[Detector], analyzer: SceneAnalyzer, reporter: EventReporter,
                 fall_detector: Detector | None = None, preview: "Preview | None" = None):
        self._detectors = detectors
        self._fall_detector = fall_detector
        self._analyzer = analyzer
        self._reporter = reporter
        self._preview = preview
        self._log = logging.getLogger("safevision.ai")

    @property
    def reporter(self) -> EventReporter:
        return self._reporter

    def close(self) -> None:
        if self._preview:
            self._preview.close()

    def process(self, frame, timestamp: float) -> list[Finding]:
        height, width = frame.shape[:2]
        items = [item for detector in self._detectors for item in detector.detect(frame)]
        fall_boxes = ([(d.box, d.confidence) for d in self._fall_detector.detect(frame)]
                      if self._fall_detector else None)
        findings = self._analyzer.process(FrameDetections(timestamp, width, height, items), fall_boxes)
        for finding in findings:
            self._reporter.report(finding, frame)
        if self._preview:
            self._preview.show(frame, self._analyzer.subjects, findings)
        return findings

    def run_live(self, buffer: FrameBuffer, fps: float, stop: threading.Event) -> None:
        """Analyzes the newest camera frame `fps` times per second until `stop` is set."""
        period = 1 / fps
        last_timestamp = 0.0
        while not stop.is_set():
            started = time.time()
            frame, timestamp = buffer.latest()
            if frame is not None and timestamp != last_timestamp:
                last_timestamp = timestamp
                self.process(frame, timestamp)
                if self._preview and self._preview.closed:
                    stop.set()
            stop.wait(max(0.0, period - (time.time() - started)))

    def run_video(self, path: str, fps: float) -> list[Finding]:
        """Analyzes a recorded video using its own timeline (for tests and demos)."""
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open video {path}")
        video_fps = capture.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, round(video_fps / fps))
        findings: list[Finding] = []
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step == 0:
                findings += self.process(frame, index / video_fps)
                if self._preview and self._preview.closed:
                    break
            index += 1
        capture.release()
        return findings


class Preview:
    """Window that draws each subject, its posture and the events (press q to close)."""

    COLORS = {"person": (60, 200, 60), "dog": (255, 160, 40), "cat": (255, 160, 40),
              "other_animal": (255, 160, 40)}

    def __init__(self, title: str = "SafeVision AI"):
        self._title = title
        self._alerts: list[tuple[str, float]] = []
        self.closed = False

    def show(self, frame, subjects, findings: list[Finding]) -> None:
        image = frame.copy()
        now = time.time()
        self._alerts = [(text, t) for text, t in self._alerts if now - t < 5]
        self._alerts += [(f"{f.event_type} ({f.detected_class})", now) for f in findings]
        for subject in subjects.values():
            obs = subject.last
            color = self.COLORS.get(subject.detected_class, (200, 200, 200))
            b = obs.box
            cv2.rectangle(image, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 2)
            label = f"#{subject.track_id} {subject.detected_class}"
            if obs.posture:
                label += f" {obs.posture.value}"
            cv2.putText(image, label, (int(b.x1), max(int(b.y1) - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            if obs.keypoints is not None:
                for (x, y), c in zip(obs.keypoints.astype(int), obs.keypoint_conf):
                    if c > 0.3:
                        cv2.circle(image, (int(x), int(y)), 3, (0, 140, 255), -1)
        for i, (text, _) in enumerate(self._alerts):
            cv2.putText(image, f"ALERTA: {text}", (12, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 255), 2)
        cv2.imshow(self._title, image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.closed = True

    def close(self) -> None:
        cv2.destroyAllWindows()
