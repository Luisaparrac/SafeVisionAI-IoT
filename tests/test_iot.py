"""Unit tests with fake camera and fake backend (no hardware or network needed)."""
import pytest

from safevision_iot.backend_client import BackendError
from safevision_iot.camera_monitor import ACTIVE, DISCONNECTED, CameraMonitor
from safevision_iot.detection_event import DetectionEvent


class FakeCamera:
    def __init__(self, frames):
        self.frames = list(frames)   # None = stream drops
        self.connected = False
        self.safe_url = "fake"

    def connect(self):
        self.connected = True
        return True

    def is_connected(self):
        return self.connected

    def read_frame(self):
        return self.frames.pop(0) if self.frames else None

    def release(self):
        self.connected = False


class FakeClient:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def update_camera_status(self, camera_id, status):
        if self.fail:
            raise BackendError("down")
        self.calls.append((camera_id, status))


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def make_monitor(frames, client=None, clock=None):
    return CameraMonitor(FakeCamera(frames), client or FakeClient(), camera_id=1,
                         heartbeat_seconds=30, clock=clock or FakeClock(), sleep=lambda s: None)


def test_reports_active_once_when_frames_arrive():
    client = FakeClient()
    monitor = make_monitor(["f"] * 5, client)
    for _ in range(5):
        monitor.step()
    assert client.calls == [(1, ACTIVE)]
    assert monitor.frames_read == 5


def test_reports_disconnected_when_stream_drops():
    client = FakeClient()
    monitor = make_monitor(["f", "f", None], client)
    for _ in range(3):
        monitor.step()
    assert client.calls == [(1, ACTIVE), (1, DISCONNECTED)]


def test_sends_heartbeat_while_active():
    client, clock = FakeClient(), FakeClock()
    monitor = make_monitor(["f"] * 3, client, clock)
    monitor.step()
    clock.now = 31
    monitor.step()
    clock.now = 40
    monitor.step()
    assert client.calls == [(1, ACTIVE), (1, ACTIVE)]


def test_backend_failure_waits_before_retrying():
    client, clock = FakeClient(fail=True), FakeClock()
    monitor = make_monitor(["f"] * 4, client, clock)
    attempts = []
    original = client.update_camera_status
    client.update_camera_status = lambda c, s: (attempts.append(clock.now), original(c, s))
    monitor.step()            # fails at t=0
    clock.now = 5
    monitor.step()            # too soon: no new attempt
    assert attempts == [0.0]
    client.fail = False
    clock.now = 11
    monitor.step()            # retry after 10 s succeeds
    assert client.calls == [(1, ACTIVE)]
    assert monitor.status == ACTIVE


def test_event_payload_skips_empty_fields():
    event = DetectionEvent(camera_id=1, event_type="Fall", confidence=0.9)
    assert event.to_payload() == {"camera_id": 1, "event_type": "Fall",
                                  "detected_class": "person", "confidence": 0.9}


def test_event_rejects_invalid_values():
    with pytest.raises(ValueError):
        DetectionEvent(camera_id=1, event_type="Fall", detected_class="robot")
    with pytest.raises(ValueError):
        DetectionEvent(camera_id=1, event_type="Fall", confidence=1.5)