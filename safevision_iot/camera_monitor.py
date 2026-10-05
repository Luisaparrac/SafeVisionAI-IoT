"""Keeps the camera connected and reports its status to the backend."""
import logging
import time
 
from safevision_iot.backend_client import BackendClient, BackendError
from safevision_iot.camera import Camera
 
ACTIVE = "active"
DISCONNECTED = "disconnected"
 
 
class CameraMonitor:
    """Reads the camera continuously; when it connects or drops, tells the backend.
 
    While connected it also sends a heartbeat every `heartbeat_seconds`, so the
    backend keeps showing the camera as active.
    """
 
    def __init__(self, camera: Camera, client: BackendClient, camera_id: int,
                 heartbeat_seconds: float = 30, reconnect_seconds: float = 5,
                 retry_seconds: float = 10, clock=time.monotonic, sleep=time.sleep):
        self._camera = camera
        self._client = client
        self._camera_id = camera_id
        self._heartbeat_seconds = heartbeat_seconds
        self._reconnect_seconds = reconnect_seconds
        self._retry_seconds = retry_seconds
        self._clock = clock
        self._sleep = sleep
        self._log = logging.getLogger("safevision.monitor")
        self.status: str | None = None   # last status sent to the backend
        self.frames_read = 0
        self._last_report = 0.0
        self._last_failure: float | None = None
        self._running = False
 
    def run(self) -> None:
        self._running = True
        self._log.info("Monitoring camera %s (id %s)", self._camera.safe_url, self._camera_id)
        try:
            while self._running:
                self.step()
        finally:
            self.stop()
 
    def stop(self) -> None:
        self._running = False
        self._camera.release()
        if self.status == ACTIVE:
            self._report(DISCONNECTED)
 
    def step(self) -> None:
        """One iteration: connect if needed, read a frame, report changes."""
        if not self._camera.is_connected() and not self._camera.connect():
            self._report(DISCONNECTED)
            self._sleep(self._reconnect_seconds)
            return
 
        frame = self._camera.read_frame()
        if frame is None:
            self._log.warning("Stream interrupted, reconnecting")
            self._camera.release()
            self._report(DISCONNECTED)
            self._sleep(self._reconnect_seconds)
            return
 
        self.frames_read += 1
        heartbeat_due = self._clock() - self._last_report >= self._heartbeat_seconds
        if self.status != ACTIVE or heartbeat_due:
            self._report(ACTIVE, force=heartbeat_due)
 
    def _report(self, status: str, force: bool = False) -> None:
        if status == self.status and not force:
            return
        now = self._clock()
        if self._last_failure is not None and now - self._last_failure < self._retry_seconds:
            return  # backend failed recently: wait before trying again
        try:
            self._client.update_camera_status(self._camera_id, status)
        except BackendError as exc:
            self._last_failure = now
            self._log.error("Could not report status '%s': %s (retrying in %.0f s)",
                            status, exc, self._retry_seconds)
            return
        if status != self.status:
            self._log.info("Camera %s -> %s", self._camera_id, status)
        self.status = status
        self._last_report = now
        self._last_failure = None
 
