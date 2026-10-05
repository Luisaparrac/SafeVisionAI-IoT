"""Command line entry point.

    python -m safevision_iot check-camera       test the camera and save a snapshot
    python -m safevision_iot check-backend      test the backend and list event types
    python -m safevision_iot send-test-event    create a test event in the backend
    python -m safevision_iot run                keep the camera connected and report its status
"""
import argparse
import logging
import sys

from safevision_iot.backend_client import BackendClient, BackendError
from safevision_iot.camera import Camera
from safevision_iot.camera_monitor import CameraMonitor
from safevision_iot.detection_event import DetectionEvent
from safevision_iot.settings import Settings


class IoTApp:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = BackendClient(settings.backend_url, settings.api_key, settings.request_timeout)

    def check_camera(self) -> int:
        camera = self._new_camera()
        print(f"Connecting to {camera.safe_url} ...")
        if not camera.connect():
            print("Could not open the stream. Check the IP, the camera account and the Wi-Fi network.")
            return 1
        path = camera.save_snapshot("snapshot.jpg")
        camera.release()
        if path is None:
            print("Connected but no image arrived. Try /stream2 instead of /stream1.")
            return 1
        print(f"OK. Snapshot saved to {path}")
        return 0

    def check_backend(self) -> int:
        print(f"Backend {self._settings.backend_url} ...")
        if not self._client.is_available():
            print("Backend not reachable. Check BACKEND_URL (Render may take ~1 min to wake up).")
            return 1
        names = [t["name"] for t in self._client.list_event_types()]
        print(f"OK. Event types: {', '.join(names)}")
        return 0

    def send_test_event(self, event_type: str) -> int:
        event = DetectionEvent(camera_id=self._settings.camera_id, event_type=event_type,
                               confidence=1.0, description="Test event sent by the IoT module")
        try:
            created = self._client.send_event(event)
        except BackendError as exc:
            print(f"Rejected: {exc}")
            return 1
        print(f"OK. Event #{created['id']} {created['type']} on {created['camera']} "
              f"({created['priority']}, {created['status']})")
        return 0

    def run(self) -> int:
        settings = self._settings
        monitor = CameraMonitor(self._new_camera(), self._client, settings.camera_id,
                                settings.heartbeat_seconds, settings.reconnect_seconds)
        try:
            monitor.run()
        except KeyboardInterrupt:
            print("\nStopped.")
        return 0

    def _new_camera(self) -> Camera:
        if not self._settings.camera_url:
            sys.exit("CAMERA_RTSP_URL is missing in .env")
        return Camera(self._settings.camera_url)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(prog="safevision_iot", description="SafeVision AI IoT module")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check-camera", help="test the camera and save a snapshot")
    commands.add_parser("check-backend", help="test the backend connection")
    test_event = commands.add_parser("send-test-event", help="create a test event")
    test_event.add_argument("--type", default="Fall", help="event type name (default: Fall)")
    commands.add_parser("run", help="keep the camera connected and report its status")
    args = parser.parse_args()

    app = IoTApp(Settings.from_env())
    actions = {
        "check-camera": app.check_camera,
        "check-backend": app.check_backend,
        "send-test-event": lambda: app.send_test_event(args.type),
        "run": app.run,
    }
    sys.exit(actions[args.command]())


if __name__ == "__main__":
    main()
