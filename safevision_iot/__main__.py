"""Command line entry point.

    python -m safevision_iot check-camera                 test the camera and save a snapshot
    python -m safevision_iot check-backend                test the backend and list event types
    python -m safevision_iot send-test-event              create a test event in the backend
    python -m safevision_iot run [--show] [--no-ai]       camera + AI: report status and risk events
    python -m safevision_iot analyze-video FILE [--show]  run the AI on a recorded video
"""
import argparse
import logging
import sys
import threading

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
        try:
            names = [t["name"] for t in self._client.list_event_types()]
        except BackendError as exc:
            print(f"Rejected: {exc}")
            return 1
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

    def run(self, use_ai: bool, show: bool, dry_run: bool) -> int:
        """Camera status in a background thread; the AI in the main thread (it can open a window)."""
        settings = self._settings
        client = None if dry_run else self._client
        if not (use_ai and settings.ai_enabled):
            monitor = CameraMonitor(self._new_camera(), self._client, settings.camera_id,
                                    settings.heartbeat_seconds, settings.reconnect_seconds)
            try:
                monitor.run()
            except KeyboardInterrupt:
                print("\nStopped.")
            return 0

        from safevision_iot.vision.factory import build_pipeline
        from safevision_iot.vision.pipeline import FrameBuffer

        logging.getLogger("safevision").info("Loading YOLO models...")
        pipeline = build_pipeline(settings, client, show)
        buffer = FrameBuffer()
        monitor = CameraMonitor(self._new_camera(), self._client, settings.camera_id,
                                settings.heartbeat_seconds, settings.reconnect_seconds,
                                on_frame=buffer.put)
        stop = threading.Event()
        camera_thread = threading.Thread(target=monitor.run, name="camera", daemon=True)
        camera_thread.start()
        try:
            pipeline.run_live(buffer, settings.analysis_fps, stop)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            stop.set()
            monitor.stop()
            pipeline.reporter.wait_until_sent()
            pipeline.close()
        return 0

    def analyze_video(self, path: str, show: bool, dry_run: bool) -> int:
        from safevision_iot.vision.factory import build_pipeline

        pipeline = build_pipeline(self._settings, None if dry_run else self._client, show)
        try:
            findings = pipeline.run_video(path, self._settings.analysis_fps)
        except FileNotFoundError as exc:
            print(exc)
            return 1
        finally:
            pipeline.close()
        pipeline.reporter.wait_until_sent()
        print(f"Done: {len(findings)} event(s) detected"
              + "".join(f"\n  - {f.event_type} ({f.detected_class}): {f.description}" for f in findings))
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
    test_event.add_argument("--type", default="Caída", help="event type name (default: Caída)")
    run = commands.add_parser("run", help="camera status + AI detection")
    run.add_argument("--show", action="store_true", help="open a window with the detections")
    run.add_argument("--no-ai", action="store_true", help="only report the camera status")
    run.add_argument("--dry-run", action="store_true", help="detect but do not send events")
    video = commands.add_parser("analyze-video", help="run the AI on a recorded video")
    video.add_argument("path")
    video.add_argument("--show", action="store_true")
    video.add_argument("--dry-run", action="store_true", help="detect but do not send events")
    args = parser.parse_args()

    app = IoTApp(Settings.from_env())
    actions = {
        "check-camera": app.check_camera,
        "check-backend": app.check_backend,
        "send-test-event": lambda: app.send_test_event(args.type),
        "run": lambda: app.run(not args.no_ai, args.show, args.dry_run),
        "analyze-video": lambda: app.analyze_video(args.path, args.show, args.dry_run),
    }
    sys.exit(actions[args.command]())


if __name__ == "__main__":
    main()