"""Local live video for the web panel (MJPEG over HTTP).

Browsers cannot play RTSP, so the IoT module re-publishes the camera image as
MJPEG, a format an <img> tag can show directly:

    http://localhost:8080/video          live video (with the AI drawings)
    http://localhost:8080/snapshot.jpg   latest image
    http://localhost:8080/health         {"streaming": true|false}
    http://localhost:8080/evidence/x.jpg photo saved with an event

The video never leaves the local network: it is not sent to the backend.
"""
import json
import logging
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import cv2


class VideoStream:
    """Keeps the newest frame as JPEG. Writers publish, HTTP clients read."""

    def __init__(self, max_fps: float = 10, jpeg_quality: int = 75):
        self._min_period = 1 / max_fps
        self._quality = jpeg_quality
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._version = 0
        self._last_publish = 0.0

    def publish(self, frame) -> None:
        now = time.monotonic()
        if now - self._last_publish < self._min_period:
            return                                    # limit CPU and bandwidth
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        if not ok:
            return
        with self._condition:
            self._jpeg, self._version, self._last_publish = encoded.tobytes(), self._version + 1, now
            self._condition.notify_all()

    def latest(self) -> bytes | None:
        with self._condition:
            return self._jpeg

    def is_live(self, max_age: float = 5) -> bool:
        return self._jpeg is not None and time.monotonic() - self._last_publish < max_age

    def wait_next(self, last_version: int, timeout: float = 5) -> tuple[bytes | None, int]:
        """Blocks until a frame newer than `last_version` exists (or timeout)."""
        with self._condition:
            self._condition.wait_for(lambda: self._version != last_version, timeout)
            return self._jpeg, self._version


class EvidenceStore:
    """Reads event photos. Only plain .jpg names directly inside the folder are allowed."""

    VALID_NAME = re.compile(r"[\w-]+\.jpg")

    def __init__(self, folder: str | Path):
        self._folder = Path(folder)

    def read(self, url_name: str) -> bytes | None:
        name = unquote(url_name)
        if not self.VALID_NAME.fullmatch(name):
            return None
        path = self._folder / name
        return path.read_bytes() if path.is_file() else None


class VideoServer:
    """Small HTTP server that serves a VideoStream in a background thread."""

    BOUNDARY = "frame"

    def __init__(self, stream: VideoStream, port: int = 8080, host: str = "127.0.0.1",
                 evidence_dir: str | Path | None = None):
        self._stream = stream
        self._evidence = EvidenceStore(evidence_dir) if evidence_dir else None
        self._log = logging.getLogger("safevision.video")
        self._server = ThreadingHTTPServer((host, port), self._handler_class())
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="video", daemon=True)

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    def start(self) -> None:
        self._thread.start()
        self._log.info("Live video at http://localhost:%s/video", self.port)

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _handler_class(self):
        stream, boundary, evidence = self._stream, self.BOUNDARY, self._evidence

        class Handler(BaseHTTPRequestHandler):
            def do_OPTIONS(self):
                self.send_response(204)
                self._cors()
                self.end_headers()

            def do_GET(self):
                path = self.path.split("?")[0]
                if path == "/video":
                    self._send_video()
                elif path == "/snapshot.jpg":
                    self._send_snapshot()
                elif path == "/health":
                    self._send(200, "application/json",
                               json.dumps({"streaming": stream.is_live()}).encode())
                elif path.startswith("/evidence/"):
                    self._send_evidence(path.removeprefix("/evidence/"))
                else:
                    self._send(404, "text/plain", b"Not found")

            def _send_video(self):
                self.send_response(200)
                self._cors()
                self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                version = -1
                try:
                    while True:
                        jpeg, version = stream.wait_next(version)
                        if jpeg is None:
                            continue
                        self.wfile.write(f"--{boundary}\r\nContent-Type: image/jpeg\r\n"
                                         f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                        self.wfile.write(jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass                                      # the browser closed the page

            def _send_snapshot(self):
                jpeg = stream.latest()
                if jpeg is None:
                    self._send(503, "text/plain", b"No image yet")
                else:
                    self._send(200, "image/jpeg", jpeg)

            def _send_evidence(self, name):
                photo = evidence.read(name) if evidence else None
                if photo is None:
                    self._send(404, "text/plain", b"Not found")
                else:
                    self._send(200, "image/jpeg", photo)

            def _send(self, code, content_type, body):
                self.send_response(code)
                self._cors()
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _cors(self):
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Private-Network", "true")

            def log_message(self, *args):
                pass                                          # keep the console clean

        return Handler
