"""Connection to the TP-Link Tapo C110 camera over RTSP."""
import os
from pathlib import Path

import cv2

# RTSP over TCP is more stable than UDP on Wi-Fi.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


class Camera:
    """Opens the video stream and reads frames from it."""

    def __init__(self, url: str):
        self._url = int(url) if url.isdigit() else url  # "0" = the computer's webcam
        self._capture: cv2.VideoCapture | None = None

    @property
    def safe_url(self) -> str:
        """URL without user and password, safe to print in logs."""
        text = str(self._url)
        return text.split("@")[-1] if "@" in text else text

    def connect(self) -> bool:
        self.release()
        self._capture = cv2.VideoCapture(self._url)
        return self._capture.isOpened()

    def is_connected(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def read_frame(self):
        """Returns the next frame, or None if the stream stopped."""
        if not self.is_connected():
            return None
        ok, frame = self._capture.read()
        return frame if ok else None

    def save_snapshot(self, path: str | Path) -> Path | None:
        frame = self.read_frame()
        if frame is None:
            return None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), frame)
        return path

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
