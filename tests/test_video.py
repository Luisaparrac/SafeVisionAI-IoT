"""Tests for the local MJPEG video server (no camera needed)."""
import urllib.error
import urllib.request

import numpy as np
import pytest

from safevision_iot.video_stream import VideoServer, VideoStream


def make_server():
    stream = VideoStream(max_fps=1000)
    server = VideoServer(stream, port=0, host="127.0.0.1")   # port 0 = any free port
    server.start()
    return stream, server


def test_snapshot_returns_latest_jpeg():
    stream, server = make_server()
    try:
        url = f"http://127.0.0.1:{server.port}"
        stream.publish(np.zeros((48, 64, 3), dtype=np.uint8))
        with urllib.request.urlopen(f"{url}/snapshot.jpg", timeout=3) as response:
            assert response.headers["Content-Type"] == "image/jpeg"
            assert response.headers["Access-Control-Allow-Origin"] == "*"
            assert response.read()[:2] == b"\xff\xd8"           # JPEG signature
    finally:
        server.stop()


def test_video_is_mjpeg():
    stream, server = make_server()
    try:
        stream.publish(np.zeros((48, 64, 3), dtype=np.uint8))
        with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/video", timeout=3) as response:
            assert response.headers["Content-Type"].startswith("multipart/x-mixed-replace")
            assert response.read(7) == b"--frame"
    finally:
        server.stop()


def test_publish_is_rate_limited():
    stream = VideoStream(max_fps=1)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    stream.publish(frame)
    first = stream.latest()
    stream.publish(np.full((10, 10, 3), 255, dtype=np.uint8))
    assert stream.latest() == first


@pytest.fixture
def evidence_server(tmp_path):
    folder = tmp_path / "evidence"
    folder.mkdir()
    (folder / "20260101_Caída.jpg").write_bytes(b"photo")
    (folder / "notes.txt").write_bytes(b"private")
    (tmp_path / "secret.jpg").write_bytes(b"secret")
    server = VideoServer(VideoStream(), port=0, host="127.0.0.1", evidence_dir=folder)
    server.start()
    yield f"http://127.0.0.1:{server.port}/evidence"
    server.stop()


def test_evidence_serves_jpg(evidence_server):
    with urllib.request.urlopen(f"{evidence_server}/20260101_Ca%C3%ADda.jpg", timeout=3) as response:
        assert response.headers["Content-Type"] == "image/jpeg"
        assert response.read() == b"photo"


@pytest.mark.parametrize("name", ["../secret.jpg", "..%2Fsecret.jpg", "%2E%2E%2Fsecret.jpg",
                                  "..%5Csecret.jpg", "notes.txt", "missing.jpg"])
def test_evidence_rejects_unsafe_or_missing_names(evidence_server, name):
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(f"{evidence_server}/{name}", timeout=3)
    assert error.value.code == 404
