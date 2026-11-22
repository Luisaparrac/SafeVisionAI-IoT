"""Fails if a real key or camera password is committed to the repository."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


class SecretScanner:
    """Finds secrets in a text. It reports only their kind, never their value."""

    API_KEY = re.compile(r"api_key\s*[=:]\s*\\?[\"']([^\"'\\\s]+)", re.IGNORECASE)
    RTSP = re.compile(r"rtsp://([^:/\s@]+):([^@\s/]+)@")
    IOT_KEY = re.compile(r"^\s*#?\s*IOT_API_KEY[ \t]*=[ \t]*([^\s#]+)", re.MULTILINE)
    API_KEY_PLACEHOLDER = "TU_API_KEY"
    RTSP_EXAMPLE = ("camera_user", "camera_password")
    IOT_KEY_EXAMPLE = "change-this-key"

    def find(self, text: str) -> list[str]:
        found = []
        if any(self._is_real_api_key(value) for value in self.API_KEY.findall(text)):
            found.append("api_key")
        if any(credentials != self.RTSP_EXAMPLE for credentials in self.RTSP.findall(text)):
            found.append("rtsp")
        if any(value != self.IOT_KEY_EXAMPLE for value in self.IOT_KEY.findall(text)):
            found.append("IOT_API_KEY")
        return found

    def _is_real_api_key(self, value: str) -> bool:
        return value != self.API_KEY_PLACEHOLDER and sum(c.isalnum() for c in value) > 12


def tracked_files() -> list[Path]:
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    listed = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=False)
    if listed.returncode:
        pytest.skip("not a git repository")
    names = [name.decode("utf-8") for name in listed.stdout.split(b"\0") if name]
    return [ROOT / name for name in names if (ROOT / name).is_file()]


def test_tracked_files_have_no_secrets():
    scanner = SecretScanner()
    problems = []
    for path in tracked_files():
        kinds = scanner.find(path.read_bytes().decode("utf-8", errors="ignore"))
        if kinds:
            problems.append(f"{path.relative_to(ROOT)}: {', '.join(kinds)}")
    assert not problems, f"Posibles claves en: {problems}"


@pytest.mark.parametrize("text, kind", [
    ("api_" + 'key="' + "a1b2c3d4e5f6g7h8" + '"', "api_key"),
    ("api_" + 'key=\\"' + "a1b2c3d4e5f6g7h8" + '\\"', "api_key"),
    ("rtsp://" + "someone:hunter2" + "@10.0.0.5:554/stream2", "rtsp"),
    ("IOT_API" + "_KEY=" + "abc123", "IOT_API_KEY"),
])
def test_scanner_detects_real_values(text, kind):
    assert SecretScanner().find(text) == [kind]


@pytest.mark.parametrize("text", [
    'rf = Roboflow(api_key="TU_API_KEY")',
    'rf = Roboflow(api_key=userdata.get("ROBOFLOW_API_KEY"))',
    'client = Client(api_key="short")',
    "CAMERA_RTSP_URL=rtsp://camera_user:camera_password@192.168.1.50:554/stream2",
    "rtsp://192.168.1.50:554/stream2",
    "IOT_API_KEY=change-this-key",
    "IOT_API_KEY=",
    '"IOT_API_KEY=",\n"another_long_text_value"',
])
def test_scanner_accepts_placeholders(text):
    assert SecretScanner().find(text) == []
