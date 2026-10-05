"""Configuration loaded from environment variables (.env file)."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    camera_url: str
    camera_id: int
    backend_url: str
    api_key: str
    heartbeat_seconds: float
    reconnect_seconds: float
    request_timeout: float

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            camera_url=os.getenv("CAMERA_RTSP_URL", "").strip(),
            camera_id=int(os.getenv("CAMERA_ID", "1")),
            backend_url=os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/"),
            api_key=os.getenv("IOT_API_KEY", "").strip(),
            heartbeat_seconds=float(os.getenv("HEARTBEAT_SECONDS", "30")),
            reconnect_seconds=float(os.getenv("RECONNECT_SECONDS", "5")),
            request_timeout=float(os.getenv("REQUEST_TIMEOUT", "90")),
        )
