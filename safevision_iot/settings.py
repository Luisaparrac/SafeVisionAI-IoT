"""Configuration loaded from environment variables (.env file)."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _optional_int(name: str) -> int | None:
    value = os.getenv(name, "").strip()
    return int(value) if value else None


@dataclass(frozen=True)
class Settings:
    # Camera and backend
    camera_url: str
    camera_id: int
    backend_url: str
    api_key: str
    heartbeat_seconds: float
    reconnect_seconds: float
    request_timeout: float

    # AI
    ai_enabled: bool
    pose_model: str
    animal_model: str
    fall_model: str | None          
    analysis_fps: float
    person_subject_id: int | None   
    pet_subject_id: int | None

    # Rules
    fall_window_seconds: float
    fall_drop_ratio: float
    fall_confirm_seconds: float
    person_immobility_seconds: float
    pet_immobility_seconds: float
    zone_dwell_seconds: float
    event_cooldown_seconds: float
    zones_refresh_seconds: float
    evidence_dir: str

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        env = os.getenv
        return cls(
            camera_url=env("CAMERA_RTSP_URL", "").strip(),
            camera_id=int(env("CAMERA_ID", "1")),
            backend_url=env("BACKEND_URL", "http://localhost:8000").rstrip("/"),
            api_key=env("IOT_API_KEY", "").strip(),
            heartbeat_seconds=float(env("HEARTBEAT_SECONDS", "30")),
            reconnect_seconds=float(env("RECONNECT_SECONDS", "5")),
            request_timeout=float(env("REQUEST_TIMEOUT", "90")),
            ai_enabled=env("AI_ENABLED", "true").lower() == "true",
            pose_model=env("POSE_MODEL", "yolo11n-pose.pt"),
            animal_model=env("ANIMAL_MODEL", "yolo11n.pt"),
            fall_model=env("FALL_MODEL", "").strip() or None,
            analysis_fps=float(env("ANALYSIS_FPS", "5")),
            person_subject_id=_optional_int("PERSON_SUBJECT_ID"),
            pet_subject_id=_optional_int("PET_SUBJECT_ID"),
            fall_window_seconds=float(env("FALL_WINDOW_SECONDS", "1.5")),
            fall_drop_ratio=float(env("FALL_DROP_RATIO", "0.3")),
            fall_confirm_seconds=float(env("FALL_CONFIRM_SECONDS", "2")),
            person_immobility_seconds=float(env("PERSON_IMMOBILITY_SECONDS", "60")),
            pet_immobility_seconds=float(env("PET_IMMOBILITY_SECONDS", "1800")),
            zone_dwell_seconds=float(env("ZONE_DWELL_SECONDS", "2")),
            event_cooldown_seconds=float(env("EVENT_COOLDOWN_SECONDS", "60")),
            zones_refresh_seconds=float(env("ZONES_REFRESH_SECONDS", "300")),
            evidence_dir=env("EVIDENCE_DIR", "evidence"),
        )
