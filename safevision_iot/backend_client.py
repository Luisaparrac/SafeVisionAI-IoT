"""HTTP client for the SafeVision AI backend."""
import requests

from safevision_iot.detection_event import DetectionEvent


class BackendError(Exception):
    """The backend rejected the request or could not be reached."""


class BackendClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = 90):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers["Content-Type"] = "application/json"
        if api_key:
            self._session.headers["X-API-Key"] = api_key

    def is_available(self) -> bool:
        try:
            return self._session.get(self._url("/health"), timeout=self._timeout).ok
        except requests.RequestException:
            return False

    def update_camera_status(self, camera_id: int, status: str) -> dict:
        """status: active, inactive, disconnected or maintenance."""
        return self._request("PATCH", f"/api/camaras/{camera_id}/estado", {"status": status})

    def send_event(self, event: DetectionEvent) -> dict:
        return self._request("POST", "/api/eventos", event.to_payload())

    def list_event_types(self) -> list[dict]:
        return self._request("GET", "/api/tipos-evento")

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def _request(self, method: str, path: str, body: dict | None = None):
        try:
            response = self._session.request(method, self._url(path), json=body, timeout=self._timeout)
        except requests.RequestException as exc:
            raise BackendError(f"Backend not reachable: {exc.__class__.__name__}") from exc
        if response.status_code == 401:
            raise BackendError("Invalid IOT_API_KEY (401)")
        if not response.ok:
            raise BackendError(f"{response.status_code}: {response.text[:200]}")
        return response.json()
