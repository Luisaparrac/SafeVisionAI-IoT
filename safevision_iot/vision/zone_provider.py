"""Keeps the restricted zones of this camera, loaded from the backend."""
import logging
import time

from safevision_iot.backend_client import BackendClient, BackendError
from safevision_iot.vision.zone import Zone


class ZoneProvider:
    """Loads the active zones of the camera and reloads them every few minutes.

    If the backend is not available, it keeps using the last zones it loaded.
    """

    def __init__(self, client: BackendClient, camera_id: int, refresh_seconds: float = 300,
                 clock=time.monotonic):
        self._client = client
        self._camera_id = camera_id
        self._refresh_seconds = refresh_seconds
        self._clock = clock
        self._zones: list[Zone] = []
        self._loaded_at: float | None = None
        self._log = logging.getLogger("safevision.zones")

    def __call__(self) -> list[Zone]:
        now = self._clock()
        if self._loaded_at is None or now - self._loaded_at >= self._refresh_seconds:
            self._loaded_at = now
            self._reload()
        return self._zones

    def _reload(self) -> None:
        try:
            rows = self._client.list_zones()
        except BackendError as exc:
            self._log.warning("Could not load zones (%s); keeping %d zones", exc, len(self._zones))
            return
        self._zones = [Zone.from_api(row) for row in rows
                       if row.get("camera_id") == self._camera_id and row.get("is_active", True)]
        self._log.info("Restricted zones for camera %s: %s", self._camera_id,
                       ", ".join(z.name for z in self._zones) or "none")
