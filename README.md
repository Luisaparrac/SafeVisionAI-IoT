# SafeVision AI — IoT module

Connects the **TP-Link Tapo C110** camera to the SafeVision AI backend. This first version is
intentionally simple: it keeps the camera connected over RTSP, reports whether it is online to
the backend, and can send events. Detection (YOLO) lives in the separate AI repository, which can
reuse `BackendClient` and `DetectionEvent` from here to report what it finds.

```
Tapo C110 --RTSP--> IoT module (PC on the same Wi-Fi) --HTTPS--> backend --> Azure PostgreSQL --> frontend
```

## Classes

| Class | File | Responsibility |
|---|---|---|
| `Settings` | `settings.py` | Reads the configuration from `.env` |
| `Camera` | `camera.py` | Opens the RTSP stream, reads frames, saves snapshots |
| `DetectionEvent` | `detection_event.py` | One event to report; validates it and builds the request body |
| `BackendClient` | `backend_client.py` | Calls the backend API (`/health`, `/api/eventos`, `/api/camaras/{id}/estado`) |
| `CameraMonitor` | `camera_monitor.py` | Keeps the camera connected, reconnects, reports `active` / `disconnected` and a heartbeat |
| `IoTApp` | `__main__.py` | Command line: wires the classes together |

## Setup

### 1. Camera (Tapo app, on the phone)

1. Open the camera in the **Tapo** app → **Settings** → **Advanced Settings** → **Camera Account**.
2. Create a camera **user and password** (different from your TP-Link account; RTSP uses these).
3. Note the camera **IP** in **Settings → Device Info** (e.g. `192.168.1.50`). Reserving that IP in the
   router (DHCP reservation) keeps it from changing.

Stream URL (`stream2` = low resolution, enough for monitoring; `stream1` = high resolution):

```
rtsp://CAMERA_USER:CAMERA_PASSWORD@192.168.1.50:554/stream2
```

### 2. Install (on a computer on the **same Wi-Fi** as the camera)

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env            # macOS/Linux: cp .env.example .env
```

Fill in `.env`:

| Variable | Value |
|---|---|
| `CAMERA_RTSP_URL` | The URL from step 1 (`0` = the computer's webcam, or a video file path, for testing) |
| `CAMERA_ID` | `camera_id` of this camera in the backend (`GET /api/camaras`) |
| `BACKEND_URL` | Backend URL, e.g. `https://safevision-backend.onrender.com` |
| `IOT_API_KEY` | Same `IOT_API_KEY` as the backend (Render → Environment) |

## Commands

```bash
python -m safevision_iot check-backend                     # backend reachable? lists event types
python -m safevision_iot check-camera                      # camera reachable? saves snapshot.jpg
python -m safevision_iot send-test-event --type "Fall"     # creates an event + alert in the backend
python -m safevision_iot run                               # keeps monitoring; Ctrl+C to stop
```

`--type` must be a name from `event_types` (the list `check-backend` prints).

`run` sets the camera to `active` in the backend when frames arrive, to `disconnected` when the
stream drops (and reconnects every 5 s), sends a heartbeat every 30 s while it is active, and sets
`disconnected` on Ctrl+C.

## Tests

```bash
pytest
```

They use a fake camera and a fake backend, so no hardware or network is needed.

## Troubleshooting

| Message | Cause |
|---|---|
| `Could not open the stream` | Wrong IP, wrong camera account, or the computer is not on the camera's Wi-Fi |
| `Connected but no image arrived` | Try `/stream2` instead of `/stream1` |
| `Backend not reachable` | Wrong `BACKEND_URL`, or Render is waking up (~1 minute): try again |
| `Invalid IOT_API_KEY (401)` | `.env` key differs from the backend's |
| `404: … no existe` | `CAMERA_ID` or the event type name does not exist in the database |

## Using it from the AI repository

```python
from safevision_iot.backend_client import BackendClient
from safevision_iot.detection_event import DetectionEvent

client = BackendClient("https://safevision-backend.onrender.com", api_key="...")
client.send_event(DetectionEvent(camera_id=1, event_type="Fall", subject_id=1,
                                 detected_class="person", confidence=0.93))
```
