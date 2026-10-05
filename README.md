# SafeVision AI — Módulo IoT

Módulo encargado de la conexión entre la cámara de seguridad **TP-Link Tapo C110** y el backend de
SafeVision AI. Mantiene la cámara conectada, informa al sistema si está en línea o desconectada y
ofrece el medio para registrar eventos de riesgo en la base de datos.

La detección de eventos con inteligencia artificial (YOLO Pose) se desarrolla en un repositorio
aparte. Ese módulo de IA reutiliza las clases `BackendClient` y `DetectionEvent` de este
repositorio para reportar lo que detecta.

## Lugar en la arquitectura

```
Cámara Tapo C110 ──RTSP──▶ Módulo IoT ──HTTPS──▶ Backend (FastAPI) ──▶ PostgreSQL en Azure ──▶ Frontend
                     (PC en la misma red)
```

1. La cámara transmite video por **RTSP** dentro de la red local.
2. El módulo IoT recibe el video en un computador conectado a la misma red.
3. Cuando la cámara se conecta o se desconecta, el módulo actualiza su estado en el backend
   (`PATCH /api/camaras/{id}/estado`).
4. Cuando hay un evento, el módulo lo envía al backend (`POST /api/eventos`), que lo guarda en la
   tabla `events` y genera la alerta correspondiente en `alerts`.

El video **no sale de la red local**: al backend solo llegan el estado de la cámara y los eventos.

## Diseño orientado a objetos

Cada clase tiene una sola responsabilidad y está en su propio archivo.

| Clase | Archivo | Responsabilidad |
|---|---|---|
| `Settings` | `settings.py` | Leer la configuración desde variables de entorno (`.env`) |
| `Camera` | `camera.py` | Conectarse a la cámara por RTSP, leer cuadros de video y guardar fotos |
| `DetectionEvent` | `detection_event.py` | Representar un evento, validar sus datos y convertirlo al formato de la API |
| `BackendClient` | `backend_client.py` | Comunicarse con la API del backend |
| `CameraMonitor` | `camera_monitor.py` | Vigilar la conexión de la cámara y reportar su estado |
| `IoTApp` | `__main__.py` | Unir las clases anteriores y ofrecer los comandos del módulo |

```
                 IoTApp
        ┌──────────┼─────────────┐
     Settings   CameraMonitor   BackendClient ◀── DetectionEvent
                ┌───┴────┐
             Camera   BackendClient
```

Principios aplicados:

- **Responsabilidad única:** la cámara no sabe que existe un backend, y el cliente HTTP no sabe que
  existe una cámara. `CameraMonitor` es la única clase que las coordina.
- **Encapsulamiento:** los atributos internos son privados (`_capture`, `_session`, `_status`…) y
  solo se accede a ellos mediante métodos.
- **Inyección de dependencias:** `CameraMonitor` recibe la cámara, el cliente, el reloj y la función
  de espera desde afuera. Esto permite probarlo con objetos falsos, sin cámara ni red.
- **Excepción propia:** `BackendError` agrupa todos los errores de comunicación con el backend, para
  que el resto del módulo los maneje en un solo lugar.

## Funciones principales

### `Camera`

| Método | Qué hace |
|---|---|
| `connect()` | Abre la transmisión RTSP de la cámara. Devuelve `True` si lo logra |
| `is_connected()` | Indica si la transmisión está abierta |
| `read_frame()` | Lee el siguiente cuadro de video; devuelve `None` si la transmisión se cortó |
| `save_snapshot(ruta)` | Toma una foto del momento actual y la guarda como imagen |
| `release()` | Cierra la conexión con la cámara |
| `safe_url` | Dirección de la cámara sin usuario ni contraseña, para mostrarla en los registros |

La conexión usa RTSP sobre **TCP**, que es más estable que UDP en redes wifi.

### `CameraMonitor`

| Método | Qué hace |
|---|---|
| `run()` | Ciclo continuo de vigilancia hasta que se detiene el programa |
| `step()` | Una iteración: conecta si hace falta, lee un cuadro y decide qué reportar |
| `stop()` | Libera la cámara y la marca como desconectada en el backend |

Comportamiento:

- Cuando llegan cuadros de video, reporta la cámara como **`active`**.
- Si la transmisión se corta, la reporta como **`disconnected`** e intenta reconectar cada 5 segundos.
- Mientras está activa, envía un **latido** (*heartbeat*) cada 30 segundos para que el sistema sepa
  que sigue funcionando.
- Solo envía el estado cuando cambia (o en el latido), para no saturar el backend.
- Si el backend no responde, no se detiene: registra el error y reintenta a los 10 segundos.

### `BackendClient`

| Método | Endpoint | Qué hace |
|---|---|---|
| `is_available()` | `GET /health` | Verifica que el backend esté en línea |
| `update_camera_status(id, estado)` | `PATCH /api/camaras/{id}/estado` | Actualiza el estado de la cámara en la tabla `cameras` |
| `send_event(evento)` | `POST /api/eventos` | Registra un evento; el backend crea también la alerta |
| `list_event_types()` | `GET /api/tipos-evento` | Consulta el catálogo de tipos de evento |

Todas las peticiones de escritura van autenticadas con la clave del módulo IoT (encabezado
`X-API-Key`).

### `DetectionEvent`

Representa un evento con los mismos campos de la tabla `events`: cámara, tipo de evento, sujeto,
zona, clase detectada (`person`, `dog`, `cat`, `other_animal`), confianza (0 a 1), evidencia y
descripción. Valida los datos al crearse y los convierte al cuerpo JSON que espera la API.

## Uso

Requisitos: Python 3.10 o superior y un computador conectado a la **misma red** que la cámara.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

En `.env` se configuran la dirección RTSP de la cámara, su `camera_id` en la base de datos, la URL
del backend y la clave del módulo IoT. La cuenta RTSP se crea en la app Tapo, en
**Configuración avanzada → Cuenta de la cámara**.

| Comando | Función |
|---|---|
| `python -m safevision_iot check-camera` | Prueba la conexión con la cámara y guarda una foto (`snapshot.jpg`) |
| `python -m safevision_iot check-backend` | Prueba la conexión con el backend y lista los tipos de evento |
| `python -m safevision_iot send-test-event --type "Fall"` | Registra un evento de prueba en la base de datos |
| `python -m safevision_iot run` | Inicia la vigilancia continua de la cámara (Ctrl+C para detener) |

## Pruebas

```bash
pytest
```

Las pruebas usan una cámara y un backend simulados, así que no necesitan hardware ni conexión a
internet. Verifican que el módulo:

- reporte la cámara como activa una sola vez cuando llegan cuadros;
- la reporte como desconectada cuando se corta la transmisión;
- envíe el latido periódico;
- espere antes de reintentar cuando el backend falla;
- construya y valide correctamente los eventos.

## Alcance

Esta versión cubre la conexión con la cámara, el monitoreo de su estado y el registro de eventos. La
detección automática de caídas, inmovilidad y movimientos anormales se realiza en el módulo de
inteligencia artificial, que envía sus resultados a través de este mismo cliente.
