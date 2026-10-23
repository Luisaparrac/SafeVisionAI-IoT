# SafeVision AI — Módulo IoT

Módulo que conecta la cámara de seguridad **TP-Link Tapo C110** con SafeVision AI y analiza su
video con inteligencia artificial. Mantiene la cámara conectada, informa al sistema si está en línea,
detecta situaciones de riesgo con **YOLO** y las registra como eventos en la base de datos a través
del backend.

| Evento | Quién | Cómo se detecta |
|---|---|---|
| Caída | Personas | YOLO Pose + reglas de postura; opcionalmente, un modelo entrenado por el equipo |
| Inmovilidad prolongada | Personas y mascotas | Los puntos del cuerpo o la posición casi no cambian durante un tiempo |
| Permanencia en zona restringida | Personas y mascotas | La persona o el animal permanece dentro de una zona marcada en la tabla `zones` |

La IA entrega una **detección preventiva**: reconoce patrones visibles de movimiento, no hace
diagnósticos médicos.

## Lugar en la arquitectura

```
Cámara Tapo C110 ──RTSP──▶ Módulo IoT + IA ──HTTPS──▶ Backend (FastAPI) ──▶ PostgreSQL ──▶ Frontend
                     (PC en la misma red)
```

1. La cámara transmite video por **RTSP** dentro de la red local.
2. El módulo recibe el video, reporta el estado de la cámara (`PATCH /api/camaras/{id}/estado`) y
   pasa las imágenes a la IA.
3. La IA detecta personas (con 17 puntos del cuerpo) y mascotas, sigue a cada una entre cuadros y
   aplica las reglas.
4. Cuando una regla se cumple, se envía el evento (`POST /api/eventos`). El backend lo guarda en
   `events`, crea la alerta en `alerts` y el panel la muestra.

El video **no sale de la red local**: se analiza en el computador y al backend solo llegan el estado
de la cámara y los eventos. Cada evento guarda una foto de evidencia en la carpeta `evidence/`.

## Cómo funciona la IA

### Modelos

| Modelo | Qué detecta |
|---|---|
| `yolo11n-pose.pt` (YOLO Pose) | Personas y sus 17 puntos del cuerpo: hombros, caderas, rodillas, tobillos… |
| `yolo11n.pt` (YOLO detección) | Perros, gatos y aves |
| `models/fall_model.pt` (opcional) | Personas caídas; es el modelo entrenado por el equipo (ver `training/`) |

Los modelos de YOLO vienen preentrenados con el dataset COCO y se descargan solos la primera vez.
Los detectores usan **ByteTrack** para asignar un número a cada persona o animal y seguirlo entre
cuadros.

### Reglas

**Postura.** Con los puntos de hombros y caderas se calcula el ángulo del torso: cerca de 0° es
erguido y cerca de 90° es horizontal. Con ese ángulo y la forma de la caja se clasifica a la persona
como *de pie*, *sentada* o *acostada*.

**Caída.** Se detecta en tres condiciones:
1. La persona pasa de *de pie* a *acostada* en menos de 1,5 s.
2. Las caderas bajan más del 30 % de su altura de pie.
3. Sigue en el suelo 2 s después.

Si se levanta antes de esos 2 s, no se considera caída.

**Inmovilidad prolongada.** Se mide cuánto se mueven los puntos del cuerpo entre cuadros, en
proporción al tamaño de la persona. Si casi no hay movimiento durante 60 s (30 min para mascotas),
se genera el evento.

**Zona restringida.** Las zonas se leen del backend: son polígonos guardados en `zones.coordinates`,
con valores de 0 a 1 respecto a la imagen. Se toma el punto donde los pies o las patas tocan el piso
y se comprueba si está dentro del polígono con el algoritmo de *ray casting*. Si permanece adentro
2 s, se genera el evento con su `zone_id`.

**Enfriamiento.** El mismo evento de la misma persona no se repite antes de 60 s.

Todos los tiempos y umbrales se pueden ajustar en el archivo `.env`.

### Modelo entrenado por el equipo

La carpeta `training/` contiene el script y un cuaderno de Google Colab para entrenar un modelo YOLO
propio que detecta personas caídas. Se parte de un modelo ya entrenado y se ajusta con un dataset de
caídas en formato YOLO; esa técnica se llama *transfer learning*. Al configurar
`FALL_MODEL=models/fall_model.pt`, el módulo usa ese modelo **además** de las reglas: una caída se
confirma cuando el modelo marca a la persona como caída durante 2 s.

## Diseño orientado a objetos

Cada clase tiene una sola responsabilidad y está en su propio archivo.

| Clase | Archivo | Responsabilidad |
|---|---|---|
| `Settings` | `settings.py` | Leer la configuración del `.env` |
| `Camera` | `camera.py` | Conectarse a la cámara por RTSP y leer cuadros |
| `CameraMonitor` | `camera_monitor.py` | Mantener la cámara conectada y reportar su estado |
| `BackendClient` | `backend_client.py` | Comunicarse con la API del backend |
| `DetectionEvent` | `detection_event.py` | Representar y validar un evento antes de enviarlo |
| `EventReporter` | `event_reporter.py` | Convertir hallazgos en eventos, guardar la evidencia y enviarlos en segundo plano |
| `Detector` → `PoseDetector`, `AnimalDetector`, `TrainedFallDetector` | `vision/detectors.py` | Ejecutar los modelos YOLO |
| `PostureClassifier` | `vision/posture.py` | Clasificar la postura de una persona |
| `TrackRegistry`, `TrackedSubject` | `vision/tracking.py` | Seguir a cada persona o animal y guardar su historial |
| `Rule` → `FallRule`, `ImmobilityRule`, `RestrictedZoneRule`, `TrainedFallRule` | `vision/rules.py` | Decidir si hay un evento de riesgo |
| `Zone`, `ZoneProvider` | `vision/zone.py`, `vision/zone_provider.py` | Representar las zonas y cargarlas del backend |
| `SceneAnalyzer`, `Cooldown` | `vision/analyzer.py` | Aplicar todas las reglas a todos los sujetos y evitar repeticiones |
| `FrameBuffer`, `DetectionPipeline`, `Preview` | `vision/pipeline.py` | Unir cámara, modelos, reglas y envío; ventana de vista previa |
| `IoTApp` | `__main__.py` | Comandos del módulo |

Principios aplicados:

- **Herencia y polimorfismo:** los detectores heredan de `Detector` y las reglas de `Rule`. El
  analizador trata todas las reglas igual (`evaluate`), así que agregar un evento nuevo es agregar una
  clase, sin cambiar las demás.
- **Responsabilidad única y encapsulamiento:** la cámara no conoce la IA, la IA no conoce HTTP y
  cada regla guarda su propio estado interno.
- **Inyección de dependencias:** las clases reciben sus dependencias desde afuera (`vision/factory.py`
  las arma). Esto permite probarlas con objetos simulados.

## Estructuras de datos

| Estructura | Dónde | Para qué |
|---|---|---|
| Cola acotada (`deque`, 300 elementos) | `TrackedSubject.history` | Historial reciente de cada persona o animal: ventana deslizante para comparar posturas en el tiempo |
| Diccionario (tabla hash) | `TrackRegistry.subjects` | Acceso O(1) al estado de cada sujeto por su número de seguimiento |
| Diccionario de alias | `TrackRegistry._aliases` | Si el rastreador pierde a una persona al caer y le da otro número, se reasocia con la más cercana para no perder su historial |
| Diccionario `(sujeto, tipo) → momento` | `Cooldown` | Evitar alertas repetidas |
| Cola FIFO (`queue.Queue`) | `EventReporter` | Envío de eventos en segundo plano, en orden y con reintentos |
| Búfer de un elemento | `FrameBuffer` | La IA siempre analiza el cuadro más reciente y no se atrasa |

## Uso

Requisitos: Python 3.10 o superior y un computador en la **misma red** que la cámara.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

En `.env` se configuran la dirección RTSP de la cámara, su `camera_id`, la URL del backend, la clave
del módulo IoT y, si se quiere, el `subject_id` de la persona y de la mascota supervisadas.

| Comando | Función |
|---|---|
| `python -m safevision_iot check-camera` | Prueba la cámara y guarda una foto (`snapshot.jpg`) |
| `python -m safevision_iot check-backend` | Prueba el backend y lista los tipos de evento |
| `python -m safevision_iot send-test-event --type "Caída"` | Registra un evento de prueba |
| `python -m safevision_iot run --show` | Cámara + IA en vivo, con una ventana que muestra las detecciones (q para cerrar) |
| `python -m safevision_iot run --no-ai` | Solo reporta el estado de la cámara |
| `python -m safevision_iot analyze-video video.mp4 --show` | Analiza un video grabado (para pruebas y demostraciones) |

Agregar `--dry-run` a `run` o `analyze-video` detecta sin enviar eventos al backend.

## Pruebas

```bash
pytest
```

Las pruebas usan detecciones simuladas, una cámara falsa y un backend falso, así que no necesitan
hardware, modelos ni internet. Verifican, entre otras cosas, que:

- se detecte una caída una sola vez, aunque el rastreador cambie el número de la persona;
- una bajada rápida seguida de levantarse **no** cuente como caída, ni tampoco alguien que ya estaba
  acostado;
- la inmovilidad use límites distintos para personas y mascotas;
- una mascota que permanece en una zona genere el evento, y una que solo la cruza, no;
- el estado de la cámara se reporte al conectarse, al desconectarse y en el latido periódico.

## Alcance y limitaciones

- La cámara está fija y el análisis es en 2D; las caídas de frente a la cámara son más difíciles de
  ver que las laterales.
- Para mascotas se detectan zona restringida e inmovilidad. Las caídas de mascotas desde muebles
  requerirían un modelo entrenado para eso.
- Los umbrales se ajustan según la ubicación de la cámara; los valores por defecto son un punto de
  partida.
