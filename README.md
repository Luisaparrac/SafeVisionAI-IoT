# SafeVision AI — Módulo IoT

> Proyecto final de **Estructuras de Datos**, Ingeniería de Software, Universidad Cooperativa de
> Colombia.
>
> Repositorios relacionados:
> [SafeVisionAI-backend](https://github.com/Luisaparrac/SafeVisionAI-backend) ·
> [SafeVisionAI-Frontend](https://github.com/Luisaparrac/SafeVisionAI-Frontend)

Módulo que conecta la cámara de seguridad **TP-Link Tapo C110** con SafeVision AI y analiza su
video con inteligencia artificial. Mantiene la cámara conectada, informa al sistema si está en línea,
detecta situaciones de riesgo con **YOLO** y las registra como eventos en la base de datos a través
del backend.

| Evento | Quién | Cómo se detecta |
|---|---|---|
| Caída | Personas | YOLO Pose + reglas de postura; opcionalmente, un modelo entrenado por el equipo |
| Movimiento anormal | Personas | Movimientos bruscos y repetidos de los brazos, como lanzar objetos |
| Inmovilidad prolongada | Personas y mascotas | Los puntos del cuerpo o la posición casi no cambian durante un tiempo |
| Permanencia en zona restringida | Personas y mascotas | La persona o el animal permanece dentro de una zona marcada en la tabla `zones` |
| Mascota sobre mueble | Perros y gatos | La mascota permanece sobre un sofá, una cama o una mesa |
| Niño subido a mueble | Niños | Los pies del niño están sobre un mueble durante un tiempo |
| Niño en zona de riesgo | Niños | Un niño permanece dentro de una zona marcada en la tabla `zones` |

La IA entrega una **detección preventiva**: reconoce patrones visibles de movimiento, no hace
diagnósticos médicos.

## Contenido

- [Lugar en la arquitectura](#lugar-en-la-arquitectura)
- [Cómo funciona la IA](#cómo-funciona-la-ia)
- [Travesuras de niños y mascotas](#travesuras-de-niños-y-mascotas)
- [Video en vivo en el panel web](#video-en-vivo-en-el-panel-web)
- [Resultados](#resultados)
- [Diseño orientado a objetos](#diseño-orientado-a-objetos)
- [Estructuras de datos](#estructuras-de-datos)
- [Uso](#uso)
- [Variables de configuración](#variables-de-configuración)
- [Pruebas](#pruebas)
- [Alcance y limitaciones](#alcance-y-limitaciones)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Créditos y licencias de datos](#créditos-y-licencias-de-datos)

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
| `yolo11n.pt` (YOLO detección) | Perros, gatos, aves y muebles (silla, sofá, cama, mesa) |
| `models/fall_model.pt` (opcional) | Personas caídas; es el modelo entrenado por el equipo (ver `training/`) |
| `models/child_model.pt` (opcional) | Clasificador niño/adulto entrenado por el equipo (ver `training/`) |

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

**Movimiento anormal.** Se mide la velocidad de las muñecas en alturas de cuerpo por segundo, para
que no dependa de la distancia a la cámara. Un movimiento es brusco si supera 0,8 alturas por
segundo. Se genera el evento con al menos 5 movimientos bruscos en 3 s. Para no confundirlo con una
caída, no cuentan los movimientos si la persona está acostada, si la cadera baja rápido o si queda
acostada en el segundo siguiente; por eso la alerta sale alrededor de 1 s después.

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

También se entrena un **clasificador niño/adulto** con YOLO11n-cls. Al configurar
`CHILD_MODEL=models/child_model.pt`, el módulo clasifica a cada persona y decide por mayoría de
votos. Los datos, las métricas y las limitaciones de cada modelo están en
[docs/modelos.md](docs/modelos.md).

## Travesuras de niños y mascotas

La detección funciona en dos etapas:

1. **Modelos.** YOLO encuentra personas, mascotas y muebles (silla, sofá, cama y mesa). Un clasificador
   YOLO11n-cls (`CHILD_MODEL`) analiza a cada persona hasta 2 veces por segundo.
2. **Reglas.** Cada persona acumula sus últimos 15 votos y se considera niño por mayoría, con al menos
   `CHILD_MIN_VOTES` votos. Después se aplican las reglas:

| Regla | Condición |
|---|---|
| Mascota sobre mueble | Perro o gato con al menos 50 % de su caja sobre un sofá, cama o mesa, y patas por encima del 80 % de la altura del mueble, durante 3 s |
| Niño subido a mueble | Pies dentro del mueble y por encima del 75 % de su altura, medida desde arriba, durante 2 s. Estar sentado en una silla no cuenta |
| Niño en zona de riesgo | Un niño permanece en una zona restringida. La regla espera hasta 3 s adicionales a que termine la votación de edad |

En la base de datos `detected_class` sigue siendo `person`. Si `CHILD_SUBJECT_ID` está configurado, se
usa como `subject_id`. Sin `CHILD_MODEL`, las reglas de niños quedan desactivadas.

## Video en vivo en el panel web

Con `--stream`, el módulo publica la imagen anotada en formato **MJPEG**, que el navegador muestra con
una etiqueta `<img>`:

```bash
python -m safevision_iot run --stream
```

| Ruta | Contenido |
|---|---|
| `http://localhost:8080/video` | Video en vivo |
| `http://localhost:8080/snapshot.jpg` | Última imagen |
| `http://localhost:8080/health` | Estado del video |
| `http://localhost:8080/evidence/<archivo>.jpg` | Foto de un evento |

El puerto se cambia con `VIDEO_PORT`. Si `EVIDENCE_BASE_URL` está configurado, cada evento envía la
dirección de su foto en `evidence_url`.

El módulo **no envía el video al backend**: se sirve desde el computador local. Cuando el panel web
está publicado en HTTPS, Chrome pide el permiso de **acceso a la red local**; hay que permitirlo para
ver el video.

**Importante:** el servidor no tiene contraseña, escucha en toda la red local y permite solicitudes
desde cualquier origen. Úsalo solo en una red de confianza.

## Resultados

Prueba con 6 videos nunca vistos, separados por video completo:

| Configuración | Aciertos | Porcentaje | Falsas alarmas |
|---|---|---|---|
| Antes: sin modelo entrenado | 3 de 6 | 50 % | 0 |
| Después: con modelo entrenado | 6 de 6 | 100 % | 0 |

El clasificador niño/adulto obtuvo una exactitud de **85,6 %** (top1_acc 0.856) en validación.

**Limitaciones:** son pocos videos, solo uno es de actividad normal y todos son del mismo entorno.
Estos resultados no garantizan el mismo desempeño en otras casas o con otras personas. El detalle está
en [training/README.md](training/README.md) y [docs/modelos.md](docs/modelos.md).

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
| `VideoStream`, `VideoServer`, `EvidenceStore` | `video_stream.py` | Publicar el video en vivo en MJPEG y servir las fotos de evidencia |
| `Detector` → `PoseDetector`, `AnimalDetector`, `TrainedFallDetector` | `vision/detectors.py` | Ejecutar los modelos YOLO; `AnimalDetector` también encuentra muebles |
| `ChildClassifier` | `vision/detectors.py` | Clasificar a cada persona como niño o adulto |
| `PostureClassifier` | `vision/posture.py` | Clasificar la postura de una persona |
| `TrackRegistry`, `TrackedSubject` | `vision/tracking.py` | Seguir a cada persona o animal y guardar su historial |
| `Rule` → `FallRule`, `AbnormalMovementRule`, `ImmobilityRule`, `RestrictedZoneRule`, `TrainedFallRule`, `FurnitureRule` → `PetOnFurnitureRule`, `ChildOnFurnitureRule` | `vision/rules.py` | Decidir si hay un evento de riesgo |
| `Zone`, `ZoneProvider` | `vision/zone.py`, `vision/zone_provider.py` | Representar las zonas y cargarlas del backend |
| `SceneAnalyzer`, `Cooldown` | `vision/analyzer.py` | Aplicar todas las reglas a todos los sujetos y evitar repeticiones |
| `FrameBuffer`, `DetectionPipeline`, `FrameAnnotator`, `Preview` | `vision/pipeline.py` | Unir cámara, modelos, reglas y envío; dibujar las detecciones; ventana de vista previa |
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

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
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
| `python -m safevision_iot run --stream` | Agrega el video en vivo para el panel web (también funciona con `--no-ai`) |
| `python -m safevision_iot analyze-video video.mp4 --show` | Analiza un video grabado (para pruebas y demostraciones) |

Agregar `--dry-run` a `run` o `analyze-video` detecta sin enviar eventos al backend.

### Cómo obtener los modelos

Los modelos propios **no están en el repositorio**. Se entrenan con los cuadernos de `training/` y se
copian a `models/`:

| Modelo | Cuaderno | Archivo | Variable en `.env` |
|---|---|---|---|
| Caídas | `training/train_fall_model.ipynb` | `models/fall_model.pt` | `FALL_MODEL=models/fall_model.pt` |
| Niño/adulto | `training/train_child_model.ipynb` | `models/child_model.pt` | `CHILD_MODEL=models/child_model.pt` |

Los modelos preentrenados de YOLO se descargan solos la primera vez.

## Variables de configuración

Se definen en `.env`. Si una variable no está, se usa el valor por defecto.

| Variable | Por defecto | Para qué sirve |
|---|---|---|
| `CAMERA_RTSP_URL` | vacío | Dirección RTSP de la cámara, con su usuario y contraseña |
| `CAMERA_ID` | `1` | `camera_id` de la cámara en el backend |
| `BACKEND_URL` | `http://localhost:8000` | Dirección del backend |
| `IOT_API_KEY` | vacío | Clave del módulo IoT para el backend |
| `AI_ENABLED` | `true` | Activa el análisis con IA |
| `POSE_MODEL` | `yolo11n-pose.pt` | Modelo de personas y puntos del cuerpo |
| `ANIMAL_MODEL` | `yolo11n.pt` | Modelo de mascotas y muebles |
| `FALL_MODEL` | vacío | Modelo de caídas; vacío = solo YOLO Pose y reglas |
| `ANALYSIS_FPS` | `5` | Cuadros analizados por segundo |
| `PERSON_SUBJECT_ID` | vacío | `subject_id` de la persona supervisada |
| `PET_SUBJECT_ID` | vacío | `subject_id` de la mascota supervisada |
| `CHILD_MODEL` | vacío | Clasificador niño/adulto; vacío = desactivado |
| `CHILD_MIN_VOTES` | `5` | Votos necesarios para decidir si una persona es un niño |
| `CHILD_SUBJECT_ID` | vacío | `subject_id` del niño supervisado |
| `FALL_WINDOW_SECONDS` | `1.5` | Tiempo máximo para pasar de pie a acostado |
| `FALL_DROP_RATIO` | `0.3` | Bajada mínima de la cadera, como fracción de la altura de pie |
| `FALL_CONFIRM_SECONDS` | `2` | Segundos en el suelo para confirmar la caída |
| `ABNORMAL_SPEED` | `0.8` | Velocidad de muñeca de un movimiento brusco, en alturas de cuerpo por segundo |
| `ABNORMAL_MIN_PEAKS` | `5` | Movimientos bruscos necesarios |
| `ABNORMAL_WINDOW_SECONDS` | `3` | Ventana en la que deben ocurrir esos movimientos |
| `ABNORMAL_MAX_HIP_SPEED` | `0.5` | Ignora movimientos si la cadera se mueve más rápido que esto |
| `ABNORMAL_SETTLE_SECONDS` | `1` | Ignora movimientos si la persona queda acostada en estos segundos |
| `PERSON_IMMOBILITY_SECONDS` | `60` | Límite de inmovilidad de una persona |
| `PET_IMMOBILITY_SECONDS` | `1800` | Límite de inmovilidad de una mascota |
| `ZONE_DWELL_SECONDS` | `2` | Permanencia necesaria en una zona restringida |
| `PET_FURNITURE_SECONDS` | `3` | Tiempo de una mascota sobre un mueble |
| `CHILD_FURNITURE_SECONDS` | `2` | Tiempo de un niño sobre un mueble |
| `EVENT_COOLDOWN_SECONDS` | `60` | Tiempo mínimo entre eventos iguales del mismo sujeto |
| `ZONES_REFRESH_SECONDS` | `300` | Cada cuánto se recargan las zonas del backend |
| `EVIDENCE_DIR` | `evidence` | Carpeta de las fotos de evidencia |
| `VIDEO_HOST` | `127.0.0.1` | `127.0.0.1` = solo este PC; `0.0.0.0` = toda la red local |
| `VIDEO_PORT` | `8080` | Puerto del video en vivo |
| `EVIDENCE_BASE_URL` | vacío | Dirección base de las fotos; vacío = no envía `evidence_url` |
| `HEARTBEAT_SECONDS` | `30` | Cada cuánto se reporta el estado de la cámara |
| `RECONNECT_SECONDS` | `5` | Espera antes de reconectar la cámara |
| `REQUEST_TIMEOUT` | `90` | Tiempo máximo de cada petición al backend |

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
