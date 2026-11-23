# Arquitectura del módulo IoT

## Flujo general

```
┌──────────────┐  RTSP   ┌───────────────────┐  HTTPS   ┌─────────┐     ┌────────────┐
│ Cámara Tapo  │────────▶│ Módulo IoT + IA   │─────────▶│ Backend │────▶│ PostgreSQL │
│ C110         │         │ (PC en la red)    │ estado y │ FastAPI │     └────────────┘
└──────────────┘         └─────────┬─────────┘ eventos  └────┬────┘
                                   │ MJPEG local             │ API
                                   ▼                         ▼
                         ┌───────────────────────────────────────┐
                         │ Frontend (panel web en el navegador)  │
                         └───────────────────────────────────────┘
```

Al backend solo llegan el estado de la cámara y los eventos. El video se analiza en el computador.

## Hilos

```
Hilo "camera"                 Hilo principal (IA)               Hilo "event-sender"
─────────────                 ───────────────────               ───────────────────
CameraMonitor                 DetectionPipeline.run_live        EventReporter._worker
  lee cuadros                   toma el cuadro más reciente       saca eventos de la cola
  reporta el estado             detectores YOLO                   POST /api/eventos
      │                         SceneAnalyzer + reglas            reintenta si falla
      ▼                               │                                  ▲
┌─────────────┐   latest()            │ report()                         │
│ FrameBuffer │──────────────▶        ├─────────▶ cola FIFO (máx. 100) ──┘
└─────────────┘                       │
                                      ├─────────▶ evidence/<fecha>_<evento>.jpg
                                      │
                                      ▼ publish() (solo con --stream)
                                 VideoStream ──▶ VideoServer ──▶ navegador
```

- **Hilo de cámara.** `CameraMonitor` mantiene la conexión, se reconecta si se cae y envía el latido
  de estado.
- **Hilo de IA.** Analiza `ANALYSIS_FPS` cuadros por segundo. Corre en el hilo principal porque
  puede abrir la ventana de vista previa.
- **`FrameBuffer`.** Guarda solo el cuadro más reciente. La IA es más lenta que la cámara, así que
  siempre analiza lo que pasa ahora y no acumula retraso.
- **Cola de envío.** `EventReporter` usa una cola FIFO de hasta 100 eventos y un hilo propio. Si el
  backend tarda, el análisis no se detiene y los eventos salen en orden, con reintentos.

Con `--no-ai` no hay hilo de IA: `CameraMonitor` publica los cuadros directamente en `VideoStream`.

## Servidor de video

Con `--stream`, `VideoServer` atiende cada conexión en un hilo propio:

| Ruta | Contenido |
|---|---|
| `/video` | Video en vivo en MJPEG, con las detecciones dibujadas por `FrameAnnotator` |
| `/snapshot.jpg` | Última imagen |
| `/health` | Estado del video |
| `/evidence/<archivo>.jpg` | Foto de un evento |

Por defecto escucha en `127.0.0.1` (solo este PC). Se cambia con `VIDEO_HOST`.

## Evidencia

1. Cuando una regla genera un evento, `EventReporter` guarda el cuadro en `evidence/`.
2. Si `EVIDENCE_BASE_URL` está configurado, el evento incluye esa dirección en `evidence_url`.
3. `EvidenceStore` solo entrega archivos `.jpg` que estén directamente en esa carpeta y rechaza
   nombres con `..` o separadores de ruta.

La foto se queda en el computador local: el backend guarda solo su dirección.
