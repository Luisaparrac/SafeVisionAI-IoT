# Historial de cambios

Las fechas corresponden a los commits del repositorio.

## Seguridad (2026-11-22)

- Los cuadernos leen la clave de Roboflow desde los secretos de Colab.
- Prueba que impide subir claves y contraseñas de cámara.
- El video en vivo escucha solo en este PC por defecto (`VIDEO_HOST`).
- Pruebas automáticas en GitHub Actions.
- Documentación de modelos, arquitectura y seguridad.

## Video en vivo (2026-11-22)

- Servidor MJPEG local para el panel web con la opción `--stream`.
- Ruta de fotos de evidencia y envío opcional de `evidence_url`.
- Dibujo de detecciones separado en `FrameAnnotator`.

## Clasificador niño/adulto y travesuras (2026-11-22)

- Script y cuaderno de entrenamiento del clasificador YOLO11n-cls.
- Votación de edad por persona y detección de muebles.
- Eventos "Mascota sobre mueble", "Niño subido a mueble" y "Niño en zona de riesgo".

## Modelo de caídas (2026-10-22 y 2026-11-22)

- 2026-10-22: script y cuaderno de entrenamiento.
- 2026-11-22: corrección de rutas de Roboflow, creación de validación y resultados de la prueba con
  6 videos nunca vistos: 3 de 6 sin modelo y 6 de 6 con modelo.

## Evaluador y línea base (2026-11-22)

- Evaluador de videos grabados y extractor de fotos para el dataset.
- Línea base con 28 clips: 14 aciertos (50,0 %).

## Movimiento anormal (2026-11-22)

- Regla de movimientos bruscos y repetidos de los brazos, con filtros para no confundirlos con caídas.

## IA con reglas (2026-10-22)

- Detección de personas y mascotas con YOLO y seguimiento entre cuadros.
- Clasificación de postura.
- Reglas de caída, inmovilidad prolongada y zona restringida.
- Envío de eventos al backend en segundo plano.

## Conexión de la cámara (2026-10-05)

- Conexión inicial con la cámara Tapo C110.
