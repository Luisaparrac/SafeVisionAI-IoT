# Fichas de los modelos

Los archivos `.pt` no están en el repositorio. Se entrenan con los cuadernos de `training/` y se
copian a `models/`.

## `fall_model.pt`: detección de caídas

| Campo | Valor |
|---|---|
| Tarea | Detección de objetos, clases `fall` y `person` |
| Arquitectura | YOLO11n |
| Datos | Unas 443 fotos propias de la cámara Tapo C110, etiquetadas en Roboflow |
| División | 310 entrenamiento, 89 validación, 44 prueba |
| Entrenamiento | 100 épocas en Google Colab con GPU T4 |
| Variable | `FALL_MODEL=models/fall_model.pt` |

**Métricas de validación**

| Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|
| 0.988 | 1.000 | 0.995 | 0.959 |

Estas métricas son **optimistas**: las fotos de validación vienen de la misma fuente de videos que
las de entrenamiento.

**Prueba con 6 videos nunca vistos**

| Configuración | Aciertos | Falsas alarmas |
|---|---|---|
| Sin modelo | 3 de 6 | 0 |
| Con modelo | 6 de 6 | 0 |

**Limitaciones:** son pocos videos, solo uno es de actividad normal y todos son del mismo entorno.
Falta medir las falsas alarmas con más situaciones normales.

## `child_model.pt`: clasificador niño/adulto

| Campo | Valor |
|---|---|
| Tarea | Clasificación de imágenes, niño o adulto |
| Arquitectura | YOLO11n-cls |
| Datos | [CHILD ADULT DATASET](https://universe.roboflow.com/vallabha-group-of-companies/child-adult-dataset), de Vallabha Group of Companies, en Roboflow Universe |
| Licencia de los datos | MIT |
| División | 559 entrenamiento, 139 validación, 99 prueba |
| Entrenamiento | 30 épocas |
| top1_acc | 0.856 |
| Variable | `CHILD_MODEL=models/child_model.pt` |

Con esa exactitud, un solo cuadro puede equivocarse. Para compensarlo, cada persona acumula sus
últimos 15 votos y se decide por **mayoría**, con al menos `CHILD_MIN_VOTES` votos.

**Limitaciones:** el dataset no viene de la cámara del proyecto y el modelo no se evaluó con videos
reales de niños en este entorno.

## Modelos preentrenados de Ultralytics

| Modelo | Uso |
|---|---|
| `yolo11n-pose.pt` | Personas y 17 puntos del cuerpo |
| `yolo11n.pt` | Perros, gatos, aves y muebles; punto de partida del modelo de caídas |
| `yolo11n-cls.pt` | Punto de partida del clasificador niño/adulto |

Ultralytics publica estos modelos con licencia **AGPL-3.0**.
