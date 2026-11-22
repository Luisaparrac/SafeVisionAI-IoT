# Entrenamiento del modelo de caídas

Aquí se entrena un modelo YOLO propio que detecta **personas caídas**. Complementa las reglas de
postura del módulo IoT.

## Opción recomendada: Google Colab

1. Abre `train_fall_model.ipynb` en [Google Colab](https://colab.research.google.com/) (*Archivo →
   Subir cuaderno*).
2. Activa la GPU: *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4*.
3. Ejecuta las celdas en orden. En el paso del dataset, pega el código de descarga de Roboflow.
4. Descarga `fall_model.pt` y guárdalo en la carpeta `models/` del módulo.
5. En `.env`, pon `FALL_MODEL=models/fall_model.pt`.

## El dataset

Hace falta un dataset de imágenes con cajas etiquetadas en **formato YOLO**:

```
dataset/
├── data.yaml        clases y carpetas
├── train/images  train/labels
└── valid/images  valid/labels
```

En [Roboflow Universe](https://universe.roboflow.com/search?q=class%3Afall) hay varios datasets
públicos de caídas. Se descargan en formato *YOLOv11*. El nombre de la clase puede variar (`fall`,
`Fall-Detected`, `fallen`…): el script reconoce cualquier clase que contenga "fall", excepto las
negativas como `no_fall`.

Para que el modelo funcione bien con la cámara del proyecto, conviene agregar fotos propias tomadas
con la Tapo C110 en el lugar donde quedará instalada.

## Desde la terminal

Con una GPU NVIDIA en el computador:

```bash
python training/train_fall_model.py --data ruta/al/data.yaml --epochs 50 --device 0
```

| Parámetro | Por defecto | Significado |
|---|---|---|
| `--epochs` | 50 | Cuántas veces el modelo recorre todo el dataset |
| `--imgsz` | 640 | Tamaño de las imágenes al entrenar |
| `--batch` | 16 | Imágenes por paso; bájalo si la GPU se queda sin memoria |
| `--base-model` | `yolo11n.pt` | Modelo de partida ya entrenado con COCO (*transfer learning*) |
| `--device` | automático | `0` para la primera GPU, `cpu` para el procesador (muy lento) |

Antes de entrenar, el script revisa que el dataset tenga una clase de caída y que haya imágenes en
`train` y `val`. Al terminar copia el mejor modelo a `models/fall_model.pt` y muestra sus métricas:

| Métrica | Significado |
|---|---|
| Precision | De las caídas que marca el modelo, qué parte eran caídas reales |
| Recall | De las caídas reales, qué parte detectó |
| mAP50 | Calidad general de la detección (0 a 1) |

Las gráficas del entrenamiento (`results.png`, matriz de confusión) quedan en la carpeta que el
script muestra al final, porque la ruta dentro de `runs/` cambia según la versión de ultralytics.

## Evaluar videos grabados

Analiza una carpeta de videos con el pipeline real, sin enviar nada al backend y sin zonas:

```bash
python training/evaluate_videos.py grabaciones --output training/results/resultados_antes.csv
```

El CSV (UTF-8 con BOM, para Excel) tiene las columnas `clip`, `evento_esperado`,
`eventos_detectados` y `resultado`. El evento esperado se deduce del nombre del video:

| Prefijo | Qué muestra | Evento esperado |
|---|---|---|
| `caida_al_frente_`, `caida_al_lado_`, `caida_atras_` | Caída real | Caída |
| `caida_silla_`, `desmayo_silla_` | Caída desde una silla | Caída |
| `actividad_anormal_` | La persona lanza objetos | Movimiento anormal |
| `actividad_normal_` | La persona se agacha o se arrodilla | Ninguno |

| Resultado | Significado |
|---|---|
| Acierto | Salió el evento esperado, o no se esperaba ninguno y no salió Caída ni Movimiento anormal |
| No detectado | Se esperaba un evento y no salió, aunque haya salido otro |
| Falsa alarma | En un clip sin evento esperado salió Caída o Movimiento anormal |

La inmovilidad prolongada solo se anota: no cuenta como error.

## Extraer fotos para el dataset

```bash
python training/extract_frames.py grabaciones frames --fps 1
```

Cada foto se llama `<nombre_video>_<numero>.jpg`. Usa OpenCV, así que no necesita ffmpeg.

## Resultados de la línea base

Evaluación de 28 clips sin modelo entrenado (`FALL_MODEL` vacío), en
`results/resultados_antes.csv`:

| Grupo | Clips | Aciertos |
|---|---|---|
| Caídas y desmayo | 21 | 8 |
| Actividad anormal | 6 | 5 |
| Actividad normal | 1 | 1 |
| **Total** | **28** | **14 (50,0 %)** |

Hubo 14 clips no detectados y 0 falsas alarmas según la definición anterior. Aun así, en
`caida_atras_4` salió Movimiento anormal en lugar de Caída; ese clip cuenta como no detectado.

Los umbrales se calibraron con estos mismos clips y solo hay un video de actividad normal, así que
estos resultados son una línea base y no una medida de precisión en situaciones nuevas.

## Resultados de los modelos entrenados

### Caídas

#### Datos de entrenamiento

| Dato | Valor |
|---|---|
| Fotos | Unas 443, propias, tomadas con la Tapo C110 |
| Etiquetado | Roboflow, clases `fall` y `person` |
| División | 310 entrenamiento, 89 validación, 44 prueba |
| Modelo | YOLO11n |
| Entrenamiento | 100 épocas en Google Colab con GPU T4, unos 10 minutos |

#### Métricas de validación

| Métrica | Valor |
|---|---|
| Precision | 0.988 |
| Recall | 1.000 |
| mAP50 | 0.995 |
| mAP50-95 | 0.959 |

Estas métricas son **optimistas**: las fotos de validación vienen de los mismos videos que las de
entrenamiento, así que se parecen mucho entre sí.

#### Prueba con videos nunca vistos

Se separaron 6 videos por video completo, de modo que ninguna de sus fotos se usó para entrenar.

| Configuración | Aciertos | Porcentaje | Falsas alarmas | Archivo |
|---|---|---|---|---|
| Sin modelo | 3 de 6 | 50 % | 0 | `results/prueba_antes.csv` |
| Con modelo | 6 de 6 | 100 % | 0 | `results/prueba_despues.csv` |

**Limitación:** son pocos videos de prueba y solo uno es de actividad normal. Falta medir las falsas
alarmas con más situaciones normales.

### Niño/adulto

| Dato | Valor |
|---|---|
| Dataset | [CHILD ADULT DATASET](https://universe.roboflow.com/vallabha-group-of-companies/child-adult-dataset), de Vallabha Group of Companies, en Roboflow Universe |
| Licencia | MIT |
| División | 559 entrenamiento, 139 validación, 99 prueba |
| Modelo | YOLO11n-cls |
| Entrenamiento | 30 épocas |
| top1_acc | 0.856 |

Con esa exactitud, un solo cuadro puede equivocarse. Por eso el sistema vota con varios cuadros por
persona y solo decide cuando tiene suficientes votos.

```bash
python training/train_child_model.py --data <carpeta> --epochs 30 --device 0
```

### Cómo obtener los modelos

Los archivos `.pt` no están en el repositorio. Para obtenerlos:

| Modelo | Cuaderno | Archivo | Variable en `.env` |
|---|---|---|---|
| Caídas | `train_fall_model.ipynb` | `models/fall_model.pt` | `FALL_MODEL=models/fall_model.pt` |
| Niño/adulto | `train_child_model.ipynb` | `models/child_model.pt` | `CHILD_MODEL=models/child_model.pt` |

No guardes los cuadernos en GitHub con tu clave de Roboflow.
