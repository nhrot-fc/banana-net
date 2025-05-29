# Audio Clip Processing and YOLO Transformation

Este módulo proporciona utilidades para procesar grabaciones de audio largas dividiéndolas en clips más pequeños con un overlap determinado, y transformar las anotaciones de estos clips al formato tensorial YOLO para entrenamiento de modelos de detección.

## Funcionalidad Principal

El módulo proporciona las siguientes funcionalidades clave:

1. **División en clips con overlap**: Divide archivos de audio largos en clips de duración fija con un overlap configurable.
2. **Alineación de anotaciones**: Filtra y ajusta las anotaciones para cada clip.
3. **Transformación a formato YOLO**: Convierte las anotaciones a tensores en formato YOLO para usar con modelos de detección de objetos.

## Funciones Principales

### `generate_clips_metadata(total_duration_sec, clip_duration=5.0, overlap=1.0)`

Genera metadatos para los clips de audio con overlap.

**Parámetros:**
- `total_duration_sec`: Duración total del archivo de audio en segundos.
- `clip_duration`: Duración de cada clip en segundos (default: 5.0).
- `overlap`: Overlap entre clips consecutivos en segundos (default: 1.0).

**Retorna:**
- Lista de tuplas con formato `(clip_id, start_time, end_time)`.

### `annotations_in_clip(annotations_df, clip_start, clip_end)`

Filtra las anotaciones que se superponen con el intervalo de tiempo del clip y ajusta los tiempos relativos al inicio del clip.

**Parámetros:**
- `annotations_df`: DataFrame con las anotaciones.
- `clip_start`: Tiempo de inicio del clip en segundos.
- `clip_end`: Tiempo de fin del clip en segundos.

**Retorna:**
- DataFrame filtrado con columnas adicionales `begin_time_clip` y `end_time_clip`.

### `create_yolo_tensor_for_clip(annotations_clip, clip_duration, max_freq_hz, S=7, B=2, class_map=None)`

Crea un tensor YOLO para un clip de audio.

**Parámetros:**
- `annotations_clip`: DataFrame con anotaciones para el clip.
- `clip_duration`: Duración del clip en segundos.
- `max_freq_hz`: Frecuencia máxima en Hz para el espectrograma.
- `S`: Tamaño de la cuadrícula YOLO (SxS).
- `B`: Número de bounding boxes por celda de la cuadrícula.
- `class_map`: Diccionario que mapea tuplas `(species, call_type)` a índices de clase.

**Retorna:**
- Tensor objetivo YOLO con forma `(S, S, B*5 + C)`, donde C es el número de clases.

### `process_dataset_to_clips(df, clip_duration=5.0, overlap=1.0, max_freq_hz=16000.0, S=7, B=2, class_map=None)`

Procesa un conjunto de datos completo en clips y tensores YOLO.

**Parámetros:**
- `df`: DataFrame con todas las anotaciones.
- `clip_duration`: Duración de cada clip en segundos.
- `overlap`: Overlap entre clips consecutivos en segundos.
- `max_freq_hz`: Frecuencia máxima en Hz para el espectrograma.
- `S`: Tamaño de la cuadrícula YOLO.
- `B`: Número de bounding boxes por celda.
- `class_map`: Diccionario que mapea tuplas `(species, call_type)` a índices de clase.

**Retorna:**
- `all_clips`: Lista de diccionarios con información de los clips para todos los archivos.
- `all_tensors`: Diccionario que mapea nombres de clips a tensores YOLO para todos los archivos.

## Ejemplo de Uso

```python
import pandas as pd
from banana_net.utils.audio_clip_processing import process_dataset_to_clips

# Configuración
CLIP_DURATION = 5.0  # segundos
OVERLAP_DURATION = 1.0  # segundos
MAX_FREQUENCY_HZ = 16000.0  # Hz
S = 7  # Tamaño de la cuadrícula
B = 2  # Número de bounding boxes por celda
CLASS_MAP = {
    ("lw", "cs"): 0,
    ("lw", "cc"): 1,
    ("lw", "tr"): 2
}

# Cargar datos
annotations_df = pd.read_csv('path/to/your/annotations.csv')

# Procesar en clips y tensores YOLO
clip_dataset, clip_yolo_tensors = process_dataset_to_clips(
    annotations_df, 
    clip_duration=CLIP_DURATION, 
    overlap=OVERLAP_DURATION, 
    max_freq_hz=MAX_FREQUENCY_HZ, 
    S=S, 
    B=B, 
    class_map=CLASS_MAP
)

# Ahora puedes usar clip_dataset y clip_yolo_tensors para entrenamiento o evaluación
```

## Notas Importantes

1. El módulo asume que las anotaciones incluyen coordenadas de tiempo (`begin_time`, `end_time`) y frecuencia (`low_freq`, `high_freq`).
2. Las anotaciones deben tener columnas `species` y `call_type` para la clasificación.
3. Se requiere un mapeo `class_map` que convierta pares `(species, call_type)` a índices de clase.
4. El módulo está diseñado para trabajar con espectrogramas, donde el eje X representa el tiempo y el eje Y representa la frecuencia.
