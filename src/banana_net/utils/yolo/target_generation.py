"""
Módulo para la generación de targets YOLO a partir de anotaciones.
"""
import numpy as np
from typing import List, Tuple, cast

# Importamos las clases Enum y Annotation
from src.banana_net.models.enums import Specie
from src.banana_net.models.annotation import Annotation

# Tipos personalizados
YoloTarget = np.ndarray
YoloTargetsTuple = Tuple[YoloTarget, YoloTarget, YoloTarget]


def generate_yolo_targets_for_clip(
    annotations_in_clip: List[Annotation],
    sample_rate: int,
    clip_duration_s: float,
    clip_dims: Tuple[int, int],
    strides: Tuple[int, ...],
    reg_max: int,
) -> YoloTargetsTuple:
    """
    Genera los tensores YOLO targets para un clip de audio con anotaciones.
    
    Args:
        annotations_in_clip: Lista de anotaciones en el clip.
        sample_rate: Tasa de muestreo del audio original.
        clip_duration_s: Duración del clip en segundos.
        clip_dims: Dimensiones del espectrograma (frecuencia, tiempo).
        strides: Tupla con los valores de stride para cada nivel de detección.
        reg_max: Valor máximo para la codificación de regresión.
        
    Returns:
        Una tupla con los tensores target para cada nivel de detección.
    """
    freq_bins, time_frames = clip_dims
    num_classes = Specie.count()
    max_freq_hz = sample_rate / 2  # La frecuencia máxima del espectrograma

    targets: List[YoloTarget] = []
    for stride in strides:
        grid_h, grid_w = freq_bins // stride, time_frames // stride
        target_shape = (grid_h, grid_w, 4 * reg_max + num_classes)
        targets.append(np.zeros(target_shape, dtype=np.float32))

    for ann in annotations_in_clip:
        # Normalización de las coordenadas
        w_norm = (ann.end_time - ann.begin_time) / clip_duration_s
        h_norm = (ann.high_freq - ann.low_freq) / max_freq_hz
        x_center_norm = ((ann.begin_time + ann.end_time) / 2) / clip_duration_s
        y_center_norm = ((ann.low_freq + ann.high_freq) / 2) / max_freq_hz

        for i, stride in enumerate(strides):
            grid_h, grid_w = targets[i].shape[:2]
            grid_x, grid_y = int(grid_w * x_center_norm), int(grid_h * y_center_norm)

            if 0 <= grid_y < grid_h and 0 <= grid_x < grid_w:
                class_vector_offset = 4 * reg_max
                targets[i][grid_y, grid_x, class_vector_offset:] = Specie.to_onehot(
                    ann.specie
                )
                # La lógica de la caja (DFL) iría aquí

    return cast(YoloTargetsTuple, tuple(targets))


def filter_annotations_for_clip(
    annotations: List[Annotation],
    clip_start_time: float,
    clip_duration_s: float
) -> List[Annotation]:
    """
    Filtra y ajusta las anotaciones para un clip específico de audio.
    
    Args:
        annotations: Lista de anotaciones originales.
        clip_start_time: Tiempo de inicio del clip en segundos.
        clip_duration_s: Duración del clip en segundos.
        
    Returns:
        Lista de anotaciones ajustadas para el clip específico.
    """
    clip_end_time = clip_start_time + clip_duration_s
    annotations_in_clip = []
    
    for ann in annotations:
        if ann.begin_time < clip_end_time and ann.end_time > clip_start_time:
            new_begin = max(0, ann.begin_time - clip_start_time)
            new_end = min(clip_duration_s, ann.end_time - clip_start_time)
            if new_begin < new_end:
                annotations_in_clip.append(
                    Annotation(
                        begin_time=new_begin,
                        end_time=new_end,
                        low_freq=ann.low_freq,
                        high_freq=ann.high_freq,
                        specie=ann.specie,
                    )
                )
    
    return annotations_in_clip
