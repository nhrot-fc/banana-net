"""
Módulo para la conversión de audio a datos de entrenamiento YOLO.
"""
import numpy as np
from typing import List, Tuple

from src.banana_net.models.annotation import Annotation
from src.banana_net.utils.audio.processing import split_audio_into_clips, get_clip_times
from src.banana_net.utils.spectrogram.processing import audio_to_spectrogram, resize_spectrogram
from src.banana_net.utils.yolo.target_generation import (
    generate_yolo_targets_for_clip, 
    filter_annotations_for_clip
)

# Tipos personalizados
Spectrogram = np.ndarray
YoloTarget = np.ndarray
YoloTargetsTuple = Tuple[YoloTarget, YoloTarget, YoloTarget]


def convert_audio_to_yolo_training_data(
    waveform: np.ndarray,
    sample_rate: int,
    annotations: List[Annotation],
    clip_duration_s: float = 10.0,
    input_size_wh: Tuple[int, int] = (640, 640),
    n_fft: int = 2048,
    hop_length: int = 512,
    strides: Tuple[int, int, int] = (8, 16, 32),
    reg_max: int = 16,
) -> List[Tuple[Spectrogram, YoloTargetsTuple]]:
    """
    Convierte un audio y sus anotaciones en datos de entrenamiento para YOLOv8.
    
    Args:
        waveform: Array NumPy con los datos del audio.
        sample_rate: Tasa de muestreo del audio.
        annotations: Lista de anotaciones para el audio.
        clip_duration_s: Duración de cada clip en segundos.
        input_size_wh: Dimensiones objetivo del espectrograma como (ancho, alto).
        n_fft: Tamaño de la ventana para la transformada de Fourier.
        hop_length: Tamaño del salto entre ventanas consecutivas.
        strides: Valores de stride para cada nivel de detección YOLO.
        reg_max: Valor máximo para la codificación de regresión.
        
    Returns:
        Lista de tuplas (espectrograma, targets_yolo) para cada clip.
    """
    # Dividir el audio en clips
    audio_clips = split_audio_into_clips(waveform, sample_rate, clip_duration_s)
    clip_times = get_clip_times(len(audio_clips), clip_duration_s)
    processed_data = []

    for i, (audio_clip, (clip_start_time, _)) in enumerate(zip(audio_clips, clip_times)):
        # Generar espectrograma para el clip actual
        spectrogram_clip = audio_to_spectrogram(audio_clip, n_fft, hop_length)
        
        # Filtrar y ajustar anotaciones para el clip actual
        annotations_in_clip = filter_annotations_for_clip(
            annotations, 
            clip_start_time, 
            clip_duration_s
        )
        
        # Generar los targets YOLO para el clip
        target_tensors = generate_yolo_targets_for_clip(
            annotations_in_clip=annotations_in_clip,
            sample_rate=sample_rate,
            clip_duration_s=clip_duration_s,
            clip_dims=spectrogram_clip.shape,
            strides=strides,
            reg_max=reg_max,
        )
        
        # Redimensionar el espectrograma
        resized_spectrogram = resize_spectrogram(
            spectrogram_clip, 
            target_size=input_size_wh
        )
        
        processed_data.append((resized_spectrogram, target_tensors))

    return processed_data
