"""
Módulo para el procesamiento de audio.
"""
import numpy as np
from typing import List, Tuple

# Tipos personalizados
AudioClip = np.ndarray


def split_audio_into_clips(
    waveform: np.ndarray, 
    sample_rate: int, 
    clip_duration_s: float
) -> List[AudioClip]:
    """
    Divide un archivo de audio en clips de duración constante.
    
    Args:
        waveform: Array NumPy con los datos del audio.
        sample_rate: Tasa de muestreo del audio.
        clip_duration_s: Duración deseada para cada clip en segundos.
        
    Returns:
        Lista de clips de audio.
    """
    clip_len_samples = int(clip_duration_s * sample_rate)
    num_clips = int(np.ceil(len(waveform) / clip_len_samples))
    clips = []
    
    for i in range(num_clips):
        start_sample = i * clip_len_samples
        audio_clip = waveform[start_sample : start_sample + clip_len_samples]
        
        # Si el último clip es más corto, rellenamos con ceros
        if len(audio_clip) < clip_len_samples:
            pad_width = clip_len_samples - len(audio_clip)
            audio_clip = np.pad(audio_clip, (0, pad_width), mode="constant")
        
        clips.append(audio_clip)
    
    return clips


def get_clip_times(
    num_clips: int, 
    clip_duration_s: float
) -> List[Tuple[float, float]]:
    """
    Calcula los tiempos de inicio y fin para cada clip.
    
    Args:
        num_clips: Número total de clips.
        clip_duration_s: Duración de cada clip en segundos.
        
    Returns:
        Lista de tuplas (tiempo_inicio, tiempo_fin) para cada clip.
    """
    return [(i * clip_duration_s, (i + 1) * clip_duration_s) for i in range(num_clips)]
