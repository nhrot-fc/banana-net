"""
Módulo para el procesamiento de espectrogramas.
"""
import numpy as np
import cv2
import librosa
from typing import Tuple

# Tipos personalizados
Spectrogram = np.ndarray


def audio_to_spectrogram(
    audio_clip: np.ndarray, 
    n_fft: int = 2048, 
    hop_length: int = 512
) -> Spectrogram:
    """
    Convierte un clip de audio en un espectrograma.
    
    Args:
        audio_clip: Array NumPy con los datos del clip de audio.
        n_fft: Tamaño de la ventana para la transformada de Fourier.
        hop_length: Tamaño del salto entre ventanas consecutivas.
        
    Returns:
        Espectrograma del audio como matriz NumPy.
    """
    return np.abs(
        librosa.stft(audio_clip, n_fft=n_fft, hop_length=hop_length)
    )


def resize_spectrogram(
    spectrogram: Spectrogram, 
    target_size: Tuple[int, int]
) -> Spectrogram:
    """
    Redimensiona un espectrograma a un tamaño específico.
    
    Args:
        spectrogram: Espectrograma original.
        target_size: Tamaño objetivo como (ancho, alto).
        
    Returns:
        Espectrograma redimensionado.
    """
    return cv2.resize(
        spectrogram,
        dsize=target_size,  # dsize=(width, height)
        interpolation=cv2.INTER_AREA,
    )
