"""
Utilidades para aumento de datos (data augmentation) en audio.

Este módulo contiene funciones para generar transformaciones en señales
de audio que permiten aumentar el conjunto de datos y mejorar la
robustez de los modelos.
"""

import numpy as np
import librosa


def apply_data_augmentation(
    waveform: np.ndarray, sample_rate: int, augmentation_type: str = "random", **kwargs
) -> np.ndarray:
    """
    Aplica transformaciones de aumento de datos a una forma de onda de audio.

    Args:
        waveform: Array numpy con la forma de onda
        sample_rate: Tasa de muestreo
        augmentation_type: Tipo de aumento ('time_stretch', 'pitch_shift', 'add_noise', 'random')
        **kwargs: Parámetros adicionales para las funciones de aumento

    Returns:
        Forma de onda aumentada
    """
    # Crear copia para evitar modificar la original
    y = waveform.copy()

    # Por defecto, elegir una transformación aleatoria
    if augmentation_type == "random":
        import random

        choices = ["time_stretch", "pitch_shift", "add_noise", "none"]
        weights = [0.3, 0.3, 0.3, 0.1]  # 10% de probabilidad de no hacer nada
        augmentation_type = random.choices(choices, weights=weights, k=1)[0]

    # Aplicar la transformación elegida
    if augmentation_type == "time_stretch":
        # Estirar/comprimir en tiempo sin cambiar el tono
        rate = kwargs.get("rate", np.random.uniform(0.8, 1.2))
        y = librosa.effects.time_stretch(y, rate=rate)

    elif augmentation_type == "pitch_shift":
        # Cambiar el tono sin cambiar el tiempo
        n_steps = kwargs.get("n_steps", np.random.uniform(-2.0, 2.0))
        y = librosa.effects.pitch_shift(y, sr=sample_rate, n_steps=n_steps)

    elif augmentation_type == "add_noise":
        # Añadir ruido gaussiano
        noise_factor = kwargs.get("noise_factor", np.random.uniform(0.001, 0.01))
        noise = np.random.randn(len(y))
        y = y + noise_factor * noise
        # Asegurar que los valores están en rango razonable
        y = np.clip(y, -1.0, 1.0)

    # Si es 'none' o cualquier otro valor, devolver la forma de onda original
    return y


def time_shift(waveform: np.ndarray, shift_factor: float = None) -> np.ndarray:
    """
    Aplica un desplazamiento temporal a la forma de onda.

    Args:
        waveform: Array numpy con la forma de onda
        shift_factor: Factor de desplazamiento como fracción de la longitud
                     (valor negativo desplaza a la izquierda, positivo a la derecha)
                     Si es None, se genera un valor aleatorio entre -0.2 y 0.2

    Returns:
        Forma de onda desplazada
    """
    # Crear copia para evitar modificar la original
    y = waveform.copy()

    # Si no se especifica el factor, generar uno aleatorio
    if shift_factor is None:
        shift_factor = np.random.uniform(-0.2, 0.2)

    # Calcular el desplazamiento en muestras
    shift = int(len(y) * shift_factor)

    # Aplicar el desplazamiento
    if shift > 0:
        # Desplazar a la derecha
        y = np.pad(y[:-shift], (shift, 0), mode="constant")
    else:
        # Desplazar a la izquierda
        y = np.pad(y[-shift:], (0, -shift), mode="constant")

    return y


def frequency_mask(spectrogram: np.ndarray, mask_factor: float = None) -> np.ndarray:
    """
    Aplica una máscara de frecuencia al espectrograma.

    Args:
        spectrogram: Espectrograma como array numpy de forma [freq, time]
        mask_factor: Factor de la máscara como fracción de las frecuencias
                    Si es None, se genera un valor aleatorio entre 0.05 y 0.2

    Returns:
        Espectrograma con máscara
    """
    # Crear copia para evitar modificar el original
    spec = spectrogram.copy()

    # Si no se especifica el factor, generar uno aleatorio
    if mask_factor is None:
        mask_factor = np.random.uniform(0.05, 0.2)

    # Obtener dimensiones
    freq_bins, time_frames = spec.shape

    # Calcular tamaño de la máscara
    mask_size = int(freq_bins * mask_factor)

    # Generar posición aleatoria para la máscara
    mask_start = np.random.randint(0, freq_bins - mask_size)

    # Aplicar la máscara
    spec[mask_start : mask_start + mask_size, :] = 0

    return spec


def time_mask(spectrogram: np.ndarray, mask_factor: float = None) -> np.ndarray:
    """
    Aplica una máscara temporal al espectrograma.

    Args:
        spectrogram: Espectrograma como array numpy de forma [freq, time]
        mask_factor: Factor de la máscara como fracción del tiempo
                    Si es None, se genera un valor aleatorio entre 0.05 y 0.2

    Returns:
        Espectrograma con máscara
    """
    # Crear copia para evitar modificar el original
    spec = spectrogram.copy()

    # Si no se especifica el factor, generar uno aleatorio
    if mask_factor is None:
        mask_factor = np.random.uniform(0.05, 0.2)

    # Obtener dimensiones
    freq_bins, time_frames = spec.shape

    # Calcular tamaño de la máscara
    mask_size = int(time_frames * mask_factor)

    # Generar posición aleatoria para la máscara
    mask_start = np.random.randint(0, time_frames - mask_size)

    # Aplicar la máscara
    spec[:, mask_start : mask_start + mask_size] = 0

    return spec


def spec_augment(
    spectrogram: np.ndarray, frequency_masks: int = 1, time_masks: int = 1
) -> np.ndarray:
    """
    Aplica SpecAugment al espectrograma (combinación de máscaras de frecuencia y tiempo).

    Esta técnica ha mostrado buenos resultados en tareas de reconocimiento de voz.

    Args:
        spectrogram: Espectrograma como array numpy de forma [freq, time]
        frequency_masks: Número de máscaras de frecuencia a aplicar
        time_masks: Número de máscaras temporales a aplicar

    Returns:
        Espectrograma aumentado
    """
    # Crear copia para evitar modificar el original
    spec = spectrogram.copy()

    # Aplicar máscaras de frecuencia
    for _ in range(frequency_masks):
        spec = frequency_mask(spec)

    # Aplicar máscaras temporales
    for _ in range(time_masks):
        spec = time_mask(spec)

    return spec


def apply_augmentation_pipeline(
    waveform: np.ndarray,
    sample_rate: int,
    augmentation_types: list = None,
    probability: float = 0.5,
) -> np.ndarray:
    """
    Aplica una secuencia de aumentos con cierta probabilidad.

    Args:
        waveform: Array numpy con la forma de onda
        sample_rate: Tasa de muestreo
        augmentation_types: Lista de tipos de aumento a aplicar
                           Si es None, se usarán ['time_stretch', 'pitch_shift', 'add_noise']
        probability: Probabilidad de aplicar cada aumento

    Returns:
        Forma de onda aumentada
    """
    # Crear copia para evitar modificar la original
    y = waveform.copy()

    # Si no se especifican tipos, usar los predeterminados
    if augmentation_types is None:
        augmentation_types = ["time_stretch", "pitch_shift", "add_noise", "time_shift"]

    # Aplicar cada aumento con la probabilidad indicada
    for aug_type in augmentation_types:
        if np.random.random() < probability:
            y = apply_data_augmentation(y, sample_rate, augmentation_type=aug_type)

    return y
