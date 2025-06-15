import numpy as np
import torch
from typing import List, Tuple
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from src.banana_net.models.enums import Specie
from src.banana_net.models.annotation import Annotation


# ==============================================================================
# 2. FUNCIONES ADAPTADAS PARA YOLOv2
# ==============================================================================


def iou(box1_wh: Tuple[float, float], box2_wh: Tuple[float, float]) -> float:
    """
    Calcula el Intersection over Union (IoU) entre dos cajas centradas en el origen.

    El IoU mide la superposición entre dos cuadros delimitadores y se utiliza para determinar
    qué ancla es más adecuada para una anotación específica.

    Args:
        box1_wh: Tupla (ancho, alto) de la primera caja.
        box2_wh: Tupla (ancho, alto) de la segunda caja.

    Returns:
        float: Valor IoU entre 0 (sin superposición) y 1 (superposición perfecta).
    """
    w1, h1 = box1_wh
    w2, h2 = box2_wh

    # Calculamos el área de la intersección
    inter_area = min(w1, w2) * min(h1, h2)

    # Calculamos el área de la unión
    union_area = (w1 * h1) + (w2 * h2) - inter_area

    # Evitamos división por cero
    if union_area < 1e-6:
        return 0.0

    return inter_area / union_area


def create_yolov2_target(
    waveform: np.ndarray,
    sample_rate: int,
    annotations: List[Annotation],
    anchors: np.ndarray,
    S: int = 13,  # YOLOv2 usa una cuadrícula más pequeña por defecto
    n_fft: int = 2048,
    hop_length: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforma un audio y sus anotaciones en el tensor objetivo para entrenamiento de YOLOv2.

    Esta función procesa un archivo de audio y sus anotaciones para crear un tensor que sirve
    como objetivo (ground truth) para entrenar una red YOLOv2 para detección de vocalizaciones
    de primates en espectrogramas.

    Args:
        waveform: Array de numpy con la forma de onda del audio.
        sample_rate: Frecuencia de muestreo del audio en Hz.
        annotations: Lista de objetos Annotation con las anotaciones de eventos acústicos.
        anchors: Array de numpy con las anclas predefenidas de forma [num_anchors, 2].
                 Cada ancla es un par (width, height) normalizado.
        S: Tamaño de la cuadrícula (grid) para YOLOv2. Por defecto es 13x13.
        n_fft: Tamaño de la ventana FFT para generar el espectrograma.
        hop_length: Tamaño del salto entre ventanas FFT consecutivas.

    Returns:
        Tupla con dos elementos:
        - espectrograma (np.ndarray): Matriz con el espectrograma del audio.
        - tensor_objetivo (np.ndarray): Tensor 4D de forma [S, S, num_anchors, 5+num_classes]
          donde 5 corresponde a [x, y, w, h, objectness] y num_classes es el número de especies.
    """
    # Generamos el espectrograma usando STFT (Short-Time Fourier Transform)
    s_complex = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(s_complex)
    freq_bins, time_frames = spectrogram.shape

    # Obtenemos el número de anclas y clases
    num_anchors = anchors.shape[0]
    num_classes = Specie.count()

    # Creamos un tensor 4D para el objetivo de YOLOv2: [grid_y, grid_x, ancla, atributos]
    # donde atributos son: [x_offset, y_offset, width_transform, height_transform, objectness, one_hot_classes]
    target_tensor = np.zeros((S, S, num_anchors, 5 + num_classes), dtype=np.float32)

    for annotation in annotations:
        # 1. Convertimos las coordenadas del evento acústico a formato normalizado (0-1)

        # Ancho y alto normalizados del evento
        width_norm = (
            (annotation.end_time - annotation.begin_time) * sample_rate / hop_length
        ) / time_frames
        height_norm = (
            (annotation.high_freq - annotation.low_freq) * n_fft / sample_rate
        ) / freq_bins

        # 2. Encontramos la mejor ancla para esta anotación (mayor IoU)
        # Comparamos la forma de la anotación con cada una de las anclas predefinidas
        ious = [
            iou((width_norm, height_norm), (anchor[0], anchor[1])) for anchor in anchors
        ]
        anchor_idx = int(np.argmax(ious))
        best_iou = ious[anchor_idx]

        # 3. Calculamos el centro normalizado del evento
        x_center_norm = (
            (annotation.begin_time + annotation.end_time) / 2 * sample_rate / hop_length
        ) / time_frames
        y_center_norm = (
            (annotation.low_freq + annotation.high_freq) / 2 * n_fft / sample_rate
        ) / freq_bins

        # 4. Identificamos la celda de la cuadrícula que contiene el centro
        grid_x = int(S * x_center_norm)
        grid_y = int(S * y_center_norm)

        # Verificamos que las coordenadas estén dentro de los límites válidos
        if not (0 <= grid_x < S and 0 <= grid_y < S):
            continue  # Ignoramos anotaciones fuera de los límites

        # 5. Calculamos los desplazamientos (offsets) desde la esquina de la celda
        # Estos valores deberían estar entre 0 y 1
        x_cell = float(S * x_center_norm - grid_x)
        y_cell = float(S * y_center_norm - grid_y)

        # 6. Calculamos las transformaciones de ancho y alto según el paper YOLOv2
        # tw = log(width_actual / width_ancla)
        # th = log(height_actual / height_ancla)
        anchor_w, anchor_h = anchors[anchor_idx]

        # Evitamos problemas con valores negativos o cero
        eps = 1e-7
        tw = float(np.log((width_norm / anchor_w) + eps))
        th = float(np.log((height_norm / anchor_h) + eps))

        # 7. Llenamos el tensor con los valores calculados
        # [x_offset, y_offset, width_transform, height_transform, objectness]
        target_tensor[grid_y, grid_x, anchor_idx, 0:5] = np.array(
            [
                x_cell,  # Desplazamiento x dentro de la celda (0-1)
                y_cell,  # Desplazamiento y dentro de la celda (0-1)
                tw,  # Transformación de ancho relativo a la ancla
                th,  # Transformación de alto relativo a la ancla
                1.0,  # Objectness (confianza de que hay un objeto)
            ],
            dtype=np.float32,
        )

        # 8. Codificamos la clase (especie) como vector one-hot
        # Utilizamos el método de EnhancedEnum para obtener la codificación one-hot
        target_tensor[grid_y, grid_x, anchor_idx, 5:] = Specie.to_onehot(
            annotation.specie
        )

    # Retornamos tanto el espectrograma como el tensor objetivo
    return spectrogram, target_tensor


def plot_yolov2_verification(
    spectrogram: np.ndarray,
    target_tensor: np.ndarray,
    anchors: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    figsize: Tuple[int, int] = (15, 6),
) -> Tuple[Figure, Axes]:
    """
    Visualiza un espectrograma y dibuja las cajas desde el tensor objetivo de YOLOv2.

    Esta función permite verificar visualmente la correcta transformación de las anotaciones
    al formato YOLOv2, dibujando las cajas delimitadoras sobre el espectrograma.

    Args:
        spectrogram: El espectrograma como matriz numpy.
        target_tensor: El tensor objetivo YOLOv2 de forma [S, S, num_anchors, 5+num_classes].
        anchors: Array de numpy con las anclas predefenidas de forma [num_anchors, 2].
        sample_rate: Frecuencia de muestreo del audio en Hz.
        n_fft: Tamaño de la ventana FFT utilizada para el espectrograma.
        hop_length: Tamaño del salto entre ventanas FFT.
        figsize: Tamaño de la figura a generar como tupla (ancho, alto).

    Returns:
        Tupla con la figura y los ejes de matplotlib para permitir personalizaciones adicionales.
    """
    # Obtenemos dimensiones del tensor y espectrograma
    S, _, num_anchors, _ = target_tensor.shape
    freq_bins, time_frames = spectrogram.shape

    # Convertimos el espectrograma a decibelios para mejor visualización
    S_db = librosa.amplitude_to_db(spectrogram, ref=np.max)

    # Creamos la figura y ejes
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Mostramos el espectrograma
    librosa.display.specshow(
        S_db,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        ax=ax,
        cmap="magma",
    )

    # Encontramos todas las celdas con predicciones activas (objectness == 1)
    responsible_predictors = np.argwhere(target_tensor[..., 4] == 1)

    # Colores para diferentes especies
    species_colors = {
        Specie.AA: "cyan",
        Specie.AC: "lime",
        Specie.AS: "yellow",
        Specie.CC: "magenta",
        Specie.LW: "red",
        Specie.PT: "blue",
        Specie.SB: "white",
        Specie.SM: "orange",
    }

    # Para cada predicción activa
    for predictor in responsible_predictors:
        grid_y, grid_x, anchor_idx = predictor

        # Extraemos los parámetros de la caja y las probabilidades de clase
        target_vector = target_tensor[grid_y, grid_x, anchor_idx]
        box_params = target_vector[:5]  # [x, y, tw, th, objectness]
        class_probs = target_vector[5:]  # Vector one-hot de especies

        # Extraemos los valores individuales de los parámetros
        x_cell, y_cell, tw, th, objectness = box_params

        # Convertimos el vector one-hot a un objeto enum Specie
        specie_enum = Specie.from_onehot(class_probs)

        # 1. Convertimos las coordenadas relativas a coordenadas normalizadas
        # Las coordenadas del centro del evento
        x_center_norm = (grid_x + x_cell) / S
        y_center_norm = (grid_y + y_cell) / S

        # 2. De-transformamos las dimensiones usando la función inversa de la transformación YOLOv2
        anchor_w, anchor_h = anchors[anchor_idx]
        width_norm = anchor_w * np.exp(tw)  # ancho = ancho_ancla * e^tw
        height_norm = anchor_h * np.exp(th)  # alto = alto_ancla * e^th

        # 3. Convertimos de coordenadas normalizadas a coordenadas reales
        # De normalizado a segundos (eje x) y Hz (eje y)
        start_time = (
            ((x_center_norm - width_norm / 2) * time_frames) * hop_length / sample_rate
        )
        end_time = (
            ((x_center_norm + width_norm / 2) * time_frames) * hop_length / sample_rate
        )
        low_freq = ((y_center_norm - height_norm / 2) * freq_bins) * sample_rate / n_fft
        high_freq = (
            ((y_center_norm + height_norm / 2) * freq_bins) * sample_rate / n_fft
        )

        # Calculamos ancho y alto en segundos y Hz
        width_sec = end_time - start_time
        height_hz = high_freq - low_freq

        # Obtenemos el color para la especie actual
        color = species_colors.get(specie_enum, "cyan")

        # Dibujamos el rectángulo que representa la predicción
        rect = patches.Rectangle(
            (start_time, low_freq),  # Esquina inferior izquierda
            width_sec,
            height_hz,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
            linestyle="--",
            alpha=0.8,
        )
        ax.add_patch(rect)

        # Añadimos texto con información sobre la especie y la celda
        confidence = f"{objectness:.2f}" if isinstance(objectness, float) else "1.00"
        # Texto: Cuadricula (X,Y), Ancla, Especie (Confianza)
        label_text = (
            f"Cuadricula ({grid_x}, {grid_y}), Ancla {anchor_idx}, "
            f"{specie_enum.name} ({confidence})"
        )
        ax.text(
            start_time,
            low_freq - 200,
            label_text,
            color=color,
            fontsize=10,
            weight="bold",
            bbox=dict(facecolor="black", alpha=0.5, pad=1),
        )

    # Configuramos título y leyenda
    title = "Verificación Visual del Tensor Objetivo YOLOv2"
    ax.set_title(title, fontsize=12)

    # Añadimos información sobre la cuadrícula
    grid_info = f"Cuadrícula: {S}x{S}, Anclas: {num_anchors}"
    ax.text(
        0.02,
        0.02,
        grid_info,
        transform=ax.transAxes,
        fontsize=10,
        color="white",
        bbox=dict(facecolor="black", alpha=0.5),
    )

    # Ajustamos la figura
    plt.tight_layout()

    # Retornamos la figura y los ejes para permitir personalización adicional
    return fig, ax
