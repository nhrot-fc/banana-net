"""
Utilidades de visualización para datos de audio y análisis de vocalizaciones.

Este módulo contiene funciones para visualizar espectrogramas, datos de
anotaciones y los resultados del cálculo de anchor boxes.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import librosa
import librosa.display
from matplotlib.patches import Rectangle
from matplotlib.colors import TABLEAU_COLORS

from src.banana_net.models.enums import Specie, CallType
from src.banana_net.utils.logger import logger


def visualize_anchor_results(
    all_anchors: Dict[Tuple[Specie, CallType], np.ndarray],
    grouped_dims: Dict[Tuple[Specie, CallType], np.ndarray],
) -> None:
    """
    Visualiza las anchor boxes calculadas junto con las dimensiones de las anotaciones.

    Args:
        all_anchors: Diccionario con las anchor boxes por grupo
        grouped_dims: Diccionario con las dimensiones por grupo
    """
    logger.info("Visualizando resultados de anchor boxes...")
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 8))

    # Colores por especie
    color_keys = list(TABLEAU_COLORS.keys())
    specie_colors = {
        specie: TABLEAU_COLORS[color_keys[i % len(color_keys)]]
        for i, specie in enumerate(Specie)
    }

    # Graficar dimensiones y anchors
    for i, ((specie, call_type), dims) in enumerate(grouped_dims.items()):
        # Obtener anchors para este grupo
        anchors = all_anchors.get((specie, call_type), np.array([]))

        # Color para esta especie
        color = specie_colors[specie]

        # Graficar dimensiones
        alpha = 0.2 if len(dims) > 200 else 0.4
        ax.scatter(
            dims[:, 0],
            dims[:, 1],
            color=color,
            alpha=alpha,
            label=f"{specie.name}-{call_type.name} ({len(dims)} anotaciones)",
            s=20,
        )

        # Graficar anchors
        if len(anchors) > 0:
            ax.scatter(
                anchors[:, 0],
                anchors[:, 1],
                color=color,
                marker="*",
                s=200,
                edgecolor="black",
                linewidth=1.5,
                label=f"{specie.name}-{call_type.name} anchors",
            )

    # Configurar gráfico
    ax.set_xlabel("Duración (segundos)")
    ax.set_ylabel("Ancho de banda (Hz)")
    ax.set_title("Dimensiones de las vocalizaciones y Anchor Boxes por Grupo")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.show()


def plot_spectrogram_with_boxes(
    audio_path: str,
    annotations_df: pd.DataFrame,
    n_fft: int = 2048,
    hop_length: int = 512,
    figsize: Tuple[int, int] = (15, 6),
) -> None:
    """
    Visualiza un espectrograma con cajas que representan las anotaciones.

    Args:
        audio_path: Ruta al archivo de audio
        annotations_df: DataFrame con anotaciones filtradas para este archivo
        n_fft: Tamaño de la ventana FFT
        hop_length: Tamaño del salto entre ventanas FFT
        figsize: Tamaño de la figura (ancho, alto)
    """
    # Cargar audio
    y, sr = librosa.load(audio_path, sr=None)

    # Crear espectrograma
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(abs(S), ref=np.max)

    # Crear figura
    fig, ax = plt.subplots(figsize=figsize)

    # Mostrar espectrograma
    img = librosa.display.specshow(
        S_db,
        x_axis="time",
        y_axis="hz",
        sr=sr,
        hop_length=hop_length,
        ax=ax,
        cmap="viridis",
    )

    # Añadir colorbar
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    # Añadir cajas para cada anotación
    for idx, ann in annotations_df.iterrows():
        width = ann.end_time - ann.begin_time
        height = ann.high_freq - ann.low_freq
        rect = Rectangle(
            (ann.begin_time, ann.low_freq),
            width,
            height,
            linewidth=1.5,
            edgecolor="r",
            facecolor="none",
            alpha=0.8,
        )
        ax.add_patch(rect)

        # Añadir etiqueta
        if hasattr(ann, "specie_enum") and hasattr(ann, "call_type_enum"):
            label = f"{ann.specie_enum.name}-{ann.call_type_enum.name}"
        else:
            label = f"{ann.specie}-{ann.call_type}"

        ax.text(
            ann.begin_time,
            ann.high_freq + 100,
            label,
            fontsize=8,
            color="white",
            bbox=dict(facecolor="black", alpha=0.6, boxstyle="round,pad=0.2"),
        )

    # Ajustar título y etiquetas
    title = f"Espectrograma con anotaciones: {os.path.basename(audio_path)}"
    ax.set_title(title)

    plt.tight_layout()
    plt.show()


def plot_distribution_by_group(df, group_cols=["specie", "call_type"], figsize=(14, 8)):
    """
    Visualiza la distribución de muestras por grupo.

    Args:
        df: DataFrame con los datos
        group_cols: Columnas para agrupar
        figsize: Tamaño de la figura
    """
    # Contar muestras por grupo
    if len(group_cols) == 1:
        counts = df[group_cols[0]].value_counts().sort_values(ascending=False)

        # Crear figura
        fig, ax = plt.subplots(figsize=figsize)

        # Graficar barras
        counts.plot(kind="bar", ax=ax, color="skyblue")

        # Configurar gráfico
        ax.set_title(f"Distribución de muestras por {group_cols[0]}")
        ax.set_xlabel(group_cols[0])
        ax.set_ylabel("Cantidad de muestras")
        ax.grid(axis="y", alpha=0.3)

        # Añadir valores sobre las barras
        for i, v in enumerate(counts):
            ax.text(i, v + 5, str(v), ha="center")

    elif len(group_cols) == 2:
        # Crear un DataFrame de pivot para visualización
        pivot_df = df.groupby(group_cols).size().unstack(fill_value=0)

        # Crear figura
        fig, ax = plt.subplots(figsize=figsize)

        # Graficar heatmap
        im = ax.imshow(pivot_df, cmap="viridis")

        # Configurar gráfico
        ax.set_xticks(np.arange(len(pivot_df.columns)))
        ax.set_yticks(np.arange(len(pivot_df.index)))
        ax.set_xticklabels(pivot_df.columns)
        ax.set_yticklabels(pivot_df.index)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Añadir valores en cada celda
        for i in range(len(pivot_df.index)):
            for j in range(len(pivot_df.columns)):
                value = pivot_df.iloc[i, j]
                text_color = "white" if value > pivot_df.values.max() / 2 else "black"
                ax.text(j, i, str(value), ha="center", va="center", color=text_color)

        ax.set_title(f"Distribución de muestras por {group_cols[0]} y {group_cols[1]}")
        fig.tight_layout()
        fig.colorbar(im, ax=ax, label="Cantidad de muestras")

    else:
        logger.warning("Esta función solo admite hasta 2 columnas de agrupación.")
        return

    plt.tight_layout()
    plt.show()


def plot_spectral_features(
    audio_path, annotations=None, n_fft=2048, hop_length=512, figsize=(15, 12)
):
    """
    Visualiza características espectrales de un archivo de audio.

    Args:
        audio_path: Ruta al archivo de audio
        annotations: DataFrame con anotaciones (opcional)
        n_fft: Tamaño de la ventana FFT
        hop_length: Tamaño del salto entre ventanas FFT
        figsize: Tamaño de la figura
    """
    # Cargar audio
    y, sr = librosa.load(audio_path, sr=None)

    # Crear figura con subplots
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # 1. Espectrograma
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(abs(S), ref=np.max)

    img1 = librosa.display.specshow(
        S_db,
        x_axis="time",
        y_axis="hz",
        sr=sr,
        hop_length=hop_length,
        ax=axes[0],
        cmap="viridis",
    )
    axes[0].set_title("Espectrograma")
    fig.colorbar(img1, ax=axes[0], format="%+2.0f dB")

    # Añadir cajas si hay anotaciones
    if annotations is not None:
        for idx, ann in annotations.iterrows():
            width = ann.end_time - ann.begin_time
            height = ann.high_freq - ann.low_freq
            rect = Rectangle(
                (ann.begin_time, ann.low_freq),
                width,
                height,
                linewidth=1.5,
                edgecolor="r",
                facecolor="none",
                alpha=0.8,
            )
            axes[0].add_patch(rect)

    # 2. Mel-espectrograma
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    M_db = librosa.power_to_db(M, ref=np.max)

    img2 = librosa.display.specshow(
        M_db,
        x_axis="time",
        y_axis="mel",
        sr=sr,
        hop_length=hop_length,
        ax=axes[1],
        cmap="magma",
    )
    axes[1].set_title("Mel-Espectrograma")
    fig.colorbar(img2, ax=axes[1], format="%+2.0f dB")

    # 3. Cromograma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)

    img3 = librosa.display.specshow(
        chroma,
        x_axis="time",
        y_axis="chroma",
        sr=sr,
        hop_length=hop_length,
        ax=axes[2],
        cmap="coolwarm",
    )
    axes[2].set_title("Cromograma")
    fig.colorbar(img3, ax=axes[2])

    plt.tight_layout()
    plt.show()
