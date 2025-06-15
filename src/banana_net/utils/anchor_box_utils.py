"""
Utilidades para el cálculo y manipulación de anchor boxes.

Este módulo contiene funciones para el cálculo de anchor boxes óptimas
para la detección de vocalizaciones de primates con algoritmos YOLO.
"""

import numpy as np
from typing import Dict, Tuple
import pandas as pd

from src.banana_net.models.enums import Specie, CallType
from src.banana_net.utils.data_preprocessing import (
    preprocess_annotations,
    extract_dimensions_by_group,
)
from src.banana_net.utils.logger import logger


def iou(box_wh: np.ndarray, cluster_wh: np.ndarray) -> np.ndarray:
    """
    Calcula la Intersección sobre Unión (IoU) para un conjunto de cajas y clusters.

    Args:
        box_wh: Array de forma (N, 2) con anchuras y alturas de cajas
        cluster_wh: Array de forma (K, 2) con anchuras y alturas de clusters

    Returns:
        np.ndarray: Matriz de IoUs de forma (N, K)
    """
    # Expandir dimensiones para permitir broadcasting
    box_wh = np.expand_dims(box_wh, axis=1)  # (N, 1, 2)
    cluster_wh = np.expand_dims(cluster_wh, axis=0)  # (1, K, 2)

    # Calcular el área de cada caja y cluster
    box_area = box_wh[:, :, 0] * box_wh[:, :, 1]  # (N, 1)
    cluster_area = cluster_wh[:, :, 0] * cluster_wh[:, :, 1]  # (1, K)

    # Calcular la intersección
    min_w = np.minimum(box_wh[:, :, 0], cluster_wh[:, :, 0])  # (N, K)
    min_h = np.minimum(box_wh[:, :, 1], cluster_wh[:, :, 1])  # (N, K)
    intersection = min_w * min_h  # (N, K)

    # Calcular la unión
    union = box_area + cluster_area - intersection  # (N, K)

    # Evitar división por cero
    union = np.maximum(union, 1e-10)

    # Calcular IoU
    return intersection / union  # (N, K)


def calculate_anchors_for_group(box_dims: np.ndarray, num_anchors: int) -> np.ndarray:
    """
    Calcula las anchor boxes óptimas para un grupo de dimensiones usando k-means.

    Esta función implementa un algoritmo de agrupación k-means adaptado para
    encontrar las anchor boxes que mejor representan las dimensiones de las anotaciones.

    Args:
        box_dims: Array de dimensiones [N, 2] con (duración_s, bandwidth_hz)
        num_anchors: Número de anchor boxes a calcular

    Returns:
        np.ndarray: Array de dimensiones [num_anchors, 2] con las anchor boxes óptimas
    """
    # Si hay menos anotaciones que anchor boxes solicitadas, retornar las anotaciones
    if len(box_dims) <= num_anchors:
        logger.warning(f"Solo {len(box_dims)} anotaciones disponibles, pero se pidieron {num_anchors} anchor boxes")
        # Rellenar con ceros si es necesario
        if len(box_dims) < num_anchors:
            padding = np.zeros((num_anchors - len(box_dims), 2))
            return np.vstack([box_dims, padding])
        return box_dims.copy()

    # Inicializar centroides aleatoriamente
    indices = np.random.choice(len(box_dims), num_anchors, replace=False)
    centroids = box_dims[indices].copy()

    # Iteraciones de k-means
    max_iterations = 100
    prev_assignments = np.ones(len(box_dims)) * -1

    for iter_num in range(max_iterations):
        # Calcular IoU entre cajas y centroides
        distances = 1 - iou(box_dims, centroids)

        # Asignar cada caja al centroide más cercano
        current_assignments = np.argmin(distances, axis=1)

        # Verificar convergencia
        if (current_assignments == prev_assignments).all():
            break

        # Actualizar centroides
        for i in range(num_anchors):
            cluster_members = box_dims[current_assignments == i]
            if len(cluster_members) > 0:
                centroids[i] = np.mean(cluster_members, axis=0)

        prev_assignments = current_assignments.copy()

    # Ordenar centroides por área (de menor a mayor)
    areas = centroids[:, 0] * centroids[:, 1]
    idx = np.argsort(areas)
    centroids = centroids[idx]

    return centroids


def run_anchor_box_pipeline(
    df: pd.DataFrame, num_anchors_per_group: int = 5
) -> Tuple[
    Dict[Tuple[Specie, CallType], np.ndarray], Dict[Tuple[Specie, CallType], np.ndarray]
]:
    """
    Ejecuta el pipeline completo de cálculo de anchor boxes para cada grupo de (specie, call_type).

    Args:
        df: DataFrame con anotaciones sin procesar
        num_anchors_per_group: Número de anchor boxes a calcular por grupo

    Returns:
        Tuple[Dict, Dict]: Tupla con dos diccionarios:
            - Diccionario de anchor boxes por grupo (specie, call_type)
            - Diccionario de dimensiones agrupadas por (specie, call_type)
    """
    logger.info("Ejecutando pipeline para cálculo de anchor boxes...")
    
    # 1. Preprocesar anotaciones
    processed_df = preprocess_annotations(df)
    logger.info(f"Anotaciones preprocesadas: {len(processed_df)} filas")

    # 2. Extraer dimensiones por grupo
    grouped_dims = extract_dimensions_by_group(processed_df)
    logger.info(f"Dimensiones extraídas para {len(grouped_dims)} grupos")

    # 3. Calcular anchor boxes para cada grupo
    all_anchors = {}
    logger.info("Calculando anchor boxes por grupo:")

    for key, dims in grouped_dims.items():
        specie, call_type = key
        
        logger.info(f"Calculando {num_anchors_per_group} anchors para {specie.name}-{call_type.name}...")
        
        # Calcular anchors para este grupo
        group_anchors = calculate_anchors_for_group(dims, num_anchors_per_group)
        
        # Log de las anchor boxes calculadas
        logger.info(f"Anchor boxes calculadas: {group_anchors.shape[0]}")
        for i, anchor in enumerate(group_anchors):
            logger.debug(f"Anchor {i+1}: duración={anchor[0]:.3f}s, bandwidth={anchor[1]:.1f}Hz")

        # Guardar en el diccionario
        all_anchors[key] = group_anchors

    return all_anchors, grouped_dims
