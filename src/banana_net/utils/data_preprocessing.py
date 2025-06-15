"""
Utilidades para el preprocesamiento de datos de anotaciones de audio.

Este módulo contiene funciones para la limpieza, normalización y
preparación de datos de anotaciones de vocalizaciones de primates.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from Levenshtein import distance as levenshtein_distance

from src.banana_net.models.enums import (
    specie_mapping,
    call_type_mapping,
    Specie,
    CallType,
)
from src.banana_net.utils.logger import logger


def preprocess_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesa un DataFrame de anotaciones crudo, calculando dimensiones y
    convirtiendo cadenas de texto a enumeraciones.

    Args:
        df (pd.DataFrame): DataFrame con anotaciones sin procesar, que debe contener
                           al menos 'begin_time', 'end_time', 'low_freq', 'high_freq',
                           'specie', y 'call_type'.

    Returns:
        pd.DataFrame: DataFrame con columnas adicionales 'duration_s', 'bandwidth_hz',
                     'specie_enum', y 'call_type_enum'. Las filas con valores de
                     especie o tipo de llamada no válidos son eliminadas.
    """
    df_processed = df.copy()

    # Calcular dimensiones directamente
    df_processed["duration_s"] = df_processed["end_time"] - df_processed["begin_time"]
    df_processed["bandwidth_hz"] = df_processed["high_freq"] - df_processed["low_freq"]

    return df_processed


def extract_dimensions_by_group(
    processed_df: pd.DataFrame,
) -> Dict[Tuple[Specie, CallType], np.ndarray]:
    """
    Agrupa el DataFrame de anotaciones por especie y tipo de llamada,
    extrayendo las dimensiones (duración, ancho de banda) para cada grupo.

    Args:
        processed_df: DataFrame preprocesado con columnas 'specie_enum',
                      'call_type_enum', 'duration_s', y 'bandwidth_hz'.

    Returns:
        Dict[Tuple[Specie, CallType], np.ndarray]: Diccionario donde las claves son
            tuplas (especie, tipo_llamada) y los valores son arrays de dimensiones
            de todas las anotaciones en ese grupo.
    """

    # Agrupar por las columnas auxiliares en lugar de los enums directamente
    grouped = processed_df.groupby(["specie", "call_type"])

    # Crear el diccionario donde las claves son tuplas de enum (Specie, CallType)
    dimensions_dict = {}
    for group_key, group_df in grouped:
        specie_name, call_type_name = group_key

        # Convertir nombres a enums
        specie_enum = Specie.from_name(specie_name)
        call_type_enum = CallType.from_name(call_type_name)

        # Extraer dimensiones como array numpy de forma [N, 2] donde N es el número de anotaciones
        # y cada fila es [duration_s, bandwidth_hz]
        dimensions = group_df[["duration_s", "bandwidth_hz"]].values

        # Guardar en el diccionario
        dimensions_dict[(specie_enum, call_type_enum)] = dimensions

    return dimensions_dict


def find_potential_typos_per_species(
    df: pd.DataFrame,
    species_list: List[str],
    distance_threshold: int = 2,
    uncommon_threshold: int = 50,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Encuentra posibles errores tipográficos en las etiquetas de especies y tipos de llamada.

    Args:
        df: DataFrame con anotaciones
        species_list: Lista de especies conocidas
        distance_threshold: Umbral de distancia de Levenshtein para considerar un typo
        uncommon_threshold: Frecuencia mínima para considerar una etiqueta como válida

    Returns:
        Diccionario con posibles typos por columna
    """
    results = {}

    # Columnas a revisar
    columns_to_check = ["specie", "call_type"]

    for col in columns_to_check:
        typos = []

        # Contar frecuencia de cada valor
        value_counts = df[col].value_counts()

        # Identificar valores poco comunes
        uncommon_values = value_counts[value_counts < uncommon_threshold].index.tolist()

        # Para cada valor poco común, buscar el valor más cercano
        for uncommon_value in uncommon_values:
            best_match = None
            min_distance = float("inf")

            # Comparar con valores conocidos
            reference_values = (
                species_list
                if col == "specie"
                else value_counts[value_counts >= uncommon_threshold].index
            )

            for ref_value in reference_values:
                dist = levenshtein_distance(uncommon_value, ref_value)

                if dist <= distance_threshold and dist < min_distance:
                    min_distance = dist
                    best_match = ref_value

            if best_match:
                typos.append(
                    {
                        "wrong_value": uncommon_value,
                        "suggested_value": best_match,
                        "distance": min_distance,
                        "occurrences": int(value_counts[uncommon_value]),
                    }
                )

        results[col] = sorted(typos, key=lambda x: x["occurrences"], reverse=True)

    return results


def correct_species_typos(
    df: pd.DataFrame, species_to_check: List[str]
) -> pd.DataFrame:
    """
    Corrige errores tipográficos en las etiquetas de especies.

    Args:
        df: DataFrame con anotaciones
        species_to_check: Lista de abreviaturas de especies a comprobar

    Returns:
        DataFrame con las etiquetas corregidas
    """
    df_corrected = df.copy()

    # Obtener valores válidos de especies
    valid_species = [s.name.lower() for s in Specie]

    # Encontrar typos potenciales
    typos = find_potential_typos_per_species(
        df_corrected, valid_species, distance_threshold=2
    )

    logger.info("Corrigiendo errores tipográficos en etiquetas")

    # Corregir typos en especies
    if "specie" in typos and typos["specie"]:
        logger.info("Correcciones de especies:")
        for typo in typos["specie"]:
            wrong = typo["wrong_value"]
            suggested = typo["suggested_value"]
            occurrences = typo["occurrences"]
            
            logger.info(f"'{wrong}' -> '{suggested}' ({occurrences} ocurrencias)")
            df_corrected["specie"] = df_corrected["specie"].replace(wrong, suggested)
    else:
        logger.info("No se encontraron errores tipográficos en especies.")

    # Corregir typos en tipos de llamada
    if "call_type" in typos and typos["call_type"]:
        logger.info("Correcciones de tipos de llamada:")
        for typo in typos["call_type"]:
            wrong = typo["wrong_value"]
            suggested = typo["suggested_value"]
            occurrences = typo["occurrences"]
            
            logger.info(f"'{wrong}' -> '{suggested}' ({occurrences} ocurrencias)")
            df_corrected["call_type"] = df_corrected["call_type"].replace(
                wrong, suggested
            )
    else:
        logger.info("No se encontraron errores tipográficos en tipos de llamada.")

    return df_corrected


def filter_uncommon_combinations(
    df: pd.DataFrame,
    threshold: int = 200,
    groupby_columns: List[str] = ["specie", "call_type"],
) -> pd.DataFrame:
    """
    Filtra combinaciones poco comunes de especies y tipos de llamada.

    Args:
        df: DataFrame con anotaciones
        threshold: Umbral mínimo de ocurrencias para conservar una combinación
        groupby_columns: Columnas por las que agrupar para el filtro

    Returns:
        DataFrame filtrado
    """
    df_filtered = df.copy()

    # Contar ocurrencias de cada combinación
    counts = df_filtered.groupby(groupby_columns).size().reset_index(name="count")

    # Identificar combinaciones poco frecuentes
    uncommon_combinations = counts[counts["count"] < threshold]

    if uncommon_combinations.empty:
        logger.info(f"No se encontraron combinaciones con menos de {threshold} ocurrencias.")
        return df_filtered

    logger.info(f"Filtrando combinaciones poco comunes (menos de {threshold} ocurrencias):")
    for _, row in uncommon_combinations.iterrows():
        group_values = [row[col] for col in groupby_columns]
        group_count = row["count"]
        
        logger.info(f"Eliminando combinación: {' + '.join(group_values)} ({group_count} ocurrencias)")

        # Crear una máscara para filtrar esta combinación específica
        mask = pd.Series(True, index=df_filtered.index)
        for col, val in zip(groupby_columns, group_values):
            mask &= df_filtered[col] == val

        # Filtrar el DataFrame
        df_filtered = df_filtered[~mask]
    
    logger.info(f"Filas restantes después del filtrado: {len(df_filtered)} de {len(df)} ({len(df_filtered)/len(df)*100:.1f}%)")
    
    return df_filtered


def sort_and_reset_dataset(
    df: pd.DataFrame,
    sort_columns: List[str] = ["specie", "recording_file", "begin_time"],
    ascending: List[bool] = [True, True, True],
) -> pd.DataFrame:
    """
    Ordena el dataset y resetea los índices.

    Args:
        df: DataFrame a ordenar
        sort_columns: Columnas por las que ordenar
        ascending: Lista de booleanos indicando si ordenar ascendente o descendente

    Returns:
        DataFrame ordenado con índices reseteados
    """
    logger.info("Ordenando dataset...")
    df_sorted = df.sort_values(by=sort_columns, ascending=ascending)
    df_sorted.reset_index(drop=True, inplace=True)
    logger.info(f"Dataset ordenado por {', '.join(sort_columns)}")

    return df_sorted
