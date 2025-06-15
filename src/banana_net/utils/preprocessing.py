import pandas as pd
import re
import re


def clean_annotation_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el DataFrame de anotaciones eliminando filas con NaNs y convirtiendo columnas a tipo numérico.
    """
    df = df.copy()
    df = df.dropna(subset=["specie", "call_type"])

    numeric_columns = [
        "begin_time",
        "end_time",
        "low_freq",
        "high_freq",
        "inband_power",
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[numeric_columns] = df[numeric_columns].round(3)

    df.dropna(subset=numeric_columns, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Drop if specie or call_type have non-letter characters
    df = df[~df["specie"].str.contains(r"[^a-zA-Z]", na=False)]
    df = df[~df["call_type"].str.contains(r"[^a-zA-Z]", na=False)]

    # Drop if noise is contained on specie or call_type
    df = df[~df["specie"].str.contains("noise", case=False, na=False)]
    df = df[~df["call_type"].str.contains("noise", case=False, na=False)]

    # Drop if specie or call_type are empty
    df = df[df["specie"].str.strip() != ""]
    df = df[df["call_type"].str.strip() != ""]
    
    # Drop if begin_time or end_time are NaN
    df = df[~df["begin_time"].isna()]
    df = df[~df["end_time"].isna()]

    return df


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza las etiquetas de especies y tipos de llamadas.
    """
    df = df.copy()
    df["call_type"] = df["call_type"].astype(str).str.strip().str.lower()
    df["specie"] = df["specie"].astype(str).str.strip().str.lower()

    corrections = {
        "cs      a": "cs",
        "contact call": "cc",
        "contact syllable": "cs",
    }

    df["call_type"] = df["call_type"].replace(corrections)
    return df


def create_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea características de duración y ancho de banda a partir de los tiempos y frecuencias.
    """
    df = df.copy()
    df["duration_s"] = df["end_time"] - df["begin_time"]
    df["bandwidth_hz"] = df["high_freq"] - df["low_freq"]
    
    df["duration_s"] = df["duration_s"].round(3)
    df["bandwidth_hz"] = df["bandwidth_hz"].round(3)

    return df


def check_species_directory_consistency(df: pd.DataFrame) -> dict:
    """
    Verifica la consistencia entre la columna 'specie' y el código de especie
    extraído de la columna 'directory'.

    Args:
        df (pd.DataFrame): DataFrame de anotaciones con columnas 'directory' y 'specie'

    Returns:
        dict: Diccionario con información de inconsistencias encontradas:
              - count: Número total de inconsistencias
              - inconsistencies: DataFrame con las filas inconsistentes
              - summary: Resumen de inconsistencias por directorio
    """
    df = df.copy()

    # Extraer el código de especie de la columna directory (los últimos 2 caracteres después de __)
    df["expected_specie"] = (
        df["directory"]
        .str.extract(r"__([A-Z]{2})$", flags=re.IGNORECASE)
        .iloc[:, 0]
        .str.lower()
    )

    # Identificar inconsistencias
    inconsistent = df[df["specie"] != df["expected_specie"]]

    # Contar inconsistencias por directorio
    summary = (
        inconsistent.groupby(["directory", "specie", "expected_specie"])
        .size()
        .reset_index(name="count")
    )

    result = {
        "count": len(inconsistent),
        "inconsistencies": inconsistent,
        "summary": summary,
    }

    return result


def fix_species_directory_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige las inconsistencias entre la columna 'specie' y el código de especie
    extraído de la columna 'directory', asignando el código de especie correcto
    basado en el directorio.

    Args:
        df (pd.DataFrame): DataFrame de anotaciones con columnas 'directory' y 'specie'

    Returns:
        pd.DataFrame: DataFrame con las inconsistencias corregidas
    """
    df = df.copy()

    # Extraer el código de especie de la columna directory (los últimos 2 caracteres después de __)
    df["expected_specie"] = (
        df["directory"]
        .str.extract(r"__([A-Z]{2})$", flags=re.IGNORECASE)
        .iloc[:, 0]
        .str.lower()
    )

    # Aplicar la corrección
    df["specie"] = df["expected_specie"]

    # Eliminar la columna temporal
    df.drop(columns=["expected_specie"], inplace=True)

    return df
