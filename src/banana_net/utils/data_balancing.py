"""
Utilidades para el balanceo y división de conjuntos de datos.

Este módulo contiene funciones para equilibrar datasets mediante
técnicas como downsampling y para dividir datos en conjuntos de
entrenamiento, validación y prueba.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from src.banana_net.utils.logger import logger


def balance_dataset_by_downsampling(
    df: pd.DataFrame,
    group_columns: List[str] = ["specie", "call_type"],
    target_count: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Equilibra el dataset reduciendo el número de muestras en los grupos más numerosos.

    Args:
        df: DataFrame a equilibrar
        group_columns: Columnas para agrupar
        target_count: Número objetivo de muestras por grupo. Si es None, se usa el mínimo.
        random_state: Semilla aleatoria para reproducibilidad

    Returns:
        DataFrame equilibrado
    """
    logger.info("Equilibrando dataset mediante downsampling...")
    
    # Agrupar y contar
    counts = df.groupby(group_columns).size()
    logger.info("Distribución original de muestras:")
    for group, count in counts.items():
        if isinstance(group, tuple):
            group_str = " + ".join(group)
        else:
            group_str = str(group)
        logger.info(f"{group_str}: {count} muestras")

    # Determinar el tamaño objetivo (mínimo por defecto)
    if target_count is None:
        target_count = counts.min()
        logger.info(f"Seleccionando {target_count} muestras por grupo (mínimo encontrado)")
    else:
        logger.info(f"Seleccionando {target_count} muestras por grupo (especificado por el usuario)")

    # Función para muestrear un grupo
    def sample_group(group):
        if len(group) <= target_count:
            return group
        return group.sample(target_count, random_state=random_state)

    # Aplicar el muestreo a cada grupo
    balanced_df = df.groupby(group_columns, group_keys=False).apply(sample_group)

    # Verificar balance
    new_counts = balanced_df.groupby(group_columns).size()
    logger.info("Distribución después de equilibrar:")
    for group, count in new_counts.items():
        if isinstance(group, tuple):
            group_str = " + ".join(group)
        else:
            group_str = str(group)
        logger.info(f"{group_str}: {count} muestras")
        
    logger.info(f"Total de muestras: {len(balanced_df)} (reducción de {len(df) - len(balanced_df)} muestras)")

    return balanced_df


def split_dataset(
    df: pd.DataFrame,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    stratify_column: str = "specie",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide el dataset en conjuntos de entrenamiento, validación y prueba.

    Args:
        df: DataFrame a dividir
        val_ratio: Proporción para el conjunto de validación
        test_ratio: Proporción para el conjunto de prueba
        stratify_column: Columna a usar para estratificación
        random_state: Semilla aleatoria para reproducibilidad

    Returns:
        Tuple con (df_train, df_val, df_test)
    """
    from sklearn.model_selection import train_test_split
    
    logger.info("Dividiendo dataset en conjuntos de entrenamiento, validación y prueba...")


    # Primera división: separar conjunto de prueba
    train_val_ratio = 1 - test_ratio
    df_train_val, df_test = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df[stratify_column] if stratify_column else None,
        random_state=random_state,
    )

    # Segunda división: separar entrenamiento y validación
    val_adjusted_ratio = val_ratio / train_val_ratio
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_adjusted_ratio,
        stratify=df_train_val[stratify_column] if stratify_column else None,
        random_state=random_state,
    )
    
    # Mostrar estadísticas
    train_pct = len(df_train) / len(df) * 100
    val_pct = len(df_val) / len(df) * 100
    test_pct = len(df_test) / len(df) * 100

    logger.info(f"Conjunto de entrenamiento: {len(df_train)} muestras ({train_pct:.1f}%)")
    logger.info(f"Conjunto de validación:    {len(df_val)} muestras ({val_pct:.1f}%)")
    logger.info(f"Conjunto de prueba:        {len(df_test)} muestras ({test_pct:.1f}%)")

    if stratify_column:
        logger.info(f"Distribución de {stratify_column} por conjunto:")
        logger.info("Entrenamiento:")
        train_dist = df_train[stratify_column].value_counts(normalize=True)
        for val, pct in train_dist.items():
            logger.info(f"  {val}: {pct*100:.1f}%")

        logger.info("Validación:")
        val_dist = df_val[stratify_column].value_counts(normalize=True)
        for val, pct in val_dist.items():
            logger.info(f"  {val}: {pct*100:.1f}%")

        logger.info("Prueba:")
        test_dist = df_test[stratify_column].value_counts(normalize=True)
        for val, pct in test_dist.items():
            logger.info(f"  {val}: {pct*100:.1f}%")

    return df_train, df_val, df_test


def balance_by_augmentation(
    df: pd.DataFrame,
    group_columns: List[str] = ["specie", "call_type"],
    target_count: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Equilibra el dataset aumentando el número de muestras en los grupos menos numerosos.

    A diferencia de balance_dataset_by_downsampling, esta función no descarta muestras,
    sino que añade marcadores para aplicar aumento de datos posteriormente.

    Args:
        df: DataFrame a equilibrar
        group_columns: Columnas para agrupar
        target_count: Número objetivo de muestras por grupo. Si es None, se usa el máximo.
        random_state: Semilla aleatoria para reproducibilidad

    Returns:
        DataFrame equilibrado con una columna adicional 'augmentation_needed'
    """
    logger.info("Equilibrando dataset mediante marcadores de augmentation...")

    # Crear una copia para no modificar el original
    df_result = df.copy()

    # Añadir columna para marcar muestras que necesitan augmentation
    df_result["augmentation_needed"] = False

    # Agrupar y contar
    counts = df.groupby(group_columns).size()

    # Determinar el tamaño objetivo (máximo por defecto)
    if target_count is None:
        target_count = counts.max()
        print(
            f"\nAugmentando hasta {target_count} muestras por grupo (máximo encontrado)"
        )
    else:
        print(
            f"\nAugmentando hasta {target_count} muestras por grupo (especificado por el usuario)"
        )

    print(f"Distribución original de muestras:")
    for group, count in counts.items():
        if isinstance(group, tuple):
            group_str = " + ".join(str(g) for g in group)
        else:
            group_str = str(group)
        print(f"  - {group_str}: {count} muestras")

        # Calcular cuántas muestras faltan
        needed = max(0, target_count - count)

        if needed > 0:
            # Obtener índices de las muestras de este grupo
            if isinstance(group, tuple):
                mask = pd.Series(True, index=df_result.index)
                for col, val in zip(group_columns, group):
                    mask &= df_result[col] == val
                group_indices = df_result[mask].index
            else:
                group_indices = df_result[df_result[group_columns[0]] == group].index

            # Si hay menos muestras que las necesarias, repetir algunas
            if len(group_indices) < needed:
                # Repetir las muestras existentes
                repeats = needed // len(group_indices)
                remainder = needed % len(group_indices)

                # Marcar para repetición completa
                for _ in range(repeats):
                    df_result.loc[group_indices, "augmentation_needed"] = True

                # Marcar muestra adicionales para repetición
                if remainder > 0:
                    additional_indices = np.random.choice(
                        group_indices, remainder, replace=False
                    )
                    df_result.loc[additional_indices, "augmentation_needed"] = True
            else:
                # Seleccionar aleatoriamente las muestras a aumentar
                aug_indices = np.random.choice(group_indices, needed, replace=False)
                df_result.loc[aug_indices, "augmentation_needed"] = True

    # Contar cuántas muestras necesitarán augmentation
    aug_count = df_result["augmentation_needed"].sum()
    print(f"\nTotal de muestras marcadas para augmentation: {aug_count}")
    print(f"Tamaño final proyectado del dataset: {len(df) + aug_count} muestras")

    return df_result
