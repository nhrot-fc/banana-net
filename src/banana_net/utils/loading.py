import os
import pandas as pd
import librosa
import warnings
from typing import List, Tuple, Set, Dict, Any
from src.banana_net.utils.logger import logger


def get_annotation_files(directory: str, extension: str = ".txt") -> List[str]:
    """
    Get a sorted list of annotation files from a directory.
    """
    try:
        files = [
            file
            for file in os.listdir(directory)
            if file.lower().endswith(extension.lower())
        ]
        files.sort()
        logger.debug(f"Found {len(files)} {extension} files in {directory}")
        return files
    except Exception as e:
        logger.error(f"Error listing {extension} files in {directory}: {e}")
        return []


def get_recording_files(directory: str, extension: str = ".wav") -> List[str]:
    """
    Get a sorted list of recording files from a directory.
    """
    try:
        files = [
            file
            for file in os.listdir(directory)
            if file.lower().endswith(extension.lower())
        ]
        files.sort()
        logger.debug(f"Found {len(files)} {extension} files in {directory}")
        return files
    except Exception as e:
        logger.error(f"Error listing {extension} files in {directory}: {e}")
        return []


def load_single_annotation_file(file_path: str) -> pd.DataFrame:
    """
    Load a single annotation file and standardize column names.
    """
    try:
        df = pd.read_csv(file_path, sep="\t", header=0)
    except Exception as e:
        logger.error(f"Failed to read annotation file {file_path}: {e}")
        return pd.DataFrame()
    
    relevant_columns = [
        "Begin Time (s)",
        "End Time (s)",
        "Low Freq (Hz)",
        "High Freq (Hz)",
        "Inband Power (dB FS)",
        "Species",
        "Call type",
    ]
    # Ensure only existing columns are selected to avoid KeyError
    existing_relevant_columns = [col for col in relevant_columns if col in df.columns]
    df = df[existing_relevant_columns].copy()

    # Rename columns for consistency
    rename_map = {
        "Begin Time (s)": "begin_time",
        "End Time (s)": "end_time",
        "Low Freq (Hz)": "low_freq",
        "High Freq (Hz)": "high_freq",
        "Inband Power (dB FS)": "inband_power",
        "Species": "specie",
        "Call type": "call_type",
    }
    df.rename(
        columns={k: v for k, v in rename_map.items() if k in existing_relevant_columns},
        inplace=True,
    )

    # Add the recording file name
    recording_file = os.path.basename(file_path).replace(".txt", ".wav")
    df["recording_file"] = recording_file
    # Get the directory basename for the recording file
    df["directory"] = os.path.basename(os.path.dirname(file_path))
    return df


def load_annotations_from_species_dirs(
    raw_data_dir: str, species_dirs: List[str]
) -> pd.DataFrame:
    """
    Loads all annotation files from specified species subdirectories within a raw data directory.
    """
    all_dfs = []
    for specie_dir_name in species_dirs:
        specie_full_path = os.path.join(raw_data_dir, specie_dir_name)
        if not os.path.isdir(specie_full_path):
            logger.warning(f"Directory {specie_full_path} not found. Skipping.")
            continue

        annotation_files = get_annotation_files(specie_full_path)
        if not annotation_files:
            logger.warning(f"No annotation files found in {specie_full_path}. Skipping.")
            continue
        
        logger.info(f"Loading annotations from: {specie_full_path}")
        for ann_file in annotation_files:
            file_path = os.path.join(specie_full_path, ann_file)
            try:
                df = load_single_annotation_file(file_path)
                if df.empty: continue
                all_dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading annotation file {file_path}: {e}", exc_info=True)

    if not all_dfs:
        logger.warning("No annotation dataframes were loaded.")
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Successfully loaded and combined {len(all_dfs)} annotation files into a DataFrame with {len(combined_df)} rows.")
    return combined_df


def load_audio_and_annotations_from_species_dirs(
    raw_data_dir: str, 
    species_dirs: List[str],
    sr: int = 22050
) -> Tuple[pd.DataFrame, Dict[str, Any], Set[str]]:
    """
    Carga archivos de audio y anotaciones desde directorios de especies especificados.
    
    Args:
        raw_data_dir (str): Directorio base con los datos raw
        species_dirs (List[str]): Lista de nombres de directorios de especies
        sr (int): Sample rate para cargar el audio (default: 22050)
    
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any], Set[str]]: 
            - DataFrame con anotaciones exitosamente cargadas
            - Diccionario con datos de audio {archivo: (audio_data, sample_rate)}
            - Conjunto de archivos descartados por errores
    """
    all_annotation_dfs = []
    discarded_files = set()
    
    for specie_dir_name in species_dirs:
        specie_full_path = os.path.join(raw_data_dir, specie_dir_name)
        if not os.path.isdir(specie_full_path):
            logger.warning(f"Directory {specie_full_path} not found. Skipping.")
            continue

        # Obtener archivos de anotaciones y audio
        annotation_files = get_annotation_files(specie_full_path)
        audio_files = get_recording_files(specie_full_path)
        
        if not annotation_files:
            logger.warning(f"No annotation files found in {specie_full_path}. Skipping.")
            continue
            
        if not audio_files:
            logger.warning(f"No audio files found in {specie_full_path}. Skipping.")
            continue
        
        logger.info(f"Processing files from: {specie_full_path}")
        
        # Crear mapeo de archivos base (sin extensión) para emparejar .txt y .wav
        annotation_base_names = {os.path.splitext(f)[0]: f for f in annotation_files}
        audio_base_names = {os.path.splitext(f)[0]: f for f in audio_files}
        
        # Procesar solo archivos que tienen tanto anotación como audio
        common_base_names = set(annotation_base_names.keys()) & set(audio_base_names.keys())
        
        for base_name in common_base_names:
            annotation_file = annotation_base_names[base_name]
            audio_file = audio_base_names[base_name]
            
            annotation_path = os.path.join(specie_full_path, annotation_file)
            audio_path = os.path.join(specie_full_path, audio_file)
            
            # Flags para rastrear errores
            annotation_error = False
            audio_error = False
            
            # Intentar cargar anotaciones
            annotation_df = None
            try:
                annotation_df = load_single_annotation_file(annotation_path)
                if annotation_df.empty:
                    annotation_error = True
                    logger.warning(f"Empty annotation file: {annotation_path}")
            except Exception as e:
                annotation_error = True
                logger.warning(f"Error loading annotation file {annotation_path}: {e}")
            
            # Intentar cargar audio
            audio_array = None
            audio_sr = None
            try:
                # Capturar warnings de librosa
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    audio_array, audio_sr = librosa.load(audio_path, sr=sr)
                    
                    # Si librosa emitió warnings, considerar como error
                    if w:
                        audio_error = True
                        warning_messages = [str(warning.message) for warning in w]
                        logger.warning(f"Librosa warnings for {audio_path}: {warning_messages}")
                        
            except Exception as e:
                audio_error = True
                logger.warning(f"Error loading audio file {audio_path}: {e}")
            
            # Si hay errores en cualquiera de los dos, descartar el archivo
            if annotation_error or audio_error:
                discarded_files.add(f"{specie_dir_name}/{base_name}")
                logger.info(f"Discarding file pair: {specie_dir_name}/{base_name} due to loading errors")
                continue
            
            # Si todo salió bien, agregar a los resultados
            if annotation_df is not None and audio_array is not None:
                all_annotation_dfs.append(annotation_df)
                logger.debug(f"Successfully loaded: {specie_dir_name}/{base_name}")

    # Combinar todas las anotaciones
    if not all_annotation_dfs:
        logger.warning("No annotation dataframes were successfully loaded.")
        combined_annotations_df = pd.DataFrame()
    else:
        combined_annotations_df = pd.concat(all_annotation_dfs, ignore_index=True)
        logger.info(f"Successfully loaded {len(all_annotation_dfs)} file pairs into DataFrame with {len(combined_annotations_df)} annotation rows.")
    
    logger.info(f"Total discarded files: {len(discarded_files)}")
    if discarded_files:
        logger.info(f"Discarded files: {sorted(discarded_files)}")
    
    return combined_annotations_df, discarded_files
