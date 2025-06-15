import os
import pandas as pd
from typing import List
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
