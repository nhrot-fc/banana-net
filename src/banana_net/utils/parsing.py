"""
Parsing utilities for audio annotation data preprocessing.

This module contains functions to parse, clean, and preprocess 
annotation files for different primate species vocalizations.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple


def get_annotation_files(directory: str, extension: str = '.txt') -> List[str]:
    """
    Get a sorted list of annotation files from a directory.
    
    Args:
        directory (str): Path to the directory containing annotation files
        extension (str): File extension to filter by (default: '.txt')
        
    Returns:
        List[str]: Sorted list of annotation filenames
    """
    files = [file for file in os.listdir(directory) 
             if file.lower().endswith(extension.lower())]
    files.sort()
    return files


def get_recording_files(directory: str, extension: str = '.wav') -> List[str]:
    """
    Get a sorted list of recording files from a directory.
    
    Args:
        directory (str): Path to the directory containing recording files
        extension (str): File extension to filter by (default: '.wav')
        
    Returns:
        List[str]: Sorted list of recording filenames
    """
    files = [file for file in os.listdir(directory) 
             if file.lower().endswith(extension.lower())]
    files.sort()
    return files


def load_annotation_file(file_path: str) -> pd.DataFrame:
    """
    Load a single annotation file and standardize column names.
    
    Args:
        file_path (str): Path to the annotation file
        
    Returns:
        pd.DataFrame: DataFrame with standardized column names
    """
    # Load the annotation file
    df = pd.read_csv(file_path, sep='\t', header=0)
    
    # Select and rename relevant columns
    relevant_columns = [
        'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 
        'High Freq (Hz)', 'Inband Power (dB FS)', 
        'Species', 'Call type'
    ]
    
    df = df[relevant_columns].copy()
    
    # Rename columns for consistency
    df.columns = [
        'begin_time', 'end_time', 'low_freq', 
        'high_freq', 'inband_power', 'species', 
        'call_type'
    ]
    
    # Add the recording file name
    recording_file = os.path.basename(file_path).replace('.txt', '.wav')
    df['recording_file'] = recording_file
    
    return df


def clean_annotation_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess an annotation DataFrame.
    
    Args:
        df (pd.DataFrame): Raw annotation DataFrame
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Drop rows with NaN values in 'species' or 'call_type'
    df = df.dropna(subset=['species', 'call_type'])
    
    # Ensure numeric columns are of the correct type
    numeric_columns = ['begin_time', 'end_time', 'low_freq', 
                       'high_freq', 'inband_power']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Reset index
    df.reset_index(drop=True, inplace=True)
    
    return df


def normalize_labels(df: pd.DataFrame, 
                    corrections: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Normalize call_type and species labels by stripping whitespace, 
    converting to lowercase, and applying corrections.
    
    Args:
        df (pd.DataFrame): DataFrame with call_type and species columns
        corrections (Dict[str, str], optional): Dictionary of label corrections
        
    Returns:
        pd.DataFrame: DataFrame with normalized labels
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Strip whitespace and convert to lowercase
    df['call_type'] = df['call_type'].str.strip().str.lower()
    df['species'] = df['species'].str.strip().str.lower()
    
    # Apply corrections if provided
    if corrections:
        df['call_type'] = df['call_type'].replace(corrections)
        df['species'] = df['species'].replace(corrections)
    
    return df


def filter_invalid_entries(df: pd.DataFrame, 
                          invalid_species: Optional[List[str]] = None,
                          invalid_call_types: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Filter out invalid species and call types.
    
    Args:
        df (pd.DataFrame): DataFrame to filter
        invalid_species (List[str], optional): List of invalid species to remove
        invalid_call_types (List[str], optional): List of invalid call types to remove
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    if invalid_species:
        df = df[~df['species'].isin(invalid_species)]
    
    if invalid_call_types:
        df = df[~df['call_type'].isin(invalid_call_types)]
    
    # Reset index after filtering
    df.reset_index(drop=True, inplace=True)
    
    return df


def get_most_common_call_types(df: pd.DataFrame, n: int = 3) -> List[str]:
    """
    Get the n most common call types from the dataset.
    
    Args:
        df (pd.DataFrame): DataFrame with call_type column
        n (int): Number of most common call types to return
        
    Returns:
        List[str]: List of most common call types
    """
    return df['call_type'].value_counts().nlargest(n).index.tolist()


def filter_by_call_types(df: pd.DataFrame, call_types: List[str]) -> pd.DataFrame:
    """
    Filter DataFrame to keep only specified call types.
    
    Args:
        df (pd.DataFrame): DataFrame to filter
        call_types (List[str]): List of call types to keep
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    df_filtered = df[df['call_type'].isin(call_types)].copy()
    df_filtered.reset_index(drop=True, inplace=True)
    return df_filtered


def plot_call_type_distribution(df: pd.DataFrame, 
                                save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Plot the distribution of call types in the dataset.
    
    Args:
        df (pd.DataFrame): DataFrame with call_type column
        save_path (str, optional): Path to save the plot
        figsize (Tuple[int, int]): Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    sns.countplot(data=df, x='call_type', 
                  order=df['call_type'].value_counts().index)
    plt.title('Call Type Distribution')
    plt.xlabel('Call Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def load_and_process_annotations(directory: str,
                                corrections: Optional[Dict[str, str]] = None,
                                invalid_species: Optional[List[str]] = None,
                                invalid_call_types: Optional[List[str]] = None,
                                filter_top_n: Optional[int] = None) -> pd.DataFrame:
    """
    Complete pipeline to load and process all annotation files in a directory.
    
    Args:
        directory (str): Path to directory containing annotation files
        corrections (Dict[str, str], optional): Label corrections to apply
        invalid_species (List[str], optional): Invalid species to filter out
        invalid_call_types (List[str], optional): Invalid call types to filter out
        filter_top_n (int, optional): Keep only top n most common call types
        
    Returns:
        pd.DataFrame: Processed dataset
    """
    # Get annotation files
    annotation_files = get_annotation_files(directory)
    
    # Default corrections for common issues
    if corrections is None:
        corrections = {
            'cs      a': 'cs',
            'contact call': 'cc',
            'contact syllable': 'cs',
            'l': 'lw',
        }
    
    # Default invalid entries
    if invalid_species is None:
        invalid_species = ['as', '}']
    
    if invalid_call_types is None:
        invalid_call_types = ['noise']
    
    # Process all annotation files
    dataset_list = []
    
    for annotation_file in annotation_files:
        file_path = os.path.join(directory, annotation_file)
        
        try:
            # Load and clean the annotation file
            df = load_annotation_file(file_path)
            df = clean_annotation_dataframe(df)
            dataset_list.append(df)
            
        except Exception as e:
            print(f"Error processing {annotation_file}: {e}")
    
    # Concatenate all DataFrames
    if not dataset_list:
        return pd.DataFrame()
    
    dataset = pd.concat(dataset_list, ignore_index=True)
    
    # Normalize labels
    dataset = normalize_labels(dataset, corrections)
    
    # Filter invalid entries
    dataset = filter_invalid_entries(dataset, invalid_species, invalid_call_types)
    
    # Filter by most common call types if specified
    if filter_top_n:
        most_common = get_most_common_call_types(dataset, filter_top_n)
        dataset = filter_by_call_types(dataset, most_common)
    
    return dataset


def print_dataset_summary(df: pd.DataFrame) -> None:
    """
    Print a summary of the dataset including unique values and counts.
    
    Args:
        df (pd.DataFrame): Dataset to summarize
    """
    print(f"Dataset shape: {df.shape}")
    print(f"Unique call types: {sorted(df['call_type'].unique())}")
    print(f"Unique species: {sorted(df['species'].unique())}")
    print(f"Unique recording files: {len(df['recording_file'].unique())}")
    print("\nCall type distribution:")
    print(df['call_type'].value_counts())
    print("\nSpecies distribution:")
    print(df['species'].value_counts())


def save_processed_dataset(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the processed dataset to a CSV file.
    
    Args:
        df (pd.DataFrame): Dataset to save
        output_path (str): Path where to save the CSV file
    """
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to: {output_path}")