"""
Utility functions for processing audio files into clips with overlap 
and transforming annotations into YOLO format.

These functions allow dividing long audio recordings into smaller clips
with a specified duration and overlap, and transforming annotations
to YOLO tensor format for object detection models.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


@dataclass
class AudioClip:
    """Representation of an audio clip with its annotations and metadata.
    
    This class stores information about a clip extracted from a longer audio recording,
    including its temporal boundaries, identifiers, and associated annotations.
    """
    original_file: str
    clip_id: int
    clip_name: str
    start_time: float
    end_time: float
    duration: float
    num_annotations: int
    annotations: pd.DataFrame
    
    def has_annotations(self) -> bool:
        """Check if the clip has any annotations."""
        return self.num_annotations > 0
    
    def get_species(self) -> List[str]:
        """Get unique species in this clip's annotations."""
        if self.has_annotations():
            return self.annotations['species'].unique().tolist()
        return []
    
    def get_call_types(self) -> List[str]:
        """Get unique call types in this clip's annotations."""
        if self.has_annotations():
            return self.annotations['call_type'].unique().tolist()
        return []
    
    def get_relative_time(self, absolute_time: float) -> float:
        """Convert an absolute time to clip-relative time."""
        if absolute_time < self.start_time or absolute_time > self.end_time:
            raise ValueError(f"Time {absolute_time} is outside clip boundaries [{self.start_time}, {self.end_time}]")
        return absolute_time - self.start_time
    
    def get_absolute_time(self, relative_time: float) -> float:
        """Convert a clip-relative time to absolute time in the original recording."""
        if relative_time < 0 or relative_time > self.duration:
            raise ValueError(f"Relative time {relative_time} is outside clip duration [0, {self.duration}]")
        return self.start_time + relative_time


def generate_clips_metadata(
    total_duration_sec: float, 
    clip_duration: float = 5.0, 
    overlap: float = 1.0, 
    min_clip_duration: float = 2.0
) -> List[Tuple[int, float, float]]:
    """
    Generate metadata for audio clips with overlap.
    
    Parameters:
    -----------
    total_duration_sec : float
        Total duration of the audio file in seconds
    clip_duration : float, default=5.0
        Duration of each clip in seconds
    overlap : float, default=1.0
        Overlap between consecutive clips in seconds
    min_clip_duration : float, default=2.0
        Minimum duration for a clip to be included
    
    Returns:
    --------
    clips : List[Tuple[int, float, float]]
        List of tuples: (clip_id, start_time, end_time)
    """
    clips: List[Tuple[int, float, float]] = []
    step = clip_duration - overlap
    
    start_time = 0.0
    clip_id = 0
    
    while start_time < total_duration_sec:
        end_time = min(start_time + clip_duration, total_duration_sec)
        
        # Only include clips that have meaningful duration
        if end_time - start_time >= min_clip_duration:
            clips.append((clip_id, start_time, end_time))
            clip_id += 1
            
        start_time += step
        
        # Break if we would have a very short final clip
        if total_duration_sec - start_time < min_clip_duration:
            break
            
    return clips


def annotations_in_clip(
    annotations_df: pd.DataFrame, 
    clip_start: float, 
    clip_end: float,
    min_duration: float = 0.1
) -> pd.DataFrame:
    """
    Filter annotations that overlap with the given clip timeframe.
    Adjust annotation times to be relative to clip start.
    
    Parameters:
    -----------
    annotations_df : pd.DataFrame
        DataFrame containing the annotations with begin_time and end_time columns
    clip_start : float
        Start time of the clip in seconds
    clip_end : float
        End time of the clip in seconds
    min_duration : float, default=0.1
        Minimum duration of annotations after clipping
    
    Returns:
    --------
    overlapping : pd.DataFrame
        Filtered DataFrame with annotations that overlap with the clip,
        with additional columns 'begin_time_clip' and 'end_time_clip'
        containing times relative to clip start
    """
    # Find annotations that overlap with this clip
    overlapping = annotations_df[
        (annotations_df['end_time'] > clip_start) & 
        (annotations_df['begin_time'] < clip_end)
    ].copy()
    
    if overlapping.empty:
        return overlapping
    
    # Adjust times to be relative to clip start and clip boundaries
    overlapping['begin_time_clip'] = np.maximum(overlapping['begin_time'] - clip_start, 0.0)
    overlapping['end_time_clip'] = np.minimum(overlapping['end_time'] - clip_start, clip_end - clip_start)
    
    # Filter out annotations that become too small after clipping
    overlapping = overlapping[overlapping['end_time_clip'] - overlapping['begin_time_clip'] >= min_duration]
    
    return overlapping


def create_yolo_tensor_for_clip(
    annotations_clip: pd.DataFrame, 
    clip_duration: float, 
    max_freq_hz: float, 
    S: int = 7, 
    B: int = 2, 
    class_map: Optional[Dict[Tuple[str, str], int]] = None
) -> np.ndarray:
    """
    Create YOLO tensor for a single audio clip.
    
    Parameters:
    -----------
    annotations_clip : pd.DataFrame
        DataFrame with annotations for this clip, including begin_time_clip and end_time_clip 
        columns with times relative to clip start
    clip_duration : float
        Duration of the clip in seconds
    max_freq_hz : float
        Maximum frequency in Hz for the spectrogram
    S : int, default=7
        Grid size for the YOLO grid (S x S)
    B : int, default=2
        Number of bounding boxes per grid cell
    class_map : Dict[Tuple[str, str], int], default=None
        Dictionary mapping (species, call_type) to class indices
        
    Returns:
    --------
    target_tensor : np.ndarray
        YOLO target tensor with shape (S, S, B*5 + C)
        Where C is the number of classes, and B*5 represents B bounding boxes
        with 5 parameters each (x_cell, y_cell, width, height, confidence)
    """
    if class_map is None:
        raise ValueError("class_map must be provided")
        
    C = len(class_map)
    target_tensor = np.zeros((S, S, B * 5 + C), dtype=np.float32)
    cell_obj_count = np.zeros((S, S), dtype=int)
    
    for _, ann_row in annotations_clip.iterrows():
        species = ann_row['species'].lower()
        call_type = ann_row['call_type'].lower()
        class_key = (species, call_type)
        
        class_idx = class_map.get(class_key)
        if class_idx is None:
            continue
            
        # Extract annotation data
        begin_t = ann_row['begin_time_clip']
        end_t = ann_row['end_time_clip']
        low_f, high_f = ann_row['low_freq'], ann_row['high_freq']
        
        # Calculate normalized coordinates
        center_t_abs = (begin_t + end_t) / 2
        center_f_abs = (low_f + high_f) / 2
        width_t_abs = end_t - begin_t
        height_f_abs = high_f - low_f
        
        if width_t_abs <= 0 or height_f_abs <= 0:
            continue
            
        # Normalize to [0, 1]
        center_t_norm = np.clip(center_t_abs / clip_duration, 0.0, 1.0)
        center_f_norm = np.clip(center_f_abs / max_freq_hz, 0.0, 1.0)
        width_t_norm = np.clip(width_t_abs / clip_duration, 0.0, 1.0)
        height_f_norm = np.clip(height_f_abs / max_freq_hz, 0.0, 1.0)
        
        if width_t_norm == 0 or height_f_norm == 0:
            continue
            
        # Get grid cell indices
        grid_col_float = center_t_norm * S
        grid_row_float = center_f_norm * S
        grid_col_idx = min(int(grid_col_float), S - 1)
        grid_row_idx = min(int(grid_row_float), S - 1)
        
        # Add to grid cell if space available
        if cell_obj_count[grid_row_idx, grid_col_idx] < B:
            b_idx = cell_obj_count[grid_row_idx, grid_col_idx]
            x_cell = grid_col_float - grid_col_idx
            y_cell = grid_row_float - grid_row_idx
            
            # Update tensor values
            box_offset = b_idx * 5
            target_tensor[grid_row_idx, grid_col_idx, box_offset:box_offset+5] = [
                x_cell, y_cell, width_t_norm, height_f_norm, 1.0
            ]
            
            # Set class probability if not already set
            class_offset = B * 5
            if np.sum(target_tensor[grid_row_idx, grid_col_idx, class_offset:]) == 0:
                target_tensor[grid_row_idx, grid_col_idx, class_offset + class_idx] = 1.0
                
            cell_obj_count[grid_row_idx, grid_col_idx] += 1
            
    return target_tensor


def process_audio_file_to_clips(
    annotations_df: pd.DataFrame, 
    file_name: str, 
    clip_duration: float = 5.0, 
    overlap: float = 1.0, 
    max_freq_hz: float = 16000.0, 
    S: int = 7, 
    B: int = 2, 
    class_map: Optional[Dict[Tuple[str, str], int]] = None
) -> Tuple[List[AudioClip], Dict[str, np.ndarray]]:
    """
    Process an audio file's annotations into clips and YOLO tensors.
    
    Parameters:
    -----------
    annotations_df : pd.DataFrame
        DataFrame containing annotations for this file
    file_name : str
        Name of the audio file
    clip_duration : float, default=5.0
        Duration of each clip in seconds
    overlap : float, default=1.0
        Overlap between consecutive clips in seconds
    max_freq_hz : float, default=16000.0
        Maximum frequency in Hz for the spectrogram
    S : int, default=7
        Grid size for the YOLO grid
    B : int, default=2
        Number of bounding boxes per grid cell
    class_map : Dict[Tuple[str, str], int], default=None
        Dictionary mapping (species, call_type) to class indices
        
    Returns:
    --------
    clip_dataset : List[AudioClip]
        List of AudioClip objects containing clip information
    clip_yolo_tensors : Dict[str, np.ndarray]
        Dictionary mapping clip names to YOLO tensors
    """
    clip_dataset: List[AudioClip] = []
    clip_yolo_tensors: Dict[str, np.ndarray] = {}
    
    # Determine total duration of the audio file
    total_duration_sec = annotations_df['end_time'].max()
    
    if pd.isna(total_duration_sec) or total_duration_sec <= 0:
        return clip_dataset, clip_yolo_tensors
    
    # Generate clips metadata
    clips_metadata = generate_clips_metadata(total_duration_sec, clip_duration, overlap)
    
    for clip_id, clip_start, clip_end in clips_metadata:
        clip_duration_actual = clip_end - clip_start
        clip_name = f"{file_name}_clip_{clip_id:03d}"
        
        # Get annotations for this clip
        annotations_clip = annotations_in_clip(annotations_df, clip_start, clip_end)
        
        # Create YOLO tensor for this clip
        clip_tensor = create_yolo_tensor_for_clip(
            annotations_clip, clip_duration_actual, max_freq_hz, S, B, class_map
        )
        
        # Create AudioClip object
        audio_clip = AudioClip(
            original_file=file_name,
            clip_id=clip_id,
            clip_name=clip_name,
            start_time=clip_start,
            end_time=clip_end,
            duration=clip_duration_actual,
            num_annotations=len(annotations_clip),
            annotations=annotations_clip
        )
        
        clip_dataset.append(audio_clip)
        clip_yolo_tensors[clip_name] = clip_tensor
        
    return clip_dataset, clip_yolo_tensors


def process_dataset_to_clips(
    df: pd.DataFrame, 
    clip_duration: float = 5.0, 
    overlap: float = 1.0, 
    max_freq_hz: float = 16000.0, 
    S: int = 7, 
    B: int = 2, 
    class_map: Optional[Dict[Tuple[str, str], int]] = None
) -> Tuple[List[AudioClip], Dict[str, np.ndarray]]:
    """
    Process an entire dataset into clips and YOLO tensors.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing all annotations
    clip_duration : float, default=5.0
        Duration of each clip in seconds
    overlap : float, default=1.0
        Overlap between consecutive clips in seconds
    max_freq_hz : float, default=16000.0
        Maximum frequency in Hz for the spectrogram
    S : int, default=7
        Grid size for the YOLO grid
    B : int, default=2
        Number of bounding boxes per grid cell
    class_map : Dict[Tuple[str, str], int], default=None
        Dictionary mapping (species, call_type) to class indices
        
    Returns:
    --------
    all_clips : List[AudioClip]
        List of AudioClip objects containing clip information for all files
    all_tensors : Dict[str, np.ndarray]
        Dictionary mapping clip names to YOLO tensors for all files
    """
    all_clips: List[AudioClip] = []
    all_tensors: Dict[str, np.ndarray] = {}
    
    # Process each file's annotations
    for file_name, annotations_for_file in df.groupby('recording_file'):
        clip_dataset, clip_yolo_tensors = process_audio_file_to_clips(
            annotations_for_file, file_name, clip_duration, overlap, 
            max_freq_hz, S, B, class_map
        )
        
        all_clips.extend(clip_dataset)
        all_tensors.update(clip_yolo_tensors)
        
    return all_clips, all_tensors
