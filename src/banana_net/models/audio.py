"""
Audio data type definitions for representing audio recordings and chunks.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import numpy as np


@dataclass
class AudioRecording:
    """
    Represents a full audio recording file.
    
    Attributes:
        path: Path to the audio file
        sample_rate: Sampling rate in Hz (typically 44100)
        duration: Duration of the recording in seconds
        channels: Number of audio channels
        metadata: Optional metadata associated with the recording
    """
    path: Path
    sample_rate: int
    duration: float
    channels: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def filename(self) -> str:
        """Return just the filename without the path."""
        return self.path.name
    
    @property
    def total_samples(self) -> int:
        """Total number of samples in the recording."""
        return int(self.duration * self.sample_rate)


@dataclass
class AudioChunk:
    """
    Represents a segment/chunk of an audio recording.
    
    This is typically used for processing long recordings in smaller chunks.
    
    Attributes:
        original: Reference to the original AudioRecording
        idx: Index of this chunk in the sequence of chunks
        start: Start time in seconds within the original recording
        end: End time in seconds within the original recording
        chunk: Tensor containing the audio data
        spectrogram: Optional pre-computed spectrogram for this chunk
    """
    original: AudioRecording
    idx: int
    start: float
    end: float
    chunk: torch.Tensor
    spectrogram: Optional[torch.Tensor] = None
    
    @property
    def duration(self) -> float:
        """Duration of this chunk in seconds."""
        return self.end - self.start
    
    @property
    def relative_position(self) -> float:
        """Relative position of this chunk in the whole recording (0.0 to 1.0)."""
        if self.original.duration > 0:
            return self.start / self.original.duration
        return 0.0
    
    def to_numpy(self) -> np.ndarray:
        """Convert audio tensor to numpy array."""
        return self.chunk.cpu().numpy()
    
    def __len__(self) -> int:
        """Number of samples in the chunk."""
        return self.chunk.shape[-1]