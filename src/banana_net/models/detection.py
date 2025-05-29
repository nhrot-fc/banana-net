"""
Detection result types for acoustic event detection.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

from .audio import AudioRecording
from .events import Event
from .sound_classes import SoundClass


@dataclass
class DetectionResult:
    """
    Represents the result of an acoustic event detection process on an audio recording.
    
    Attributes:
        recording: Reference to the analyzed AudioRecording
        events: List of acoustic Events detected in the recording
        model_name: Optional name of the detection model used
        detection_timestamp: When the detection was performed
        metadata: Additional metadata about the detection process
    """
    recording: AudioRecording
    events: List[Event]
    model_name: Optional[str] = None
    detection_timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def filter_by_class(self, c: SoundClass) -> List[Event]:
        """Return only events of the specified class."""
        return [e for e in self.events if e.sound_class == c]

    def filter_by_time(self, t_min: float, t_max: float) -> List[Event]:
        """Return events whose interval intersects [t_min, t_max]."""
        return [e for e in self.events if not (e.t_end < t_min or e.t_start > t_max)]
    
    def filter_by_confidence(self, threshold: float) -> List[Event]:
        """Return events with confidence score above the threshold."""
        return [e for e in self.events if e.confidence >= threshold]
    
    def get_class_distribution(self) -> Dict[SoundClass, int]:
        """Return a distribution of event counts by class."""
        result = {}
        for event in self.events:
            if event.sound_class in result:
                result[event.sound_class] += 1
            else:
                result[event.sound_class] = 1
        return result
    
    @property
    def event_count(self) -> int:
        """Total number of events detected."""
        return len(self.events)
    
    @property
    def total_event_duration(self) -> float:
        """Sum of the duration of all events in seconds."""
        return sum(event.duration for event in self.events)
    
    def __str__(self) -> str:
        """String representation of the detection result."""
        return (f"DetectionResult for {self.recording.filename}: "
                f"{len(self.events)} events detected")