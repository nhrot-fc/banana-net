"""
Event class definitions for acoustic events detected in audio recordings.
"""
from dataclasses import dataclass, field
from typing import Optional

from .sound_classes import SoundClass, VocalizationType


@dataclass
class Event:
    """
    Represents an acoustic event detected in an audio recording.
    
    An event is defined by its temporal and frequency boundaries,
    classification information, and detection confidence.
    
    Attributes:
        t_start: Start time in seconds from the beginning of the recording
        t_end: End time in seconds
        f_min: Minimum frequency of the event in Hertz
        f_max: Maximum frequency of the event in Hertz
        sound_class: Classification of the sound source (primate species or anthropogenic)
        vocal_type: Type of vocalization or subcategory
        confidence: Detection and classification confidence score (0.0 to 1.0)
        notes: Optional additional information about the event
    """
    t_start: float           # segundos desde inicio
    t_end: float             # segundos
    f_min: float             # Hertz
    f_max: float             # Hertz
    sound_class: SoundClass  # etiqueta c(i)
    vocal_type: VocalizationType  # v(i)
    confidence: float        # p(i), between 0.0 and 1.0
    notes: Optional[str] = field(default=None)  # Optional additional information

    @property
    def duration(self) -> float:
        """Duración del evento en segundos."""
        return self.t_end - self.t_start

    @property
    def frequency_band(self) -> float:
        """Ancho de banda en Hz."""
        return self.f_max - self.f_min

    def overlaps_with(self, other: 'Event') -> bool:
        """Check if this event temporally overlaps with another event."""
        return not (self.t_end <= other.t_start or self.t_start >= other.t_end)

    def __str__(self) -> str:
        """String representation of the event."""
        return (f"{self.sound_class.value} ({self.vocal_type.value}): "
                f"{self.t_start:.2f}s-{self.t_end:.2f}s, "
                f"{self.f_min:.1f}-{self.f_max:.1f}Hz, "
                f"conf={self.confidence:.2f}")