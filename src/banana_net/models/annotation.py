from dataclasses import dataclass
from .enums import Specie, CallType


@dataclass
class Annotation:
    """Representa una única anotación de un evento acústico."""

    begin_time: float
    end_time: float
    low_freq: float
    high_freq: float
    inband_power: float
    specie: Specie
    call_type: CallType
    recording_file: str
    directory: str
