"""
Definición de modelos y constructores.
"""

from .annotation import Annotation
from .enums import Specie, CallType, EnhancedEnum
from .enums import get_specie_from_abbreviation, get_call_type_from_abbreviation

__all__ = [
    'Annotation',
    'Specie',
    'CallType',
    'EnhancedEnum',
    'get_specie_from_abbreviation',
    'get_call_type_from_abbreviation',
]
