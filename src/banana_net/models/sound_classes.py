"""
Sound class definitions for primate vocalizations and anthropogenic sounds.
"""
from enum import Enum


class SoundClass(Enum):
    """
    Enum representing the taxonomic source (primate species) or anthropogenic sound type.
    
    Primate species:
    - LW: Lemur wolfi
    - PT: Primate tarsier
    - CC: Cercopithecus campbellingi
    - AC: Alouatta caraya
    - AA: Ateles arctoides
    - SM: Saguinus mystax
    - AS: Aotus seniculus
    - SB: Sapajus badius
    
    Anthropogenic sounds:
    - MOTOSIERRA: Chainsaw sounds
    - DISPARO: Gunshot sounds
    - VOZ_HUMANA: Human voice
    - OTRO_ANTHROPO: Other human-originated sounds
    """
    # Primate species
    LW = "LW"           # Lemur wolfi
    PT = "PT"           # Primate tarsier
    CC = "CC"           # Cercopithecus campbellingi
    AC = "AC"           # Alouatta caraya
    AA = "AA"           # Ateles arctoides
    SM = "SM"           # Saguinus mystax
    AS = "AS"           # Aotus seniculus
    SB = "SB"           # Sapajus badius
    
    # Anthropogenic sounds
    MOTOSIERRA = "Motosierra"     # Chainsaw
    DISPARO = "Disparo"           # Gunshot
    VOZ_HUMANA = "Voz Humana"     # Human Voice
    OTRO_ANTHROPO = "Otro"        # Other anthropogenic sounds


class VocalizationType(Enum):
    """
    Enum representing vocalization types or subcategories of acoustic events.
    
    For primates:
    - SILABA: Basic acoustic units
    - CANTO: More complex sequences
    
    For non-biological sounds:
    - NA: Not applicable
    """
    SILABA = "Sílaba"  # Basic acoustic units
    CANTO = "Canto"    # More complex sequences
    NA = "N/A"         # Not applicable for anthropogenic or other sounds