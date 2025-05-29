
from enum import Enum

class Species(Enum):
    """
    Enum for species.
    """
    LW = "weddells_saddleBack_tamarin"
    PT = "toppins_titi_monkey"
    CC = "shock_headed_capuchin_monkey"
    AC = "peruvian_spider_monkey"
    AA = "night_monkey"
    SM = "large_headed_capuchin"
    SB = "bolivian_squirrel_monkey"

class CallTypes(Enum):
    """
    Enum for call types.
    """
    CS = "contact_syllable"
    TA = "territorial_alarm"
    