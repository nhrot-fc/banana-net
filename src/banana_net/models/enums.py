from enum import Enum
from typing import TypeVar, Type, Any, Final, Dict


# Define a type variable for EnhancedEnum subclasses
T = TypeVar("T", bound="EnhancedEnum")


class EnhancedEnum(Enum):
    """
    Clase base para enums con funcionalidades adicionales como conversión
    por índice, valor o nombre, y comparación basada en orden.
    """

    def __str__(self) -> str:
        """Retorna el valor del miembro del enum como una cadena."""
        return str(self.value)

    def __int__(self) -> int:
        """Retorna el índice (basado en cero) del miembro del enum según su orden de definición."""
        return list(type(self)).index(self)

    # Implementación de los métodos de comparación para permitir ordenamiento por índice
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return int(self) < int(other)
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return int(self) > int(other)
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return int(self) <= int(other)
        return NotImplemented

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return int(self) >= int(other)
        return NotImplemented

    @classmethod
    def count(cls: Type[T]) -> int:
        """Retorna el número total de miembros en el enum."""
        return len(cls)

    @classmethod
    def from_index(cls: Type[T], index: int) -> T:
        """Retorna el miembro del enum correspondiente al índice dado."""
        try:
            return list(cls)[index]
        except IndexError:
            raise ValueError(
                f"Índice {index} fuera de rango para el enum {cls.__name__}"
            ) from None

    @classmethod
    def from_value(
        cls: Type[T], value_to_find: Any, case_insensitive: bool = True
    ) -> T:
        """
        Retorna el miembro del enum correspondiente al valor dado.
        Intenta una coincidencia sin distinguir mayúsculas/minúsculas para valores de cadena.
        """
        if isinstance(value_to_find, str) and case_insensitive:
            value_lower = value_to_find.lower()
            for member in cls:
                if (
                    isinstance(member.value, str)
                    and member.value.lower() == value_lower
                ):
                    return member
        # Fallback al constructor estándar de Enum (coincidencia exacta, o para valores no cadena)
        try:
            return cls(value_to_find)
        except ValueError:
            raise ValueError(
                f"No se encontró ningún miembro del enum para el valor '{value_to_find}' en {cls.__name__}"
            ) from None

    @classmethod
    def from_name(cls: Type[T], name_to_find: str, case_insensitive: bool = True) -> T:
        """
        Retorna el miembro del enum correspondiente al nombre dado.
        Permite coincidencia sin distinguir mayúsculas/minúsculas.
        """
        if case_insensitive:
            name_lower = name_to_find.lower()
            for member in cls:
                if member.name.lower() == name_lower:
                    return member
        else:
            # Coincidencia exacta (sensible a mayúsculas/minúsculas) usando Enum.__getitem__
            try:
                return cls[name_to_find]
            except KeyError:
                pass  # Se lanzará ValueError personalizado abajo

        raise ValueError(
            f"No se encontró ningún miembro del enum para el nombre '{name_to_find}' en {cls.__name__}"
        )


class Specie(EnhancedEnum):
    """
    Enum para especies de monos.
    """

    AA = "night_monkey"
    AC = "peruvian_spider_monkey"
    AS = "howler_monkey"
    CC = "shock_headed_capuchin_monkey"
    LW = "weddells_saddleBack_tamarin"
    PT = "toppins_titi_monkey"
    SB = "bolivian_squirrel_monkey"
    SM = "large_headed_capuchin"


class CallType(EnhancedEnum):
    """
    Enum para tipos de llamadas de monos.
    """

    BC = "bc_call"
    CHC = "chc_call"
    CC = "contact_call"
    CS = "contact_syllable"
    DC = "distress_call"
    FS = "fs_call"
    GC = "general_call"
    HC = "hc_call"
    HIC = "hic_call"
    PC = "pc_call"
    PCC = "pcc_call"
    PPC = "ppc_call"
    SPC = "spc_call"
    SQC = "social_call"
    TA = "territorial_alarm"
    TR = "territorial_call"


# Mapeo de abreviaturas a especies
specie_mapping: Final[Dict[str, Specie]] = {s.name.lower(): s for s in Specie}
call_type_mapping: Final[Dict[str, CallType]] = {ct.name.lower(): ct for ct in CallType}


def get_specie_from_abbreviation(abbreviation: str) -> Specie:
    """
    Obtiene la especie a partir de su abreviatura.

    Args:
        abbreviation: Abreviatura de la especie (e.g., 'lw', 'aa')

    Returns:
        Specie correspondiente a la abreviatura

    Raises:
        ValueError: Si la abreviatura no es válida
    """
    if abbreviation not in specie_mapping:
        raise ValueError(
            f"Abreviatura '{abbreviation}' no válida para ninguna especie."
        )
    return specie_mapping[abbreviation]


def get_call_type_from_abbreviation(abbreviation: str) -> CallType:
    """
    Obtiene el tipo de llamada a partir de su abreviatura.

    Args:
        abbreviation: Abreviatura del tipo de llamada (e.g., 'cs', 'ta')

    Returns:
        CallType correspondiente a la abreviatura

    Raises:
        ValueError: Si la abreviatura no es válida
    """
    if abbreviation not in call_type_mapping:
        raise ValueError(
            f"Abreviatura '{abbreviation}' no válida para ningún tipo de llamada."
        )
    return call_type_mapping[abbreviation]
