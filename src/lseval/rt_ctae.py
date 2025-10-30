from dataclasses import dataclass
from enum import Enum
from typing import Any

from .datatypes import Entity, Relation


class NaranjoScale(Enum):
    DOUBTFUL = "Doubtful"
    POSSIBLE = "Possible"
    PROBABLE = "Probable"
    CERTAIN = "Certain"


@dataclass
class CausalRelation(Relation):
    def __post_init__(self):
        if not validate_radiation_treatment(self.arg1):
            ValueError(
                f"{self.arg1} is not a radiation treatment - convention is radiation treatment is the anchor"
            )
        if not validate_adverse_event(self.arg1):
            ValueError(
                f"{self.arg1} is not a adverse event - convention is adverse event is the anchor"
            )
        if not validate_naranjo_label(self.label):
            ValueError(f"Invalid causality label {self.label}")


def validate_radiation_treatment(entity: Entity) -> bool:
    return False


def validate_adverse_event(entity: Entity) -> bool:
    return False


def validate_naranjo_label(label: Any) -> bool:
    return label in NaranjoScale
