from dataclasses import dataclass
from enum import Enum


@dataclass
class Entity:
    span: tuple[int, int]
    text: str | None = None
    cuis: set[str] = set()

    def __post_init__(self):
        if self.span[1] <= self.span[0]:
            ValueError(f"Invalid span {self.span}")


@dataclass
class Relation:
    arg1: Entity
    arg2: Entity
    label: Enum


Annotation = Entity | Relation


@dataclass
class AnnotatedFile:
    entities: set[Entity]
    relations: set[Relation]


@dataclass
class SingleAnnotatorCorpus:
    annotated_files: set[AnnotatedFile]
