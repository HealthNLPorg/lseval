from dataclasses import dataclass
from enum import Enum
from typing import Any


@dataclass
class Entity:
    span: tuple[int, int]
    text: str | None = None
    cuis: set[str] = set()

    def __post_init__(self):
        if self.span[1] <= self.span[0]:
            ValueError(f"Invalid span {self.span}")

    def span_match(self, other: Any, overlap: bool = False) -> bool:
        if not isinstance(other, Entity):
            return False
        if overlap:
            return self.overlap_match(other)
        return self.span == other.span

    def overlap_match(self, other: "Entity") -> bool:
        return self.span[0] < other.span[1] and self.span[1] > other.span[0]


@dataclass
class Relation:
    arg1: Entity
    arg2: Entity
    label: Enum
    directed: bool = False

    def __eq__(self, other: Any):
        if not isinstance(other, Relation):
            return False
        if self.directed and other.directed:
            return (other.arg1.span, other.arg2.span, other.label) == (
                self.arg1.span,
                self.arg2.span,
                self.label,
            )
        if not self.directed and not other.directed:
            spans = {self.arg1.span, self.arg2.span}
            order_ignored_match = other.arg1.span in spans and other.arg2.span in spans
            return other.label == self.label and order_ignored_match
        return False

    def overlap_match(self, other: Any) -> bool:
        if not isinstance(other, Relation):
            return False
        return False


Annotation = Entity | Relation


@dataclass
class AnnotatedFile:
    entities: set[Entity]
    relations: set[Relation]


@dataclass
class SingleAnnotatorCorpus:
    annotated_files: set[AnnotatedFile]
