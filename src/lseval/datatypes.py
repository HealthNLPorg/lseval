from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations, product
from operator import itemgetter
from typing import Any


def overlap_match(arg1_span: tuple[int, int], arg2_span: tuple[int, int]) -> bool:
    return arg1_span[0] < arg2_span[1] and arg1_span[1] > arg2_span[0]


def admits_bijection(preimage: Iterable[Any], image: Iterable[Any]) -> bool:
    # reduced function is a permutation but allows for repeats
    return sorted(Counter(preimage).values()) == sorted(Counter(image).values())


def get_nth(mapping: Iterable[tuple[Any, ...]], n: int) -> Iterable[Any]:
    return map(itemgetter(n), mapping)


def get_preimage(mapping: Iterable[tuple[Any, Any]]) -> Iterable[Any]:
    return get_nth(mapping, n=0)


def get_image(mapping: Iterable[tuple[Any, Any]]) -> Iterable[Any]:
    return get_nth(mapping, n=1)


def overlap_exists(
    first_spans: set[tuple[int, int]], second_spans: set[tuple[int, int]]
) -> bool:
    for mapping in combinations(product(first_spans, second_spans), r=2):
        # Don't need to worry about iterator exhaustion
        # since itertools.combinations returns tuples
        preimage = get_preimage(mapping)
        image = get_image(mapping)
        if admits_bijection(preimage, image) and all(
            overlap_match(*pair) for pair in mapping
        ):
            return True
    return False


@dataclass
class Entity:
    span: tuple[int, int]
    text: str | None
    dtr: str | None
    cuis = field(default_factory=set)

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
            order_ignored_match = {self.arg1.span, self.arg2.span} == {
                other.arg1.span,
                other.arg2.span,
            }
            return other.label == self.label and order_ignored_match
        return False

    def overlap_match(self, other: Any) -> bool:
        if not isinstance(other, Relation):
            return False
        if self.directed and other.directed:
            order_sensitive_match = self.arg1.overlap_match(
                other.arg1
            ) and self.arg2.overlap_match(other.arg2)

            return other.label == self.label and order_sensitive_match
        if not self.directed and not other.directed:
            this_spans = {self.arg1.span, self.arg2.span}
            other_spans = {other.arg1.span, other.arg2.span}
            order_ignored_match = overlap_exists(this_spans, other_spans)
            return other.label == self.label and order_ignored_match
        return False


@dataclass
class AnnotatedFile:
    file_id: int | None
    entities: set[Entity] = field(default_factory=set)
    relations: set[Relation] = field(default_factory=set)


@dataclass
class SingleAnnotatorCorpus:
    annotated_files: set[AnnotatedFile] = field(default_factory=set)
