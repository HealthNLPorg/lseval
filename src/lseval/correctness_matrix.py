from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from functools import lru_cache
from typing import Any


@lru_cache
def precision(
    true_positives: int,
    true_negatives: int,
    false_positives: int,
    false_negatives: int,
) -> float:
    denominator = true_positives + false_positives
    if denominator == 0:
        return float("nan")
    return true_positives / denominator


@lru_cache
def recall(
    true_positives: int,
    true_negatives: int,
    false_positives: int,
    false_negatives: int,
) -> float:
    denominator = true_positives + false_negatives
    if denominator == 0:
        return float("nan")
    return true_positives / denominator


@lru_cache
def f_beta(precision: float, recall: float, beta: float) -> float:
    beta_squared = pow(beta, 2.0)
    denominator = (beta_squared * precision) + recall
    if denominator == 0:
        return float("nan")
    return ((1 + beta_squared) * precision * recall) / denominator


@lru_cache
def f1(precision: float, recall: float) -> float:
    return f_beta(precision, recall, beta=1.0)


class Correctness(IntEnum):
    TRUE_POSITIVE = 0
    TRUE_NEGATIVE = 1
    FALSE_POSITIVE = 2
    FALSE_NEGATIVE = 3
    NA = 4


def score_totals(
    correctness_totals: Mapping[Correctness, int],
) -> tuple[float, float, float, int]:
    _precision = precision(
        true_positives=correctness_totals[Correctness.TRUE_POSITIVE],
        true_negatives=correctness_totals[Correctness.TRUE_NEGATIVE],
        false_positives=correctness_totals[Correctness.FALSE_POSITIVE],
        false_negatives=correctness_totals[Correctness.FALSE_NEGATIVE],
    )
    _recall = recall(
        true_positives=correctness_totals[Correctness.TRUE_POSITIVE],
        true_negatives=correctness_totals[Correctness.TRUE_NEGATIVE],
        false_positives=correctness_totals[Correctness.FALSE_POSITIVE],
        false_negatives=correctness_totals[Correctness.FALSE_NEGATIVE],
    )
    support = (
        correctness_totals[Correctness.TRUE_POSITIVE]
        + correctness_totals[Correctness.FALSE_NEGATIVE]
    )
    return f1(_precision, _recall), _precision, _recall, support


@dataclass
class CorrectnessMatrix[T]:
    true_positives: set[T] = field(default_factory=set)
    true_negatives: set[T] = field(default_factory=set)
    false_positives: set[T] = field(default_factory=set)
    false_negatives: set[T] = field(default_factory=set)
    support: int = 0

    def get_correctness(self, datum: Any) -> Correctness:
        if self.is_true_positive(datum):
            return Correctness.TRUE_POSITIVE
        elif self.is_true_negative(datum):
            return Correctness.TRUE_NEGATIVE
        elif self.is_false_positive(datum):
            return Correctness.FALSE_POSITIVE
        elif self.is_false_negative(datum):
            return Correctness.FALSE_NEGATIVE
        else:
            return Correctness.NA

    def __contains__(self, datum: Any) -> bool:
        return (
            self.is_true_positive(datum)
            or self.is_true_negative(datum)
            or self.is_false_positive(datum)
            or self.is_false_negative(datum)
        )

    def is_true_positive(self, datum: Any) -> bool:
        return datum in self.true_positives

    def is_true_negative(self, datum: Any) -> bool:
        return datum in self.true_negatives

    def is_false_positive(self, datum: Any) -> bool:
        return datum in self.false_positives

    def is_false_negative(self, datum: Any) -> bool:
        return datum in self.false_negatives

    def get_precision(self) -> float:
        return precision(
            true_positives=len(self.true_positives),
            true_negatives=len(self.true_negatives),
            false_positives=len(self.false_positives),
            false_negatives=len(self.false_negatives),
        )

    def get_recall(self) -> float:
        return recall(
            true_positives=len(self.true_positives),
            true_negatives=len(self.true_negatives),
            false_positives=len(self.false_positives),
            false_negatives=len(self.false_negatives),
        )

    def get_f_beta(self, beta: float) -> float:
        return f_beta(
            precision=self.get_precision(), recall=self.get_recall(), beta=beta
        )

    def get_f1(self) -> float:
        return f1(
            precision=self.get_precision(),
            recall=self.get_recall(),
        )

    def get_support(self) -> int:
        if self.support != len(self.true_positives) + len(self.false_negatives):
            ValueError(
                f"Supports should be {self.support}, actual total of true positives and false negatives is {len(self.true_positives) + len(self.false_negatives)}"
            )
            return -1
        return self.support

    def to_correctness_totals(self) -> Mapping[Correctness, int]:
        return {
            Correctness.TRUE_POSITIVE: len(self.true_positives),
            Correctness.TRUE_NEGATIVE: len(self.true_negatives),
            Correctness.FALSE_POSITIVE: len(self.false_positives),
            Correctness.FALSE_NEGATIVE: len(self.false_negatives),
        }

    def __len__(self) -> int:
        return (
            len(self.false_negatives)
            + len(self.false_positives)
            + len(self.true_negatives)
            + len(self.true_positives)
        )
