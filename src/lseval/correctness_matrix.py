from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class Correctness(IntEnum):
    TRUE_POSITIVE = 0
    TRUE_NEGATIVE = 1
    FALSE_POSITIVE = 2
    FALSE_NEGATIVE = 3
    NA = 4


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
        denominator = len(self.true_positives) + len(self.false_positives)
        if denominator == 0:
            return float("nan")
        return len(self.true_positives) / denominator

    def get_recall(self) -> float:
        denominator = len(self.true_positives) + len(self.false_negatives)
        if denominator == 0:
            return float("nan")
        return len(self.true_positives) / denominator

    def get_f_beta(self, beta: float) -> float:
        precision = self.get_precision()
        recall = self.get_recall()
        beta_squared = pow(beta, 2.0)
        denominator = (beta_squared * precision) + recall
        if denominator == 0:
            return float("nan")
        return ((1 + beta_squared) * precision * recall) / denominator

    def get_f1(self) -> float:
        return self.get_f_beta(beta=1.0)

    def get_support(self) -> int:
        if self.support != len(self.true_positives) + len(self.false_negatives):
            ValueError(
                f"Supports should be {self.support}, actual total of true positives and false negatives is {len(self.true_positives) + len(self.false_negatives)}"
            )
            return -1
        return self.support

    def __len__(self) -> int:
        return (
            len(self.false_negatives)
            + len(self.false_positives)
            + len(self.true_negatives)
            + len(self.true_positives)
        )
