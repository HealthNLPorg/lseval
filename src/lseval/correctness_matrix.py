from dataclasses import dataclass, field
from typing import Any


@dataclass
class CorrectnessMatrix[T]:
    true_positives: set[T] = field(default_factory=set)
    true_negatives: set[T] = field(default_factory=set)
    false_positives: set[T] = field(default_factory=set)
    false_negatives: set[T] = field(default_factory=set)
    support: int = 0

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
        return self.support
