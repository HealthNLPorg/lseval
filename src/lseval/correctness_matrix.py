from dataclasses import dataclass

from .datatypes import Annotation


@dataclass
class CorrectnessMatrix:
    true_positives: set[Annotation] = set()
    true_negatives: set[Annotation] = set()
    false_positives: set[Annotation] = set()
    false_negatives: set[Annotation] = set()

    def is_true_positive(self, annotation: Annotation) -> bool:
        return annotation in self.true_positives

    def is_true_negative(self, annotation: Annotation) -> bool:
        return annotation in self.true_negatives

    def is_false_positive(self, annotation: Annotation) -> bool:
        return annotation in self.false_positives

    def is_false_negative(self, annotation: Annotation) -> bool:
        return annotation in self.false_negatives

    def get_precision(self) -> float:
        return len(self.true_positives) / (
            len(self.true_positives) + len(self.false_positives)
        )

    def get_recall(self) -> float:
        return len(self.true_positives) / (
            len(self.true_positives) + len(self.false_negatives)
        )

    def get_f_beta(self, beta: float) -> float:
        precision = self.get_precision()
        recall = self.get_recall()
        beta_squared = pow(beta, 2.0)
        return ((1 + beta_squared) * precision * recall) / (
            (beta_squared * precision) + recall
        )

    def get_f1(self) -> float:
        return self.get_f_beta(beta=1.0)
