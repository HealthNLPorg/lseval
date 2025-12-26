import json
from lseval.correctness_matrix import CorrectnessMatrix
from lseval.datatypes import Entity, Relation
from lseval.score import (
    build_entity_correctness_matrix,
    build_relation_correctness_matrix,
)
from lseval.utils import get_unique_label_studio_id
import xml.etree.ElementTree as ET


def update_schema(
    current_schema: ET.ElementTree,
    reference_annotator: str,
    prediction_annotator: str,
) -> ET.ElementTree | None:
    NotImplementedError("Figure this out")
    return None


def confirm_linked_annotations(
    relations: list[Relation], entities: set[Entity]
) -> bool:
    NotImplementedError("Figure this out")
    return False


def recoordinate_annotation_ids(
    entity_correctness_matrix: CorrectnessMatrix[Entity],
    relation_correctness_matrix: CorrectnessMatrix[Relation],
    used_annotation_ids: set[str],
) -> tuple[CorrectnessMatrix[Entity], CorrectnessMatrix[Relation], set[str]]:
    NotImplementedError("Figure this out")
    return CorrectnessMatrix(), CorrectnessMatrix(), set()


# TP, FP, are from the predicted annotations, FN from the reference,
# as a result the TP and FP annotation IDs will be from predicted,
# and FN will be from reference

# If FN relation, could be because one or more
# of the relevant entities were annotated but
# a relation wasn't annotated, or there were no relevant annotations annotated

# Dually with FP


# Thus will need to recoordinate the IDs
def build_adjudication_file(
    file_id: int,
    reference_annotator: str,
    prediction_annotator: str,
    prediction_entities: set[Entity],
    reference_entities: set[Entity],
    prediction_relations: list[Relation],
    reference_relations: list[Relation],
    used_annotation_ids: set[str],
    overlap: bool,
) -> dict:
    if not confirm_linked_annotations(prediction_relations, prediction_entities):
        ValueError(
            f"Issues with entity - relation linking in {prediction_annotator}'s annotations"
        )
        return {}
    if not confirm_linked_annotations(reference_relations, reference_entities):
        ValueError(
            f"Issues with entity - relation linking in {reference_annotator}'s annotations"
        )
        return {}
    entity_correctness_matrix = (
        build_entity_correctness_matrix(
            predicted_entities=prediction_entities,
            reference_entities=reference_entities,
            overlap=overlap,
        ),
    )
    relation_correctness_matrix = (
        build_relation_correctness_matrix(
            predicted_relations=prediction_relations,
            reference_relations=reference_relations,
            overlap=overlap,
        ),
    )

    return {}
