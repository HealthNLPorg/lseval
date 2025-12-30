import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from enum import Enum, EnumType
from functools import partial
from itertools import chain

from lseval.correctness_matrix import CorrectnessMatrix
from lseval.datatypes import Entity, Relation


def update_schema(
    current_schema: ET.ElementTree,
    reference_annotator: str,
    prediction_annotator: str,
) -> ET.ElementTree | None:
    NotImplementedError("Figure this out")
    return None


def relation_is_linked(entities: set[Entity], relation: Relation) -> bool:
    return relation.arg1 in entities and relation.arg2 in entities


def confirm_linked_annotations(
    relations: Iterable[Relation], entities: set[Entity]
) -> bool:
    local_is_linked = partial(relation_is_linked, entities)
    return all(map(local_is_linked, relations))


def valid_false_postive_relation_arguments(
    entity_correctness_matrix: CorrectnessMatrix[Entity], relation: Relation
) -> bool:
    arg1_valid = entity_correctness_matrix.is_true_positive(
        relation.arg1
    ) or entity_correctness_matrix.is_true_positive(relation.arg1)
    arg2_valid = entity_correctness_matrix.is_true_positive(
        relation.arg2
    ) or entity_correctness_matrix.is_true_positive(relation.arg2)
    return arg1_valid and arg2_valid


def recoordinate_annotation_ids(
    entity_correctness_matrix: CorrectnessMatrix[Entity],
    relation_correctness_matrix: CorrectnessMatrix[Relation],
    overlap: bool,
    used_annotation_ids: set[str],
) -> tuple[CorrectnessMatrix[Entity], CorrectnessMatrix[Relation], set[str]]:
    # Valid true positives
    if not confirm_linked_annotations(
        relation_correctness_matrix.true_positives,
        entity_correctness_matrix.true_positives,
    ):
        ValueError("True positive relation not rooted in true positive entities")
        return CorrectnessMatrix(), CorrectnessMatrix(), set()
    # Valid false negatives
    if not confirm_linked_annotations(
        relation_correctness_matrix.false_negatives,
        entity_correctness_matrix.true_positives
        | entity_correctness_matrix.false_negatives,
    ):
        ValueError(
            "False negative relation not rooted in true positive or false negative entities"
        )
        return CorrectnessMatrix(), CorrectnessMatrix(), set()
    # Valid false positives
    local_fp_valid = partial(
        valid_false_postive_relation_arguments, entity_correctness_matrix
    )
    if not all(map(local_fp_valid, relation_correctness_matrix.false_positives)):
        ValueError(
            "False postive relation not rooted in true positive or false positive entities"
        )
        return CorrectnessMatrix(), CorrectnessMatrix(), set()
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
    file_text: str,
    total_files: int,  # For their weird ID generation scheme for data
    # which aren't annotations
    reference_annotator: str,
    prediction_annotator: str,
    prediction_entities: set[Entity],
    reference_entities: set[Entity],
    prediction_relations: list[Relation],
    reference_relations: list[Relation],
    # Temporary (?) dumb way to handle type enforcement
    # from the "agnostic" API end
    entity_to_typed_correctness_matrix: Mapping[Entity, CorrectnessMatrix[Entity]],
    # You know what?  Handle the FN wrangling upstream too
    relation_to_typed_correctness_matrix: Mapping[
        Relation, CorrectnessMatrix[Relation]
    ],
) -> dict:
    return {
        "id": file_id,
        "data": {"text": file_text},
        "predictions": build_preannotations(
            prediction_id=file_id + total_files,
            reference_annotator=reference_annotator,
            prediction_annotator=prediction_annotator,
            prediction_entities=prediction_entities,
            reference_entities=reference_entities,
            prediction_relations=prediction_relations,
            reference_relations=reference_relations,
            entity_to_typed_correctness_matrix=entity_to_typed_correctness_matrix,
            relation_to_typed_correctness_matrix=relation_to_typed_correctness_matrix,
        ),
    }


def build_preannotations(
    prediction_id: int,
    reference_annotator: str,
    prediction_annotator: str,
    prediction_entities: set[Entity],
    reference_entities: set[Entity],
    prediction_relations: list[Relation],
    reference_relations: list[Relation],
    entity_to_typed_correctness_matrix: Mapping[Entity, CorrectnessMatrix[Entity]],
    relation_to_typed_correctness_matrix: Mapping[
        Relation, CorrectnessMatrix[Relation]
    ],
) -> list[dict]:
    return [
        {
            "id": prediction_id,
            "result": insert_adjudication_data(
                reference_annotator=reference_annotator,
                prediction_annotator=prediction_annotator,
                prediction_entities=prediction_entities,
                reference_entities=reference_entities,
                prediction_relations=prediction_relations,
                reference_relations=reference_relations,
                entity_to_typed_correctness_matrix=entity_to_typed_correctness_matrix,
                relation_to_typed_correctness_matrix=relation_to_typed_correctness_matrix,
            ),
        }
    ]


def labels_annotation_to_adjudication_annotation(
    annotator: Enum, text_annotation: dict
) -> dict:
    return {
        "id": text_annotation["id"],
        "value": {
            "start": text_annotation["value"]["start"],
            "end": text_annotation["value"]["end"],
            "text": text_annotation["value"]["text"],
            "labels": [annotator.value],
        },
        "from_name": "IAA",
        "to_name": "text",
        "type": "labels",
        "origin": "manual",
    }


def insert_adjudication_data(
    reference_annotator: str,
    prediction_annotator: str,
    prediction_entities: set[Entity],
    reference_entities: set[Entity],
    prediction_relations: list[Relation],
    reference_relations: list[Relation],
    entity_to_typed_correctness_matrix: Mapping[Entity, CorrectnessMatrix[Entity]],
    relation_to_typed_correctness_matrix: Mapping[
        Relation, CorrectnessMatrix[Relation]
    ],
) -> list[dict]:
    annotators = Enum(
        "Annotator",
        [
            (reference_annotator, "Reference"),
            (prediction_annotator, "Prediction"),
            ("Agreement", "Agreement"),
        ],
    )
    return list(
        chain(
            adjudicate_entities(
                annotators,
                prediction_entities,
                reference_entities,
                entity_to_typed_correctness_matrix,
            ),
            adjudicate_relations(
                annotators,
                prediction_relations,
                reference_relations,
                relation_to_typed_correctness_matrix,
            ),
        )
    )


def adjudicate_entities(
    annotators: EnumType,
    prediction_entities: Iterable[Entity],
    reference_entities: Iterable[Entity],
    entity_to_typed_correctness_matrix: Mapping[Entity, CorrectnessMatrix[Entity]],
) -> Iterable[dict]:
    return []


def adjudicate_relations(
    annotators: EnumType,
    prediction_entities: Iterable[Relation],
    reference_entities: Iterable[Relation],
    relation_to_typed_correctness_matrix: Mapping[
        Relation, CorrectnessMatrix[Relation]
    ],
) -> Iterable[dict]:
    return []
