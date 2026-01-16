import json
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from enum import Enum, EnumType
from itertools import chain, groupby
from operator import attrgetter
from typing import cast

from lseval.correctness_matrix import Correctness, CorrectnessMatrix
from lseval.datatypes import Entity, Relation


def update_schema(
    current_schema: ET.ElementTree,
    reference_annotator: str,
    prediction_annotator: str,
) -> ET.ElementTree | None:
    raise NotImplementedError("Figure this out")
    return None


def relation_is_linked(entities: set[Entity], relation: Relation) -> bool:
    return relation.arg1 in entities and relation.arg2 in entities


def build_adjudication_file(
    file_id: int,
    file_text: str,
    total_files: int,  # For their weird ID generation scheme for data
    # which aren't annotations
    reference_annotator: str,
    prediction_annotator: str,
    # Temporary (?) dumb way to handle type enforcement
    # from the "agnostic" API end
    entity_correctness_matrices: Iterable[CorrectnessMatrix[Entity]],
    # You know what?  Handle the FN wrangling upstream too
    relation_correctness_matrices: Iterable[CorrectnessMatrix[Relation]],
) -> dict:
    return {
        "id": file_id,
        "data": {"text": file_text},
        "predictions": build_preannotations(
            prediction_id=file_id + total_files,
            reference_annotator=reference_annotator,
            prediction_annotator=prediction_annotator,
            entity_correctness_matrices=entity_correctness_matrices,
            relation_correctness_matrices=relation_correctness_matrices,
        ),
    }


def build_preannotations(
    prediction_id: int,
    reference_annotator: str,
    prediction_annotator: str,
    entity_correctness_matrices: Iterable[CorrectnessMatrix[Entity]],
    relation_correctness_matrices: Iterable[CorrectnessMatrix[Relation]],
) -> list[dict]:
    return [
        {
            "id": prediction_id,
            "result": insert_adjudication_data(
                reference_annotator=reference_annotator,
                prediction_annotator=prediction_annotator,
                entity_correctness_matrices=entity_correctness_matrices,
                relation_correctness_matrices=relation_correctness_matrices,
            ),
        }
    ]


def labels_entity_to_adjudication_entity(
    annotator: Enum,
    labels_entity: dict,
    from_name: str = "IAA",
    to_name: str = "text",
    entity_type: str = "choices",
    origin: str = "prediction",
) -> dict:
    return {
        "value": {
            "start": labels_entity["value"]["start"],
            "end": labels_entity["value"]["end"],
            "text": labels_entity["value"]["text"],
            entity_type: [
                annotator.name
            ],  # At first I thought to mix this with an extant entity but we(I) want to maintain separate label spaces
        },
        "id": labels_entity["id"],
        "from_name": from_name,
        "to_name": to_name,
        "type": entity_type,
        "origin": origin,
    }


def labels_relation_to_json_adjudication_relation(
    annotator: Enum, labels_relation: dict
) -> dict:
    return {
        "from_id": labels_relation["from_id"],
        "to_id": labels_relation["to_id"],
        "type": "relation",
        "direction": labels_relation["direction"],
        "labels": [annotator.name],
    }


def labels_relation_to_json_relation(labels_relation: dict) -> dict:
    return {
        "from_id": labels_relation["from_id"],
        "to_id": labels_relation["to_id"],
        "type": "relation",
        "direction": labels_relation["direction"],
        "labels": labels_relation["labels"],
    }


def get_correctness[T](
    t_to_typed_correctness_matrix: Mapping[T, CorrectnessMatrix[T]],
    t: T,
) -> Correctness:
    correctness_matrix = t_to_typed_correctness_matrix.get(t)
    if correctness_matrix is None:
        raise ValueError("All entities should be accounted for")
        return Correctness.NA
    return correctness_matrix.get_correctness(t)


def insert_adjudication_data(
    reference_annotator: str,
    prediction_annotator: str,
    entity_correctness_matrices: Iterable[CorrectnessMatrix[Entity]],
    relation_correctness_matrices: Iterable[CorrectnessMatrix[Relation]],
) -> list[dict]:
    annotators = Enum(
        "Annotator",
        [
            (reference_annotator, "Reference"),
            (prediction_annotator, "Prediction"),
            ("Agreement", "Agreement"),
        ],
    )
    result = list(
        chain(
            adjudicate_entities(
                annotators,
                entity_correctness_matrices,
            ),
            adjudicate_relations(
                annotators,
                relation_correctness_matrices,
            ),
        )
    )
    return result


def adjudicate_correctness_grouped_entities(
    annotator: Enum, entity_group: Iterable[Entity]
) -> Iterable[dict]:
    for _, annotation_id_group in groupby(
        sorted(entity_group, key=attrgetter("label_studio_id")),
        key=attrgetter("label_studio_id"),
    ):
        entities = list(annotation_id_group)
        if len(entities) != 1:
            raise ValueError(f"Wrong number of entities {len(entities)}")
            return []
        entity = cast(Entity, entities[0])
        source_entities = [
            json.loads(entity_source) for entity_source in entity.source_annotations
        ]
        label_entities = [
            entity for entity in source_entities if entity["type"] == "labels"
        ]
        if len(label_entities) != 1:
            raise ValueError(
                f"Wrong number of label entities in source annotations {len(label_entities)}"
            )
            return []
        yield labels_entity_to_adjudication_entity(annotator, label_entities[0])
        for entity in source_entities:
            entity["origin"] = "prediction"
            yield entity


def adjudicate_entities(
    annotators: EnumType,
    correctness_matrices: Iterable[CorrectnessMatrix[Entity]],
) -> Iterable[dict]:
    for correctness_matrix in correctness_matrices:
        yield from adjudicate_correctness_grouped_entities(
            cast(Enum, annotators("Agreement")), correctness_matrix.true_positives
        )
        yield from adjudicate_correctness_grouped_entities(
            cast(Enum, annotators("Prediction")), correctness_matrix.false_positives
        )
        yield from adjudicate_correctness_grouped_entities(
            cast(Enum, annotators("Reference")), correctness_matrix.false_negatives
        )


def get_relation_arg_ids(relation: Relation) -> tuple[str, str]:
    return relation.arg1.label_studio_id, relation.arg2.label_studio_id


def adjudicate_correctness_grouped_relations(
    annotator: Enum, relation_group: Iterable[Relation]
) -> Iterable[dict]:
    for id_directions, id_directions_group in groupby(
        sorted(relation_group, key=get_relation_arg_ids),
        key=get_relation_arg_ids,
    ):
        from_id, to_id = id_directions
        relations = list(id_directions_group)
        if len(relations) != 1:
            raise ValueError(
                f"Wrong number of relations from {from_id} to {to_id: {len(relations)}}"
            )
            return []
        relation = cast(Relation, relations[0])
        source_relations = [
            json.loads(relation_source)
            for relation_source in relation.source_annotations
        ]
        label_relations = [
            entity for entity in source_relations if entity["type"] == "relation"
        ]
        if len(label_relations) != 1:
            raise ValueError(
                f"Wrong number of label relations in source annotations {len(label_relations)}"
            )
            return []
        yield labels_relation_to_json_adjudication_relation(
            annotator, label_relations[0]
        )
        yield labels_relation_to_json_relation(label_relations[0])


def adjudicate_relations(
    annotators: EnumType,
    correctness_matrices: Iterable[CorrectnessMatrix[Relation]],
) -> Iterable[dict]:
    for correctness_matrix in correctness_matrices:
        yield from adjudicate_correctness_grouped_relations(
            cast(Enum, annotators("Agreement")), correctness_matrix.true_positives
        )
        yield from adjudicate_correctness_grouped_relations(
            cast(Enum, annotators("Prediction")), correctness_matrix.false_positives
        )
        yield from adjudicate_correctness_grouped_relations(
            cast(Enum, annotators("Reference")), correctness_matrix.false_negatives
        )
