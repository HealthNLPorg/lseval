import json
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from enum import Enum, EnumType
from functools import partial
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
    NotImplementedError("Figure this out")
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
    prediction_entities: Iterable[Entity],
    reference_entities: Iterable[Entity],
    prediction_relations: Iterable[Relation],
    reference_relations: Iterable[Relation],
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
    prediction_entities: Iterable[Entity],
    reference_entities: Iterable[Entity],
    prediction_relations: Iterable[Relation],
    reference_relations: Iterable[Relation],
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


def labels_entity_to_adjudication_entity(annotator: Enum, labels_entity: dict) -> dict:
    return {
        "id": labels_entity["id"],
        "value": {
            "start": labels_entity["value"]["start"],
            "end": labels_entity["value"]["end"],
            "text": labels_entity["value"]["text"],
            "labels": [
                annotator.value
            ],  # At first I thought to mix this with an extant entity but we(I) want to maintain separate label spaces
        },
        "from_name": "IAA",
        "to_name": "text",
        "type": "labels",
        "origin": "manual",
    }


def labels_relation_to_adjudication_relation(
    annotator: Enum, labels_relation: dict
) -> dict:
    return {
        "from_id": labels_relation["from_id"],
        "to_id": labels_relation["to_id"],
        "type": "relation",
        "direction": labels_relation["direction"],
        "labels": [annotator.value],
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
    prediction_entities: Iterable[Entity],
    reference_entities: Iterable[Entity],
    prediction_relations: Iterable[Relation],
    reference_relations: Iterable[Relation],
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
    result = list(
        chain(
            # map(
            #     json.loads,
            #     chain.from_iterable(map(attrgetter("source_annotations"), prediction_entities)),
            # ),
            # map(
            #     json.loads,
            #     chain.from_iterable(map(attrgetter("source_annotations"), prediction_relations)),
            # ),
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
    return result


def adjudicate_correctness_grouped_entities(
    annotator: Enum, entity_group: Iterable[Entity]
) -> Iterable[dict]:
    for _, annotation_id_group in groupby(
        sorted(entity_group, key=attrgetter("label_studio_id")),
        key=attrgetter("label_studio_id"),
    ):
        entities = list(annotation_id_group)
        print(f"total entities: {len(entities)}")
        if len(entities) != 1:
            ValueError(f"Wrong number of entities {len(entities)}")
            return []
        entity = cast(Entity, entities[0])
        source_entities = [
            json.loads(entity_source) for entity_source in entity.source_annotations
        ]
        label_entities = [
            entity for entity in source_entities if entity["type"] == "labels"
        ]
        if len(label_entities) != 1:
            ValueError(
                f"Wrong number of label entities in source annotations {len(label_entities)}"
            )
            return []
        yield labels_entity_to_adjudication_entity(annotator, label_entities[0])
        yield from source_entities


def adjudicate_entities(
    annotators: EnumType,
    prediction_entities: Iterable[Entity],
    reference_entities: Iterable[Entity],
    entity_to_typed_correctness_matrix: Mapping[Entity, CorrectnessMatrix[Entity]],
) -> Iterable[dict]:
    local_get_correctness = partial(get_correctness, entity_to_typed_correctness_matrix)
    for correctness, entity_group in groupby(
        sorted(
            chain(prediction_entities, reference_entities), key=local_get_correctness
        ),
        key=local_get_correctness,
    ):
        entity_group = list(entity_group)
        print(correctness)
        print(len(entity_group))
        match correctness:
            case Correctness.TRUE_POSITIVE:
                print("True positives")
                yield from adjudicate_correctness_grouped_entities(
                    cast(Enum, annotators("Agreement")), entity_group
                )

            case Correctness.FALSE_POSITIVE:
                print("False positives")
                yield from adjudicate_correctness_grouped_entities(
                    cast(Enum, annotators("Prediction")), entity_group
                )
            case Correctness.FALSE_NEGATIVE:
                print("False negatives")
                yield from adjudicate_correctness_grouped_entities(
                    cast(Enum, annotators("Reference")), entity_group
                )
            case other:
                ValueError(f"There shouldn't be any of these {other}")


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
            ValueError(
                f"Wrong number of relations from {from_id} to {to_id: {len(relations)}}"
            )
            return []
        relation = cast(Relation, relations[0])
        source_relations = [
            json.loads(relation_source)
            for relation_source in relation.source_annotations
        ]
        label_relations = [
            entity for entity in source_relations if entity["type"] == "labels"
        ]
        if len(label_relations) != 1:
            ValueError(
                f"Wrong number of label relations in source annotations {len(label_relations)}"
            )
            return []
        yield labels_relation_to_adjudication_relation(annotator, label_relations[0])
        yield relation


def adjudicate_relations(
    annotators: EnumType,
    prediction_relations: Iterable[Relation],
    reference_relations: Iterable[Relation],
    relation_to_typed_correctness_matrix: Mapping[
        Relation, CorrectnessMatrix[Relation]
    ],
) -> Iterable[dict]:
    NotImplementedError()
    local_get_correctness = partial(
        get_correctness, relation_to_typed_correctness_matrix
    )
    for correctness, relation_group in groupby(
        sorted(prediction_relations, key=local_get_correctness),
        key=local_get_correctness,
    ):
        relation_group = list(relation_group)
        # print(correctness)
        # print(len(relation_group))
        match correctness:
            case Correctness.TRUE_POSITIVE:
                yield from adjudicate_correctness_grouped_relations(
                    cast(Enum, annotators("Agreement")), relation_group
                )

            case Correctness.FALSE_POSITIVE:
                yield from adjudicate_correctness_grouped_relations(
                    cast(Enum, annotators("Prediction")), relation_group
                )
            case Correctness.FALSE_NEGATIVE:
                yield from adjudicate_correctness_grouped_relations(
                    cast(Enum, annotators("Reference")), relation_group
                )
            case other:
                ValueError(f"There shouldn't be any of these {other}")
