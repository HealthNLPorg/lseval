import json
import xml.etree.ElementTree as ET
from collections.abc import Collection, Iterable, Mapping, Sequence
from enum import Enum, EnumType
from itertools import chain
from operator import attrgetter, itemgetter
from typing import cast

from more_itertools import map_reduce

from lseval.correctness_matrix import Correctness, CorrectnessMatrix
from lseval.datatypes import Entity, Relation


def update_schema(
    current_schema: ET.ElementTree,
    reference_annotator: str,
    prediction_annotator: str,
) -> ET.ElementTree | None:
    raise NotImplementedError("TBD")


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
    filter_agreements: bool = True,
) -> dict | None:
    predictions = build_preannotations(
        prediction_id=file_id + total_files,
        reference_annotator=reference_annotator,
        prediction_annotator=prediction_annotator,
        entity_correctness_matrices=entity_correctness_matrices,
        relation_correctness_matrices=relation_correctness_matrices,
        filter_agreements=filter_agreements,
    )
    if filter_agreements and len(predictions) == 0:
        return None
    return {
        "id": file_id,
        "data": {"text": file_text},
        "predictions": build_preannotations(
            prediction_id=file_id + total_files,
            reference_annotator=reference_annotator,
            prediction_annotator=prediction_annotator,
            entity_correctness_matrices=entity_correctness_matrices,
            relation_correctness_matrices=relation_correctness_matrices,
            filter_agreements=filter_agreements,
        ),
    }


def build_preannotations(
    prediction_id: int,
    reference_annotator: str,
    prediction_annotator: str,
    entity_correctness_matrices: Iterable[CorrectnessMatrix[Entity]],
    relation_correctness_matrices: Iterable[CorrectnessMatrix[Relation]],
    filter_agreements: bool = True,
) -> Sequence[dict]:
    result = insert_adjudication_data(
        reference_annotator=reference_annotator,
        prediction_annotator=prediction_annotator,
        entity_correctness_matrices=entity_correctness_matrices,
        relation_correctness_matrices=relation_correctness_matrices,
        filter_agreements=filter_agreements,
    )
    if filter_agreements and len(result) == 0:
        return []
    return [
        {
            "id": prediction_id,
            "result": result,
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


def labels_relation_to_json_relation(
    from_id: str, to_id: str, direction: str, labels: list[str]
) -> dict:
    return {
        "from_id": from_id,
        "to_id": to_id,
        "type": "relation",
        "direction": direction,
        "labels": labels,
    }


def get_correctness[T](
    t_to_typed_correctness_matrix: Mapping[T, CorrectnessMatrix[T]],
    t: T,
) -> Correctness:
    correctness_matrix = t_to_typed_correctness_matrix.get(t)
    if correctness_matrix is None:
        raise ValueError("All entities should be accounted for")
    return correctness_matrix.get_correctness(t)


def insert_adjudication_data(
    reference_annotator: str,
    prediction_annotator: str,
    entity_correctness_matrices: Iterable[CorrectnessMatrix[Entity]],
    relation_correctness_matrices: Iterable[CorrectnessMatrix[Relation]],
    filter_agreements: bool = True,
) -> Sequence[dict]:
    annotators = Enum(
        "Annotator",
        [
            (reference_annotator, "Reference"),
            (prediction_annotator, "Prediction"),
            ("Agreement", "Agreement"),
        ],
    )
    adjudicated_relations = list(
        adjudicate_relations(
            annotators,
            relation_correctness_matrices,
            filter_agreements,
        )
    )
    argument_entity_ids = set(
        chain.from_iterable(map(itemgetter("from_id", "to_id"), adjudicated_relations))
    )
    result = list(
        chain(
            adjudicated_relations,
            adjudicate_entities(
                annotators=annotators,
                correctness_matrices=entity_correctness_matrices,
                argument_entity_ids=argument_entity_ids,
                filter_agreements=filter_agreements,
            ),
        )
    )
    return result


def adjudicate_correctness_grouped_entities(
    annotator: Enum, entity_group: Iterable[Entity]
) -> Iterable[dict]:
    for entities in map_reduce(
        entity_group, keyfunc=attrgetter("label_studio_id")
    ).values():
        if len(entities) != 1:
            raise ValueError(f"Wrong number of entities {len(entities)}")
        entity = entities[0]
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
        yield labels_entity_to_adjudication_entity(annotator, label_entities[0])
        for entity in source_entities:
            entity["origin"] = "prediction"
            yield entity


def adjudicate_entities(
    annotators: EnumType,
    correctness_matrices: Iterable[CorrectnessMatrix[Entity]],
    argument_entity_ids: Collection[str],
    filter_agreements: bool = True,
) -> Iterable[dict]:
    def is_argument_of_adjudicated_relation(entity: Entity) -> bool:
        return entity.label_studio_id in argument_entity_ids

    for correctness_matrix in correctness_matrices:
        if not filter_agreements:
            yield from adjudicate_correctness_grouped_entities(
                cast(Enum, annotators("Agreement")), correctness_matrix.true_positives
            )
        elif len(argument_entity_ids) > 0:
            yield from adjudicate_correctness_grouped_entities(
                cast(Enum, annotators("Agreement")),
                filter(
                    is_argument_of_adjudicated_relation,
                    correctness_matrix.true_positives,
                ),
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
    for id_directions, relations in map_reduce(
        relation_group, keyfunc=get_relation_arg_ids
    ).items():
        # Using recoordinated IDs since those had to be adjusted
        # from scoring overlaps etc.
        from_id, to_id = id_directions
        if len(relations) != 1:
            raise ValueError(
                f"Wrong number of relations from {from_id} to {to_id}: {len(relations)}"
            )
        relation = relations[0]
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
        yield labels_relation_to_json_relation(
            from_id=from_id,
            to_id=to_id,
            direction=label_relations[0]["direction"],
            labels=[annotator.name],
        )
        yield labels_relation_to_json_relation(
            from_id=from_id,
            to_id=to_id,
            direction=label_relations[0]["direction"],
            labels=label_relations[0]["labels"],
        )


def adjudicate_relations(
    annotators: EnumType,
    correctness_matrices: Iterable[CorrectnessMatrix[Relation]],
    filter_agreements: bool = True,
) -> Iterable[dict]:
    for correctness_matrix in correctness_matrices:
        if not filter_agreements:
            yield from adjudicate_correctness_grouped_relations(
                cast(Enum, annotators("Agreement")), correctness_matrix.true_positives
            )
        yield from adjudicate_correctness_grouped_relations(
            cast(Enum, annotators("Prediction")), correctness_matrix.false_positives
        )
        yield from adjudicate_correctness_grouped_relations(
            cast(Enum, annotators("Reference")), correctness_matrix.false_negatives
        )
