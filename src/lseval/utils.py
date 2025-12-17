import json
import logging
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from itertools import groupby
from operator import itemgetter
from typing import cast

from more_itertools import partition

from .datatypes import (
    AnnotatedFile,
    DocTimeRel,
    Entity,
    Relation,
    SingleAnnotatorCorpus,
)

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


CORE_ATTRIBUTES = {"DocTimeRel", "CUI", "Event"}


def parse_dtr(entity: dict) -> DocTimeRel:
    if entity.get("from_name") != "DocTimeRel":
        ValueError(f"Wrong entity type for parse_dtr: {entity['from_name']}")
        return DocTimeRel.NA
    entity_value = entity.get("value")
    if entity_value is None:
        ValueError(f"Missing value field for DTR entity: {entity}")
        return DocTimeRel.NA
    dtr_choices = entity_value.get("choices", [])
    if len(dtr_choices) != 1:
        ValueError(f"Invalid values for DTR choices: {dtr_choices}")
        return DocTimeRel.NA
    # Don't worry there's a _missing_ method
    return DocTimeRel(dtr_choices[0])


def parse_cuis(entity: dict) -> tuple[str, ...]:
    if entity.get("from_name") != "CUI":
        ValueError(f"Wrong entity type for parse_cuis: {entity['from_name']}")
        return ()
    entity_value = entity.get("value")
    if entity_value is None:
        ValueError(f"Missing value field for CUIS entity: {entity}")
        return ()
    return tuple(sorted(entity_value.get("text", [])))


def parse_text(entity: dict) -> str | None:
    if entity.get("from_name") != "DocTimeRel" and entity.get("from_name") != "Event":
        ValueError(f"Wrong entity type for parse_text: {entity['from_name']}")
        return None
    entity_value = entity.get("value")
    if entity_value is None:
        ValueError(f"Missing value field for DTR/Event entity: {entity}")
        return None
    return entity_value.get("text", [])


def organize_corpus_annotations_by_annotator[T](
    raw_json_corpus: Iterable[dict],
    id_to_unique_annotator: Mapping[int, T],
    annotator_ids_to_ignore: list[int],
) -> dict[T, SingleAnnotatorCorpus]:
    annotator_to_files = defaultdict(lambda: deque())
    for raw_json_file in raw_json_corpus:
        raw_file_dictionary = organize_file_by_annotator_id(raw_json_file)
        annotator_merged_file_dictionary = organize_file_annotations_by_annotator(
            raw_file_dictionary, id_to_unique_annotator, annotator_ids_to_ignore
        )
        for annotator, annotated_file in annotator_merged_file_dictionary.items():
            annotator_to_files[annotator].append(annotated_file)
    return {
        annotator: SingleAnnotatorCorpus(annotated_files=frozenset(annotated_files))
        for annotator, annotated_files in annotator_to_files.items()
    }


def organize_file_annotations_by_annotator[T](
    raw_file_dictionary: dict,
    id_to_unique_annotator: Mapping[int, T],
    annotator_ids_to_ignore: list[int],
) -> dict[T, AnnotatedFile]:
    annotator_to_annotated_files = defaultdict(lambda: deque())
    for annotator_id, annotated_file in raw_file_dictionary.items():
        if annotator_id not in annotator_ids_to_ignore:
            annotator_to_annotated_files[id_to_unique_annotator[annotator_id]].append(
                annotated_file
            )
    merged = {}
    for annotator, annotated_files in annotator_to_annotated_files.items():
        if len(annotated_files) > 1:
            ValueError(
                f"Single annotator {annotator} has {len(annotated_files)} annotations done under multiple IDs - merge logic not presently supported - consult {annotator} for selection"
            )
        merged[annotator] = annotated_files.pop()
    return merged


def organize_file_by_annotator_id(
    raw_file_dictionary: dict,
) -> dict[int, AnnotatedFile]:
    file_id = int(raw_file_dictionary["id"])
    id_annotations_ls = raw_file_dictionary["annotations"]

    return {
        annotations["completed_by"]: id_annotations_to_file(
            file_id, annotations["result"]
        )
        for annotations in id_annotations_ls
    }


def id_annotations_to_file(file_id: int, id_annotations: list[dict]) -> AnnotatedFile:
    def is_relation(annotation: dict) -> bool:
        return annotation["type"] == "relation"

    entity_iter, relation_iter = partition(is_relation, id_annotations)
    ann_id_to_entity = organize_entities_by_ann_id(file_id, entity_iter)
    linked_relations = frozenset(
        parse_and_coordinate_relations(file_id, relation_iter, ann_id_to_entity)
    )
    return AnnotatedFile(
        file_id=file_id,
        entities=frozenset(
            entity for entity in ann_id_to_entity.values()
        ),  # Mapping has the values method as a mixin https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping
        relations=linked_relations,
    )


def organize_entities_by_ann_id(
    file_id: int,
    entity_annotations: Iterable[dict],
) -> Mapping[str, Entity]:
    def get_annotation_id(entity_annotation: dict) -> str:
        annotation_id = entity_annotation.get("id")
        if annotation_id is None:
            ValueError(f"Entity: {entity_annotation} is missing id")
        return cast(str, annotation_id)

    annotation_id_to_entity = {}
    for annotation_id, entity_iter in groupby(
        entity_annotations, key=get_annotation_id
    ):
        entities = list(entity_iter)

        single_entity = coordinate_attribute_entities_to_single(file_id, entities)
        if single_entity is not None:
            annotation_id_to_entity[annotation_id] = single_entity
    return annotation_id_to_entity


def get_indices(entity: dict) -> tuple[int, int]:
    e_value = entity.get("value")
    if e_value is None:
        ValueError(f"Entity missing value field {entity}")
        return -1, -1
    else:
        return e_value["start"], e_value["end"]


def parse_event_type(entity: dict) -> str | None:
    if entity.get("from_name") != "Event":
        ValueError(f"Wrong entity type for parse_event_type: {entity['from_name']}")
        return None
    entity_value = entity.get("value")
    if entity_value is None:
        ValueError(f"Missing value field for event type entity: {entity}")
        return None
    event_type_labels = entity_value.get("labels", [])
    if len(event_type_labels) != 1:
        ValueError(f"Invalid values for event type labels: {event_type_labels}")
        return None
    # Don't worry there's a _missing_ method
    return str(event_type_labels[0])


def coordinate_attribute_entities_to_single(
    file_id: int, entities: list[dict], attributes: set[str] = CORE_ATTRIBUTES
) -> Entity | None:
    entity_attribute_to_instances = {}
    for attribute_type, entity_iter in groupby(
        sorted(entities, key=itemgetter("from_name")), key=itemgetter("from_name")
    ):
        if attribute_type in attributes:
            attribute_entities = list(entity_iter)
            if len(attribute_entities) > 1:
                logger.error(
                    "%s has more than one entry for a particular entity %s",
                    attribute_type,
                    attribute_entities,
                )
            entity_attribute_to_instances[attribute_type] = attribute_entities[0]

    first_inds = get_indices(entities[0])

    if not all(get_indices(entity) == first_inds for entity in entities[1:]):
        ValueError(f"Entities not matching on indices {entities}")
        return None

    return Entity(
        file_id=file_id,
        span=first_inds,
        text=parse_text(entity_attribute_to_instances["Event"]),
        dtr=parse_dtr(entity_attribute_to_instances["DocTimeRel"]),
        label=parse_event_type(entity_attribute_to_instances["Event"]),
        cuis=parse_cuis(entity_attribute_to_instances["CUI"]),
        source_annotations=[json.dumps(entity) for entity in entities],
    )


def parse_and_coordinate_relations(
    file_id: int,
    relation_annotations: Iterable[dict],
    ann_id_to_entity: Mapping[str, Entity],
) -> Iterable[Relation]:
    def json_annotation_to_relation(annotation: dict) -> Relation:
        label = annotation["labels"]
        assert isinstance(label, list)
        return Relation(
            file_id=file_id,
            arg1=ann_id_to_entity[annotation["from_id"]],
            arg2=ann_id_to_entity[annotation["to_id"]],
            label=label,
            source_annotations=[json.dumps(annotation)],
        )

    for annotation in relation_annotations:
        yield json_annotation_to_relation(annotation)
