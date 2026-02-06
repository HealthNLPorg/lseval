import json
import logging
import operator
from collections import defaultdict, deque
from collections.abc import Container, Iterable, Mapping, Sequence
from functools import partial
from operator import itemgetter
from typing import cast

from more_itertools import all_equal, map_reduce, partition

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
        raise ValueError(f"Wrong entity type for parse_dtr: {entity['from_name']}")
    entity_value = entity.get("value")
    if entity_value is None:
        raise ValueError(f"Missing value field for DTR entity: {entity}")
    dtr_choices = entity_value.get("choices", [])
    if len(dtr_choices) != 1:
        raise ValueError(f"Invalid values for DTR choices: {dtr_choices}")
    # Don't worry there's a _missing_ method
    return DocTimeRel(dtr_choices[0])


def parse_cuis(entity: dict) -> tuple[str, ...]:
    if entity.get("from_name") != "CUI":
        raise ValueError(f"Wrong entity type for parse_cuis: {entity['from_name']}")
    entity_value = entity.get("value")
    if entity_value is None:
        raise ValueError(f"Missing value field for CUIS entity: {entity}")
    return tuple(
        sorted(
            map(
                str.upper,
                map(
                    str.strip,
                    filter(operator.is_not_none, entity_value.get("text", [])),
                ),
            )
        )
    )


def parse_text(entity: dict) -> str | None:
    if entity.get("from_name") != "DocTimeRel" and entity.get("from_name") != "Event":
        entity_space = entity["from_name"]
        raise ValueError(f"Wrong entity type for parse_text: {entity_space}")
    entity_value = entity.get("value")
    if entity_value is None:
        raise ValueError(f"Missing value field for DTR/Event entity: {entity}")
    return entity_value.get("text", [])


def organize_corpus_annotations_by_annotator[T](
    raw_json_corpus: Iterable[dict],
    id_to_unique_annotator: Mapping[int, T],
    annotator_ids_to_ignore: Sequence[int],
) -> Mapping[T, SingleAnnotatorCorpus]:
    annotator_to_files = defaultdict(deque)
    for raw_file_dictionary in raw_json_corpus:
        annotator_id_to_file = organize_file_by_annotator_id(raw_file_dictionary)
        annotator_merged_file_dictionary = organize_file_annotations_by_annotator(
            annotator_id_to_file, id_to_unique_annotator, annotator_ids_to_ignore
        )
        for annotator, annotated_file in annotator_merged_file_dictionary.items():
            annotator_to_files[annotator].append(annotated_file)
    return {
        annotator: SingleAnnotatorCorpus(annotated_files=frozenset(annotated_files))
        for annotator, annotated_files in annotator_to_files.items()
    }


def organize_file_annotations_by_annotator[T](
    annotator_id_to_file: Mapping[int, AnnotatedFile],
    id_to_unique_annotator: Mapping[int, T],
    annotator_ids_to_ignore: Container[int],
) -> Mapping[T, AnnotatedFile]:
    annotator_to_annotated_files = defaultdict(deque)
    for annotator_id, annotated_file in annotator_id_to_file.items():
        if annotator_id not in annotator_ids_to_ignore:
            annotator_to_annotated_files[id_to_unique_annotator[annotator_id]].append(
                annotated_file
            )
    merged = {}
    for annotator, annotated_files in annotator_to_annotated_files.items():
        if len(annotated_files) > 1:
            raise ValueError(
                f"Single annotator {annotator} has {len(annotated_files)} annotations done under multiple IDs - merge logic not presently supported - consult {annotator} for selection"
            )
        merged[annotator] = annotated_files.pop()
    return merged


def organize_file_by_annotator_id(
    raw_file_dictionary: dict,
) -> Mapping[int, AnnotatedFile]:
    file_id = int(raw_file_dictionary["id"])
    id_annotations_ls = raw_file_dictionary["annotations"]

    return {
        annotations["completed_by"]: id_annotations_to_file(
            file_id, annotations["result"], raw_file_dictionary["data"]["text"]
        )
        for annotations in id_annotations_ls
    }


def id_annotations_to_file(
    file_id: int, id_annotations: Iterable[dict], file_text: str
) -> AnnotatedFile:
    def is_relation(annotation: dict) -> bool:
        return annotation["type"] == "relation"

    entity_iter, relation_iter = partition(is_relation, id_annotations)
    ann_id_to_entity = organize_entities_by_ann_id(file_id, entity_iter)
    linked_relations = frozenset(
        parse_and_coordinate_relations(file_id, relation_iter, ann_id_to_entity)
    )
    return AnnotatedFile(
        file_id=file_id,
        file_text=file_text,
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
            raise ValueError(f"Entity: {entity_annotation} is missing id")
        return cast(str, annotation_id)

    coordindate_to_single = partial(coordinate_attribute_entities_to_single, file_id)
    return {
        annotation_id: entity
        for annotation_id, entity in map_reduce(
            entity_annotations,
            keyfunc=get_annotation_id,
            reducefunc=coordindate_to_single,
        ).items()
        if entity is not None
    }


def get_indices(entity: dict) -> tuple[int, int]:
    e_value = entity.get("value")
    if e_value is None:
        raise ValueError(f"Entity missing value field {entity}")
    else:
        return e_value["start"], e_value["end"]


def parse_event_type(entity: dict) -> str | None:
    if entity.get("from_name") != "Event":
        entity_space = entity["from_name"]
        raise ValueError(f"Wrong entity type for parse_event_type: {entity_space}")
    entity_value = entity.get("value")
    if entity_value is None:
        raise ValueError(f"Missing value field for event type entity: {entity}")
    event_type_labels = entity_value.get("labels", [])
    if len(event_type_labels) != 1:
        raise ValueError(f"Invalid values for event type labels: {event_type_labels}")
    # Don't worry there's a _missing_ method
    return str(event_type_labels[0])


def coordinate_attribute_entities_to_single(
    file_id: int, entities: Sequence[dict], attributes: Container[str] = CORE_ATTRIBUTES
) -> Entity | None:
    def get_first(entities: Sequence[dict]) -> dict:
        if len(entities) > 1:
            logger.error(
                "%d entities with id %s sharing type %s",
                len(entities),
                entities[0]["id"],
                entities[0]["from_name"],
            )
        return entities[0]

    entity_attribute_to_instances = map_reduce(
        (entity for entity in entities if entity["from_name"] in attributes),
        keyfunc=itemgetter("from_name"),
        reducefunc=get_first,
    )

    if not all_equal(map(get_indices, entities)):
        raise ValueError(f"Entities not matching on indices {entities}")

    raw_event = entity_attribute_to_instances.get("Event")
    raw_dtr = entity_attribute_to_instances.get("DocTimeRel")
    raw_cuis = entity_attribute_to_instances.get("CUI")
    return Entity(
        file_id=file_id,
        label_studio_id=entities[0]["id"],
        span=get_indices(entity=entities[0]),
        text=parse_text(raw_event) if raw_event is not None else None,
        dtr=parse_dtr(raw_dtr) if raw_dtr is not None else None,
        label=parse_event_type(raw_event) if raw_event is not None else None,
        cuis=parse_cuis(raw_cuis) if raw_cuis is not None else (),
        source_annotations=tuple(json.dumps(entity) for entity in entities),
    )


def parse_and_coordinate_relations(
    file_id: int,
    relation_annotations: Iterable[dict],
    ann_id_to_entity: Mapping[str, Entity],
) -> Iterable[Relation]:
    def json_annotation_to_relation(annotation: dict) -> Relation:
        label = annotation["labels"]
        assert isinstance(label, list)
        label = tuple(label)
        return Relation(
            file_id=file_id,
            arg1=ann_id_to_entity[annotation["from_id"]],
            arg2=ann_id_to_entity[annotation["to_id"]],
            label=label,
            source_annotations=(json.dumps(annotation),),
        )

    for annotation in relation_annotations:
        yield json_annotation_to_relation(annotation)
