from collections.abc import Iterable, Mapping
from itertools import groupby
from typing import cast

from more_itertools import partition
from collections import deque, defaultdict

from .datatypes import AnnotatedFile, Entity, Relation, SingleAnnotatorCorpus

CORE_ATTRIBUTES = {"DocTimeRel", "CUI"}


def attribute_select(attribute: str, instances: list[str]) -> list[str] | str | None:
    match attribute:
        case "DocTimeRel":
            if len(instances) == 0:
                return None
            elif len(instances) == 1:
                return instances[0]
            else:
                first = instances[0]
                if not all(instance == first for instance in instances):
                    ValueError(f"Not all DTRs match for the same entity: {instances}")
                    return None
                return first
        case "CUI":
            if len(instances) == 0:
                return None
            return instances
        case _:
            ValueError(f"Attribute: {attribute} not presently supported")
            return None


def organize_corpus_annotations_by_annotator[T](
    raw_json_corpus: Iterable[dict], id_to_unique_annotator: Mapping[int, T]
) -> dict[T, SingleAnnotatorCorpus]:
    return {}


def organize_file_annotations_by_annotator[T](
    raw_file_dictionary: dict, id_to_unique_annotator: Mapping[int, T]
) -> dict[T, AnnotatedFile]:
    annotator_to_annotated_files = defaultdict(lambda: deque)
    for annotator_id, annotated_file in raw_file_dictionary.items():
        annotator_to_annotated_files[id_to_unique_annotator[annotator_id]].append(
            annotated_file
        )
    return {}


def organize_file_by_annotator_id(
    raw_file_dictionary: dict,
) -> dict[int, AnnotatedFile]:
    file_id = int(raw_file_dictionary["id"])
    id_annotations_ls = raw_file_dictionary["annotations"]

    def build_annotated_file(annotations: list[dict]) -> AnnotatedFile:
        annotated_file = id_annotations_to_file(annotations)
        annotated_file.file_id = file_id
        return annotated_file

    return {
        annotations["completed_by"]: build_annotated_file(annotations)
        for annotations in id_annotations_ls
    }


def id_annotations_to_file(id_annotations: list[dict]) -> AnnotatedFile:
    def is_relation(annotation: dict) -> bool:
        return annotation["type"] == "relation"

    entity_iter, relation_iter = partition(is_relation, id_annotations)
    ann_id_to_entity = organize_entities_by_ann_id(entity_iter)
    linked_relations = parse_and_coordinate_relations(relation_iter, ann_id_to_entity)
    return AnnotatedFile(
        entities=set(
            ann_id_to_entity.values()
        ),  # Mapping has the values method as a mixin https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping
        relations=linked_relations,
    )


def organize_entities_by_ann_id(
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
        annotation_id_to_entity[annotation_id] = (
            coordinate_attribute_entities_to_single(entities)
        )
    return annotation_id_to_entity


def coordinate_attribute_entities_to_single(
    entities: list[dict], attributes: set[str] = CORE_ATTRIBUTES
) -> Entity | None:
    def get_indices(entity: dict) -> tuple[int, int]:
        e_value = entity.get("value")
        if e_value is None:
            ValueError(f"Entity missing value field {entity}")
            return -1, -1
        else:
            return e_value["start"], e_value["end"]

    first_inds = get_indices(entities[0])

    def ind_agree(entity: dict) -> bool:
        return get_indices(entity) == first_inds

    if not all(ind_agree(entity) for entity in entities[1:]):
        ValueError(f"Entities not matching on indices {entities}")
        return None
    attribute_name_to_ls = {}
    for attribute in attributes:
        attribute_name_to_ls[attribute] = [
            entity.get(attribute)
            for entity in entities
            if entity.get(attribute) is not None
        ]
    dtr = attribute_select("DocTimeRel", attribute_name_to_ls["DocTimeRel"])
    if dtr is not None and isinstance(dtr, list):
        ValueError(f"DTR should be str or None instead is {dtr}")
        dtr = None

    cuis = attribute_select("CUIs", attribute_name_to_ls["CUIs"])

    if isinstance(cuis, list):
        cuis = set(cuis)
    else:
        ValueError(f"CUIs should be None or list[str] instead is {cuis}")
        cuis = set()

    return Entity(
        span=first_inds, text=entities[0]["value"]["text"], dtr=dtr, cuis=cuis
    )


def parse_and_coordinate_relations(
    relation_annotations: Iterable[dict], ann_id_to_entity: Mapping[str, Entity]
) -> set[Relation]:
    def json_annotation_to_relation(annotation: dict) -> Relation:
        if len(annotation["labels"]) > 1:
            ValueError(f"More than one label for relation annotation {annotation}")
        label = annotation["labels"][0]
        return Relation(
            arg1=ann_id_to_entity[annotation["from_id"]],
            arg2=ann_id_to_entity[annotation["to_id"]],
            label=label,
        )

    return {
        json_annotation_to_relation(annotation) for annotation in relation_annotations
    }
