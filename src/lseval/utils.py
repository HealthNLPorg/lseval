from collections.abc import Iterable, Mapping

from more_itertools import partition

from .datatypes import AnnotatedFile, Entity, Relation, SingleAnnotatorCorpus


def organize_corpus_annotations_by_annotator[T](
    raw_json_corpus: Iterable[dict], id_to_unique_annotator: Mapping[int, T]
) -> dict[T, SingleAnnotatorCorpus]:
    return {}


def organize_file_annotations_by_annotator[T](
    raw_file_dictionary: dict, id_to_unique_annotator: Mapping[int, T]
) -> dict[T, AnnotatedFile]:
    return {}


def organize_file_by_annotator_id(
    raw_file_dictionary: dict,
) -> dict[int, AnnotatedFile]:
    id_annotations_ls = raw_file_dictionary["annotations"]
    return {
        annotations["completed_by"]: id_annotations_to_file(annotations)
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
    return dict()


def parse_and_coordinate_relations(
    relation_annotations: Iterable[dict], ann_id_to_entity: Mapping[str, Entity]
) -> set[Relation]:
    return set()
