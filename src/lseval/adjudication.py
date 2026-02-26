from __future__ import annotations

import json
import logging
import operator
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from enum import Enum, EnumType, StrEnum
from functools import partial, reduce
from itertools import chain, groupby
from operator import attrgetter, itemgetter

from frozendict import frozendict
from more_itertools import (
    flatten,
    map_reduce,
    one,
    unique_everseen,
)

from lseval.correctness_matrix import Correctness, CorrectnessMatrix
from lseval.datatypes import Entity, Relation

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


class AnnotatorChoice(StrEnum):
    AGREEMENT = "AGREEMENT"
    PREDICTION = "PREDICTION"
    REFERENCE = "REFERENCE"


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
    result = get_adjudication_data(
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


def labels_entity_to_adjudication_entity[T](
    annotator_value: T,
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
                annotator_value
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


def get_adjudication_data(
    reference_annotator: str,
    prediction_annotator: str,
    entity_correctness_matrices: Iterable[CorrectnessMatrix[Entity]],
    relation_correctness_matrices: Iterable[CorrectnessMatrix[Relation]],
    filter_agreements: bool = True,
) -> Sequence[dict]:
    annotators = Enum(
        "Annotator",
        [
            (reference_annotator, AnnotatorChoice.REFERENCE),
            (prediction_annotator, AnnotatorChoice.PREDICTION),
            ("Agreement", AnnotatorChoice.AGREEMENT),
        ],
    )

    def fix_annnotator_name(entity: dict) -> dict:
        if entity["from_name"] == "IAA":
            originals = entity["value"]["choices"]
            entity["value"]["choices"] = [annotators(av.value).name for av in originals]
            return entity
        return entity

    adjudicated_relations = list(
        adjudicate_relations(
            annotators,
            relation_correctness_matrices,
            filter_agreements,
        )
    )
    unique_adjudicated_relations = list(
        unique_everseen(adjudicated_relations, key=frozendict)
    )
    if len(adjudicated_relations) != len(unique_adjudicated_relations):
        raise ValueError(
            f"Of {len(adjudicated_relations)} adjudicated relations {len(unique_adjudicated_relations)} are unique"
        )
    argument_entity_ids = set(
        flatten(map(itemgetter("from_id", "to_id"), adjudicated_relations))
    )
    return order_adjudication_data(
        entities=map(
            fix_annnotator_name,
            adjudicate_entities(
                entity_correctness_matrices=entity_correctness_matrices,
                argument_entity_ids=argument_entity_ids,
                filter_agreements=filter_agreements,
            ),
        ),
        relations=adjudicated_relations,
    )


def order_adjudication_data(
    entities: Iterable[dict],
    relations: Iterable[dict],
) -> Sequence[dict]:
    def entity_sort(entity: dict) -> tuple[tuple[int, int], str]:
        return entity_offsets(entity), entity["id"]

    sorted_entities = sorted(entities, key=entity_sort)
    entity_id_to_index = {
        entity_id: index
        for index, (entity_id, _) in enumerate(
            groupby(sorted_entities, key=itemgetter("id"))
        )
    }

    def relation_sort(relation_dict: dict) -> tuple[int, int]:
        from_id = relation_dict["from_id"]
        to_id = relation_dict["to_id"]
        from_index = entity_id_to_index.get(from_id)
        to_index = entity_id_to_index.get(to_id)
        if not isinstance(from_index, int):
            raise ValueError(f"Phantom from argument {from_id}")
        if not isinstance(to_index, int):
            raise ValueError(f"Phantom to argument {to_id}")
        return from_index, to_index

    return list(chain(sorted_entities, sorted(relations, key=relation_sort)))


def adjudicate_single_id_entity_group[T](
    entities: Sequence[Entity], annotator_value: T
) -> Iterable[dict]:
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
    yield labels_entity_to_adjudication_entity(annotator_value, label_entities[0])
    for entity in source_entities:
        entity["origin"] = "prediction"
        yield entity


def adjudicate_correctness_grouped_entities[T](
    annotator_value: T, entity_group: Iterable[Entity]
) -> Iterable[dict]:
    local_adjudicate_single_id = partial(
        adjudicate_single_id_entity_group, annotator_value=annotator_value
    )
    return flatten(
        map_reduce(
            entity_group,
            keyfunc=attrgetter("label_studio_id"),
            reducefunc=local_adjudicate_single_id,
        ).values()
    )


def adjudicate_entities(
    entity_correctness_matrices: Iterable[CorrectnessMatrix],
    argument_entity_ids: Collection[str],
    filter_agreements: bool,
) -> Iterable[dict]:
    return coordinate_adjudicated_entities(
        adjudicated_entities=adjudicate_individual_entities(
            entity_correctness_matrices=entity_correctness_matrices,
            argument_entity_ids=argument_entity_ids,
            filter_agreements=filter_agreements,
        ),
    )


def entity_offsets(entity: dict) -> tuple[int, int]:
    value = entity.get("value")
    if value is None:
        raise ValueError(f"Entity missing 'value' field: {entity}")
    start = value.get("start")
    end = value.get("end")
    if not (isinstance(start, int) and isinstance(end, int)):
        raise ValueError(f"Problematic offsets {(start, end)}")
    return start, end


def get_annotator[T](ls_dict: dict) -> T:
    annotators = ls_dict["value"]["choices"]
    try:
        return one(annotators)
    except Exception:
        raise ValueError(f"Too many or few few values for annotators {annotators}")


# Assumes annotators will be AnnotatorChoice.PREDICTION and AnnotatorChoice.REFERENCE,
# since agreements, improper values, and singletons will be filtered out upstream
# default is select singleton from reference
def wrangle_conflicts(disagreements: Iterable[dict]) -> dict:
    annotator_to_annotations = map_reduce(disagreements, keyfunc=get_annotator)
    for annotator, annotation_collection in annotator_to_annotations.items():
        if len(annotation_collection) > 1:
            scapegoat = annotation_collection[0]
            scapegoat_offsets = entity_offsets(scapegoat)
            print(annotator_to_annotations)
            raise ValueError(
                f"{annotator} {len(annotation_collection)} IAA entities for one root entity with offsets {scapegoat_offsets}"
            )
    reference_annotations = annotator_to_annotations.get(AnnotatorChoice.REFERENCE)
    if reference_annotations is None or len(reference_annotations) == 0:
        scapegoat = next(
            flatten(annotator_to_annotations.values())
        )  # to avoid consuming exhaustible
        scapegoat_offsets = entity_offsets(scapegoat)
        raise ValueError(
            f"Missing reference annotations for one root entity offsets {scapegoat_offsets}"
        )
    return reference_annotations[0]


def deduplicate_shared_offset_id_entities(entities: Iterable[dict]) -> Iterable[dict]:
    def warned_first(clustered_entities: Sequence[dict]) -> dict:
        try:
            return one(clustered_entities, too_short=RuntimeError, too_long=IndexError)
        except IndexError:
            first = clustered_entities[0]
            first_offsets = entity_offsets(first)
            first_from_name = first["from_name"]
            logger.warning(
                "%d clustered_entities for same from_name %s for entity offsets %s",
                len(clustered_entities),
                first_from_name,
                str(first_offsets),
            )
            return first
        except RuntimeError:
            raise ValueError("No clustered_entities for a key, shouldn't be possible")

    return map_reduce(
        entities, keyfunc=itemgetter("from_name"), reducefunc=warned_first
    ).values()


# For now just select which has most annotations
def select_most_informative_cluster[T, S](
    ids: Collection[T], id_to_entities: Mapping[T, Collection[S]]
) -> T:
    return max(ids, key=lambda _id: len(id_to_entities.get(_id, [])))


def get_consistent_cluster(cluster: Iterable[dict]) -> Sequence[dict]:
    cluster = list(cluster)
    cleaned_cluster = list(unique_everseen(cluster, key=itemgetter("from_name")))
    if len(cluster) != len(cleaned_cluster):
        logger.warning("Duplicate entity types in cluster")
    return cleaned_cluster


def select_most_prominent_id(type_to_annotations: Mapping[str, Sequence[dict]]) -> str:
    type_to_id_totals = {
        _type: Counter(map(itemgetter("id"), annotations))
        for _type, annotations in type_to_annotations.items()
    }

    # Ideally want to use an ID which is in all of them
    def get_coverage(_id: str) -> int:
        return sum(1 for id_totals in type_to_id_totals.values() if _id in id_totals)

    all_ids = set(flatten(type_to_id_totals.values()))
    id_to_coverage = Counter({_id: get_coverage(_id) for _id in all_ids})
    id_to_type_agnostic_coverage = reduce(operator.add, type_to_id_totals.values())

    def coverage_then_total(_id: str) -> tuple[int, int]:
        return id_to_coverage[_id], id_to_type_agnostic_coverage[_id]

    return sorted(all_ids, key=coverage_then_total, reverse=True)[0]


def wrangle_mixed[T](
    annotator_to_annotations: Mapping[T, Sequence[dict]],
) -> Iterable[dict]:
    type_to_annotations = map_reduce(
        flatten(annotator_to_annotations.values()), keyfunc=itemgetter("from_name")
    )
    type_to_disagreement_annotations = map_reduce(
        flatten(
            values
            for annotator, values in annotator_to_annotations.items()
            if annotator != AnnotatorChoice.AGREEMENT
        ),
        keyfunc=itemgetter("from_name"),
    )
    most_prominent_disagreement_id = select_most_prominent_id(
        type_to_disagreement_annotations
    )
    for _type, type_grouped_annotations in type_to_annotations.items():
        candidates = [
            annotation
            for annotation in type_grouped_annotations
            if annotation["id"] == most_prominent_disagreement_id
        ]
        try:
            yield one(candidates, too_short=IndexError, too_long=ValueError)
        except IndexError:
            with_wrong_id = type_grouped_annotations[0]
            with_wrong_id["id"] = most_prominent_disagreement_id
            yield with_wrong_id
        except ValueError:
            candidate = candidates[0]
            logger.warning(
                "%d candidates found for entity with id %s and type %s",
                len(candidates),
                candidate["id"],
                _type,
            )
            yield candidate


def adjudicate_offset_entity_cluster[T](
    offsets_entity_cluster: Iterable[dict],
) -> Iterable[dict]:
    id_to_entities = map_reduce(
        offsets_entity_cluster,
        keyfunc=itemgetter("id"),
        reducefunc=get_consistent_cluster,
    )
    annotator_to_ids = defaultdict(set)
    annotator_to_id = {}
    for entity in flatten(id_to_entities.values()):
        if entity["from_name"] == "IAA":
            annotator_to_ids[get_annotator(entity)].add(entity["id"])
    for annotator, ids in annotator_to_ids.items():
        try:
            annotator_to_id[annotator] = one(
                ids, too_short=IndexError, too_long=ValueError
            )
        except IndexError:
            raise ValueError(
                f"No IDs associated with annotator {annotator} - shouldn't happen"
            )
        except ValueError:
            logger.warning(
                "Mutliple ids annotation %s found for annotator %s. Selecting most informative",
                ", ".join(ids),
                annotator.value,
            )
            annotator_to_id[annotator] = select_most_informative_cluster(
                ids=ids, id_to_entities=id_to_entities
            )
    # TODO clustering should be easier from here
    annotator_to_annotations = {
        annotator: id_to_entities.get(_id, [])
        for annotator, _id in annotator_to_id.items()
    }
    agreements = annotator_to_annotations.get(AnnotatorChoice.AGREEMENT, [])
    predictions = annotator_to_annotations.get(AnnotatorChoice.PREDICTION, [])
    references = annotator_to_annotations.get(AnnotatorChoice.REFERENCE, [])
    match len(agreements), len(predictions), len(references):
        case 0, 0, 0:
            raise ValueError("Shouldn't happen")
        case (
            total_agreements,
            0,
            0,
        ) if total_agreements > 0:
            return agreements
        case (
            0,
            total_predictions,
            0,
        ) if total_predictions > 0:
            return predictions
        case (
            0,
            0,
            total_references,
        ) if total_references > 0:
            return references
        case _:
            yield from wrangle_mixed(annotator_to_annotations)


# def adjudicate_id_entity_cluster[T](
# def old_adjudicate_offset_entity_cluster[T](
#     offsets_entity_cluster: Iterable[dict],
# ) -> Iterable[dict]:
#     normal_entity_iter, iaa_entity_iter = partition(
#         lambda ent: ent["from_name"] == "IAA", offsets_entity_cluster
#     )

#     agreements, disagreements = map(
#         list,
#         partition(
#             lambda ent: get_annotator(ent) != AnnotatorChoice.AGREEMENT,
#             iaa_entity_iter,
#         ),
#     )
#     cleaned_normal_entities = deduplicate_shared_offset_id_entities(normal_entity_iter)
#     match len(agreements), len(disagreements):
#         case 0, 0:
#             # Should have adjudication entities even if they're agreements
#             scapegoat = next(iter(offsets_entity_cluster))
#             scapegoat_offsets = entity_offsets(scapegoat)
#             raise ValueError(
#                 f"Missing IAA entities for root entity with offsets {scapegoat_offsets}"
#             )
#         case (0, 1) | (1, 0):
#             # Haven't seen any inconsistencies for singletons TODO yet...
#             yield (
#                 agreements[0] if len(agreements) == 1 else disagreements[0]
#             )  # this gives us all the non-IAA entities along with the singleton IAA entity
#             yield from cleaned_normal_entities
#             return
#         case (
#             _,
#             0,
#         ) if len(agreements) > 1:
#             # TODO - figure out why this nonsense happens
#             yield agreements[0]
#             yield from cleaned_normal_entities
#             return
#         # Disagreements override agreements if they exist since we're clustering
#         case _:
#             # The hard part - this might be worth factoring out into another function
#             disagreement_types = list(map(get_annotator, disagreements))
#             if len(disagreement_types) > 1 and all_equal(disagreement_types):
#                 yield disagreements[0]
#                 yield from cleaned_normal_entities
#                 return
#             unique_non_agreement = set(disagreement_types)
#             out_of_scope = unique_non_agreement - {
#                 AnnotatorChoice.PREDICTION,
#                 AnnotatorChoice.REFERENCE,
#             }
#             if len(out_of_scope) > 0:
#                 raise ValueError(f"Erroneous annotator choices {out_of_scope}")
#             match len(unique_non_agreement):
#                 case 0:
#                     raise ValueError("Technically impossible")
#                 case 1:
#                     if len(disagreements) > 1:
#                         scapegoat = disagreements[0]
#                         scapegoat_offsets = entity_offsets(scapegoat)
#                         raise ValueError(
#                             f"{len(disagreements)} disagreement IAA entities for root entity with offsets {scapegoat_offsets}"
#                         )
#                     yield disagreements[0]
#                 case 2:
#                     yield wrangle_conflicts(disagreements=disagreements)
#                 case _:
#                     # Shouldn't be able to get here given the previous guard
#                     # but I'd rather be specific
#                     raise ValueError(f"Erroneous annotator choices {out_of_scope}")

#     yield from cleaned_normal_entities


def annotator_name_update(
    entities: Iterable[dict], annotators: EnumType
) -> Iterable[dict]:
    for entity in entities:
        if entity["from_name"] == "IAA":
            originals = entity["value"]["choices"]
            entity["value"]["choices"] = [v.value for v in originals]
        yield entity


def coordinate_adjudicated_entities(
    adjudicated_entities: Iterable[dict],
) -> Iterable[dict]:
    sized_adjudicated_entities = list(adjudicated_entities)
    unique_adjudicated_entities = list(
        unique_everseen(sized_adjudicated_entities, key=frozendict)
    )
    if len(sized_adjudicated_entities) != len(unique_adjudicated_entities):
        logger.warning(
            "Of %d adjudicated entities %d are unique",
            len(sized_adjudicated_entities),
            len(unique_adjudicated_entities),
        )
    return flatten(
        map_reduce(
            unique_adjudicated_entities,
            keyfunc=entity_offsets,
            reducefunc=adjudicate_offset_entity_cluster,
        ).values()
    )


def adjudicate_individual_entities(
    entity_correctness_matrices: Iterable[CorrectnessMatrix[Entity]],
    argument_entity_ids: Collection[str],
    filter_agreements: bool = True,
) -> Iterable[dict]:
    def is_argument_of_adjudicated_relation(entity: Entity) -> bool:
        return entity.label_studio_id in argument_entity_ids

    for correctness_matrix in entity_correctness_matrices:
        if not filter_agreements:
            yield from adjudicate_correctness_grouped_entities(
                AnnotatorChoice.AGREEMENT,
                correctness_matrix.true_positives,
            )

        elif len(argument_entity_ids) > 0:
            yield from adjudicate_correctness_grouped_entities(
                AnnotatorChoice.AGREEMENT,
                filter(
                    is_argument_of_adjudicated_relation,
                    correctness_matrix.true_positives,
                ),
            )
        yield from adjudicate_correctness_grouped_entities(
            AnnotatorChoice.PREDICTION, correctness_matrix.false_positives
        )
        yield from adjudicate_correctness_grouped_entities(
            AnnotatorChoice.REFERENCE, correctness_matrix.false_negatives
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
            labels=[annotator.name, *label_relations[0]["labels"]],
        )


def adjudicate_relations(
    annotators: EnumType,
    correctness_matrices: Iterable[CorrectnessMatrix[Relation]],
    filter_agreements: bool = True,
) -> Iterable[dict]:
    for correctness_matrix in correctness_matrices:
        if not filter_agreements:
            yield from adjudicate_correctness_grouped_relations(
                AnnotatorChoice.AGREEMENT, correctness_matrix.true_positives
            )
        yield from adjudicate_correctness_grouped_relations(
            AnnotatorChoice.PREDICTION, correctness_matrix.false_positives
        )
        yield from adjudicate_correctness_grouped_relations(
            AnnotatorChoice.REFERENCE, correctness_matrix.false_negatives
        )
