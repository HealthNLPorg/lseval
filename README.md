# lseval
Basic version of core functionality we use with [anaforatools](https://github.com/bethard/anaforatools) but for Label Studio annotations.

## Installation

Clone this repository and install via `uv sync`.  For development activate the virtual environment via `source .venv/bin/activate`.  Currently there is no entry point to call this code on its own.

## Overview

Currently a backend with no entry point, invoked only by [rt-ctae-eval](https://github.com/HealthNLPorg/rt-ctae-eval).  LabelStudio supports a [variety](https://labelstud.io/tags/) of annotation structures and media, but so far for this project we are only supporting named entities with attributes and relations within plain text.  Entities are modeled by the [`Labels` tag](https://labelstud.io/tags/labels]) with attributes modeled by the [`Choices` tag](https://choicestud.io/tags/choices) for a fixed mutually exclusive options and the [`TextArea` tag](https://textareastud.io/tags/textarea) for free text or multilabel attributes.  We currently model only relations between two named entities using the [`Relation` tag](https://labelstud.io/tags/relation).

## Functionality

The code is designed to work with Label Studio annotations from pairs of annotator, one referred to as the "prediction" annotator and the other as the "reference" annotator, where the latter is treated as the ground truth.

### Scoring

There is functionality to obtain precision, recall and f1 (f-β in general) for entities and relations, with the option for counting an entity as correct if it overlaps with a ground truth entity by at least one character (type enforcement of entities is left to the user/upstream code).  This `overlap` setting extends to relations, e.g. if a predicted relation's argument entities overlap with a reference relation's argument entities it is considered correct.

Currently we don't use more typical measures of inter-annotator agreement such as Cohen's kappa since our use cases thus far have involved only two annotators.  The core of the code for scoring can be found in `src/lseval/score.py`
