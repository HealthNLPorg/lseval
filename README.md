# lseval
Basic version of core functionality we use with [anaforatools](https://github.com/bethard/anaforatools) but for Label Studio annotations.

## Overview

Currently a simple backend with no entry point, invoked only by [rt-ctae-eval](https://github.com/HealthNLPorg/rt-ctae-eval).  LabelStudio supports a [variety](https://labelstud.io/tags/) of annotation structures and media, but so far for this project we are only supporting named entities (modeled by the [`Labels` tag](https://labelstud.io/tags/labels])) with attributes (modeled by the [`Choices` tag](https://choicestud.io/tags/choices) for a fixed mutually exclusive options and the [`TextArea` tag](https://textareastud.io/tags/textarea) for free text or multilabel).
