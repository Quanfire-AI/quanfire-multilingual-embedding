# evaluation

> Scores tokenizers and embedding matrices, and assembles the results into a report that carries the configuration which produced them.

## Purpose

A trained model without measurement is an assertion. This layer supplies the metric
primitives, the two evaluators that apply them to the artefacts the layers below
produce, and the report object that binds metrics to the settings that produced them.
It is a separate layer because scoring must be able to read an embedding matrix and a
tokenizer without either of them knowing they are being scored, and because the metric
primitives are useful on their own — they are plain functions over plain data and import
nothing from the framework except `utils.validation`.

## Modules

| Module | Responsibility |
|---|---|
| `metrics.py` | Metric primitives: cosine similarity, `precision_at_k`, `recall_at_k`, `f1_score`, `average_precision`, `reciprocal_rank`, `mean_reciprocal_rank`, `ndcg_at_k`, `accuracy`, `pearson_correlation`, `spearman_correlation`. |
| `tokenizer_eval.py` | `TokenizerEvaluator`, `TokenizerMetrics`, `evaluate_tokenizer` and `language_fairness` — compression, fertility, unknown rate and the per-language spread. |
| `embedding_eval.py` | `EmbeddingEvaluator`, `EmbeddingMetrics`, the `SimilarityPair` / `AnalogyQuestion` records, their JSON Lines loaders, and `sample_neighbour_probes`. |
| `report.py` | `EvaluationReport`: bundles corpus, tokenizer and embedding metrics with the resolved config, and writes `report.json` plus `report.md`. |

## Key design decisions

**Metric primitives know nothing about the framework.** Every function in `metrics.py`
takes sequences, sets and numpy arrays. Nothing there mentions a tokenizer, a matrix or
a corpus. That keeps them independently testable, lets them be reused across evaluation
suites, and means a bug in a metric can be reproduced from three literals rather than
from a training run.

**`precision_at_k` divides by `k`, not by the number of results returned.** A system
that returns three results for `k=10` is penalised for the shortfall rather than being
scored as if it had confidently returned three. `recall_at_k` divides by
`len(relevant)`, which is the complementary question. Both call `require_positive(k)`
so a zero or negative cutoff fails loudly instead of producing a division error deep in
an averaging loop.

**Ranking metrics short-circuit on an empty relevant set.** `precision_at_k`,
`recall_at_k`, `average_precision` and `ndcg_at_k` all return `0.0` immediately when
`relevant` is empty, which keeps the denominators well defined. The module docstring
states the stronger intent — that a query with no correct answer is undefined rather
than a zero, because averaging in a zero silently drags a mean down — and a caller
computing an average over queries should filter empty-relevant queries out before
averaging rather than relying on these functions to do it.

**Spearman, not Pearson, for word-similarity benchmarks.** `EmbeddingEvaluator.
similarity_correlation` uses `spearman_correlation`. Human similarity judgements are
ordinal: an annotator saying 8 and another saying 9 is not a meaningful gap, but both
ranking a pair above another pair is. What matters is whether the model orders the pairs
as annotators did, not whether it reproduces their absolute numbers. `pearson_correlation`
remains exported because Spearman is implemented as Pearson over average ranks, and
because a caller may legitimately want the linear form.

**`TokenizerEvaluator` takes a plain `tokenize` callable, not a `Tokenizer` object.**
The field is `tokenize: Tokenize`, where `Tokenize = Callable[[str], list[str]]`. Any
segmentation function can therefore be scored — a trained SentencePiece model, a
whitespace baseline (`str.split`), a regex splitter — with no adapter class and no
dependency on the tokenizer layer's interface. Comparing a trained model against a
whitespace baseline is the first thing anyone wants to do, and this makes it a one-line
change rather than a subclass.

**Per-language tokenizer fairness is the headline multilingual metric.** A single
averaged `characters_per_token` hides the failure that matters most: a vocabulary trained
on mostly-English text encodes English in few tokens and everything else in many, so the
other languages get a fraction of the effective context length for the same content.
`evaluate_by_language` scores each language separately and `language_fairness` reduces
the spread to `minimum`, `maximum` and `ratio`. A ratio near 1.0 means comparable
efficiency; a ratio of 3 means one language needs three times as many tokens for the same
text. `evaluate_by_script` covers the common case of scraped text with no language labels,
grouping by dominant script via `corpus.script.detect_script`.

**Missing benchmarks leave metrics as `None`, never `0.0`.** `EmbeddingMetrics.
similarity_correlation`, `similarity_coverage`, `analogy_accuracy` and `analogy_coverage`
all default to `None`, and `evaluate` only populates them when a dataset was actually
supplied. `TokenizerMetrics.vocabulary_utilisation` behaves the same way when the
vocabulary size is unknown. A zero would be indistinguishable from a model that scored
zero, and `_rounded` preserves `None` through `to_dict` so the distinction survives into
the JSON report.

**Coverage is reported alongside every labelled metric.** `similarity_correlation`
returns `(correlation, coverage)` and skips pairs containing an out-of-vocabulary word
rather than scoring them as zero, which would understate a model whose only fault is a
smaller vocabulary. `analogy_accuracy` likewise returns accuracy over attempted questions
plus the attempted share. A strong correlation over 10% of the pairs says very little,
so the two numbers are only meaningful read together.

**Structural metrics need no labels, which is the normal case.** `mean_pairwise_similarity`,
`spectrum` (returning isotropy and effective dimensions) and `zero_vector_count` describe
the geometry itself and are therefore available for any language, including the great
majority this framework targets for which no benchmark dataset exists. Isotropy is the
ratio of smallest to largest singular value; effective dimensions is how many principal
components explain 95% of the variance. Both catch the failure where the matrix collapses
toward a narrow cone, every pair looks similar and cosine similarity stops discriminating —
a high mean pairwise similarity on random tokens is the symptom. `_informative_vectors`
excludes special tokens and all-zero rows first, since those would drag the figures
toward zero for reasons unrelated to the geometry. The sampling is seeded so a report is
reproducible.

**The report carries its configuration.** `EvaluationReport` holds a `config` dictionary
next to the metrics. A metric without the settings that produced it cannot be compared
against anything, so the two are written together as `report.json` for machines and
`report.md` for humans.

## Usage

```python
from multilingual_embedding.evaluation.metrics import precision_at_k, spearman_correlation
from multilingual_embedding.evaluation.tokenizer_eval import (
    TokenizerEvaluator,
    language_fairness,
)

print("precision_at_k:", precision_at_k(["a", "b"], {"a", "z"}, k=5))
print("spearman:", round(spearman_correlation([1.0, 2.0, 3.0], [2.0, 3.0, 9.0]), 4))

evaluator = TokenizerEvaluator(tokenize=str.split, vocabulary_size=100)

by_language = evaluator.evaluate_by_language(
    {
        "en": ["the river is wide", "the cat sat down"],
        "de": ["Donaudampfschifffahrtsgesellschaft faehrt"],
    }
)

for language, metrics in sorted(by_language.items()):
    print(language, "chars/token:", round(metrics.characters_per_token, 3))

print("fairness:", language_fairness(by_language))
```

Actual output:

```
precision_at_k: 0.2
spearman: 1.0
de chars/token: 20.5
en chars/token: 4.125
fairness: {'minimum': 4.125, 'maximum': 20.5, 'ratio': 4.9697}
```

The precision is 0.2 rather than 0.5 because one of two returned items was relevant and
the denominator is `k=5`. The Spearman correlation is 1.0 despite the second sequence
being non-linear, because the two agree perfectly on ordering. The fairness ratio of
nearly 5 is exactly the signal the metric exists to surface — here from a whitespace
baseline on a deliberately unfair pair of inputs.

## Dependencies

May import from `common`, `core`, `utils`, `config`, `corpus`, `vocabulary`, `tokenizer`
and `embedding`. In practice: `metrics.py` imports only `utils.validation`;
`tokenizer_eval.py` imports `core.logging` and `corpus.script`; `embedding_eval.py`
imports `core.logging`, `embedding.matrix` and `utils.io`; `report.py` imports
`common.version`, `core.logging`, `utils.filesystem` and `utils.io`. Notably this layer
does not import `tokenizer` — see the plain-callable decision above.

Only `pipelines` imports this package. Nothing below it may.

## Tests

| File | Tests |
|---|---|
| `tests/evaluation/test_metrics.py` | 30 |
| `tests/evaluation/test_evaluators.py` | 37 |

`.venv/bin/python -m pytest tests/evaluation -q` reports **67 passed**. The evaluators are
additionally exercised against real trained artefacts by
`tests/integration/test_end_to_end.py` (25 tests).
