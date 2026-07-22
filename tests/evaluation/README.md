# tests/evaluation

> Tests for [`multilingual_embedding.evaluation`](../../src/multilingual_embedding/evaluation/README.md) — metrics, scoring and reports.

**89 tests.** Run with `pytest tests/evaluation -q`.

## Files

| File | Tests | Covers |
|---|---:|---|
| `test_evaluators.py` | 37 | Tokenizer and embedding evaluators, per-language fairness, dataset loading, report rendering |
| `test_metrics.py` | 30 | The metric primitives: cosine, ranking metrics, accuracy, correlation |
| `test_retrieval.py` | 14 | Scoring an encoder against held-out pairs: recall@k, MRR, nDCG, Wilson intervals, breakdowns by kind, language and overlap band |

`test_retrieval.py` is what makes an adaptation report trustworthy. Every number
`scripts/adapt_pretrained.py` prints comes through `evaluation/retrieval.py`, including the
Wilson intervals that decide whether a gain is significant and the `by_overlap` breakdown
that is the control against a model learning to match strings rather than meaning.

## What matters here

**Metrics are checked against closed-form references, not against themselves.** The
Spearman test derives its expected value from the rank-difference formula
`1 - 6·Σd² / (n(n²−1))` rather than from the implementation. That distinction caught a
real error: the reference value was initially wrong, and computing it independently
showed the code was right and the test was not.

**Spearman must be monotonic, not linear.** `test_spearman_is_monotonic_not_linear`
feeds a quadratic relationship and expects a perfect correlation, which Pearson would
not give. This is exactly why word-similarity benchmarks are scored with rank
correlation: human judgements are ordinal.

**Undefined must not be reported as zero.** An empty relevant set, a constant input to
a correlation, a query with no correct answer — each returns a documented value rather
than silently contributing a zero that drags down an average.

**A missing benchmark must stay `None`.** `test_missing_datasets_stay_none` asserts an
absent similarity dataset leaves the metric unset rather than defaulting to `0.0`, which
would read as a failing score rather than an unmeasured one.

**Out-of-vocabulary pairs are skipped, and coverage is reported alongside.** Scoring
them as zero would penalise a model merely for having a smaller vocabulary; skipping
them without reporting coverage would let a strong correlation over 10% of pairs look
authoritative.

**Precision uses `k` as its denominator.** Returning fewer than `k` results is a
shortfall, not a free pass — tested directly.

**Zero vectors must not produce NaN** anywhere in the similarity paths.

**Per-language fairness is tested as a first-class metric**, since it is the headline
number for a multilingual model and the one a single average conceals.
