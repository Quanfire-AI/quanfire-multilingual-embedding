# tests/embedding

> Tests for [`multilingual_embedding.embedding`](../../src/multilingual_embedding/embedding/README.md) — word vectors, sentence encoders, similarity search.

**76 tests.** Run with `pytest tests/embedding -q`.

## Files

| File | Covers |
|---|---|
| `test_matrix.py` | Size validation, lookup, normalisation, similarity, `most_similar`, analogies, persistence |
| `test_word2vec.py` | Training, reproducibility, learned structure, hyperparameter edge cases |
| `test_sentence.py` | Mean pooling and SIF encoders, batch shapes, degenerate input |
| `test_index.py` | Index construction, exact search, ranking, persistence |

## What matters here

**The model must actually learn, and that is asserted.** A synthetic corpus is built
from two disjoint topic vocabularies, and the test asserts that mean within-topic cosine
similarity exceeds cross-topic similarity by a clear margin. Without this, every other
test here would pass against an implementation that returns noise — which is the usual
failure mode of a hand-written word2vec.

**Reproducibility is tested in both directions.** The same seed must give identical
vectors; a different seed must not. Half of that assertion is easy to satisfy
accidentally, so both are checked.

**Matrix and vocabulary size mismatch must raise.** This is the classic silent
corruption in embedding code: rows and ids drift apart, and every lookup afterwards
returns a plausible vector belonging to a different token.

**Normalisation must not produce NaN.** The padding row is all zeros, so dividing by its
norm would poison every downstream similarity with NaN. Tests assert the guard holds.

**`most_similar` must exclude the query itself** and, by default, the special-token rows
— otherwise the top result is always the query and the list is useless.

**Persistence must be exact.** Round trips compare vectors with
`assert_array_equal`, not approximate equality: saving and loading a model must not
perturb it at all.

**Degenerate input must give a zero vector, not NaN.** A sentence whose every token is
out of vocabulary has no meaningful encoding; returning zeros lets a caller detect that
with a norm check instead of a NaN test.

## Speed

Tests use small dimensions, few epochs and tiny corpora, so the whole group runs in
about ten seconds despite training real models repeatedly.
