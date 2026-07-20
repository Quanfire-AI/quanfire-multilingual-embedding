# tests/embedding

> Tests for [`multilingual_embedding.embedding`](../../src/multilingual_embedding/embedding/README.md) — word vectors, sentence encoders, the contextual encoder, similarity search.

**148 tests.** Run with `pytest tests/embedding -q`.

## Files

| File | Covers |
|---|---|
| `test_matrix.py` | Size validation, lookup, normalisation, similarity, `most_similar`, analogies, persistence |
| `test_word2vec.py` | Training, reproducibility, learned structure, hyperparameter edge cases |
| `test_sentence.py` | Mean pooling and SIF encoders, batch shapes, degenerate input |
| `test_index.py` | Index construction, exact search, ranking, persistence |
| `test_encoder_contract.py` | The `TextEncoder` protocol, its guarantees, dimension discovery, and the decoupling proof |
| `test_neural.py` | The transformer encoder: architecture, contract conformance, retrieval quality, persistence, pipeline service, precision |
| `test_lora_gradcache.py` | LoRA initialisation, freezing, learning and merging, adapter checkpoints, exact gradient caching, chunk sizing |

The last two require torch, which lives behind the optional `neural` extra. They call
`pytest.importorskip` at module level, so a core-only checkout skips them instead of
erroring.

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

## The encoder contract

`test_encoder_contract.py` covers `TextEncoder`, the protocol that lets the search
pipeline work with a model that has no embedding matrix. A contextual encoder computes a
vector at call time and has no per-token table to hand over, so a pipeline that demanded
one could never serve it.

`TestPipelineIsGenuinelyDecoupled` is the exit criterion, not a formality. Every test in
it would have been impossible to write before the decoupling: it builds a
`SemanticSearchPipeline` from an encoder backed by no model at all and searches with it.
A refactor that moved the matrix behind an interface without removing the dependency
would still pass the protocol-conformance tests and fail these.

## The contextual encoder

`test_neural.py` covers the transformer. Its architecture, shape and persistence tests
are the cheap half — a model that has learned nothing passes all of them just as easily.
`TestItActuallyLearns` is the half that matters: a transformer emitting well-formed
vectors containing no information is the default failure mode of a hand-written training
loop, and only a retrieval assertion catches it.

The model used throughout is two layers at 32 dimensions, so the module runs in seconds
on CPU. That is the design, not a compromise: the same code path trains a real model,
differing only in configuration.

`TestPrecision` covers the join between a machine profile and a run. Everything executes
on CPU, and bf16 autocast is exercised genuinely — the operations really do run in
bfloat16. What is *not* verified here is CUDA kernel selection or the speed and memory
claims that motivate bf16 at all; those remain unconfirmed until a run happens on GPU
hardware. The tests are written to make that boundary explicit rather than to imply
coverage they do not have. A separate case asserts bf16 is requested but ignored on
Metal, and another that an invalid precision is caught when the config is built rather
than mid-training.

## LoRA and gradient caching

Two claims in `test_lora_gradcache.py` are the ones worth testing, because both are easy
to implement in a way that appears to work and is silently wrong.

**LoRA must be a no-op at initialisation.** If the up-projection is not zeroed, the
adapted model starts as a corrupted version of its base, and the first optimizer steps go
on undoing that damage. Nothing looks broken — the model still trains — it merely starts
from a worse place than it should and nobody can tell.

**Gradient caching must be exact.** It is sold as identical to a large batch, not an
approximation of one. If it were merely close, the justification for the extra forward
pass collapses, and the discrepancy would hide in the noise of training. The test
compares against the uncached gradient rather than against a tolerance chosen to pass.

## Speed

Tests use small dimensions, few epochs and tiny corpora. With the `neural` extra
installed the whole group runs in about sixteen seconds despite training real models
repeatedly; without it, in a couple.
