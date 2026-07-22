# tests/embedding

> Tests for [`multilingual_embedding.embedding`](../../src/multilingual_embedding/embedding/README.md) — word vectors, sentence encoders, the contextual encoder, published checkpoints, similarity search.

**177 tests.** Run with `pytest tests/embedding -q`.

## Files

| File | Tests | Covers |
|---|---:|---|
| `test_neural.py` | 38 | The transformer encoder: architecture, contract conformance, retrieval quality, persistence, pipeline service, precision |
| `test_lora_gradcache.py` | 24 | LoRA initialisation, freezing, learning and merging, adapter checkpoints, exact gradient caching, chunk sizing |
| `test_matrix.py` | 20 | Size validation, lookup, normalisation, similarity, `most_similar`, analogies, persistence |
| `test_sentence.py` | 20 | Mean pooling and SIF encoders, batch shapes, degenerate input |
| `test_index.py` | 19 | Index construction, exact search, ranking, persistence |
| `test_word2vec.py` | 17 | Training, reproducibility, learned structure, hyperparameter edge cases |
| `test_encoder_contract.py` | 15 | The `TextEncoder` protocol, its guarantees, dimension discovery, and the decoupling proof |
| `test_pretrained.py` | 15 | Loading a published checkpoint, pooling strategies, prefixes, offline mode, contract conformance |
| `test_adapter.py` | 9 | Saving and loading an adapter artefact: base name, LoRA config, pooling, prefixes |

The four torch-dependent modules — `test_neural.py`, `test_lora_gradcache.py`,
`test_pretrained.py`, `test_adapter.py` — call `pytest.importorskip` at module level, so a
core-only checkout skips them instead of erroring. `test_pretrained.py` additionally needs
the `pretrained` extra (transformers) and, for the cases that load real weights, a
previously cached checkpoint.

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

## Published checkpoints

`test_pretrained.py` covers `neural/pretrained.py`, which loads someone else's weights
through their own library rather than into this project's transformer. The whole module is
organised around one question: does an external model behave like any other encoder here,
or does it need special cases everywhere?

**`TestNoSpecialCasesDownstream` is the answer, and it is the reason the class exists.**
Four cases assert that the retrieval evaluation accepts it unchanged, that LoRA applies to
the *upstream* module names, that the existing trainer trains it and leaves the base
frozen, and that gradient caching works against it. If any of those needed a branch, the
`TextEncoder` contract would not have been worth having.

**Contract conformance is asserted, not assumed.** It is a `TextEncoder`, its dimension
comes from the model config rather than a constant, `encode_batch` preserves order,
output is L2-normalised, and an empty batch keeps its shape instead of collapsing.

**Pooling is tested for the failure that does not raise.**
`test_padding_does_not_change_a_vector` is the one that matters — mean-pooling over
padding tokens produces a vector that is plausible, wrong, and indistinguishable from a
correct one. `test_mean_and_cls_are_different` guards against a strategy argument that is
accepted and ignored, and an unknown strategy is refused rather than silently defaulted.

**Loading fails loudly.** A missing checkpoint and a model whose width cannot be read are
each reported clearly rather than as a library traceback several frames deep.

## The adapter artefact

`test_adapter.py` covers `neural/adapter.py` — the 3.4 MB thing a training run actually
ships.

**A round trip must reproduce the model exactly**, and the companion case is what gives
that meaning: `test_an_unadapted_reload_would_differ` asserts the base model *without* the
adapter produces different vectors. Without it, a round-trip test passes against a loader
that silently loads nothing.

**The artefact carries how to use it, not just what it weighs.** Prefixes survive,
provenance survives, and `test_only_the_adapter_is_stored` pins the size claim — a 419 MB
base model must not end up inside a 3.4 MB directory. The prefix case is the practically
important one: an E5 adapter served without `query: ` and `passage: ` degrades silently,
so the artefact records them rather than relying on whoever loads it remembering.

**Four ways to fail, all of them refused.** Saving without an adapter, loading a directory
that is not one, loading with mismatched LoRA settings, and an incompatible format version.
The mismatch case is the subtle one: rank and target names must agree or the weights land
in the wrong places with matching shapes and no error.

## Speed

Tests use small dimensions, few epochs and tiny corpora. With the `neural` extra
installed the whole group runs in about sixteen seconds despite training real models
repeatedly; without it, in a couple.
