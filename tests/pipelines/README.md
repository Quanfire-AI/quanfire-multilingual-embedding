# tests/pipelines

> Tests for [`multilingual_embedding.pipelines`](../../src/multilingual_embedding/pipelines/README.md) — currently the search pipeline's query/passage asymmetry and the adapter it can be built from.

**18 tests.** Run with `pytest tests/pipelines -q`.

## Files

| File | Covers |
|---|---|
| `test_search.py` | Prefixes reaching the encoder, prefixes not leaking into results, `prefixes` and `repr` visibility, batched indexing, the SIF regression |
| `test_search_adapter.py` | `from_adapter`: prefixes travelling out of a saved artefact, and the pipeline it produces |

`test_search_adapter.py` needs torch and transformers, both behind the optional `neural`
extra. It calls `pytest.importorskip` at module level, so a core-only checkout skips it
instead of erroring.

`TrainingPipeline` has no unit tests here. It is orchestration over components that are
each tested in their own package, and `tests/integration/test_end_to_end.py` runs it for
real; a unit test of a composition root mostly asserts that mocks were called.

## What matters here

**The defect these tests exist for is invisible everywhere else.** An E5-family model is
trained with `query:` on one side and `passage:` on the other. Served without them it
returns vectors of the right shape and the right norm, free of NaN, that encode the wrong
thing. No exception, no warning, no shape error — the only symptom is a lower retrieval
score, which looks exactly like a model that is not very good. Nothing downstream can
catch it, so it is caught here.

**Assertions are on the strings handed to the encoder, not on retrieval quality.**
`RecordingEncoder` keeps every text it was given, so `test_the_two_sides_get_different_prefixes`
can assert that `index` saw `passage: alpha beta` while `search` saw `query: alpha`.
Asserting that scores improved would be asserting on the signal that goes quietly wrong.

**The adapter tests compare two pipelines rather than reading an attribute back.**
Storing a prefix and then not applying it would satisfy `pipeline.prefixes == (...)`. So
each side is checked by building a second pipeline over the *same* encoder and the *same*
corpus, differing only in that one prefix, and asserting the scores differ. Each of those
tests carries a control asserting the prefix changes this model's vectors at all —
without it, the comparison would hold whether or not the prefix was ever applied.

**One test guards a bug the batching change fixed.**
`test_sif_fits_its_common_component_during_indexing` asserts `SifEncoder.is_fitted` after
`index`. `SifEncoder` estimates the direction shared by all sentences from a batch and
removes it; `encode` reuses that estimate. Indexing one sentence at a time never supplied
a batch, so the component was never fitted and SIF silently degraded to a plain weighted
average — with the right shapes and plausible results throughout.

**Test fixtures are deterministic across processes.** `RecordingEncoder` seeds its
generator from `zlib.crc32` rather than the built-in `hash`, whose result for a `str` is
salted per interpreter by `PYTHONHASHSEED`. Nothing here depends on the vector values,
but a fixture that changes its data every run has already caused one intermittent failure
in this suite.

**The checkpoint is built, not downloaded.** `test_search_adapter.py` writes a 1,000-token
vocabulary and a two-layer, 64-dimension BERT to `tmp_path` and adapts that. Tests reach
no network, and the whole file runs in about eight seconds.
