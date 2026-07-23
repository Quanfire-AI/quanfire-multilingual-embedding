# tests/integration

> End-to-end tests exercising the real pipeline across every layer.

**49 tests**, all marked `slow`. Run with `pytest tests/integration -q`, or skip them
with `pytest -m "not slow"`.

Fifteen of them — the whole of `test_saved_adapter.py` — skip unless a trained adapter
has been copied to `models/indic-v1/` and its base checkpoint is in the local Hugging
Face cache. That is the normal state of a fresh checkout, not a failure.

## Files

| File | Covers |
|---|---|
| `test_end_to_end.py` | Training pipeline, evaluation results, search pipeline, CLI |
| `test_saved_adapter.py` | A trained adapter copied from the GPU machine: that it is intact, that it reloads, and that it serves |
| `test_hard_negatives.py` | Documents → pairs → negatives mined with a real torch encoder → a model that trains on them |

## Purpose

Unit tests verify each module against its own contract. They cannot catch two layers
whose contracts have quietly diverged — the tokenizer emitting something the embedding
trainer does not expect, or a model saved in a layout the loader does not read. That
class of failure only appears when the whole thing runs.

These tests therefore use no mocks. They train a real SentencePiece model, train real
word vectors, write real files, reload them from disk and answer real queries.

## Structure

The corpus and the trained model are module-scoped fixtures, so training happens once
and every test inspects the same artefacts. The whole group runs in about five seconds.

| Group | Asserts |
|---|---|
| `TestTrainingPipeline` | The corpus is read and segmented, all three languages and scripts survive, artefacts land on disk, the resolved config is persisted and reloadable, reports are written in both formats |
| `TestEvaluationResults` | Unknown rate is negligible, per-language metrics cover every language, no dead embedding rows beyond padding, results serialise |
| `TestSearchPipeline` | The model loads from its directory, search returns correctly ranked hits, topically related text is found, an unindexed pipeline returns empty, a missing directory raises |
| `TestCommandLine` | `stats`, `evaluate`, `--set` overrides, and the error exit codes for a missing source or experiment |
| `TestConfiguredNormalizersReachTheWholePipeline` | A configured normalizer chain shapes the training corpus, the tokenizer handed to the embedding stage, and the tokenizer the search pipeline reloads from disk |

| `TestTheArtefactIsComplete` | Both files present, `format_version` is 1, the base checkpoint is named, the LoRA shape is recorded, prefixes are recorded as a pair, an E5 base carries `query: `/`passage: `, the weight count matches the manifest, and the up-projections are not all zero |
| `TestItReloadsIntoAWorkingEncoder` | Dimension survives the round trip, all three scripts encode finitely and non-degenerately, the reloaded model differs from its base, and two loads agree exactly |
| `TestItServesThroughTheSearchPipeline` | `from_adapter` carries the recorded prefixes, indexing and ranking work, and an exact query finds its own passage |

## What matters here

**The corpus is genuinely multilingual** — English, Hindi and Japanese, chosen because
they segment differently. A three-European-language fixture would pass while the
framework was broken for everything else.

**Reproducibility is asserted, not assumed.**
`test_resolved_config_is_persisted` reloads `config.yaml` from the artefact directory
and checks the values match what was trained with. A model whose settings cannot be
recovered is not reproducible however good its metrics are.

**Search results must be ranked.** The test asserts scores are non-increasing and ranks
are `1..k`, rather than merely that something was returned.

**Failure paths get exit codes.** A missing experiment directory and a missing corpus
must exit `1`, not raise a traceback — the CLI is a program, and its contract with a
shell script is its exit status.

**A setting must hold across every stage, not just the first.**
`TestConfiguredNormalizersReachTheWholePipeline` trains with a lowercasing chain and
follows it through. A chain that shaped the training corpus but not the tokenizer handed
to the embedding stage — or not the one the search pipeline reloads from disk — would
leave the persisted `config.yaml` claiming something untrue, and queries would be encoded
into pieces the vectors were never trained on. Nothing raises in that scenario; retrieval
just quietly gets worse. Only a test that spans training and search can see it, which is
why it lives here rather than in `tests/tokenizer`.

**A copied model is verified, not trusted.** `test_saved_adapter.py` exists because the
hand-copy from the training machine is the one step in the pipeline with no config, no
checksum and no run to reproduce it. The two assertions that carry the weight are the
parameter count — the sum of `numel` across `adapter.pt` must equal `adapter_parameters`
in `adapter.json`, which catches a truncated or mismatched copy — and the check that the
up-projections are not all zero. LoRA initialises those to zero so that an untrained
adapter reloads *identical* to its base; that is the right default and also the quietest
possible failure, since every shape, norm and score stays plausible while the model is
simply the checkpoint again. Checking the down-projections instead would pass on exactly
that model, because they are random-initialised and never zero.

**The base model is a skip gate; the adapter is an assertion.** A base checkpoint missing
from the cache means this machine is not set up, and says nothing about the copy. An
adapter that fails to load beside a base that loaded fine means the copy is broken. The
two must not produce the same outcome.

**These tests assert the artefact contract, not any run's numbers.** A future adapter,
trained on other data at another rank, passes them unchanged. Retrieval quality is what
the evaluation reports are for; `test_an_exact_query_finds_its_own_passage` is a wiring
check, not a claim.

## Sizing

`vocab_size` is deliberately small (110). SentencePiece fails outright when the target
exceeds what the corpus can support, and the fixture corpus is small by design so the
tests stay fast. If you enlarge the fixture, the vocabulary size can rise with it.
