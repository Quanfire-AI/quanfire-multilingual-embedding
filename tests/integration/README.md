# tests/integration

> End-to-end tests exercising the real pipeline across every layer.

**29 tests**, all marked `slow`. Run with `pytest tests/integration -q`, or skip them
with `pytest -m "not slow"`.

## Files

| File | Covers |
|---|---|
| `test_end_to_end.py` | Training pipeline, evaluation results, search pipeline, CLI |

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

## Sizing

`vocab_size` is deliberately small (110). SentencePiece fails outright when the target
exceeds what the corpus can support, and the fixture corpus is small by design so the
tests stay fast. If you enlarge the fixture, the vocabulary size can rise with it.
