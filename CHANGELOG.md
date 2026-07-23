# Changelog

This file exists because other repositories now pin this one by version. A consumer
deciding whether to move a pin needs to know what moved, and `git log` is the wrong
granularity for that question.

Versions follow semantic versioning. Before 1.0 the minor number carries breaking
changes, so treat `0.2 → 0.3` the way you would treat `1.0 → 2.0`.

## 0.2.0 — 2026-07-23

First tagged release. `0.1.0` was the version the project carried while it was only ever
installed from a working copy; nothing was ever published under it, so there is no upgrade
path to describe and no changes are listed against it.

### For consumers

The base install is `numpy`, `pandas`, `pyyaml`, `sentencepiece` and `tqdm` — deliberately
no `torch`. The corpus, vocabulary, tokenizer and evaluation layers work from that alone,
so a caller that wants text preparation keeps full control of its own training stack.
Torch and the model hub client sit behind the `neural` and `pretrained` extras.

Requires Python `>=3.12,<3.13`. The ceiling is not arbitrary: torch stopped shipping
x86_64 macOS wheels after 2.2.2, which predates 3.13, so 3.13 leaves an Intel development
machine with no torch at all.

### Added

- **Adapting published checkpoints.** LoRA over a frozen pretrained encoder, behind the
  same `TextEncoder` protocol as the from-scratch models, with `qfme adapt` as a command
  rather than a script.
- **A saved adapter is a real artefact.** `format_version: 1`, a manifest naming the base
  checkpoint, pooling, prefixes and LoRA shape, and `load_adapter` /
  `SemanticSearchPipeline.from_adapter` to reload it with its prefixes applied on the
  right sides.
- **Declared adaptations.** A run states which of six adaptations it claims to measure and
  is checked against its own data before the model loads, so a language-transfer claim
  cannot quietly be a task-adaptation result.
- **Retrieval measurement with confidence intervals**, replacing a falling loss as the
  evidence that anything improved.
- **Corpus provenance fingerprinting**, so which dump produced a corpus stops being a
  matter of recollection.
- **`models/indic-v1/`** — the adapter every published number in this repository was
  measured on, tracked because it predates `qfme adapt` and no committed configuration
  rebuilds it.

### Changed

- The version now has one source of truth. `pyproject.toml` declares
  `dynamic = ["version"]` and hatchling reads
  `src/multilingual_embedding/common/version.py`. Previously both held independent
  literals with nothing asserting they matched — a wheel could publish one number while
  every evaluation report stamped another, making the provenance beside a published metric
  wrong with nothing raising.

### Fixed

- A pair-kind filter that could silently shrink the training set rather than failing.
- A test tokenizer that was not deterministic across processes.
- A reported gain that was within noise, now withdrawn and reported as such.
