# examples

> Runnable demonstrations of the framework end to end.

## Purpose

Documentation drifts; a script that must run does not. Everything here executes against
the bundled sample corpus with no setup beyond `uv sync`, so an example that has rotted
fails visibly rather than quietly misleading a reader.

## Contents

| Script | What it demonstrates |
|---|---|
| `train_and_search.py` | The complete lifecycle: configure, train a tokenizer and embeddings, report per-language tokenizer fairness, reload the model from disk, and run semantic queries in four languages |

## Running

```bash
uv run python examples/train_and_search.py
```

Takes a few seconds. Artefacts are written to `artifacts/example/` and the evaluation
report to `reports/example/`; both are gitignored, and both can be deleted freely.

## What `train_and_search.py` shows

**Configuration is assembled in code**, not read from a file, so the script is
self-contained and every setting is visible at the point of use.

**Settings are sized for the sample corpus.** `vocab_size` is 300 because the sample is
only 750 sentences and SentencePiece fails outright when the target exceeds what the
corpus can support. This is called out in the script rather than left as a surprise.

**The model is reloaded from disk before searching**, rather than reusing the
in-memory objects the training run returned. That is deliberate: it exercises the same
path a deployed service takes, which is the path most worth knowing works.

**Per-language tokenizer efficiency is printed.** This is the number that matters most
for a multilingual model and the one a single average hides. On the sample corpus the
spread runs from about 1.07 characters per token for Japanese to about 4.07 for Tamil —
roughly 3.8×, meaning Japanese text consumes far more of a model's context for the same
content.

**Queries run in four languages** — English, Hindi, French and Tamil — because a
framework that only demonstrates English has not demonstrated the thing it claims.

## A note on the results

The sample corpus is small and template-generated, so similarity scores cluster near
0.99 and the embedding space is markedly anisotropic. The *ranking* is meaningful; the
absolute scores are not. The example demonstrates that the pipeline works, not that a
model trained on 750 synthetic sentences is good. Point it at a real corpus to judge
model quality.

## Adding an example

Keep it runnable with no arguments and no external data, prefer the bundled sample
corpus, and write output to a gitignored directory. If an example needs settings that
would be wrong for real use, say so in the script where the setting appears.
