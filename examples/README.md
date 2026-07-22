# examples

> Runnable demonstrations of the framework end to end.

## Purpose

Documentation drifts; a script that must run does not. Everything here executes against
the bundled sample corpus, so an example that has rotted fails visibly rather than quietly
misleading a reader.

`train_and_search.py` and the first six walkthrough steps need only the base install
(`uv sync`). Beyond that, each extra buys one step:

| Walkthrough steps | Needs |
|---|---|
| 1–6, and `train_and_search.py` | base install |
| 7–8 — the contextual encoder and LoRA | `--extra neural` |
| 9 — extracting a real dump | `--extra wikipedia` |
| 10 — mining pairs | base install |
| 11 — adapting a published checkpoint | `--extra neural --extra pretrained`, and a GPU |

`use_adapter.py` needs `--extra neural --extra pretrained` and an adapter at
`models/indic-v1/`; unlike step 11 it needs no GPU, because using a model is not training
one. See [`models/README.md`](../models/README.md).

```bash
uv sync --extra neural --extra pretrained --extra wikipedia
```

## Contents

| File | What it demonstrates |
|---|---|
| `train_and_search.py` | The complete lifecycle: configure, train a tokenizer and embeddings, report per-language tokenizer fairness, reload the model from disk, and run semantic queries in four languages |
| `adaptation.yaml` | A complete `qfme adapt` experiment, annotated: what each setting decides, why the evaluation file is held fixed, and why the E5 prefixes are part of the model rather than of the command line |
| `use_adapter.py` | Loading a trained adapter and querying it — the raw encoder path, what forgetting the prefixes costs, and the pipeline path that cannot forget them |

`adaptation.yaml` is the only file here that will not run as it stands — it points at a
mined pair file, which is yours to produce:

```bash
qfme mine-pairs --source corpus.jsonl --output pairs.jsonl.gz
qfme adapt --config examples/adaptation.yaml --profile configs/gpu.yaml \
    --set adaptation.pairs=pairs.jsonl.gz
```

It is loaded by `tests/config/test_compute_profiles.py`, so a renamed field or a mode that
no longer exists fails in CI rather than on the training box.

## Running

```bash
uv run python examples/train_and_search.py
uv run python examples/use_adapter.py      # needs models/indic-v1/
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

## Walkthrough

[`walkthrough/`](walkthrough/README.md) is an eleven-step tour of the whole project —
corpus stats, auditing a damaged extraction, training, multilingual search, per-language
fairness, the static model's structural limit, evidence that the contextual encoder
learns, what domain adaptation costs, extracting a real Wikipedia dump, mining it into
contrastive pairs, and adapting a published checkpoint with a measured before/after.

Every command and every output in it was executed rather than predicted, and the numbers
reproduce. The last step's figures come from a GPU run rather than this machine, and are
labelled as such.
