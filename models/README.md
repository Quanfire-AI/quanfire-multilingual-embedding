# models/

> Where a trained adapter lives once it has been brought back from the training machine.

Training happens where the GPU is. Serving, inspection and every example in this
repository happen wherever you are reading this. What crosses between the two is a
directory of a few megabytes, and this is where it goes.

## Where to put it

```
models/indic-v1/
├── adapter.json     the manifest — base checkpoint, pooling, prefixes, LoRA shape
└── adapter.pt       the low-rank weights, about 3.4 MB
```

Copy the whole directory, both files. That exact path is not a suggestion: it is the one
already written into `SemanticSearchPipeline.from_adapter`'s docstring, into
[`data/README.md`](../data/README.md), and into
[`tests/integration/test_saved_adapter.py`](../tests/integration/test_saved_adapter.py).
A copy somewhere else works, but nothing will find it for you.

## Then verify it

```bash
pytest tests/integration/test_saved_adapter.py -q -rs
```

Fifteen tests. They skip when nothing is here, so a fresh clone is quiet; once the
directory exists they assert that the copy is intact, that it reloads into a working
encoder, and that it serves through the search pipeline with the right prefixes.

They also skip if the base checkpoint named in `adapter.json` is not in the local Hugging
Face cache, because nothing in the test suite reaches the network. Warm it once:

```bash
python -c "from transformers import AutoModel, AutoTokenizer; \
n='intfloat/multilingual-e5-small'; AutoTokenizer.from_pretrained(n); AutoModel.from_pretrained(n)"
```

`-rs` prints the reason for each skip, which is how you tell "not set up yet" from
"quietly not testing anything".

## Why the copy needs verifying at all

Every other artefact in this repository is reproducible: a config file plus a corpus
regenerates it. An adapter is not, for two reasons. Its base is an external name whose
contents can change upstream, and `indic-v1` predates `qfme adapt` — it was produced by a
script, so no committed configuration rebuilds it. It is also the artefact every
published number was measured on.

That makes the hand-copy the weakest link in the whole pipeline: the one step with no
config, no checksum and no run behind it. And its failure mode is silent. A truncated
`adapter.pt` or a manifest from a different run still reloads into an encoder that
returns vectors of the right shape and the right norm, free of NaN, encoding something
other than what was measured. Nothing raises.

## Why this directory is tracked, mostly

`.gitignore` ignores `models/*` — trained artefacts are build outputs and do not belong
in git. `models/indic-v1/` and this file are the documented exceptions: 3.4 MB is trivial
for git, and an artefact that carries published claims and cannot be rebuilt has to be
kept somewhere durable. Retrained successors are reproducible from a config and stay
ignored by default.
