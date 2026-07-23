# QuanFire Multilingual Embedding

**QuanFire's own embedding models: corpus in, trained embedding model out.**

Turns a multilingual corpus into meaningful vectors — generic, or adapted to a specific
domain — with the corpus handling, tokenization, vocabulary management, training and
evaluation in between built to be inspected, configured and reproduced.

```
Wikipedia dump  ->  corpus  ->  mined pairs  ->  adapted encoder  ->  evaluation  ->  search
                       |
                  tokenizer  ->  vocabulary  ->  static baseline
```

Three routes to a vector share that pipeline: a **static** word2vec baseline in pure numpy,
a **contextual** transformer written out in this repository and trained contrastively, and
an **adapted** published checkpoint — LoRA over frozen `multilingual-e5-small`-class weights.
The last is what produces the models this project ships; the first is the floor the others
are measured against. The second and third need the optional `neural` extra.

**1357 tests · 94% coverage · `ruff` clean · `mypy --strict` clean · layer graph verified acyclic**

**Proven on real data:** a published checkpoint adapted on mined Hindi and Tamil Wikipedia
pairs beats itself by **+28.6% (Hindi)** and **+40.9% (Tamil)** recall@1 on held-out
retrieval, training **0.50%** of its parameters into a **3.4 MB** artefact.
[Full numbers below.](#what-this-has-achieved)

---

## Table of contents

- [Purpose](#purpose)
- [Objective](#objective)
- [What this has achieved](#what-this-has-achieved)
- [What it can do today](#what-it-can-do-today)
- [Goals](#goals)
- [Non-goals](#non-goals)
- [Status](#status)
- [Running locally](#running-locally)
- [Quick start](#quick-start)
- [Training on Wikipedia, in multiple languages](#training-on-wikipedia-in-multiple-languages)
- [What makes it multilingual](#what-makes-it-multilingual)
- [Architecture](#architecture)
- [The three model families](#the-three-model-families)
- [Design decisions worth knowing](#design-decisions-worth-knowing)
- [Configuration](#configuration)
- [Running in production](#running-in-production)
- [Pros and cons](#pros-and-cons)
- [Project layout](#project-layout)
- [Development](#development)
- [Limitations](#limitations)

> **Where this is going:** [ROADMAP.md](ROADMAP.md) tracks the remaining work — domain pair
> miners, hard negatives, a serving API, and from-scratch pretraining.
> [ECOSYSTEM.md](ECOSYSTEM.md) places this repository within the wider QuanFire AI stack;
> everything outside embeddings lives there and nowhere in this README.

> **Looking for the full reference?**
> [`knowledge-base/QuanFire-Multilingual-Embedding-Handbook.pdf`](knowledge-base/QuanFire-Multilingual-Embedding-Handbook.pdf)
> is a 90-page handbook covering purpose, design, architecture, components, usage, local
> and production operation, benefits, and an honest pros-and-cons assessment. Read that
> if you want the whole picture in one document; read on for the quick version.

---

## Purpose

Almost every applied AI system — retrieval-augmented generation, semantic search,
deduplication, clustering, recommendation, document intelligence — rests on a text
embedding. The embedding is rarely the part teams control. It arrives as an opaque
model file or a paid API, and when it behaves badly on a particular language or
domain there is no way to see why, and no lever to pull.

This project exists to make that layer transparent and owned. It provides the whole
path from raw text to a queryable vector space as inspectable, configurable,
reproducible code:

- **You can see what happened.** Every stage reports what it did — how the corpus was
  filtered, how efficiently each language tokenizes, how the vector space is shaped.
- **You can reproduce it.** Runs are seeded and the resolved configuration is written
  next to the artefacts it produced, so a model file always traces back to its settings.
- **You can change it.** Components are selected by name from configuration, so
  swapping a normalizer or a pre-tokenizer does not mean editing the pipeline.

The second reason it exists is that most embedding tooling is Latin-first, with other
scripts handled as an afterthought. Here, multilingual behaviour is a core requirement
that shaped the implementation — see [what makes it multilingual](#what-makes-it-multilingual)
for the specifics, and note that the framework reports its own fairness across
languages rather than hiding it behind an average.

## Objective

Be the place QuanFire's **embedding models** are built: a corpus goes in, a trained,
evaluated, reproducible embedding model comes out — generic, or adapted to a specific
domain. The vectors it produces are the input other systems build on, whether that is
retrieval, semantic search, clustering, or a downstream model consuming the embedding
layer.

Concretely, the framework must be able to:

1. Read a corpus from plain text or JSON Lines, on disk or gzipped, larger than memory.
2. Audit that corpus and refuse to train on one that is broken.
3. Segment it correctly across Latin, Devanagari, Tamil, Han, Kana, Arabic and more.
4. Train a subword tokenizer and a shared vocabulary over it.
5. Train an embedding model on the tokenizer's own output — static word vectors, or a
   contextual transformer encoder fitted with a contrastive objective.
6. Adapt an existing model to a domain cheaply, without a full fine-tune.
7. Score the result — including per-language fairness — and write a report.
8. Answer semantic queries against the trained model, in any language it was trained on.
9. Do all of the above deterministically, on a development machine or a GPU box, without
   changing anything but a profile.

**Where it falls short of that today.** Two of the three routes now have a command.
`qfme train` trains the static model; `qfme adapt` runs the full adaptation experiment from
a config file and a compute profile. The one still missing is the *from-scratch* contextual
encoder: `TrainingPipeline` has no neural path, so a transformer trained from nothing is
built by driving `ContrastiveTrainer` from the Python API. `SemanticSearchPipeline.from_directory`
likewise always reconstructs a static model, though serving an adapted one is a single call
to `from_adapter`. The families share the `TextEncoder` contract, and now share most of the
command line. Closing the rest is tracked in [ROADMAP.md](ROADMAP.md); until it is, "from a
single command" is true of word2vec and of adaptation, and not of a transformer trained from
scratch.

**Scope.** This repository does embeddings. Nothing else. The other modalities in the
QuanFire stack are separate repositories and are described in [ECOSYSTEM.md](ECOSYSTEM.md).

---

## What this has achieved

Not capability claims — measured results, each with the run that produced it. Full working
in [ROADMAP.md](ROADMAP.md).

### The headline: adaptation works, on real Indic text

`intfloat/multilingual-e5-small` adapted with LoRA on 20,000 mined Wikipedia pairs per
language, scored against ~2,000 held-out pairs it never saw. Rank 32, two epochs, **0.50%
of parameters trained**, on an RTX 4070 Ti SUPER.

| | Hindi base | Hindi adapted | Tamil base | Tamil adapted |
|---|---:|---:|---:|---:|
| recall@1 | 0.4238 | **0.5451** (+28.6%) | 0.3219 | **0.4535** (+40.9%) |
| recall@10 | 0.6690 | 0.7929 (+18.5%) | 0.5269 | 0.6966 (+32.2%) |
| MRR | 0.5136 | 0.6364 (+23.9%) | 0.3931 | 0.5397 (+37.3%) |

**The weaker language gained more, which is the argument for doing this at all.** E5 serves
Tamil at 76% of its Hindi score; after adaptation Tamil reaches 83% of Hindi's. The corpus
helps most exactly where the published model is thinnest — which is where a proprietary
corpus earns its keep.

### The control: it is not learning to match strings

Gains run *inversely* to lexical overlap, in both languages, and Tamil is Dravidian while
Hindi is Indo-Aryan:

| overlap band | Hindi | Tamil |
|---|---:|---:|
| low `<0.3` | +145.5% | +126.7% |
| mid `0.3–0.7` | +39.6% | +56.9% |
| high `>0.7` | *not significant* | +21.6% |

A model memorising surface form improves most where strings already match. Neither does.
One language could be an accident; two unrelated ones make it a property of the method.

### The finding that changes how corpora get planned

Four controlled runs, each varying exactly one facet with everything else held fixed and
the evaluation set pinned:

| varied | held fixed | achievable gain captured |
|---|---|---:|
| **task shape** — `adjacent` → `heading_section` | language, corpus | **−17%** |
| **language** — Hindi → Tamil | task shape, corpus | **+95%** |

**The adaptation is language-general and task-specific** — the reverse of the intuitive
assumption. Pairs transfer across languages almost completely: training on Hindi alone
scored 381/1272 on Tamil against in-language training's 388/1272, seven queries apart. Pairs
do **not** transfer across query shapes. So: mine wherever the text is cleanest, but mine
*several pair shapes*, because every shape to be served must be present in the mixture. A
mixture works even when a single wrong shape does not — `indic-v1` trained on all three
kinds recovered +38.0% on `heading_section` *and* +40.8% on `adjacent` from one adapter.

### The engineering that made it fit

Measured on the training box, batch 256, a 5.3M-parameter encoder over 4,000 mined pairs:

| | no caching | chunk 32 |
|---|---:|---:|
| fp32 | 4.89 GB / 4.3s | 0.40 GB / 4.7s |
| bf16 | 2.99 GB / 2.7s | 0.29 GB / 4.7s |

Gradient caching carries the memory saving — **12.2× alone**, 1.6× for bf16, **16.9×
together** — and final losses spanned **0.51%** across all four cells, so the exactness
claim holds off the test bench. Initial loss matched `ln(batch_size)` to within 4–6% at both
batch 16 and 256, which is what an untrained contrastive model must show and independent
evidence the objective is wired the right way round.

LoRA at BERT-base shape, rank 16: **0.81% of parameters trainable**, a **3.4 MB** adapter
against a 419 MB model, optimiser state from **0.82 GB to 6.8 MB**.

### The data path, end to end on real dumps

A full `extract → validate → mine-pairs → train` run over both dumps, 9/9 stages passing in
1h 30m on a laptop with **under 201 MB peak resident memory** throughout:

| | Hindi | Tamil |
|---|---:|---:|
| Dump | 227 MB | 258 MB |
| Articles extracted | 118,571 in 7.4s | 163,768 in 8.2s |
| Sentences | 2,235,798 | 2,677,328 |
| Pairs mined | **642,536** in 25m | **893,523** in 29m |
| — `adjacent` / `heading_section` / `title_lead` | 414,166 / 130,243 / 98,127 | 507,058 / 237,049 / 149,416 |
| Mean overlap by kind | 0.50 / 0.77 / 0.98 | 0.47 / 0.76 / 0.98 |

That `title_lead` overlap of 0.98 is the leakage the pair miner exists to measure: a
Wikipedia lead restates its title almost verbatim, so the largest pair source is also the
most solvable by string matching. It is reported, not hidden, and `--max-overlap` filters it.

---

## What it can do today

| Capability | State | Entry point |
|---|---|---|
| Read a corpus larger than memory, plain or gzipped, text/lines/JSON Lines | ✅ | `qfme stats`, `stream_documents` |
| Extract a Wikipedia dump into corpus format, sections preserved | ✅ | `qfme extract` |
| Audit a corpus and refuse to train on a broken one | ✅ | `qfme validate` (non-zero exit) |
| Segment and script-detect across 22 scheduled Indian languages + more | ✅ | `corpus/` |
| Train a SentencePiece tokenizer and shared vocabulary | ✅ | `qfme train` |
| Train a static word2vec model and search it | ✅ | `qfme train`, `qfme search` |
| Mine contrastive pairs from unlabelled text, leakage measured | ✅ | `qfme mine-pairs` |
| Train a transformer encoder contrastively from scratch | ✅ | Python API |
| Adapt a published checkpoint with LoRA on mined pairs | ✅ | `qfme adapt` |
| Fit a large contrastive batch on 16 GB (gradient caching, bf16) | ✅ | `compute` profile |
| Save an adapted model as a ~3.4 MB artefact | ✅ | `save_adapter` |
| Serve an adapted model, prefixes applied correctly | ✅ | `SemanticSearchPipeline.from_adapter` |
| Score retrieval per language, per pair kind, per overlap band, with Wilson intervals | ✅ | `evaluate_retrieval` |
| Declare an experiment's design and have the data checked against it | ✅ | `--adaptation` |
| Hard-negative mining, domain-specific miners, synthetic pairs | ❌ | Phase C |
| HTTP embeddings endpoint, ONNX export, container image | ❌ | Phase D |
| Neural path in `TrainingPipeline` (a transformer from scratch, from one command) | ❌ | Phase D |
| From-scratch pretraining (MLM then contrastive) | ❌ | Phase E |
| Approximate nearest-neighbour index | ❌ | out of scope — export instead |

## Goals

| Goal | How it is met |
|---|---|
| **Correct across scripts** | Script-aware segmentation and tokenization; Unicode combining marks, ZWJ/ZWNJ and non-whitespace-delimited scripts handled explicitly |
| **Scales past memory** | Every stage streams from a re-iterable source; corpus size is bounded by disk |
| **Reproducible** | Seeded runs, deterministic vocabulary ordering, resolved config persisted with artefacts |
| **Fails early and clearly** | Typed config validated at load; framework errors carry structured context, not opaque tracebacks |
| **Safe to interrupt** | All writes are atomic — a killed run never leaves a truncated model in place |
| **Extensible without forks** | Registries resolve components by name so configuration selects implementations |
| **Maintainable** | Strict layering with an enforced acyclic import graph; full type coverage; 94% test coverage |
| **Honest about quality** | Evaluation reports per-language metrics and structural geometry, and leaves absent benchmarks as `None` rather than `0.0` |

## Non-goals

Stated up front, because a framework that claims everything is useful for nothing:

- **Not a general deep learning framework.** There is a transformer and a training loop,
  but they exist to produce embeddings. torch is an optional extra, not a foundation —
  the corpus, tokenizer and vocabulary layers install and run without it.
- **Not a generative model.** No decoder, no text generation. Embeddings only.
- **Not an approximate nearest-neighbour engine.** Search is exact brute-force cosine.
  Adequate to a few hundred thousand vectors; past that, export to a vector database.
- **Not a language identification library.** Language inference is script-based and
  deliberately returns `None` where a script is shared across languages.
- **Not a general text-cleaning toolkit.** Filtering is conservative by design;
  over-aggressive cleaning silently destroys valid non-Latin text. `qfme validate`
  *reports* extraction damage rather than repairing it, because a corpus that needs
  repairing should be re-extracted.

---

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Foundation — errors, logging, registries, config, utilities | **Implemented** |
| 2 | Corpus — document tree, segmentation, readers/writers, statistics, auditing | **Implemented** |
| 3 | Tokenization — normalizers, pre-tokenizers, SentencePiece | **Implemented** |
| 4 | Vocabulary — token/id mapping, special tokens, persistence | **Implemented** |
| 5 | Static embeddings — word2vec, sentence encoders, similarity search | **Implemented** |
| A | Transformer encoder, contrastive InfoNCE training | **Implemented** |
| B | LoRA, gradient caching, mixed precision, **external checkpoint adaptation** | **Implemented** — exit criterion met on hardware, 21 July 2026 |
| C | Pair mining from unlabelled text | **Substantially done** — Wikipedia structure miners and `qfme mine-pairs` ship; hard negatives, domain miners and synthetic pairs outstanding |
| D | Serving | **Started** — `from_adapter` serves a saved model locally; endpoint, ONNX, container outstanding |
| E | From-scratch pretraining | **Planned** — capability, not the default; see [ROADMAP.md](ROADMAP.md) |

**The dependency split is deliberate.** The base install is `numpy`, `pandas`, `pyyaml`,
`sentencepiece`, `tqdm` — no torch. Everything through vocabulary and static embeddings
runs on that alone, which keeps text preparation a small install for callers that need
nothing else. The transformer lives behind an optional extra:

```bash
uv sync --extra neural        # adds torch
```

Skipping it costs you the contextual encoder and nothing else; the suite skips those
tests rather than failing.

**Honest limit on verification.** Development happens on a machine with no NVIDIA GPU, so
**the CUDA paths are not exercised by any automated test**. They are verified by hand on an
RTX 4070 Ti SUPER — that is where the memory, speed and retrieval numbers above come from,
and `scripts/verify_e2e.py` exists to reproduce the whole path there on demand. But a
device-specific regression will still reach the training box before it reaches CI.

**And a limit on scale.** Every measurement above is at 5.3M or 118M parameters. The target
is a 568M encoder, and 0.29 GB of a 16 GB card says nothing about where that ceiling sits.

---

## Running locally

### Requirements

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.12.x | Pinned to `>=3.12,<3.13` in `pyproject.toml` |
| [uv](https://docs.astral.sh/uv/) | latest | Dependency and environment management |
| OS | Linux, macOS, Windows | Pure Python plus SentencePiece wheels; no compiler needed |
| Disk | ~200 MB base, ~3 GB with `neural` | torch dominates the second figure |
| RAM | 1 GB for the sample corpus | Training streams, so requirements scale with vocabulary size and not corpus size |
| GPU | **Optional** | Not needed for anything in the base install. Contextual training runs on CPU, and will be slow. |

**No GPU is required** and no external service needs configuring. Nothing is downloaded at
runtime beyond the Python dependencies — no model weights, no API keys.

A GPU changes what is *practical*, not what runs. Contextual training on CPU works and is
how the tests verify it; producing a model worth serving wants a card. See
[compute profiles](#running-on-more-than-one-machine).

### Setup

**To use it** — a `qfme` command available anywhere, like any other CLI tool:

```bash
uv tool install 'git+ssh://git@github.com/<owner>/quanfire-multilingual-embedding[neural,wikipedia]'
qfme --version
```

No clone needed — anyone whose SSH key has access to the repository can run that, private
or not. From a clone you already have, `uv tool install '.[neural,wikipedia]'` does the
same.

That is the whole install: no environment to activate, no `PATH` to edit. `uv` gives the
tool its own isolated environment and links a shim into `~/.local/bin`.

**To develop it** — edit the code and run the tests:

```bash
git clone <repository-url>
cd quanfire-multilingual-embedding

uv sync --extra neural --extra wikipedia   # create .venv and install
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
```

**Name the extras.** Plain `uv sync` does not merely skip them, it *uninstalls* them — you
would silently lose the contextual encoder and `qfme extract`. Omit them deliberately if
you want the small install: everything through vocabulary, word2vec, search and evaluation
works without either.

With the *development* install, `qfme` lives in `.venv/bin/` rather than on the system
path, so an unactivated terminal reports `command not found: qfme`. That is expected — use
`uv run qfme …`, `.venv/bin/qfme …`, or activate as above. The *tool* install has no such
caveat. Full detail in [`docs/installing.md`](docs/installing.md).

### Verify the installation

```bash
qfme --version                # the CLI entry point is registered by uv sync
pytest -m "not slow"          # fast suite, no model training
pytest                        # full suite including end-to-end training
```

The full suite trains real tokenizers and embedding models on small fixtures and
completes in under twenty seconds.

### See it work, step by step

[`examples/walkthrough/`](examples/walkthrough/README.md) is a fifteen-minute tour with
real commands and their real output: auditing a damaged corpus, training, searching in
three scripts, per-language fairness, the static model's structural ceiling, proof the
contextual encoder learns, and the cost of domain adaptation. It also states plainly
what does not work yet.

### Run the worked example

```bash
uv run python examples/train_and_search.py
```

This trains on the bundled six-language sample corpus and runs queries in English,
Hindi, French and Tamil. Artefacts land in `artifacts/example/` and the evaluation
report in `reports/example/`; both directories are gitignored.

### Optional: quality tooling and docs

```bash
pre-commit install            # run lint, format and type checks on every commit
mkdocs serve                  # documentation at http://127.0.0.1:8000
```

### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Vocabulary size too high (N). Please set it to a value <= M` | `tokenizer.vocab_size` exceeds what your corpus supports. Lower it to the suggested value, or use more text. |
| `Vocabulary size is smaller than required_chars` | The opposite problem: your corpus has more distinct characters than the vocabulary can hold. **Raise** `vocab_size`, or lower `character_coverage`. Common with many scripts at once. |
| `Corpus produced no sentences after filtering` | `corpus.min_sentence_characters` is too aggressive for your input, or the source path matched no files. |
| `qfme: command not found` | The environment is not active, or `uv sync` has not been run. |
| Search returns no results | The query contains no in-vocabulary tokens. Check `unknown_rate` in the evaluation report. |

---

## Quick start

A six-language sample corpus ships in `data/sample/corpus.jsonl` (English, Hindi,
Tamil, Japanese, Arabic, French — 150 documents, 750 sentences).

### Inspect a corpus

```bash
qfme stats --source data/sample/corpus.jsonl
```

### Check a corpus before training on it

```bash
qfme validate --source data/wikipedia/hi.jsonl.gz
```

Extraction pipelines fail quietly. Markup that survived cleaning, a wrong encoding guess,
the same article ingested twice, an unpopulated language column — none of those raise.
They produce a corpus that loads, trains, and yields a worse model for reasons that are
invisible by the time anyone reads the metrics.

`validate` names them, each with a count, examples and a remedy, and exits non-zero on
errors so a data pipeline can gate on it:

```bash
qfme validate --source "$OUT" || exit 1              # blocks on errors
qfme validate --source "$OUT" --strict || exit 1     # blocks on warnings too
```

The format an extraction must produce is specified in
[`docs/data-format.md`](docs/data-format.md).

### Train

```yaml
# experiment.yaml
name: demo
seed: 42
corpus:
  source: data/sample/corpus.jsonl
  format: jsonl
tokenizer:
  vocab_size: 300
embedding:
  dimension: 64
  window: 4
  min_count: 2
  epochs: 8
```

```bash
qfme train --config experiment.yaml
```

```json
{
  "name": "demo",
  "documents": 150,
  "sentences": 750,
  "vocabulary_size": 226,
  "dimension": 64,
  "characters_per_token": 2.922,
  "unknown_rate": 0.0,
  "experiment_directory": "artifacts/demo"
}
```

### Search

```bash
qfme search --experiment artifacts/demo \
            --source data/sample/corpus.jsonl \
            --query "अभियंता मशीन लर्निंग पढ़ता है" --top-k 3
```

```
 1. [0.9996] अभियंता लिखता है मशीन लर्निंग।
 2. [0.9996] अभियंता पढ़ता है मशीन लर्निंग।
 3. [0.9996] अभियंता देखता है मशीन लर्निंग।
```

### Adapt a published checkpoint

`qfme train` builds a model from your corpus. `qfme adapt` takes one that already exists
and specialises it, which is the route that produces the models this project ships. It needs
the `neural` and `pretrained` extras, a checkpoint, and a pair file from `qfme mine-pairs`.

```bash
qfme adapt --config examples/adaptation.yaml --profile configs/cpu.yaml \
    --set adaptation.pairs=data/pairs/hi.jsonl.gz
```

It prints three things in order: the published checkpoint's score on held-out pairs, the
training run, and the same score again afterwards. The first is the number to beat — beating
chance, or beating an untrained model, proves nothing about whether adaptation was worth
doing. The verdict block, with the recorded Hindi figures filled in:

```
published checkpoint     recall@1 0.4238   MRR 0.5136
after LoRA adaptation    recall@1 0.5451   MRR 0.6364

recall@1 +0.1213  (+28.6%)   -> BETTER
the model itself moved by <max change in a probe vector>
```

Two details there are the point rather than decoration. **The probe** — the last line —
re-encodes sixteen anchors before and after and reports the largest change, which
distinguishes "the adaptation did not help" from "the adaptation did not happen". Those two
have opposite remedies, and without the probe they are the same line of output. **The
declared mode**: `adaptation: in-distribution` in that config says what the run measures, it
is checked against what the filters actually vary, and a run whose label and data disagree is
refused before the model loads. The label outlives the command line, so it must not be able
to be wrong.

The command exits non-zero when the adapted model did not beat the checkpoint, so a shell
pipeline that chains adaptation into a deployment step stops rather than shipping a
regression. [The full Wikipedia run is below.](#training-on-wikipedia-in-multiple-languages)

### From Python

```python
from multilingual_embedding import ExperimentConfig, TrainingPipeline, SemanticSearchPipeline
from multilingual_embedding.corpus import stream_sentences

config = ExperimentConfig(name="demo")
config.corpus.source = "data/sample/corpus.jsonl"

result = TrainingPipeline(config).run()

pipeline = SemanticSearchPipeline.from_directory(result.experiment_directory)
pipeline.index(stream_sentences(config.corpus))

for hit in pipeline.search("machine learning", top_k=5):
    print(hit.rank, round(hit.score, 3), hit.text)
```

A complete worked example is in [`examples/train_and_search.py`](examples/train_and_search.py).

---

## Training on Wikipedia, in multiple languages

**When can this start? It already has.** Hindi and Tamil are done end to end — dumps
extracted, audited, mined, adapted, measured, and the adapter saved as `models/indic-v1`.
Nothing is blocking a third language, or a twentieth. What follows is the recipe and its
real costs.

### The four commands

```bash
# 1. Fetch a dump. ~200-300 MB per mid-sized Indic wiki.
curl -O https://dumps.wikimedia.org/hiwiki/latest/hiwiki-latest-pages-articles.xml.bz2

# 2. Extract. Streams; peak memory is one article.
qfme extract --dump data/dumps/hiwiki-latest-pages-articles.xml.bz2 \
             --output data/corpora/hi.jsonl.gz --language hi

# 3. Gate on quality before spending GPU time.
qfme validate --source data/corpora/hi.jsonl.gz --output reports/hi-audit.json

# 4. Mine pairs, all three kinds, with leakage reported per kind.
qfme mine-pairs --source data/corpora/hi.jsonl.gz \
                --output data/pairs/hi.jsonl.gz \
                --max-overlap 0.9 --report reports/hi-pairs.json
```

Then adapt, on the GPU box:

```bash
qfme adapt --config examples/adaptation.yaml --profile configs/gpu.yaml \
    --set adaptation.pairs=data/pairs/hi.jsonl.gz \
    --save-adapter models/hi-v1 --output reports/hi-v1.json
```

The experiment file holds what decides the result and the profile holds what the box
dictates, so the same command runs on a laptop by naming `configs/cpu.yaml` instead. `adapt`
exits non-zero when the adapted model did not beat the checkpoint it started from, which is
what makes it safe to chain into a deployment step.

The same run as flags, which is how every figure below was produced:

```bash
python scripts/adapt_pretrained.py \
    --checkpoint intfloat/multilingual-e5-small \
    --pairs data/pairs/hi.jsonl.gz \
    --query-prefix "query: " --passage-prefix "passage: " \
    --rank 32 --epochs 2 --batch-size 64 \
    --sample-pairs 120000 --train-pairs 20000 --eval-pairs 2000 \
    --output reports/hi-v1.json --save-adapter models/hi-v1
```

For **multilingual** training, concatenate the pair files rather than training one adapter
per language. That question was settled by a controlled experiment: joint training is
numerically best on both languages, never worse than either specialist, and produces one
artefact instead of two.

```bash
cat data/pairs/hi.jsonl.gz data/pairs/ta.jsonl.gz > data/pairs/indic.jsonl.gz
```

Gzip members concatenate, and every reader here decompresses transparently. Sampling is a
reservoir over the whole file, so a mixed file gives a mixed sample without interleaving.

### What it costs, measured

Per language, on an Intel MacBook with no GPU (steps 2–4):

| Step | Hindi | Tamil |
|---|---:|---:|
| `extract` | 7.4s → 118,571 articles | 8.2s → 163,768 articles |
| `validate` | 11m 02s | 12m 15s |
| `mine-pairs` | 25m 04s → 642,536 pairs | 28m 46s → 893,523 pairs |
| Peak resident memory | **< 201 MB** | **< 201 MB** |
| Disk (corpus + pairs) | ~288 MB | ~276 MB |

So **roughly 40 minutes and 300 MB of disk per language, on a laptop, with no GPU.** The
adaptation itself is minutes on the 4070 Ti at 20,000 pairs — the data preparation dominates.
Two languages at once is under two hours of wall-clock, most of it unattended.

### What the results say about which languages to add

The controlled task/language experiment above changes the obvious plan. Because adaptation
is **language-general**, the first language buys most of the benefit and each additional one
adds less than its collection cost implies. Because it is **task-specific**, every query
shape to be served must appear in the training mixture.

Practical consequences for a 22-language programme:

1. **Mine where the text is cleanest and most abundant first.** Hindi, Tamil, Bengali,
   Telugu, Marathi have wikis large enough to matter. The smallest — Santali (Ol Chiki),
   Meitei (Meetei Mayek), Dogri — will yield few pairs and, on this evidence, would have
   been largely covered by the larger languages anyway.
2. **Always mine all three pair kinds.** `--kinds` defaults to all of them for this reason.
   A single-shape adapter cost 17% of the achievable gain when the shape was wrong.
3. **Cap the leakiest kind rather than dropping it.** `title_lead` averages 0.98 overlap and
   is still the second-largest source; `--max-overlap 0.9` keeps volume while removing the
   pairs a string matcher solves outright.
4. **Hold the evaluation set fixed with `--eval-pairs-file`** whenever comparing runs, or
   the held-out split moves with the training filter and the comparison measures the wrong
   thing.
5. **Use `--sample-pairs` several times `--train-pairs`** whenever a facet filter is set.
   Filters run after reservoir sampling, so without it a run naming a minority kind silently
   trains on less data — this happened, at 25,000 pairs against 7,000.

### What is *not* yet possible

- **A non-Wikipedia corpus axis is untested.** Every comparison so far has Wikipedia on both
  sides. `--adaptation domain` exists for it and needs a pair file from real QuanFire
  documents to run — that is the experiment that would justify "this will help on our
  contracts".
- **No hard negatives.** Negatives are in-batch only. Mining hard ones against a base
  encoder is Phase C's remaining piece and is the most likely next source of gain.
- **No cross-lingual pairs.** Nothing mines translation pairs, so cross-lingual retrieval
  works only to the extent the corpus contained comparable content.
- **`qfme extract` needs the `wikipedia` extra** (`mwparserfromhell`). Without it the
  command raises a message naming the fix rather than an `ImportError`.

---

## What makes it multilingual

The word "multilingual" is easy to claim. Concretely, these are the places where
treating it as a core requirement changed the implementation:

**Sentence segmentation knows more than the full stop.** The terminator inventory
covers the Devanagari danda (`।`) and double danda (`॥`), the CJK ideographic full
stop (`。`), the Arabic question mark (`؟`), the Urdu full stop (`۔`), the Ethiopic
full stop (`።`), the Ol Chiki mucaad (`᱾`) and the Meetei Mayek cheikhei (`꯫`). CJK
and Indic terminators end a sentence without a following space, so the usual
"period, space, capital letter" heuristic never fires for them and is bypassed.

**All 22 scheduled languages of India are supported**, plus English — verified
end to end by [`tests/corpus/test_indian_languages.py`](tests/corpus/test_indian_languages.py),
which asserts script detection, segmentation, word splitting and language naming for
each. That spans ten scripts, including Ol Chiki (Santali) and Meetei Mayek (Meitei),
and the six languages that have no ISO 639-1 two-letter code and must be identified by
their three-letter ISO 639-2/3 form.

**Word splitting handles combining marks.** Python's `\w` does not match Unicode
combining marks, so a naive `\w+` both fragments words *and silently drops characters*:

| Input | Naive `\w+` | Correct |
|---|---|---|
| `नमस्ते दुनिया` (2 words) | `['नमस', 'त', 'द', 'न', 'य']` | `['नमस्ते', 'दुनिया']` |
| `हैं` (1 word) | `['ह']` — two of three codepoints lost | `['हैं']` |
| `مُحَمَّد` (1 word) | `['م', 'ح', 'م', 'د']` | `['مُحَمَّد']` |

The splitter builds its own character class from the Unicode database instead, and
treats ZWJ/ZWNJ as word-internal because they are meaningful in Devanagari and Arabic.

**Script detection drives behaviour, not just labels.** Han, Hiragana, Katakana and
Thai are flagged as not whitespace-delimited, so the pre-tokenizer segments them per
character instead of producing one token per sentence.

**Language inference refuses to guess.** Devanagari implies Hindi and Hangul implies
Korean, but Latin, Arabic, Cyrillic and Han are each shared by many languages, so
those return `None` rather than a plausible-looking wrong answer.

**Evaluation reports tokenizer fairness.** A vocabulary trained on a corpus that is
mostly English will encode English efficiently and everything else poorly, and a
single average hides that. Metrics are reported per language:

```
lang    chars/token   fertility   unknown
ar            2.864       2.200    0.0000
en            3.978       1.667    0.0000
fr            3.642       1.963    0.0000
hi            2.750       2.000    0.0000
ja            1.066      13.320    0.0000
ta            4.070       2.048    0.0000
```

The 3.8× spread between Japanese and Tamil is the number you would want to act on
before training anything larger.

---

## Architecture

Layered, with an acyclic import graph. Each layer may only import from layers below
it — enforced by [`tests/test_architecture.py`](tests/test_architecture.py), which
parses the source and fails on any upward or sideways import.

```
pipelines        training and search workflows
    |
evaluation       metrics, scoring, reports
    |
embedding        word2vec, transformer encoder, contrastive training, similarity index
    |
tokenizer        normalizers, pre-tokenizers, SentencePiece
    |
vocabulary       token <-> id mapping
    |
corpus           document tree, segmentation, readers, statistics
    |
config           typed, validated configuration
    |
core / utils     errors, logging, registries, I/O
    |
common           spans, enums, type aliases, constants
```

Every package carries its own README with its modules, design decisions and a runnable
example:

| Package | Responsibility |
|---|---|
| [`common`](src/multilingual_embedding/common/README.md) | Spans, enums, type aliases, constants |
| [`core`](src/multilingual_embedding/core/README.md) | Exceptions, logging, registry, factory |
| [`config`](src/multilingual_embedding/config/README.md) | Typed configuration and loading |
| [`utils`](src/multilingual_embedding/utils/README.md) | Validation, hashing, filesystem, I/O, serialization |
| [`corpus`](src/multilingual_embedding/corpus/README.md) | Document tree, segmentation, readers, statistics, Wikipedia extraction, pair mining |
| [`vocabulary`](src/multilingual_embedding/vocabulary/README.md) | Token/id mapping, special tokens |
| [`tokenizer`](src/multilingual_embedding/tokenizer/README.md) | Normalizers, pre-tokenizers, SentencePiece |
| [`embedding`](src/multilingual_embedding/embedding/README.md) | word2vec, sentence encoders, similarity index, the `TextEncoder` contract |
| [`embedding/neural`](src/multilingual_embedding/embedding/neural/README.md) | Transformer encoder, contrastive training, LoRA, gradient caching, pretrained adaptation, the adapter artefact |
| [`evaluation`](src/multilingual_embedding/evaluation/README.md) | Metrics, scoring, reports |
| [`pipelines`](src/multilingual_embedding/pipelines/README.md) | Training and search workflows |

Outside the package, [`scripts/`](scripts/README.md) holds the adaptation experiment, the
end-to-end verifier and one diagnostic, and [`data/`](data/README.md) documents the corpus
and dump layout.

The corpus is a tree:

```
Corpus -> Document -> Paragraph -> Sentence -> Token
```

Each node stores its span **relative to its immediate parent**. Segmentation stays
local — a paragraph can be re-segmented without renumbering the rest of the document
— at the cost of needing `corpus/offsets.py` to resolve absolute positions.

Container nodes keep their own `text` rather than deriving it by joining children,
because the material *between* children (whitespace, punctuation, markup) is part of
the source and would be lost. `verify()` checks the two views agree.

See [`docs/architecture.md`](docs/architecture.md) for the full walkthrough.

---

## The three model families

All three are produced by this repository, from the same corpus. They differ in what a
vector can represent and in where the pretraining came from.

| | Static (`word2vec`) | Contextual (ours) | Adapted (published + LoRA) |
|---|---|---|---|
| Vector per | token type | token occurrence, pooled to a text | same |
| Runtime | pure numpy | torch (`neural` extra) | torch + `transformers` |
| Training | skip-gram, negative sampling | contrastive InfoNCE from scratch | contrastive InfoNCE over frozen weights |
| Needs pairs | no — raw text is enough | yes | yes |
| Pretraining scale | none | whatever you can afford | **someone else's, free** |
| Trains on CPU | comfortably | slowly, but yes | slowly, but yes |
| Downloads at runtime | no | no | **yes**, once, cached |
| Artefact size | vocab × dim × 4 bytes | full model | **3.4 MB** adapter |
| Reachable from `qfme` | yes | **no — Python API only** | **no — `scripts/adapt_pretrained.py`** |
| What it is for | the baseline every claim is measured against | owning a trustworthy training loop | **the models actually shipped** |

**The adapted route is the product.** Writing the transformer out was the right way to build
a training loop worth trusting — a borrowed checkpoint would have masked a broken loop, and
a good model trains adequately in spite of bugs. But pretraining scale is precisely what a
single consumer GPU cannot reproduce, so the encoder worth serving starts from someone
else's weights and earns its advantage from a corpus nobody else has.

**The static model has a structural ceiling, and it is worth seeing rather than reading
about.** In `river bank` and `savings bank`, word2vec assigns `bank` one vector — the two
are byte-identical, because the model has one row per token type and no notion of
context. No amount of data fixes it. That limitation is the entire reason the contextual
encoder exists, and it is why the static model is kept as a *baseline* rather than
retired: a contextual model that cannot beat it has not learned anything.

The transformer is written out in this repository rather than imported — pre-norm
residuals, fused scaled dot-product attention, GELU, learned positions, mean pooling over
the true mask. Pre-norm trains more stably, at one real cost: most published encoders are
post-norm, so external weights **do not transfer into it** — and because the shapes match,
such a load *succeeds* and produces a model that is structurally valid and numerically
wrong. That is why the adapted route goes through the upstream library instead
(`neural/pretrained.py`) rather than inventing that failure.

**Adapting a model to a domain does not require retraining it.** LoRA freezes the base
and learns a low-rank update. Measured at BERT-base shape **with rank 16** — the rank is
what sets these numbers, so a figure quoted without it means nothing: **0.81% of
parameters trainable**, a 3.4 MB adapter against a 419 MB model, and optimiser state
falling from 0.82 GB to 6.8 MB. Halving the rank halves all three. That is what makes one base model plus several domain adapters
practical on a single card.

**Batch size is a quality parameter here, not just a throughput one.** Contrastive
training contrasts each query against every other passage in the batch, so a batch of 256
poses a far harder task than a batch of 16. Gradient caching is what makes a large batch
fit: it encodes in chunks and caches the vector gradients, so peak memory follows the
chunk rather than the batch. It is mathematically exact — verified gradient-for-gradient
identical to one large backward pass, and invariant to chunk size.

---

## Design decisions worth knowing

**Streaming by default.** Tokenizer training, vocabulary construction and every
embedding epoch pull from a re-iterable `SentenceStream`. Corpus size is bounded by
disk, not memory. Build an in-memory `Corpus` only when you need random access or
splitting.

**Reproducibility is enforced, not encouraged.** Runs are seeded; the resolved
configuration is written next to the artefacts it produced. Vocabulary ordering is
deterministic (frequency descending, ties broken on the token string), so the same
corpus yields a byte-identical vocabulary across runs.

**Special token ids are fixed.** `pad=0, unk=1, bos=2, eos=3`, matched between
SentencePiece and `Vocabulary`. Padding is id 0 so a zero-filled array is a valid
padded batch. These are baked into every trained model, so they are not configurable.

**Splits are at document level.** Sentences within a document are highly correlated;
dividing them across a train/eval boundary would let the model see near-duplicates of
what it is scored on.

**Writes are atomic.** A partially written model is worse than no model, because the
next run will happily load the truncated version.

**Failures are typed and contextual.** Every framework error carries structured
key/value context. A tokenizer that fails because `vocab_size` exceeds what the
corpus supports says so, and says what the corpus supports.

---

## Configuration

Precedence, lowest to highest: dataclass defaults → config file → compute profile →
`QFME_` environment variables → `--set` overrides.

```bash
export QFME_EMBEDDING__DIMENSION=256          # double underscore nests
qfme train --config experiment.yaml --set embedding.epochs=20
```

Every value is validated at load time against the dataclass that owns it, so a bad
setting fails immediately rather than an hour into training. An error also records which
of those layers introduced it, because a value that is fine in the file and broken by a
profile is otherwise hard to place. Full field reference in
[`docs/configuration.md`](docs/configuration.md).

### Running on more than one machine

A configuration has two halves. The **experiment** — corpus, tokenizer, embedding,
evaluation — determines the result. The **machine** — device, precision, batch size,
gradient-cache chunking — determines what fits and how fast it runs. Only the
second differs between a laptop and a training box, so only the second lives in a profile:

```bash
qfme train --config experiments/indic.yaml --profile configs/cpu.yaml   # development
qfme train --config experiments/indic.yaml --profile configs/gpu.yaml   # training box
```

One branch, one experiment file, one set of code. The alternative — a branch per machine —
makes every fix land twice and quietly destroys the guarantee that the code you tested is
the code that trained.

One wrinkle worth knowing: `batch_size` is machine-shaped but *not* result-neutral,
because in contrastive training it sets how many negatives each query is contrasted
against. The `cpu` profile trains a worse model on purpose. Full treatment in
[`docs/compute-profiles.md`](docs/compute-profiles.md).

---

## Running in production

The framework is a library and a CLI, not a service. There is no server, scheduler or
database to operate. Production use means two separable concerns: an offline training
job, and an online inference process that loads its artefacts.

### Separate training from serving

Training is a batch job — CPU-bound, memory-flat, minutes to hours depending on corpus
size. Serving loads the artefacts and answers queries. Do not train inside a request
path.

```
   [ batch job ]                        [ long-lived process ]

   qfme train --config prod.yaml   ->   artifacts/<name>/
                                          tokenizer/     ->  SemanticSearchPipeline
                                          embedding/         .from_directory(...)
                                          config.yaml
```

`SemanticSearchPipeline.from_directory()` deliberately loads from disk rather than
accepting in-memory objects, because that is the path a deployed service takes and it
is therefore the path the tests exercise.

### Artefact layout and versioning

A training run writes a self-describing directory:

```
artifacts/<name>/
├── config.yaml              the fully resolved configuration that produced this run
├── tokenizer/
│   ├── tokenizer.model      SentencePiece model
│   └── tokenizer.vocab      its vocabulary listing
└── embedding/
    ├── vectors.npy          float32 matrix, rows indexed by vocabulary id
    ├── vocabulary.json      token <-> id mapping with frequencies
    ├── metadata.json        dimension and format version
    └── word2vec.json        the hyperparameters used, so a reload restores them

reports/<name>/
├── report.json              machine-readable metrics
└── report.md                human-readable summary
```

Note that a reloaded model can do lookup and search but cannot resume training: the
output matrix is discarded after fitting, as it is in the original word2vec.

Treat this directory as an immutable, versioned build artefact. Publish it to object
storage or a model registry keyed by a build identifier, and have the serving process
pull a pinned version rather than the latest. Because `config.yaml` travels with the
model, any deployed artefact can be traced to the exact settings and corpus revision
that produced it.

### Operational guidance

**Logging.** Call `configure_logging(log_format="json")` at process start. Records are
emitted one JSON object per line with structured fields, ready for log aggregation.
The framework never configures logging as an import side effect, so it will not fight
your application's setup, and records do not propagate to the root logger.

```python
import logging
from multilingual_embedding import configure_logging

configure_logging(level=logging.INFO, log_format="json")
```

**Configuration.** Supply settings through `QFME_`-prefixed environment variables so
that container orchestration is the source of truth and no config file needs to be
baked into the image.

**Error handling.** Catch `MultilingualEmbeddingError` to distinguish framework
failures from everything else; each carries structured `.context` suitable for logging
as fields rather than as a message blob.

**Memory.** Training memory scales with vocabulary size, not corpus size: the
embedding matrices are `2 × vocab_size × dimension × 4` bytes. A 32k vocabulary at 300
dimensions is roughly 77 MB. Serving needs one matrix plus your indexed vectors.

**Concurrency.** `SemanticSearchPipeline` is read-only after `index()` and safe to
share across threads. Training is single-process.

**Determinism.** Pin the seed and the artefact version. Given both, a run reproduces
byte-identically.

### Before you go live

- [ ] Check `unknown_rate` in the evaluation report — it should be near zero
- [ ] Check the **per-language** `characters_per_token` spread, not just the average;
      a wide spread means some languages are being served much worse than others
- [ ] Check `zero_vector_count` — anything beyond the padding row means vocabulary
      entries never appeared in training
- [ ] Check `isotropy` and `effective_dimensions`; a collapsed space makes cosine
      similarity stop discriminating
- [ ] Confirm the corpus licence permits the use you intend — a model inherits the
      licensing constraints of its training text, which is why `DocumentMetadata`
      carries a `license` field
- [ ] Size the index: search is exact, so latency grows linearly with the number of
      indexed vectors

### Scaling limits to plan around

Exact cosine search is the right choice up to roughly 10⁵–10⁶ vectors. Beyond that
you need an approximate index, which this framework does not provide; the embeddings
themselves are plain `float32` numpy arrays, so they feed directly into any approximate
nearest-neighbour library or vector database without conversion.

---

## Pros and cons

An honest assessment. The cons are structural choices with reasons, not a defect list.

### Pros

| | |
|---|---|
| **Owned end to end** | No opaque model file, no paid API. Every stage is inspectable code, and a bad result on one language can be traced to the segmentation, the vocabulary or the pairs that caused it. |
| **Genuinely multilingual** | 22 scheduled Indian languages verified end to end across ten scripts. The combining-mark, danda, ZWJ and non-whitespace-delimited handling are structural, not patches. |
| **Cheap to specialise** | 3.4 MB per domain adapter over one shared base. Many domain models cost roughly one model's storage, and a run is minutes rather than days. |
| **Fits real hardware** | Gradient caching (12.2× memory) plus bf16 (1.6×) makes a competitive contrastive batch fit on a 16 GB consumer card. Without it this hardware could not train a competitive model however long it ran. |
| **Honest measurement** | Wilson intervals on every retrieval number, per language, per pair kind, per lexical-overlap band, always against a named baseline. Claims that did not survive a recount were corrected in `ROADMAP.md` rather than quietly dropped. |
| **Fails loudly where it can** | Typed errors with structured context, atomic writes, a corpus audit that exits non-zero, an experiment-design check that refuses a run whose label and data disagree. |
| **Small base install** | No torch needed for corpus, tokenizer, vocabulary, static embeddings or evaluation. Text preparation is a light dependency. |
| **Reproducible** | Seeded runs, deterministic vocabulary ordering, resolved config written beside every artefact, `local_files_only` to refuse the network. |
| **Streams** | Under 201 MB peak resident memory over a 227 MB Wikipedia dump, at every stage. Corpus size is bounded by disk. |

### Cons

| | Why it is this way |
|---|---|
| **The best models are not ours** | The shipped route starts from a published checkpoint. Pretraining scale cannot be reproduced on one consumer GPU, so the differentiation has to come from corpus and domain instead. Phase E exists for independence, not because it would be better. |
| **One command short of complete** | `qfme` covers the corpus-to-static path and the adaptation path. A transformer trained *from scratch* is still Python-API only: `TrainingPipeline` has no neural stage. Adaptation was the same story until recently — the experiment design was changing weekly and freezing it into a subcommand would have meant a contract that had to break — and `qfme adapt` is what that settling produced. |
| **Search is exact only** | Brute-force cosine, right to ~10⁵–10⁶ vectors. Wrapping a poor ANN implementation would be worse than being honest about the ceiling; the vectors are plain float32 and export anywhere. |
| **No CUDA in CI** | Development has no NVIDIA GPU. GPU claims are hand-verified and reproducible via `scripts/verify_e2e.py`, but a device regression reaches the training box before it reaches CI. |
| **Everything measured is Wikipedia** | Both sides of every comparison so far. The corpus axis — does this survive contact with real contracts and invoices — is the untested one, and it is the one the business case rests on. |
| **Everything measured is small** | 5.3M and 118M parameters. The 568M target is unvalidated. |
| **Segmentation is rule-based** | Fast, predictable, dependency-free, and it will split on an unknown abbreviation before a capitalised noun. Readers accept pre-segmented input when that is not good enough. |
| **Deduplication is exact-match only** | Near-duplicate detection carries a false-positive risk, and a false positive here silently deletes legitimate text — most likely in the least-represented language, where it is hardest to spot. |
| **Language inference refuses to guess** | Returns `None` for Latin, Arabic, Cyrillic and Han. A plausible wrong answer propagates into metadata, segmentation rules and normalizers with nothing downstream able to notice. |
| **In-batch negatives only** | No hard-negative mining yet, which is the most likely remaining source of gain. |
| **Single-process training** | No data-loader parallelism, no distributed training. One process, one device. |
| **Runtime download on the adapted route** | Base weights are fetched and cached on first use, which breaks the otherwise-absolute "downloads nothing at runtime" property. Opt-in behind an extra, and `local_files_only=True` disables it. |

---

## Project layout

```
quanfire-multilingual-embedding/
├── src/multilingual_embedding/     the framework (see the package table above)
│   ├── common/  core/  config/  utils/
│   ├── corpus/                     tree, segmentation, readers, wikipedia.py, pairs.py
│   ├── vocabulary/  tokenizer/  evaluation/  pipelines/
│   ├── embedding/                  word2vec and the TextEncoder contract
│   │   └── neural/                 transformer, LoRA, gradcache, pretrained, adapter
│   ├── cli.py                      the `qfme` command
│   └── py.typed                    marks the package as typed for consumers
├── tests/                          1357 tests mirroring the source layout
├── scripts/                        adapt_pretrained.py, verify_e2e.py, diagnose_audit.py
├── configs/                        compute profiles — cpu.yaml, gpu.yaml
├── examples/                       runnable end-to-end example and a walkthrough
├── data/
│   ├── sample/                     six-language sample corpus (committed)
│   └── dumps/                      Wikipedia dumps (gitignored, ~485 MB for hi + ta)
├── docs/                           MkDocs documentation
├── knowledge-base/                 the reference handbook (PDF), its source and build script
├── models/                         saved LoRA adapters, e.g. indic-v1 (gitignored)
├── artifacts/                      trained static experiments (gitignored)
├── reports/                        evaluation output (gitignored)
├── verify-output/                  end-to-end verification products (gitignored)
├── .github/workflows/ci.yml        lint, types, tests, build, docs
├── pyproject.toml                  dependencies and tool configuration
└── uv.lock                         pinned dependency versions
```

`models/`, `artifacts/`, `reports/`, `verify-output/` and `data/dumps/` are gitignored
build products — they exist on the training box and not in a fresh clone.

---

## Development

```bash
pytest                      # full suite
pytest -m "not slow"        # skip model-training integration tests
pytest --cov                # with coverage

ruff check src tests        # lint
ruff format src tests       # format
mypy                        # strict type checking

pre-commit install          # run all of the above on commit
mkdocs serve                # docs at http://127.0.0.1:8000
```

CI runs the same gates on every push and pull request, plus a wheel build that asserts
`py.typed` is packaged and a smoke test that installs the wheel in a clean environment.

### Extending the framework

Components are resolved by name from registries, so adding one does not mean editing
the pipeline:

```python
from multilingual_embedding.tokenizer.normalizer import NORMALIZERS, Normalizer

@NORMALIZERS.register("my-normalizer")
class MyNormalizer(Normalizer):
    def normalize(self, text: str) -> str:
        return text.replace(" ", " ")
```

It is then selectable from configuration:

```yaml
tokenizer:
  normalizers:
    - type: nfkc
    - type: my-normalizer
```

The same pattern applies to pre-tokenizers, tokenizers, corpus readers and sentence
encoders. When adding a package, respect the layering rule — the architecture test
will fail the build otherwise.

---

## Limitations

Stated plainly, because knowing where a tool stops is part of using it well.

- **Search is exact, not approximate.** Brute-force cosine over a normalized matrix is
  the right choice up to roughly 10⁵–10⁶ vectors. Past that you need an ANN index,
  which this framework does not provide.
- **Sentence segmentation is rule-based.** Fast, predictable and dependency-free, but
  it will not resolve genuinely ambiguous cases. Readers accept pre-segmented input
  for when you need better.
- **Deduplication is exact-match only.** Near-duplicate detection needs MinHash or
  SimHash; exact matching was chosen because it carries no false-positive risk.
- **Cross-lingual alignment is not guaranteed.** All languages share one vector space,
  but a query in one language retrieves another's sentences only to the extent the
  training corpus contained parallel or comparable content.
- **Language inference is script-based**, not statistical, and returns `None` for
  scripts shared across languages.
- **A transformer trained from scratch has no CLI path.** `qfme train` produces the static
  model and `qfme adapt` runs the adaptation experiment, but `TrainingPipeline` has no
  neural stage, so a contextual encoder trained from nothing is still driven through the
  Python API.
- **Training is single-process.** There is no data-loader parallelism and no
  distributed training; a run is bounded by one process on one device.
- **External weights do not load into our own transformer.** It is pre-norm and most
  published encoders are post-norm; the shapes match, so such a load succeeds and is
  numerically wrong. External checkpoints *are* supported — through their own library, in
  `neural/pretrained.py` — which is a different thing from that loader existing.
- **Negatives are in-batch only.** No hard-negative mining yet.
- **CUDA is unverified by any automated test**, though verified by hand on an RTX 4070 Ti
  SUPER: gradient caching cut peak VRAM 12.2×, bf16 a further 1.6×, with final losses
  within 0.51%. Development still happens without an NVIDIA GPU, so device-specific bugs
  surface first on the training box.
- **The corpus axis is untested.** Every adaptation result so far has Wikipedia on both
  sides. `--adaptation domain` exists to test transfer to real documents and has not been
  run, so "this will help on our contracts" is not yet a claim this repository can make.
- **Everything is measured small.** 5.3M and 118M parameters against a 568M target.

---

## License

To be decided.
