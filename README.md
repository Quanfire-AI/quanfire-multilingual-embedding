# QuanFire Multilingual Embedding

**A production-grade framework for turning multilingual text into searchable embeddings.**

Raw text in, semantic search out — with the corpus handling, tokenization, vocabulary
management and evaluation in between built to be inspected, configured and reproduced.

```
corpus  ->  tokenizer  ->  vocabulary  ->  embeddings  ->  evaluation  ->  search
```

**963 tests · 94% coverage · `ruff` clean · `mypy --strict` clean · layer graph verified acyclic**

---

## Table of contents

- [Purpose](#purpose)
- [Objective](#objective)
- [Goals](#goals)
- [Non-goals](#non-goals)
- [Status](#status)
- [Running locally](#running-locally)
- [Quick start](#quick-start)
- [What makes it multilingual](#what-makes-it-multilingual)
- [Architecture](#architecture)
- [Design decisions worth knowing](#design-decisions-worth-knowing)
- [Configuration](#configuration)
- [Running in production](#running-in-production)
- [Project layout](#project-layout)
- [Development](#development)
- [Limitations](#limitations)

> **Where this is going:** [ROADMAP.md](ROADMAP.md) sets out the path to an embedding
> model factory — corpus in, trained and evaluated model out, generic or adapted to a
> specific domain. The word2vec model below is the static baseline that future encoders
> are measured against. [ECOSYSTEM.md](ECOSYSTEM.md) places this repository within the
> wider QuanFire AI stack and explains which modalities are worth training rather than
> integrating.

> **Looking for the full reference?**
> [`knowledge-base/QuanFire-Multilingual-Embedding-Handbook.pdf`](knowledge-base/QuanFire-Multilingual-Embedding-Handbook.pdf)
> is a 47-page handbook covering purpose, design, architecture, components, usage, local
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

Provide a **complete, self-contained, production-quality pipeline** that takes a
multilingual corpus and produces a trained tokenizer, a vocabulary, an embedding
matrix, an evaluation report and a working semantic search interface — with no
external model downloads, no API keys, and no deep-learning runtime.

Concretely, the framework must be able to:

1. Read a corpus from plain text or JSON Lines, on disk or gzipped, larger than memory.
2. Segment it correctly across Latin, Devanagari, Tamil, Han, Kana, Arabic and more.
3. Train a subword tokenizer and a shared vocabulary over it.
4. Train word vectors on the tokenizer's own output.
5. Score the result — including per-language fairness — and write a report.
6. Answer semantic queries against the trained model, in any language it was trained on.
7. Do all of the above from a single command, or from a Python API, deterministically.

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

- **Not a deep learning framework.** No autograd, no GPU, no PyTorch. See [Status](#status).
- **Not an approximate nearest-neighbour engine.** Search is exact brute-force cosine.
- **Not a language identification library.** Language inference is script-based and
  deliberately returns `None` where a script is shared across languages.
- **Not a general text-cleaning toolkit.** Filtering is conservative by design;
  over-aggressive cleaning silently destroys valid non-Latin text.

---

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Foundation — errors, logging, registries, config, utilities | **Implemented** |
| 2 | Corpus — document tree, segmentation, readers/writers, statistics | **Implemented** |
| 3 | Tokenization — normalizers, pre-tokenizers, SentencePiece | **Implemented** |
| 4 | Vocabulary — token/id mapping, special tokens, persistence | **Implemented** |
| 5 | Embeddings — word2vec, sentence encoders, similarity search | **Implemented** |
| 6 | Transformer encoder, contrastive learning, fine-tuning | **Not implemented** — planned in [ROADMAP.md](ROADMAP.md) |

Phase 6 is deliberately absent rather than stubbed. The dependency set is
`numpy`, `pandas`, `pyyaml`, `sentencepiece`, `tqdm` — there is **no PyTorch**, and
embedding training is pure numpy (skip-gram with negative sampling). Adding a
transformer means adding a deep learning runtime, which is a decision about the
project's dependency posture, not a missing function body.

---

## Running locally

### Requirements

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.12.x | Pinned to `>=3.12,<3.13` in `pyproject.toml` |
| [uv](https://docs.astral.sh/uv/) | latest | Dependency and environment management |
| OS | Linux, macOS, Windows | Pure Python plus SentencePiece wheels; no compiler needed |
| Disk | ~200 MB | Virtual environment and dependencies |
| RAM | 1 GB is enough for the sample corpus | Training streams, so requirements scale with vocabulary size and not corpus size |

There is no GPU requirement and no external service to configure. Nothing is
downloaded at runtime beyond the Python dependencies.

### Setup

```bash
git clone <repository-url>
cd quanfire-multilingual-embedding

uv sync                       # create .venv and install everything, including dev tools
source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

### Verify the installation

```bash
qfme --version                # the CLI entry point is registered by uv sync
pytest -m "not slow"          # fast suite, no model training
pytest                        # full suite including end-to-end training
```

The full suite trains real tokenizers and embedding models on small fixtures and
completes in under twenty seconds.

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
embedding        word2vec, sentence encoders, similarity index
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
| [`corpus`](src/multilingual_embedding/corpus/README.md) | Document tree, segmentation, readers, statistics |
| [`vocabulary`](src/multilingual_embedding/vocabulary/README.md) | Token/id mapping, special tokens |
| [`tokenizer`](src/multilingual_embedding/tokenizer/README.md) | Normalizers, pre-tokenizers, SentencePiece |
| [`embedding`](src/multilingual_embedding/embedding/README.md) | word2vec, sentence encoders, similarity index |
| [`evaluation`](src/multilingual_embedding/evaluation/README.md) | Metrics, scoring, reports |
| [`pipelines`](src/multilingual_embedding/pipelines/README.md) | Training and search workflows |

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

Precedence, lowest to highest: dataclass defaults → config file → `QFME_`
environment variables → `--set` overrides.

```bash
export QFME_EMBEDDING__DIMENSION=256          # double underscore nests
qfme train --config experiment.yaml --set embedding.epochs=20
```

Every value is validated at load time against the dataclass that owns it, so a bad
setting fails immediately rather than an hour into training. Full field reference in
[`docs/configuration.md`](docs/configuration.md).

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
share across threads. Training is single-process; the `workers` setting is currently
informational.

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

## Project layout

```
quanfire-multilingual-embedding/
├── src/multilingual_embedding/     the framework (see the package table above)
│   ├── common/  core/  config/  utils/
│   ├── corpus/  vocabulary/  tokenizer/  embedding/  evaluation/  pipelines/
│   ├── cli.py                      the `qfme` command
│   └── py.typed                    marks the package as typed for consumers
├── tests/                          963 tests mirroring the source layout
├── examples/                       runnable end-to-end example
├── data/sample/                    six-language sample corpus
├── docs/                           MkDocs documentation
├── knowledge-base/                 the reference handbook (PDF), its source and build script
├── reports/                        evaluation output (gitignored)
├── .github/workflows/ci.yml        lint, types, tests, build, docs
├── pyproject.toml                  dependencies and tool configuration
└── uv.lock                         pinned dependency versions
```

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
- **Training is single-process.** The `workers` setting is reserved and currently
  informational.

---

## License

To be decided.
