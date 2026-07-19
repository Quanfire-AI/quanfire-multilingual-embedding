# Configuration

Configuration is a tree of dataclasses defined in
`src/multilingual_embedding/config/base.py`, each validating itself in
`__post_init__`. A mistake surfaces at load time, next to the file that caused it,
rather than an hour into a training run.

```
ExperimentConfig
├── name, seed, output_directory
├── corpus       CorpusConfig
├── tokenizer    TokenizerConfig
├── embedding    EmbeddingConfig
└── evaluation   EvaluationConfig
```

!!! warning "Treat config objects as immutable"
    Dataclasses do not re-run `__post_init__` on mutation. Assigning
    `config.embedding.dimension = -1` after construction bypasses every validation
    rule silently. Derive variants with `ExperimentConfig.merged(overrides)`, which
    round-trips through primitives and revalidates on the way back.

---

## `ExperimentConfig` — top level

| Field | Type | Default | Meaning | Validation |
|---|---|---|---|---|
| `name` | `str` | `"default"` | Experiment identifier; names the output directory | Must contain a non-whitespace character |
| `seed` | `int` | `42` | Seed for evaluation's random pair sampling | `>= 0` |
| `output_directory` | `Path` | `artifacts` | Root for all artefacts of this experiment | Coerced to `Path` |
| `corpus` | `CorpusConfig` | see below | Corpus source and segmentation | Delegated |
| `tokenizer` | `TokenizerConfig` | see below | Subword model settings | Delegated |
| `embedding` | `EmbeddingConfig` | see below | Word2vec hyperparameters | Delegated |
| `evaluation` | `EvaluationConfig` | see below | Metrics and reporting | Delegated |

Derived read-only properties:

| Property | Value |
|---|---|
| `experiment_directory` | `output_directory / name` |
| `tokenizer_directory` | `experiment_directory / "tokenizer"` |
| `embedding_directory` | `experiment_directory / "embedding"` |

!!! danger "`seed` does not propagate to `embedding.seed`"
    The `ExperimentConfig.seed` docstring claims it is "propagated to the embedding
    config when that config does not override it." **No code performs that
    propagation.**

    ```python
    >>> ExperimentConfig(seed=7).embedding.seed
    42
    ```

    In practice `ExperimentConfig.seed` seeds only `EmbeddingEvaluator`'s random pair
    sampling. Word2vec weight initialisation and negative sampling read
    `embedding.seed` and nothing else. **Set both** if you want a fully seeded run.

---

## `CorpusConfig`

| Field | Type | Default | Meaning | Validation |
|---|---|---|---|---|
| `source` | `Path \| None` | `None` | File or directory to read | Coerced to `Path`; required by any command that reads text |
| `format` | `str` | `"auto"` | Reader selection. `"auto"` chooses by file extension | Must be one of `auto`, `text`, `jsonl` |
| `patterns` | `list[str]` | `["*.txt", "*.jsonl"]` | Glob patterns applied when `source` is a directory | None |
| `language` | `str \| None` | `None` | Default ISO 639-1 code for documents that declare none | None |
| `encoding` | `str` | `"utf-8"` | Character encoding of source files | None |
| `text_field` | `str` | `"text"` | JSON Lines record key holding the text | None — **see caveat below** |
| `min_sentence_characters` | `int` | `1` | Sentences shorter than this are dropped | `> 0`; `<= max_sentence_characters` |
| `max_sentence_characters` | `int` | `10000` | Sentences longer than this are dropped | `> 0` |
| `lowercase` | `bool` | `False` | Whether the reader lowercases on load | None |

`min_sentence_characters` filters out segmentation debris; `max_sentence_characters`
guards against text where segmentation failed entirely — unsegmented markup, a table,
a file with no terminators.

`lowercase` is off by default on purpose: casing is a tokenizer concern, and
destroying it at load time is irreversible.

!!! bug "`text_field` is not wired through the config path"
    `build_reader()` constructs the reader with `format`, `language`, `encoding` and
    `patterns` only. It never passes `text_field`, so `JsonlReader` always falls back
    to its own default of `"text"`:

    ```python
    >>> c = CorpusConfig(source="alt.jsonl", format="jsonl", text_field="content")
    >>> build_reader(c).text_field
    'text'
    ```

    Loading a corpus whose records use a different key therefore fails with
    `CorpusFormatError: JSON Lines record is missing the text field
    (text_field='text', keys=['content', 'id', 'language'])` — naming the default,
    not what you configured.

    **Workaround:** construct `JsonlReader(source, text_field="content")` directly, or
    rename the field in your data.

---

## `TokenizerConfig`

| Field | Type | Default | Meaning | Validation |
|---|---|---|---|---|
| `model_type` | `TokenizerModel` | `unigram` | Subword algorithm | Coerced from string; one of `unigram`, `bpe`, `word`, `char` |
| `vocab_size` | `int` | `32000` | Target vocabulary size, including special tokens | `> 0`; also constrained by the corpus at train time |
| `character_coverage` | `float` | `0.9995` | Fraction of corpus characters the model must cover | Strictly inside `(0.0, 1.0)` — **1.0 is rejected** |
| `normalizers` | `list[dict]` | `[{"type": "nfkc"}, {"type": "whitespace"}]` | Ordered normalizer specs | Resolved through the normalizer registry — **see caveat** |
| `pretokenizer` | `dict` | `{"type": "whitespace"}` | Pre-tokenizer spec | Resolved through the pre-tokenizer registry — **see caveat** |
| `max_sentence_length` | `int` | `16384` | Longest training sentence in bytes, passed to SentencePiece | `> 0` |
| `model_prefix` | `str` | `"tokenizer"` | Base filename for the trained model artefacts | None |

Unigram is the default because it handles scripts without whitespace word boundaries
more gracefully than BPE. Leaving `character_coverage` below 1.0 lets rare characters
fall back to byte encoding, which matters far more for large-alphabet scripts such as
Han than for Latin.

!!! warning "`character_coverage: 1.0` is rejected"
    The field docstring suggests 1.0 is reasonable for Latin corpora, but validation
    uses an **exclusive** range:

    ```python
    >>> TokenizerConfig(character_coverage=1.0)
    ValidationError: character_coverage must lie in (0.0, 1.0) (value=1.0)
    ```

    Use `0.99999` if you want effectively full coverage.

!!! warning "`vocab_size` must fit the corpus"
    SentencePiece hard-fails if `vocab_size` exceeds the distinct pieces the training
    text can support. The default of `32000` is sized for a real corpus and will fail
    on anything sample-sized. The error names the ceiling:

    ```
    error: vocab_size exceeds what the training corpus can support; reduce vocab_size
    or supply more text (... Please set it to a value <= 327 ...)
    ```

    The opposite failure — a `vocab_size` too *small* to cover the corpus's character
    inventory, which multilingual corpora hit easily — is reported separately and
    needs `vocab_size` **raised** or `character_coverage` lowered.

!!! bug "`normalizers` and `pretokenizer` do not affect `qfme train`"
    These fields configure `WordTokenizer`, which builds its normalizer chain and
    pre-tokenizer from specs. But `TrainingPipeline` uses
    `SentencePieceTrainerAdapter`, which reads only `model_type`, `vocab_size`,
    `character_coverage`, `max_sentence_length` and `model_prefix`. SentencePiece
    applies its own internal normalization.

    Setting `pretokenizer: {type: script}` in an experiment YAML has **no effect on a
    `qfme train` run**. The fields are persisted into `config.yaml` regardless, which
    makes them easy to mistake for settings that took effect. They are live only when
    you construct a `WordTokenizer` yourself.

---

## `EmbeddingConfig`

| Field | Type | Default | Meaning | Validation |
|---|---|---|---|---|
| `dimension` | `int` | `128` | Length of each vector | `> 0` |
| `window` | `int` | `5` | Maximum context distance either side of the centre token | `> 0` |
| `min_count` | `int` | `5` | Tokens rarer than this are excluded from the embedding vocabulary | `> 0` |
| `negative_samples` | `int` | `5` | Negative examples drawn per positive pair | `>= 0` |
| `epochs` | `int` | `5` | Passes over the corpus | `> 0` |
| `learning_rate` | `float` | `0.025` | Initial rate, decayed linearly | `> 0`; `>= min_learning_rate` |
| `min_learning_rate` | `float` | `0.0001` | Floor for the decayed rate | `> 0` |
| `subsample_threshold` | `float` | `0.001` | Frequency above which tokens are randomly discarded. `0` disables | `>= 0` |
| `batch_size` | `int` | `32` | Documented as "sentences per progress update" | `> 0` — **unused** |
| `seed` | `int` | `42` | Seed for weight init and sampling | **None — not validated** |
| `workers` | `int` | `1` | Reserved for future parallel training | `> 0` — **unused** |

`window` is a maximum, not a fixed value: the effective window is redrawn uniformly
from `[1, window]` per centre token, which weights nearer context more heavily without
an explicit distance term.

`min_count` is the field that most often surprises. Tokens below it are **dropped**,
not folded into `<unk>` — training one vector on every rare word produces a centroid
of unrelated meanings. It is also why the trained vocabulary is smaller than
`vocab_size`: in the [getting started](getting-started.md) run, `vocab_size: 300` with
`min_count: 2` yields 226 embedding rows.

!!! note "Fields that are declared but not used"
    - **`batch_size`** appears only in `base.py` (declaration, docstring, validation).
      Nothing in `word2vec.py` or elsewhere reads it. The training loop processes one
      sentence at a time and reports progress via tqdm.
    - **`workers`** is documented as "reserved for future parallel training. Currently
      informational" and is likewise never read. Training is single-threaded.
    - **`seed`** is the one `EmbeddingConfig` field with no validation rule, so a
      negative value is accepted here even though `ExperimentConfig.seed` rejects one.

---

## `EvaluationConfig`

| Field | Type | Default | Meaning | Validation |
|---|---|---|---|---|
| `top_k` | `int` | `10` | Neighbourhood size for retrieval metrics | `> 0` |
| `similarity_dataset` | `Path \| None` | `None` | JSON Lines word-similarity dataset | Coerced to `Path` when set |
| `report_directory` | `Path` | `reports` | Where evaluation reports are written | Coerced to `Path` |
| `sample_size` | `int` | `0` | Corpus sentences sampled for tokenizer stats. `0` means the whole corpus | `>= 0` |

Reports land in `report_directory / name`, as `report.json` and `report.md`.

A `similarity_dataset` file holds one JSON object per line with `word_a`, `word_b` and
`score`:

```json
{"word_a": "cat", "word_b": "dog", "score": 7.5}
```

When it is absent, `similarity_correlation` and `similarity_coverage` stay `None`
rather than defaulting to zero — a missing benchmark is never reported as a failing
score. Note that words must match the tokenizer's vocabulary, which for SentencePiece
means pieces such as `▁cat` rather than `cat`.

---

## The precedence chain

Four sources, each overriding the last:

```
dataclass defaults  →  YAML/JSON file  →  QFME_ environment  →  --set overrides
     lowest                                                        highest
```

Nested mappings merge recursively, so overriding `embedding.dimension` leaves the rest
of the embedding section intact.

### 1. Dataclass defaults

```python
>>> from multilingual_embedding.config.base import ExperimentConfig
>>> c = ExperimentConfig()
>>> c.name, c.seed, c.embedding.dimension, c.tokenizer.vocab_size
('default', 42, 128, 32000)
```

### 2. The config file

YAML (`.yaml`, `.yml`) or JSON (`.json`), selected by suffix. Any other suffix raises
`ConfigurationError`. The file must contain a mapping at the top level; an empty file
is treated as `{}`.

```python
>>> from multilingual_embedding.config.loader import load_config
>>> c = load_config("experiments/quickstart.yaml", use_environment=False)
>>> c.name, c.embedding.dimension, c.tokenizer.vocab_size
('quickstart', 64, 300)
```

### 3. `QFME_` environment variables

Prefix `QFME_`, **double underscore denotes nesting**, and the remainder is lowercased:

```
QFME_NAME=trial-a                 ->  {"name": "trial-a"}
QFME_EMBEDDING__DIMENSION=256     ->  {"embedding": {"dimension": 256}}
QFME_CORPUS__MIN_SENTENCE_CHARACTERS=5
                                  ->  {"corpus": {"min_sentence_characters": 5}}
```

Note that single underscores inside a field name are preserved — only `__` splits a
level. Values are parsed as YAML scalars, so `256` arrives as an `int` and `true` as a
`bool` rather than as strings that would fail validation.

```console
$ QFME_EMBEDDING__DIMENSION=256 QFME_NAME=from-env python -c "
from multilingual_embedding.config.loader import load_config, config_from_env
print(config_from_env())
c = load_config('experiments/quickstart.yaml')
print(c.name, c.embedding.dimension)"

{'embedding': {'dimension': 256}, 'name': 'from-env'}
from-env 256
```

Pass `use_environment=False` to `load_config` to skip this source entirely — which is
what the framework does when reloading a persisted `config.yaml`, so that an
environment variable cannot silently rewrite the record of a completed run.

### 4. `--set key.path=value` overrides

Highest precedence. Repeatable. Dots denote nesting; values are parsed as YAML
scalars, so lists and booleans work:

```python
>>> from multilingual_embedding.config.loader import parse_override
>>> parse_override("embedding.dimension=256")
{'embedding': {'dimension': 256}}
>>> parse_override("corpus.lowercase=true")
{'corpus': {'lowercase': True}}
>>> parse_override("corpus.patterns=[a.txt, b.txt]")
{'corpus': {'patterns': ['a.txt', 'b.txt']}}
```

```bash
qfme train --config experiments/quickstart.yaml \
           --set embedding.dimension=256 \
           --set embedding.epochs=20
```

An assignment with no `=`, or with an empty key path, raises `ConfigurationError`.

### Verifying precedence

With both an environment variable and a `--set` override in play, the override wins:

```console
$ QFME_EMBEDDING__DIMENSION=256 python -c "
from multilingual_embedding.config.loader import load_config, parse_override
c = load_config('experiments/quickstart.yaml',
                overrides=parse_override('embedding.dimension=512'))
print('dimension =', c.embedding.dimension)"

dimension = 512
```

### Named CLI flags

`--source`, `--format`, `--language` and `--name` are folded in as overrides, so
precedence stays in one place. They are applied **after** `--set`, making them the
effective top of the chain:

```bash
qfme stats --source data/sample/corpus.jsonl --language en
```

### The persisted record

Every training run writes its fully resolved configuration to
`<experiment_directory>/config.yaml` before the first stage runs — file plus
environment plus overrides, every default made explicit:

```yaml
name: quickstart
seed: 42
output_directory: artifacts
corpus:
  source: data/sample/corpus.jsonl
  format: jsonl
  patterns:
  - '*.txt'
  - '*.jsonl'
  language: null
  encoding: utf-8
  text_field: text
  min_sentence_characters: 2
  max_sentence_characters: 10000
  lowercase: false
tokenizer:
  model_type: unigram
  vocab_size: 300
  character_coverage: 0.9995
  normalizers:
  - type: nfkc
  - type: whitespace
  pretokenizer:
    type: whitespace
  max_sentence_length: 16384
  model_prefix: tokenizer
embedding:
  dimension: 64
  window: 3
  min_count: 2
  negative_samples: 5
  epochs: 8
  learning_rate: 0.025
  min_learning_rate: 0.0001
  subsample_threshold: 0.001
  batch_size: 32
  seed: 42
  workers: 1
evaluation:
  top_k: 5
  similarity_dataset: null
  report_directory: reports
  sample_size: 0
```

Reload it with `load_config(path, use_environment=False)` to reproduce the run. A model
file whose settings cannot be recovered is not reproducible, however good its metrics
are.

---

## Validation errors you will actually see

Every one of these is raised at construction, before any work begins:

| Setting | Error |
|---|---|
| `embedding.learning_rate: -1` | `ValidationError: learning_rate must be > 0 (name='learning_rate', value=-1)` |
| `tokenizer.character_coverage: 1.0` | `ValidationError: character_coverage must lie in (0.0, 1.0) (value=1.0)` |
| `tokenizer.model_type: fasttext` | `ConfigurationError: Unsupported tokenizer model (model_type='fasttext', supported=['unigram', 'bpe', 'word', 'char'])` |
| `corpus.format: csv` | `ConfigurationError: Unsupported corpus format (format='csv', supported=['auto', 'text', 'jsonl'])` |
| `min_sentence_characters: 50` with `max: 10` | `ConfigurationError: min_sentence_characters must not exceed max_sentence_characters (maximum=10, minimum=50)` |
| `min_learning_rate: 0.5` with `learning_rate: 0.01` | `ConfigurationError: min_learning_rate must not exceed learning_rate` |
| `name: "  "` | `ConfigurationError: Experiment name must not be empty` |
| `seed: -1` | `ValidationError: seed must be >= 0 (name='seed', value=-1)` |
| no source configured | `MultilingualEmbeddingError: No corpus source configured; pass --source or set corpus.source` |

Errors carry structured context, so the offending value is always in the message.

---

## Complete examples

### A. Small experiment against the sample corpus

Verified to train in a few seconds. Useful as a smoke test.

```yaml
name: quickstart
seed: 42
output_directory: artifacts

corpus:
  source: data/sample/corpus.jsonl
  format: jsonl
  # Drop single-character fragments left by segmentation, but keep
  # everything else: this corpus is clean and small.
  min_sentence_characters: 2

tokenizer:
  model_type: unigram
  # 300 is close to the ceiling this corpus supports (327). The framework
  # default of 32000 would fail outright on a corpus this size.
  vocab_size: 300
  character_coverage: 0.9995

embedding:
  dimension: 64          # small vocabulary needs no more
  window: 3              # sentences average 4.2 words
  min_count: 2           # 5 would discard nearly everything here
  negative_samples: 5
  epochs: 8              # more epochs compensate for very little data
  learning_rate: 0.025

evaluation:
  top_k: 5
  report_directory: reports
```

### B. A realistic multilingual corpus

Settings for a directory of JSON Lines shards in several scripts.

```yaml
name: multilingual-v1
seed: 1234
output_directory: /data/experiments

corpus:
  source: /data/corpora/multilingual/
  format: jsonl
  # Only these shards; the directory also holds README files.
  patterns: ["*.jsonl", "*.jsonl.gz"]
  encoding: utf-8
  # Below 10 characters is nearly always navigation text or debris.
  min_sentence_characters: 10
  # Anything longer means segmentation failed — unsegmented markup or a table.
  max_sentence_characters: 2000
  # Casing preserved: it is a tokenizer concern and folding here is irreversible.
  lowercase: false

tokenizer:
  model_type: unigram
  vocab_size: 32000
  # Below 1.0 so rare characters fall back to bytes. Essential for
  # large-alphabet scripts; 1.0 is rejected by validation in any case.
  character_coverage: 0.9995
  max_sentence_length: 16384

embedding:
  dimension: 300
  window: 5
  # Rare tokens get too few gradient updates to acquire a useful vector
  # and each costs a full embedding row.
  min_count: 5
  negative_samples: 10       # more negatives suit a large vocabulary
  epochs: 5
  learning_rate: 0.025
  min_learning_rate: 0.0001
  subsample_threshold: 0.001 # classic word2vec value
  # Must be set explicitly: the top-level `seed` does NOT propagate here.
  seed: 1234

evaluation:
  top_k: 10
  # Cap the tokenizer evaluation pass; scoring the full corpus is wasteful
  # once the estimate has stabilised.
  sample_size: 50000
  similarity_dataset: /data/benchmarks/wordsim.jsonl
  report_directory: /data/experiments/reports
```

### C. Fast iteration on plain text

For sweeping hyperparameters, where each run should be cheap and disposable.

```yaml
name: sweep-baseline
seed: 7
# Scratch location keeps the repository clean.
output_directory: /tmp/qfme-artifacts

corpus:
  source: /data/corpora/wiki-en/
  # Chooses TextFileReader by extension. Use `lines` explicitly via a
  # reader if the source is one sentence per line — `format` accepts only
  # auto/text/jsonl.
  format: text
  patterns: ["*.txt"]
  language: en
  min_sentence_characters: 20

tokenizer:
  model_type: bpe
  vocab_size: 8000
  character_coverage: 0.9999   # monolingual Latin: a small alphabet

embedding:
  dimension: 100
  window: 5
  min_count: 3
  negative_samples: 5
  epochs: 2                    # deliberately few; this is a sweep
  learning_rate: 0.05          # higher rate to converge faster
  seed: 7

evaluation:
  top_k: 10
  sample_size: 10000
  report_directory: /tmp/qfme-reports
```

Then vary one axis at a time without editing the file:

```bash
for d in 100 200 300; do
  qfme train --config sweep.yaml --set embedding.dimension=$d --set name=sweep-d$d
done
```
