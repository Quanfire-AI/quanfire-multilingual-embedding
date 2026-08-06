# Getting started

Every command and output on this page was executed against the sample corpus that
ships with the repository. Numbers are real, not illustrative.

---

## Install

### What you need first

| | | |
|---|---|---|
| Python 3.12 | `python3 --version` | Pinned to `>=3.12,<3.13` |
| [uv](https://docs.astral.sh/uv/) | `uv --version` | Installs everything else |

If `uv` is missing:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

You do **not** need to install Python yourself, create a virtual environment, or run
`pip`. `uv` does all three.

### Install the project

```bash
cd quanfire-multilingual-embedding
uv sync --extra neural --extra wikipedia
```

That creates `.venv/`, installs the dependencies, and registers the `qfme` command.
Expect roughly a minute the first time, mostly downloading PyTorch.

**Pass the extras.** They are optional, and plain `uv sync` does not merely skip them —
it *removes* them if they are already there:

```
$ uv sync
 - mwparserfromhell==0.7.2
 - torch==2.2.2
```

Those minus signs are uninstalls. The contextual encoder and the Wikipedia extractor
would stop working, with nothing to explain why.

| Extra | Adds | Without it |
|---|---|---|
| `neural` | PyTorch | No transformer encoder; its tests skip rather than fail |
| `wikipedia` | mwparserfromhell | `qfme extract` refuses to run |

Everything else — corpus handling, tokenizer, vocabulary, word2vec, search, evaluation —
works on the base install alone.

### Check it worked

```bash
uv run qfme --version
```

```
qfme 0.5.0
```

### Running `qfme`

`qfme` is not a system-wide command. It lives inside the project's virtual environment at
`.venv/bin/qfme`, so a fresh terminal will not find it:

```
$ qfme --version
zsh: command not found: qfme
```

**That is expected, not a broken install.** Three ways to run it, all equivalent:

```bash
# 1. Explicit path — always works, nothing to remember
.venv/bin/qfme --version

# 2. uv run — uv locates the environment itself
uv run qfme --version

# 3. Activate once, then use it plainly for the whole session
source .venv/bin/activate
qfme --version
```

Use the third for a working session; your prompt gains a `(.venv)` prefix, and
`deactivate` leaves. The rest of this page assumes it.

### If something is wrong

| Symptom | Cause and fix |
|---|---|
| `command not found: qfme` | The environment is not active. Use one of the three forms above. |
| `qfme extract` says it needs the `wikipedia` extra | Run `uv sync --extra neural --extra wikipedia`. Plain `uv sync` removed it. |
| Neural tests are skipped | Same cause — the `neural` extra is not installed. |
| `No solution found` during sync | Python is not 3.12. Check `python3 --version`; `uv python install 3.12` fixes it. |

---

## The sample corpus

`data/sample/corpus.jsonl` holds 150 documents in six languages — English, Hindi,
Tamil, Japanese, Arabic and French — as JSON Lines, one document per record:

```json
{"id": "sample-001", "language": "en", "text": "A student reviews the results. A student explains the results. ...", "source": "synthetic-sample"}
```

It is synthetic and templated. That makes it small enough to train on in seconds and
deliberately easy for a model to fit, which is worth remembering when you read the
similarity scores further down.

---

## 1. Inspect the corpus

Always look before you train. `stats` reads the corpus, segments it, applies the
configured filters and reports what the training input would actually be — never
touching a model.

```bash
qfme stats --source data/sample/corpus.jsonl
```

```json
{
  "document_count": 150,
  "paragraph_count": 150,
  "sentence_count": 750,
  "character_count": 22675,
  "word_count": 3150,
  "unique_words": 235,
  "type_token_ratio": 0.0746,
  "truncated_vocabulary": false,
  "languages": {
    "ar": 25, "en": 25, "fr": 25, "hi": 25, "ja": 25, "ta": 25
  },
  "scripts": {
    "Arab": 25, "Deva": 25, "Hani": 15, "Hira": 10, "Latn": 50, "Taml": 25
  },
  "sentence_characters": {
    "count": 750, "total": 22175, "minimum": 11, "maximum": 52,
    "mean": 29.567, "median": 31.0, "p95": 41.0, "p99": 50.0
  },
  "sentence_words": {
    "count": 750, "total": 3150, "minimum": 1, "maximum": 7,
    "mean": 4.2, "median": 5.0, "p95": 6.0, "p99": 7.0
  },
  "top_words": [
    ["the", 125], ["है", 125], ["le", 100], ["a", 25], ["student", 25]
  ]
}
```

*(The `languages`, `scripts` and length blocks are printed on one key per line; they
are compacted here. `top_words` holds the 50 most frequent words — only the first five
are shown.)*

Two things to read off this output:

- `type_token_ratio` of 0.075 is very low, which is what templated text looks like.
  On real prose expect something far higher; a ratio close to 1.0 instead means the
  corpus is too small to train on.
- Japanese splits across `Hani` (15) and `Hira` (10) because script detection reports
  the *dominant* script per document, and Japanese mixes kanji with kana. The
  `languages` count of `ja: 25` is the declared label and is the reliable figure.

Add `--output stats.json` to write the payload to a file instead of printing it.

---

## 2. Write a training config

Save this as `experiments/quickstart.yaml`:

```yaml
name: quickstart
seed: 42
output_directory: artifacts

corpus:
  source: data/sample/corpus.jsonl
  format: jsonl
  min_sentence_characters: 2

tokenizer:
  model_type: unigram
  vocab_size: 300
  character_coverage: 0.9995

embedding:
  dimension: 64
  window: 3
  min_count: 2
  negative_samples: 5
  epochs: 8
  learning_rate: 0.025

evaluation:
  top_k: 5
  report_directory: reports
```

!!! warning "`vocab_size` must fit the corpus"
    SentencePiece hard-fails when `vocab_size` exceeds the number of distinct pieces
    the training text can support. `300` is verified to work on this corpus; the
    ceiling is 327. Ask for more and the run stops:

    ```bash
    qfme train --config experiments/quickstart.yaml --set tokenizer.vocab_size=5000
    ```

    ```
    error: vocab_size exceeds what the training corpus can support; reduce vocab_size
    or supply more text (model_type='unigram', reason='INTERNAL: ... Vocabulary size
    too high (5000). Please set it to a value <= 327.', requested_vocab_size=5000,
    sentences=750)
    ```

    The framework translates SentencePiece's opaque `RuntimeError` into this message,
    and distinguishes it from the opposite failure — a `vocab_size` too *small* to
    cover the corpus's character inventory, which multilingual corpora hit easily and
    which needs `vocab_size` raised rather than lowered.

    The framework default is `32000`, sized for a real corpus. It will fail on
    anything sample-sized, so this is not a value to leave at its default when
    experimenting.

---

## 3. Train

```bash
qfme train --config experiments/quickstart.yaml
```

Progress logs go to stderr; the summary is printed to stdout as JSON:

```json
{
  "name": "quickstart",
  "documents": 150,
  "sentences": 750,
  "vocabulary_size": 226,
  "dimension": 64,
  "characters_per_token": 2.922,
  "unknown_rate": 0.0,
  "experiment_directory": "artifacts/quickstart"
}
```

`vocabulary_size` is 226, not 300: the embedding vocabulary is built from the
tokenizer's pieces filtered by `min_count: 2`, so pieces the corpus produces only
once do not get a row. `unknown_rate` of 0.0 is expected from a subword model with
byte fallback — an appreciable value there would mean the vocabulary does not fit the
corpus at all.

The run writes:

```
artifacts/quickstart/
├── config.yaml              the fully resolved configuration
├── tokenizer/
│   ├── tokenizer.model
│   └── tokenizer.vocab
└── embedding/
    ├── vectors.npy
    ├── vocabulary.json
    ├── metadata.json
    └── word2vec.json

reports/quickstart/
├── report.json
└── report.md
```

`config.yaml` is written *before* training starts, so an interrupted run still leaves
a record of what was attempted. It is the complete resolved config — file, plus
environment, plus overrides — which is what makes the model traceable back to its
settings.

Useful flags: `--no-evaluate` skips the scoring passes and produces only artefacts;
`--name` overrides the experiment name; `--set key.path=value` overrides any config
value (see [Configuration](configuration.md)).

### What the report tells you

`reports/quickstart/report.md` contains a per-language tokenizer table. This is the
number that matters most for a multilingual model:

| Language | Chars/token | Fertility |
|---|---:|---:|
| ar | 2.864 | 2.200 |
| en | 3.978 | 1.667 |
| fr | 3.642 | 1.963 |
| hi | 2.750 | 2.000 |
| ja | 1.066 | 13.320 |
| ta | 4.070 | 2.048 |

Japanese compresses at 1.07 characters per token against Tamil's 4.07 — a 3.8× spread
recorded in the report as `language_fairness`. Japanese content costs nearly four
times as many tokens as Tamil content of the same length, so it gets a fraction of
the effective context for the same text. That is the figure to act on before scaling
anything up.

The embedding section reports structural metrics:

```json
{
  "vocabulary_size": 226,
  "dimension": 64,
  "mean_pairwise_similarity": 0.57196,
  "isotropy": 0.00698,
  "effective_dimensions": 9,
  "zero_vector_count": 0
}
```

`isotropy` of 0.007 and 9 effective dimensions out of 64 say the vectors have
collapsed into a narrow cone — 55 of the 64 dimensions are doing nothing. That is the
expected outcome on 750 templated sentences and it explains the near-identical
similarity scores in the next section. On a real corpus, a result like this would be
a signal to look at the data, not to celebrate the retrieval quality.

---

## 4. Search

`search` loads the trained artefacts, encodes and indexes a corpus, and answers one
query.

```bash
qfme search --experiment artifacts/quickstart \
            --source data/sample/corpus.jsonl \
            --query "the engineer studies machine learning" --top-k 5
```

```
Query: the engineer studies machine learning
Indexed 750 sentences

 1. [0.9983] The researcher writes about machine learning.
 2. [0.9982] The engineer writes about machine learning.
 3. [0.9981] The teacher writes about machine learning.
 4. [0.9979] A student writes about machine learning.
 5. [0.9977] My friend writes about machine learning.
```

The query is not a sentence in the corpus, and every result is about machine
learning, so the model learned something. But the scores sit between 0.9977 and
0.9983 — a spread of 0.0006. That is the isotropy collapse above showing through:
cosine similarity is still ranking correctly, but it has almost no dynamic range
left. Treat the ordering as meaningful and the absolute values as not.

Non-English queries work the same way — one shared vector space, one shared
tokenizer:

```bash
qfme search --experiment artifacts/quickstart \
            --source data/sample/corpus.jsonl \
            --query "शिक्षक पढ़ाता है मशीन लर्निंग" --top-k 5
```

```
Query: शिक्षक पढ़ाता है मशीन लर्निंग
Indexed 750 sentences

 1. [0.9989] मेरा मित्र समझाता है मशीन लर्निंग।
 2. [0.9989] मेरा मित्र देखता है मशीन लर्निंग।
 3. [0.9989] मेरा मित्र पढ़ता है मशीन लर्निंग।
 4. [0.9989] एक छात्र समझाता है मशीन लर्निंग।
 5. [0.9988] एक छात्र देखता है मशीन लर्निंग।
```

```bash
qfme search --experiment artifacts/quickstart \
            --source data/sample/corpus.jsonl \
            --query "技術者は機械学習を研究します" --top-k 5
```

```
Query: 技術者は機械学習を研究します
Indexed 750 sentences

 1. [0.9997] 技術者は研究します機械学習を。
 2. [0.9994] 研究者は研究します機械学習を。
 3. [0.9993] 技術者は書きます機械学習を。
 4. [0.9992] 技術者は説明します機械学習を。
 5. [0.9991] 技術者は確認します機械学習を。
```

The Japanese query contains no whitespace at all. It works because the tokenizer is a
subword model over script-aware pre-tokenization, not a whitespace splitter.

!!! note "Results stay within the query's language"
    Each query returns sentences in its own language. All six languages share one
    vector space, but this corpus contains no parallel or comparable content across
    languages, so nothing ever taught the model that `machine learning` and
    `मशीन लर्निंग` denote the same thing. A shared space is a prerequisite for
    cross-lingual retrieval, not evidence of it.

Note also that `search` indexes the corpus from scratch on every invocation. It is a
demonstration path, not a served index.

---

## 5. Evaluate an existing experiment

Scoring a trained experiment against any corpus, without retraining:

```bash
qfme evaluate --experiment artifacts/quickstart \
              --source data/sample/corpus.jsonl \
              --output report.json
```

Omit `--output` to print a Markdown report to the terminal instead.

---

## 6. The Python API

The CLI is a thin wrapper over two pipeline classes. This script does everything
above, end to end:

```python
from multilingual_embedding.config.base import (
    CorpusConfig,
    EmbeddingConfig,
    EvaluationConfig,
    ExperimentConfig,
    TokenizerConfig,
)
from multilingual_embedding.corpus.loader import stream_sentences
from multilingual_embedding.pipelines.search import SemanticSearchPipeline
from multilingual_embedding.pipelines.training import TrainingPipeline

config = ExperimentConfig(
    name="api-demo",
    output_directory="artifacts",
    corpus=CorpusConfig(source="data/sample/corpus.jsonl", format="jsonl"),
    tokenizer=TokenizerConfig(vocab_size=300),
    embedding=EmbeddingConfig(dimension=64, window=3, min_count=2, epochs=8),
    evaluation=EvaluationConfig(report_directory="reports", top_k=5),
)

result = TrainingPipeline(config).run()
print(result.summary())

pipeline = SemanticSearchPipeline.from_directory(result.experiment_directory)
indexed = pipeline.index(stream_sentences(config.corpus))
print("indexed:", indexed)

for hit in pipeline.search("le professeur enseigne le langage naturel", top_k=3):
    print(f"{hit.rank}. [{hit.score:.4f}] {hit.text}")
```

Output:

```
{'name': 'api-demo', 'documents': 150, 'sentences': 750, 'vocabulary_size': 226,
 'dimension': 64, 'characters_per_token': 2.922, 'unknown_rate': 0.0,
 'experiment_directory': 'artifacts/api-demo'}
indexed: 750
1. [0.9964] Le chercheur examine le langage naturel.
2. [0.9963] Le professeur examine le langage naturel.
3. [0.9961] Le chercheur enseigne le langage naturel.
```

The most convenient sanity check on a trained model is to look at word-vector
neighbourhoods directly, which `similar_tokens` does without going through sentence
encoding:

```python
pipeline.similar_tokens("▁student", top_k=5)
```

```python
[('▁teacher', 0.9954500198364258),
 ('▁researcher', 0.9953668117523193),
 ('▁engineer', 0.9950007200241089),
 ('▁teaches', 0.9934359192848206),
 ('▁friend', 0.9923708438873291)]
```

Four of the five neighbours are person-nouns, which is the right answer. Note the
`▁` prefix (U+2581): SentencePiece marks word-initial pieces with it, so the
vocabulary key is `▁student`, not `student`. Querying `similar_tokens("student")`
returns `[]` because that piece is not in the vocabulary — the method returns an empty
list for unknown tokens rather than raising.

A few API notes:

- `TrainingPipeline(config)` raises `ConfigurationError` immediately if
  `corpus.source` is unset, rather than failing later.
- `TrainingPipeline.run(evaluate=False)` skips scoring, matching `--no-evaluate`.
- `SemanticSearchPipeline.from_directory` raises `ResourceNotFoundError` naming the
  missing component if `tokenizer/` or `embedding/` is absent.
- `pipeline.search(...)` before any `index(...)` call returns `[]` rather than
  raising — an unanswerable query is a normal condition for a search service, not an
  error.
- `load_config("path.yaml")` builds an `ExperimentConfig` from a file with the full
  precedence chain applied; see [Configuration](configuration.md).

A runnable version of this is in `examples/train_and_search.py`:

```bash
uv run python examples/train_and_search.py
```

---

## 7. Run the tests

```bash
pytest
```

```
710 passed in 14.98s
```

The integration tests under `tests/integration/` train real models and are marked
`slow`. Skip them for a fast feedback loop:

```bash
pytest -m "not slow"
```

```
685 passed, 25 deselected in 12.21s
```

With coverage:

```bash
pytest --cov
```

Other development commands:

```bash
ruff check src tests        # lint
ruff format src tests       # format
mypy                        # strict type checking
mkdocs serve                # these docs at http://127.0.0.1:8000
```

---

## Cleaning up

Everything the runs above produced lives under `artifacts/` and `reports/`, both of
which are gitignored. To remove the experiments from this page:

```bash
rm -rf artifacts/quickstart artifacts/api-demo reports/quickstart reports/api-demo
```

To keep the repository clean from the start, point `output_directory` and
`evaluation.report_directory` at a scratch location instead:

```bash
qfme train --config experiments/quickstart.yaml \
           --set output_directory=/tmp/qfme-artifacts \
           --set evaluation.report_directory=/tmp/qfme-reports
```
