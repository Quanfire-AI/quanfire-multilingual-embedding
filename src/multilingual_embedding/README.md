# multilingual_embedding

> Package index. For installation, CLI usage and project background see the [root README](../../README.md).

A layered pipeline from raw multilingual text to searchable embeddings:

```
corpus -> tokenizer -> vocabulary -> embeddings -> evaluation
```

## Layers

Ordered lowest to highest. A package may import from packages strictly below it and from
nothing else inside the framework.

```
common      shared constants, enums, spans, type aliases, version
   |
core        exceptions, structured logging, registries, factories
   |
utils       validation, hashing, filesystem, I/O, serialisation
   |
config      typed dataclass configuration and YAML loading
   |
corpus      text representation, segmentation, reading, statistics
   |
vocabulary  the token to id mapping shared by tokenizer and model
   |
tokenizer   normalisation, pre-tokenisation, subword model, encoding
   |
embedding   the text-to-vector contract, word and sentence vectors,
            the contextual encoder and its training, similarity search
   |
evaluation  metrics, tokenizer and embedding scoring, reports
   |
pipelines   end-to-end training and semantic search workflows
```

## Subpackages

| Package | Purpose |
|---|---|
| [common](common/README.md) | Constants, enums, `Span`, type aliases and `__version__`. Depends on nothing else in the framework. |
| [core](core/README.md) | The exception hierarchy, structured logging, the generic `Registry` and factory helpers. Standard library plus `common` only. |
| [utils](utils/README.md) | Cross-cutting helpers: `validation`, `hashing`, `filesystem`, `io`, `serialization`. Never imports a domain package. |
| [config](config/README.md) | `ExperimentConfig` and its `CorpusConfig`, `TokenizerConfig`, `EmbeddingConfig`, `EvaluationConfig` sections, each self-validating in `__post_init__`, plus YAML load and save. |
| [corpus](corpus/README.md) | The document tree, sentence segmentation, script and language detection, readers and writers, and streaming statistics. |
| [vocabulary](vocabulary/README.md) | `Vocabulary`, `VocabularyBuilder` and the special-token block that every layer above indexes against. |
| [tokenizer](tokenizer/README.md) | Normalizer pipeline, pre-tokenizers, the SentencePiece tokenizer and its trainer, and `Encoding`. |
| [embedding](embedding/README.md) | The `TextEncoder` contract; numpy skip-gram word2vec, the embedding matrix and sentence encoders as the static baseline; a transformer encoder with contrastive training, LoRA and gradient caching under `embedding/neural/`; exact cosine search over either. |
| [evaluation](evaluation/README.md) | Metric primitives, tokenizer and embedding evaluators, and the evaluation report. |
| [pipelines](pipelines/README.md) | `TrainingPipeline` and `SemanticSearchPipeline`. |

## One optional dependency

Every layer above runs on the core dependencies — numpy, pandas, pyyaml, sentencepiece,
tqdm. The single exception is `embedding/neural/`, which needs torch and is installed with
`uv sync --extra neural`.

The boundary is drawn deliberately rather than by accident. `embedding/__init__.py` does not
import `neural`, so nothing pulls torch in transitively, and the corpus, tokenizer,
vocabulary and evaluation layers stay a small install for callers that only need text
preparation. Importing `multilingual_embedding.embedding.neural` without torch raises an
`ImportError` naming the extra, rather than a bare `ModuleNotFoundError`. The layering rule
below is unaffected: `neural` is part of the `embedding` layer and obeys the same ordering.

## The corpus tree

Every text unit is a node, and each node stores its span relative to its parent, so any
unit can be mapped back to its exact position in the source text.

```
Corpus
 └── Document
      └── Paragraph
           └── Sentence
                └── Token
```

## Package root files

`__init__.py` re-exports only the most commonly used names — `ExperimentConfig`,
`TrainingPipeline`, `SemanticSearchPipeline`, `Corpus`, `Document`, `Paragraph`,
`Sentence`, `Token`, `Vocabulary`, the core exceptions and the logging helpers.
Everything else is reached through its own package, which keeps this module's import
cost proportional to what a caller actually uses.

`cli.py` is the entry point for the `qfme` console script, declared in `pyproject.toml`
as `qfme = "multilingual_embedding.cli:main"`. Its subcommands are `stats`, `validate`,
`train`, `search` and `evaluate`.

`py.typed` marks the package as typed under PEP 561. Without it, mypy in a downstream
project silently ignores every annotation this framework provides;
`tests/test_architecture.py::test_package_ships_py_typed_marker` asserts the file is
present.

## The layering rule

`cli.py` and `__init__.py` sit directly under the package root and are composition roots,
so they may import anything. Every other module belongs to exactly one layer and may
import only from layers strictly below it — no upward imports, which would introduce a
cycle, and no sideways ones, which would blur the boundary between two peers.

This is not a convention maintained by review. `tests/test_architecture.py` parses every
source file with `ast`, builds the import graph between layers and asserts the ordering
directly, along with `common` and `core` having no internal dependencies at all and the
whole graph being acyclic.
