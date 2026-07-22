# pipelines

> Composes the layers below into two end-to-end workflows: training a model from raw text, and serving semantic search over a trained one.

## Purpose

Every stage of this framework is usable on its own, which means something has to decide
the order they run in, thread the artefacts between them and record what happened. That
is all this layer does. `TrainingPipeline` runs corpus to evaluated model;
`SemanticSearchPipeline` loads what it wrote and answers queries. Keeping orchestration
in its own top layer is what stops ordering logic from leaking into the tokenizer or the
embedding trainer, where it would make either of them unusable outside a full run.

## Modules

| Module | Responsibility |
|---|---|
| `training.py` | `TrainingPipeline`, the `TrainingResult` record of everything one run produced, and the `train` convenience wrapper. |
| `search.py` | `SemanticSearchPipeline` over any `TextEncoder`, with its `from_static` / `from_directory` / `from_config` / `from_adapter` constructors, and the `SearchHit` result record. |

## Key design decisions

**Orchestration only.** `TrainingPipeline` implements no corpus reading, no subword
algorithm, no gradient and no metric. Each stage method is a handful of lines that
constructs the component from the relevant config section and calls it:
`StatisticsAccumulator`, `SentencePieceTrainerAdapter`, `Word2Vec`, `TokenizerEvaluator`,
`EmbeddingEvaluator`, `EvaluationReport`. Every one of those is constructible and runnable
without this package.

**The corpus is never materialised.** Tokenizer training, vocabulary construction and
every embedding epoch pull from a re-iterable `SentenceStream` obtained once via
`stream_sentences(config.corpus)`, and statistics are gathered by streaming documents
through a `StatisticsAccumulator`. Corpus size is therefore bounded by disk rather than
by memory. The corollary is that the stream is traversed several times — once for
statistics, once for tokenizer training, once per embedding epoch plus one to build the
vocabulary, and again for evaluation — which is why it must be re-iterable and why
`EvaluationConfig.sample_size` exists to cap the evaluation passes.

**The resolved configuration is persisted first.** `run` calls `save_config(config,
experiment_directory / "config.yaml")` before any stage executes. A run interrupted
during tokenizer training still leaves a record of what was attempted, which is the
difference between a reproducible failure and a directory of unexplained partial
artefacts. Statistics are collected after filtering and deduplication, so the recorded
figures describe the corpus actually trained on rather than the raw source.

**Embeddings are trained over the tokenizer's subword output, not over whitespace
words.** `_train_embeddings` passes `tokenize=tokenizer.tokenize` into `Word2Vec.train`,
so the embedding vocabulary is built from subword pieces. This is the decision that makes
the framework work at all for scripts without whitespace word boundaries, where splitting
on spaces would yield roughly one token per sentence and a vocabulary of unique sentences.
It also means the embedding vocabulary and the tokenizer are permanently coupled: encoding
with a different tokenizer produces pieces that index into the wrong embedding rows, which
`SemanticSearchPipeline`'s docstring calls out explicitly.

**Failing loudly on an empty corpus.** `_collect_statistics` raises `ConfigurationError`
when filtering leaves zero sentences, naming the source and the `min_sentence_characters`
that caused it. Training on nothing would otherwise proceed to a tokenizer failure much
further down, with a message about vocabulary size rather than about the filter.

**`evaluate=False` is a supported mode.** Skipping evaluation avoids the extra corpus
passes scoring requires, and the report records why the metrics are absent by appending
a note rather than by writing zeros.

**`SemanticSearchPipeline` depends on `TextEncoder`, not on `EmbeddingMatrix`.** This is
the decision that keeps the serving path open to the contextual encoder in
`embedding/neural/`. A matrix is a `vocabulary × dimension` table, which a transformer
does not have and cannot be given: it computes a vector for the whole input at call time.
Had the pipeline been written against the matrix, serving a contextual model would have
meant rewriting everything downstream of it — not for quality reasons but for shape ones.

The consequence is visible in the constructor: `encoder` is required, while `matrix` and
`tokenizer` are keyword-only and default to `None`. Both are `None` for a contextual model,
whose tokenizer is internal to it and which has no table. `similar_tokens` is the one method
that genuinely needs the matrix — inspecting word neighbours is a static-model question —
and it returns an empty list rather than raising when there is none. `from_static` builds
the static combination and defaults the encoder to `MeanPoolingEncoder` over the matrix.

**`from_directory` deliberately loads from disk.** It could accept in-memory objects from a
`TrainingResult`, and the constructor still does. But `from_directory` reads `tokenizer/`
and `embedding/` back off disk because that is the path a deployed service takes, and it is
the path that needs exercising. It raises `ResourceNotFoundError` naming which of the two
subdirectories is missing, rather than failing later on a partial load. It builds the static
path specifically; a contextual encoder is loaded with `NeuralTextEncoder.load` and passed
to the constructor directly.

**`from_adapter` loads an adaptation run's output into a working search path.** An
adaptation run that reports a 39% improvement and writes 3.4 MB to disk has produced an
artefact nothing can query. This constructor closes that gap: it calls `load_adapter`,
which names the frozen base checkpoint, rebuilds it, applies the low-rank update, and
returns the encoder alongside its metadata. The torch import is inside the method rather
than at module scope, because the neural stack is an optional extra and the static search
path must stay installable without it.

**The query and passage prefixes live on the pipeline, because nothing else knows which
side it is on.** An E5-family model is trained with `query:` on one side and `passage:` on
the other, and served without them it returns vectors that are the right shape, the right
norm, free of NaN, and encode the wrong thing. Nothing raises; the score is simply lower,
which is indistinguishable from the model not being very good. The encoder cannot apply
them — `encode` receives text and has no idea whether it is a query — and the caller
should not have to, since the requirement is a property of the checkpoint rather than of
the application. What does know is `index` versus `search`, which is why the two prefixes
are constructor arguments, applied automatically on their respective sides, and readable
back off the `prefixes` property. `from_adapter` reads them out of the artefact, so the
information `save_adapter` recorded specifically to prevent this failure is actually used.
The stored text is the sentence as given, so a hit returns what was indexed rather than
what the model had to see. `repr` shows the prefixes only when set, so a symmetric model
and a forgotten prefix do not look alike.

**`index` encodes the corpus as one batch.** For a transformer this is the difference
between one padded batch per forward pass and one sentence per forward pass. It also
matters for `SifEncoder`, whose common component is estimated from a batch and reused by
`encode`: indexing one sentence at a time never gave it a batch to estimate from, so the
component was never removed and SIF quietly degraded to a weighted average. Corpus in
`index`, query in `search`, is the shape that encoder documents.

**Zero vectors are skipped rather than indexed.** `index` drops any sentence that encodes
to an all-zero vector — every token out of vocabulary — because such a vector matches
every query equally poorly and pollutes results. It passes `dimension` explicitly to
`SimilarityIndex.build` so that indexing a set where every sentence was skipped yields an
empty index rather than failing on a dimension inference with nothing to infer from.
Symmetrically, `search` returns an empty list when nothing is indexed or the query encodes
to zero: an unanswerable query is a normal condition for a search service, not an error.

**Cross-lingual caveat, stated honestly.** Because the tokenizer and the embeddings are
trained jointly over one multilingual corpus, all languages share a single vector space,
and a query in one language can retrieve sentences in another. That happens only to the
extent the training corpus contained parallel or comparable content. Without that signal
the languages occupy separate regions of the same space and cross-lingual retrieval will
not work, however shared the space is. A shared vector space does not imply alignment;
the way to know is to measure retrieval across languages, not to assume it.

## The asymmetry between the two pipelines

`SemanticSearchPipeline` already serves both kinds of model. `TrainingPipeline` trains only
the static one: `_train_embeddings` constructs a `Word2Vec` and nothing in this package
touches `embedding.neural`, so `ContrastiveTrainer` is driven directly rather than through
an orchestrated run.

That asymmetry was originally justified by an input that did not exist: contrastive
training consumes `TextPair`s, labelled query-passage pairs will not exist for the domains
this is aimed at, and the framework had no way to manufacture them from document structure.
Wiring a contrastive stage into `run` before there was anything to feed it would have
produced a stage with no input.

**That justification has expired.** `corpus/pairs.py` mines pairs, `qfme mine-pairs` fronts
it, and 1,536,059 pairs have been mined from Hindi and Tamil Wikipedia. What remains is
plumbing rather than a missing capability — a config schema for the contrastive path, and a
`qfme adapt` subcommand, both tracked in `ROADMAP.md`. Until they land, the honest statement
is that an adapted model is trained by calling `scripts/adapt_pretrained.py` and served by
pointing `from_adapter` at the directory it saved; a from-scratch contextual model is
trained by driving `ContrastiveTrainer` from the Python API. Neither path goes through
`TrainingPipeline`, and neither is reachable from the CLI.

## Usage

```python
from pathlib import Path

from multilingual_embedding.config.base import (
    CorpusConfig,
    EmbeddingConfig,
    EvaluationConfig,
    ExperimentConfig,
    TokenizerConfig,
)
from multilingual_embedding.pipelines.search import SemanticSearchPipeline
from multilingual_embedding.pipelines.training import TrainingPipeline

output = Path("/tmp/qfme-demo")

config = ExperimentConfig(
    name="demo",
    output_directory=output,
    corpus=CorpusConfig(source=Path("data/sample/corpus.jsonl"), format="jsonl"),
    tokenizer=TokenizerConfig(vocab_size=220),
    embedding=EmbeddingConfig(dimension=32, window=3, min_count=1, epochs=4),
    evaluation=EvaluationConfig(report_directory=output / "reports", top_k=5),
)

result = TrainingPipeline(config).run()

for key, value in result.summary().items():
    print(f"{key}: {value}")

pipeline = SemanticSearchPipeline.from_directory(result.experiment_directory)

sentences = ["The teacher explains machine learning.", "A student reviews the results."]

print("indexed:", pipeline.index(sentences))

for hit in pipeline.search("machine learning", top_k=2):
    print(hit.rank, round(hit.score, 3), hit.text)
```

Actual output, with SentencePiece's own training log to stderr omitted:

```
name: demo
documents: 150
sentences: 750
vocabulary_size: 201
dimension: 32
characters_per_token: 1.28
unknown_rate: 0.0
experiment_directory: /tmp/qfme-demo/demo
indexed: 2
1 1.0 The teacher explains machine learning.
2 0.998 A student reviews the results.
```

The `characters_per_token` of 1.28 is a genuine signal, not a defect in the run: a
220-piece vocabulary over this corpus segments almost to characters. It is exactly what
the tokenizer metrics exist to make visible. The two search scores are close together
because the demo indexes two sentences from a small synthetic corpus of highly similar
text; do not read the absolute values as a quality claim.

`vocab_size` is set to 220 rather than something smaller because SentencePiece raises if
the requested vocabulary cannot cover the corpus's required characters; the pipeline
surfaces that as a `ConfigurationError` telling you to raise `vocab_size` or lower
`character_coverage`.

## Dependencies

The top layer. It may import from all nine layers below — `common`, `core`, `utils`,
`config`, `corpus`, `vocabulary`, `tokenizer`, `embedding` and `evaluation` — and
`training.py` in fact imports from most of them, which is the expected shape for a
composition root.

Nothing inside the framework may import `pipelines`. The only permitted importers are
the package root `__init__.py` and `cli.py`, which sit directly under
`src/multilingual_embedding/` and are exempt from the layering check as composition
roots.

## Tests

`tests/pipelines/` holds **18 tests**, all on the search pipeline and all on the one
failure the rest of the suite cannot see: a prefix that never reaches the encoder. They
assert on the strings the encoder was actually handed rather than on retrieval quality,
because retrieval quality is precisely the signal that goes quietly wrong.
`test_search_adapter.py` needs the `neural` extra and skips without it.

Both pipelines are additionally covered end to end by
`tests/integration/test_end_to_end.py`, which trains once against
`data/sample/corpus.jsonl` and shares the result across its assertions —
`TestTrainingPipeline` covers the training stages and artefacts, `TestSearchPipeline`
covers loading from a directory, indexing and querying.

`.venv/bin/python -m pytest tests/pipelines tests/integration -q` reports
**47 passed**. The rule that nothing below may import this package is enforced by
`tests/test_architecture.py` (16 tests).
