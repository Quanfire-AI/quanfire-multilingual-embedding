# Quanfire Multilingual Embedding

A framework for turning multilingual text into searchable embeddings, built so that
every stage between raw text and a ranked result list can be inspected, configured
and reproduced.

```
corpus  ->  tokenizer  ->  vocabulary  ->  embeddings  ->  evaluation  ->  search
```

---

## The problem it solves

Getting from a directory of text files to "which of these sentences is closest in
meaning to this query" requires five stages, and each one has a way of failing
quietly.

A corpus reader has to decide what a document, a paragraph and a sentence are. A
period-and-space rule handles English and destroys Hindi, which ends sentences with
the danda (`।`), and Japanese, which uses `。` and inserts no space afterwards. A
tokenizer has to produce units that are stable across scripts, which whitespace
splitting cannot do for languages written without spaces. A vocabulary has to assign
ids that stay fixed for the life of a trained model, because those ids are indices
into an embedding matrix and a shift of one makes every vector wrong without raising
an exception. Then the embeddings have to be trained, scored honestly, and served.

This framework implements that whole path, with the multilingual cases treated as the
normal case rather than as exceptions handled later.

---

## Design principles actually followed

**Layered architecture with an acyclic import graph.** Each layer may only import
from layers below it. `common` and `core` import nothing internal at all. This is not
a diagram drawn after the fact — it is why `evaluation` can score a bare `tokenize`
callable without knowing what a `Tokenizer` is, and why the corpus tree has no idea
embeddings exist. See [Architecture](architecture.md).

**Streaming everywhere.** Tokenizer training, vocabulary construction and every
embedding epoch pull from a re-iterable `SentenceStream` that restarts the reader on
each pass. Corpus size is bounded by disk, not memory. An in-memory `Corpus` exists
and is the right tool for random access and splitting, but nothing in the training
path requires one.

**Typed configuration validated at load time.** Configuration is a tree of
dataclasses, each validating itself in `__post_init__`. A `learning_rate` of `-1` or
a `character_coverage` of `2.0` fails next to the file that caused it, not ninety
minutes into a training run. See [Configuration](configuration.md).

**Reproducible runs.** Runs are seeded. Vocabulary ordering is deterministic —
descending frequency, ties broken on the token string — so the same corpus produces a
byte-identical vocabulary across runs. The fully resolved configuration is written to
`config.yaml` next to the artefacts it produced, before training starts, so even an
interrupted run leaves a record of what was attempted.

**Registries for pluggable components.** Normalizers, pre-tokenizers, tokenizers,
readers and sentence encoders each own a `Registry`, and a config file selects an
implementation by name without importing it. An unknown key produces an error listing
the available ones, which turns a typo into an obvious fix.

**Atomic writes.** Model artefacts are staged and moved into place, so an interrupted
run never leaves a half-written model where the next run will load it.

---

## Scope, honestly

Implemented and tested end to end:

| Area | What is there |
|---|---|
| Corpus | `Corpus → Document → Paragraph → Sentence → Token` tree, script-aware segmentation, JSON Lines / plain text / line readers, streaming statistics, filtering, exact deduplication, graded quality auditing |
| Corpus building | MediaWiki dump extraction with markup stripping and section preservation; contrastive pair mining in three kinds with lexical leakage measured per pair. Run on real Hindi and Tamil dumps: 282,339 articles → 1,536,059 pairs, under 201 MB resident |
| Tokenization | Unicode normalizers, four pre-tokenizers including a script-aware one, SentencePiece training and inference, a dependency-free `WordTokenizer` |
| Vocabulary | Deterministic token/id mapping, fixed special token ids, streaming builder, JSON persistence |
| Embeddings | Skip-gram word2vec with negative sampling in pure numpy, mean-pooling and SIF sentence encoders, exact cosine similarity index |
| Evaluation | Tokenizer compression/fertility/unknown-rate metrics with a per-language fairness breakdown, structural embedding metrics (isotropy, effective dimensions), optional similarity and analogy datasets |
| Interfaces | `qfme` CLI (`extract`, `stats`, `validate`, `mine-pairs`, `mine-negatives`, `train`, `adapt`, `search`, `evaluate`, `serve`) and a Python API |

**Behind optional extras.** Three capabilities are installed separately, so the base
install stays small for callers that only need text preparation:

| Extra | Adds | Gives you |
|---|---|---|
| `neural` | torch | The contextual encoder: transformer, contrastive InfoNCE training, LoRA, gradient caching |
| `pretrained` | transformers | Loading a published checkpoint and LoRA-adapting it |
| `wikipedia` | mwparserfromhell | `qfme extract` |

```bash
uv sync --extra neural --extra pretrained --extra wikipedia
```

The base install is `numpy`, `pandas`, `pyyaml`, `sentencepiece`, `tqdm`. Everything
through vocabulary, static word2vec and pair mining runs on that alone. Skipping an extra
costs exactly the capability it names and nothing else; the affected tests skip rather
than fail.

**Adapting a published checkpoint is implemented and proven.** `intfloat/multilingual-e5-small`
adapted with LoRA on mined Wikipedia pairs gained **+28.6% recall@1 in Hindi and +40.9% in
Tamil** from a 3.4 MB adapter training 0.50% of parameters. It runs as `qfme adapt` from a
config file and a compute profile, and is served with `SemanticSearchPipeline.from_adapter`.
`scripts/adapt_pretrained.py` is a thin front end over the same pipeline, kept so the
command lines that produced those figures still work.

Note what that does *not* say: the weights are not loaded into this project's transformer.
That encoder is pre-norm while most published ones are post-norm, and the shapes match, so
a cross-load would succeed and be numerically wrong with nothing raising. External
checkpoints are therefore run through their own library instead, behind the same
`TextEncoder` contract. See [ROADMAP.md](https://github.com/quanfire-ai/quanfire-multilingual-embedding/blob/main/ROADMAP.md).

**Still absent.** These are missing rather than stubbed:

- **No subword-averaging model.** There is no character n-gram embedding model; word2vec
  operates on SentencePiece pieces, which is a different thing.
- **No decoder and no generation.** This produces embeddings, not text.
- **No hard negatives.** Contrastive training uses in-batch negatives only. Mining hard
  negatives against a base encoder is the most likely next source of gain.
- **No cross-lingual pair mining.** Nothing produces translation pairs, so cross-lingual
  retrieval depends entirely on what the corpus happened to contain.
- **No from-scratch pretraining at scale.** Every measurement here is at 5.3M or 118M
  parameters; the target is a 568M encoder, and nothing has been run at that size.

Further limits worth knowing before you rely on the framework:

- **Search is exact, not approximate.** Brute-force cosine over an L2-normalised
  matrix is the right choice up to roughly 10⁵–10⁶ vectors, where a query still lands
  in single-digit milliseconds and the index needs no build step, no tuning and no
  recall measurement. Beyond that you need an approximate index (HNSW, IVF-PQ), which
  this framework does not provide.
- **Sentence segmentation is rule-based.** Fast, predictable and dependency-free, but
  it will not resolve genuinely ambiguous cases. An unknown abbreviation followed by a
  capitalised word will split. Every reader accepts pre-segmented input.
- **Deduplication is exact-match only.** Near-duplicate detection needs MinHash or
  SimHash; exact matching was chosen because it carries no false-positive risk.
- **Cross-lingual alignment is not guaranteed.** Tokenizer and embeddings are trained
  jointly over the whole corpus so all languages share one vector space, but a query
  in one language retrieves another language's sentences only to the extent the
  training corpus contained parallel or comparable content. A shared space does not
  imply alignment, and the honest way to check is a cross-lingual retrieval metric.
- **Language inference is script-based**, not statistical. It returns `None` for
  scripts shared across languages (Latin, Arabic, Cyrillic, Han) rather than
  guessing.
- **Some configuration fields are declared but not yet wired.** They are listed
  explicitly in [Configuration](configuration.md) so you do not set one and assume it
  took effect.

---

## Where to go next

- **[Getting started](getting-started.md)** — install, then a worked run against the
  six-language sample corpus: statistics, training, search in English and in Hindi,
  Japanese and French, and the equivalent Python API.
- **[Architecture](architecture.md)** — the layer diagram, the dependency rule, and a
  walkthrough of each package including the non-obvious decisions: relative spans,
  why container nodes store their own text, the multilingual terminator inventory,
  fixed special token ids, and the word2vec details that separate an implementation
  that learns from one that only appears to.
- **[Configuration](configuration.md)** — every field of every config section, with
  types, defaults and validation rules, plus the precedence chain and complete
  annotated examples.
