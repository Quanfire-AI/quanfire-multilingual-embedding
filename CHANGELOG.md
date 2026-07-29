# Changelog

This file exists because other repositories now pin this one by version. A consumer
deciding whether to move a pin needs to know what moved, and `git log` is the wrong
granularity for that question.

Versions follow semantic versioning. Before 1.0 the minor number carries breaking
changes, so treat `0.2 → 0.3` the way you would treat `1.0 → 2.0`.

## Public API

As of 0.2.0 these are the supported surface. Renaming or removing anything here is a
breaking change and gets a minor bump. `tests/test_public_api.py` asserts each one, so
the promise is checked rather than merely written down.

| Import path | What a consumer gets |
|---|---|
| `multilingual_embedding.corpus` | Reading, cleaning, segmenting, streaming, pair mining, provenance |
| `multilingual_embedding.vocabulary` | Vocabulary construction, special tokens, id spaces |
| `multilingual_embedding.tokenizer` | Normalizers, pre-tokenizers, SentencePiece training and encoding |
| `multilingual_embedding.evaluation` | `TokenizerEvaluator`, `TokenizerMetrics`, `language_fairness`, `evaluate_tokenizer` |
| `multilingual_embedding.core.exceptions` | `MultilingualEmbeddingError` and its subclasses |

Some names are pinned individually as well as by package, because a consumer depends on
them specifically: `tokenizer.trainer_for`, `tokenizer.SentencePieceTrainerAdapter`,
`corpus.reader_for`, `corpus.sentences_from`, `corpus.documents_from` and
`corpus.corpus_from` — the supported way to reach the things whose other entry points take
internal config types — plus the evaluation names below.

**A name is public only if every type in its signature is.** That rule is the lesson of
0.3.2 and applies to anything added here: a published function reachable only by importing
something from `config` is a promise with a hole in it, because a change this repository
would correctly call internal breaks a consumer's build.

**Guaranteed at the package boundary, not the module.** `TokenizerEvaluator` and
`language_fairness` are promised as `from multilingual_embedding.evaluation import ...`
— never from `...evaluation.tokenizer_eval`. The file may move; the import will not.
Depending on the module path forfeits the guarantee.

That distinction is the answer to "do not relocate `tokenizer_eval.py`": the request is
granted in the form that matters and refused in the form that would freeze the layout.
The instrument in question — `evaluate_by_language`, `evaluate_by_script`,
`language_fairness` — is the one that says whether Devanagari and Tamil are tokenized
worse than Latin, and it is why consumers depend on this package instead of copying from
it. It sits in `evaluation/` rather than `tokenizer/` because it evaluates; a vendoring
set of `corpus, vocabulary, tokenizer` misses it by one directory, which is an argument
for the dependency, not for the file's address.

**Torch-free on a base install.** All five import without pulling in torch. Torch is
confined to `embedding/neural/` behind the `neural` extra, so a consumer keeps control of
its own training stack. This is asserted in a clean subprocess against `sys.modules`, so
it is tested here — where every extra is installed — rather than only where torch is
absent.

**Not public.** Everything else, `embedding/` and `pipelines/` included. They may move
without a minor bump.

## Unreleased

### Added

- **`--positive-margin`, a relative hard-negative guard.** `qfme mine-negatives` gained a
  `positive_margin` setting (and `NegativeConfig.positive_margin`) that rejects any candidate
  scoring within a margin of, or above, its own positive. The existing guards are absolute —
  a same-document check and a fixed similarity ceiling — and they miss the false negatives
  that sit *near the positive* rather than near a paraphrase ceiling. The statistics gained
  `rejected_outranks_positive` to count what the margin drops. Off by default (`None`), so
  existing mining is byte-identical. Three tests pin the gate, the default-off behaviour, and
  cosine-range validation.

- **Ten-language cross-lingual adaptation, measured.** `qfme mine-aligned` was run across
  ten Indian-language Wikipedia editions (hi, ta, bn, gu, kn, ml, mr, sa, te, ur) joined
  through langlinks, and a single LoRA adapter trained on the resulting cross-lingual pairs.
  Cross-lingual retrieval rises from **0.7875 → 0.8964 recall@1** (non-Hindi X↔Y, held out)
  and hi-pivot recall@10 from 0.7495 → 0.8852. The public-bitext transfer limit is recorded:
  on FLORES-200 (never trained on) the base scores 0.985 and the adapter 0.961, a small
  regression. This reverses the earlier caution where a *within-language* adapter lowered
  cross-lingual retrieval; the fix was training on aligned pairs, not adapting and hoping to
  transfer. Consolidated scores in `reports/optionb/hn-verdict.json`.

- **A hard-negative verdict, at scale.** The 250k-pair ablation makes mined negatives a
  measured trade-off: ~−1.5 recall@1 in-domain, ~+1.5 on FLORES-200 — a regularizer, not an
  improvement. The mining report shows why the in-domain cost is intrinsic here: 49.8% of the
  kept negatives outrank their own positive. The canonical adapter carries no hard negatives;
  the capability stays in place (it is the strict generalisation of in-batch and costs nothing
  when a pair set has no negatives) and is not claimed as a win.

- **A no-customer-text policy, asserted rather than described.**
  `tests/test_data_policy.py` requires every JSON Lines file git tracks — anywhere in the
  repository — to declare a `source` on every record, and every one of those to begin
  `synthetic-`. A sample added with no `source` fails, because the file most likely to be
  real is the one whose provenance nobody wrote down. It also holds `data/` to exactly the
  sample corpora plus its README, and refuses any tracked `.gz`, `.bz2`, `.csv`,
  `.parquet` or `.xml`.

  `.gitignore` already said most of this, but an ignore rule is a convention held up by
  whoever writes the next commit, and the failure mode is permanent: text committed once
  stays in the history after the file is deleted. `examples/walkthrough/broken-extraction.jsonl`
  gained `"source": "synthetic-broken"` on all six records so it satisfies the rule; the
  deliberate encoding damage and the `qfme validate` output the walkthrough quotes are
  unchanged.

  Two gaps the test cannot close and that stay a matter of judgement: text pasted somewhere
  that is not a corpus file, and the weights — a model adapted on customer text carries it,
  and `adapter.json` records `trained_on` as a path rather than an origin.

## 0.3.2 — 2026-07-23

A hole in the public surface, found by a consumer and closed. Additive: nothing that
worked on 0.3.1 stops working.

### Added

- **`multilingual_embedding.tokenizer.trainer_for`** — builds a `SentencePieceTrainerAdapter`
  from plain settings, with no config object:

  ```python
  from multilingual_embedding.tokenizer import trainer_for

  trainer = trainer_for(vocab_size=8000, model_type="bpe")
  ```

  The class was already public. The only type its constructor accepted, `TokenizerConfig`,
  lives in `config` — which the section above puts outside the guarantee. So the published
  class was reachable only by importing a type this repo may change in a patch release,
  which means a change this repo would call internal could break a consumer's build
  without being breaking on this repo's own terms. That is a hole, not a design.

  `trainer_for` is the same shape as `corpus.reader_for`, which solved this on the corpus
  side. Its signature contains only builtins, `Mapping`s of them, and the already-public
  `SpecialTokenSet`; `model_type` is a `str` because `TokenizerModel` is a `StrEnum`, so
  both the string and the member are accepted. Every argument defaults to `None` meaning
  *keep the framework default*, rather than restating those defaults where they would
  drift silently.

  A test asserts no annotation in the signature names `TokenizerConfig` or `config`, so
  the guarantee is checked rather than reviewed.

- **`SentencePieceTrainerAdapter.vocab_size`** and **`.model_type`** — an `int` and a
  `str`, so reading back what a trainer was built with also needs no private import.
  `.config` remains and still returns the internal type; its docstring now says so.

- **`multilingual_embedding.corpus.sentences_from`, `.documents_from` and `.corpus_from`**
  — the same three access patterns as `stream_sentences`, `stream_documents` and
  `load_corpus`, from a source and plain settings:

  ```python
  from multilingual_embedding.corpus import sentences_from

  for sentence in sentences_from("data/corpus.jsonl", min_sentence_characters=20):
      ...
  ```

  Those five loader functions had the identical hole: all exported, all taking only a
  `CorpusConfig`. The consumer reported the shape and had already worked around it. The
  config forms stay — inside this repository a `CorpusConfig` already exists because a
  YAML file produced it, and threading ten keyword arguments through `pipelines` to avoid
  a type it already holds would be worse. `build_reader` needs no twin: `reader_for`
  already is one. `build_filter` returns a `SentenceFilter` whose own constructor takes
  plain integers.

  Same conventions as `trainer_for`: `None` means keep the framework default, the caller's
  `patterns` list is copied rather than aliased, and a test asserts no annotation on any of
  the three names `CorpusConfig` or `config`.

- `trainer_for`, `SentencePieceTrainerAdapter`, `corpus.reader_for` and the three loading
  twins are named individually in `tests/test_public_api.py`, alongside the evaluation
  names.

### Documentation

- A note next to `CorpusReader` on what construction does *not* raise. `iter_documents`
  is a generator: its body does not run until the caller iterates, so a `try` around
  `reader_for(path)` translates no missing file and no malformed line — the handler has
  already exited. Wrap the iteration. The one exception is an explicit `format` naming no
  registered reader, which raises `RegistryError` at construction; under `auto` even that
  is silent, since an unrecognised extension falls back to `TextFileReader`. Reported by a
  consumer who hit it.

## 0.3.1 — 2026-07-23

Hard-negative mining. A patch bump rather than a minor: nothing in the public surface
changed shape, and a pair file written by 0.3.0 round-trips through this version to a
byte-identical line.

### Added

- **`qfme mine-negatives`** — ranks a pair set's own positives against each anchor using a
  saved adapter and attaches the hardest survivors to each pair. No second corpus is
  fetched; the candidate pool is already resident and already human-written.

  The reason this is a module rather than a function is the failure it has to avoid. A
  mined "hard negative" that is really a correct answer trains the model to push the right
  passage away, with the largest gradient in the batch, and the loss curve *improves* while
  retrieval degrades. Three guards reject the obvious cases — the pair's own positive by
  identity, the anchor's own document by provenance, anything above 0.95 similarity as a
  likely paraphrase — and unencodable text is rejected by vector norm rather than by score,
  so the guard does not depend on a default threshold happening to catch it.

  **No false-negative rate is reported.** The statistics count
  `outranking_the_positive` — the population such errors are drawn from — and contain no
  field named for the rate, which a test enforces. `--audit` writes the hardest sample as
  JSONL with an `is_actually_correct: null` field for a person to fill in. That is the only
  route to the real number, and it needs an afternoon of labelling rather than a flag.

- **`MinedPair.negatives`** and **`TextPair.negatives`.** Mined negatives become extra
  candidate columns in the contrastive loss, so the similarity matrix is
  `batch × (batch + extras)` and the targets stay the diagonal. Columns are deduplicated
  across the batch, which matters here more than for positives: the pool a negative is
  mined from *is* the set of positives, so a collision is routine rather than unlikely.

  Both fields default to empty and produce zero extra columns, so a pair set without
  negatives trains exactly as it did on 0.3.0. `MinedPair.to_record()` omits the key when
  empty rather than writing `"negatives": []` on every line of a million-line file.

- `multilingual_embedding.embedding` exports `mine_negatives`, `NegativeConfig`,
  `NegativeStatistics` and `AuditRecord`. It takes the `TextEncoder` protocol and never
  imports torch, so the algorithm — and its whole unit suite — runs on a base install.
  Only the CLI command needs the `neural` and `pretrained` extras, because loading an
  adapter does.

### Not done

Whether training on mined negatives improves `models/indic-v1` is unmeasured. That needs a
GPU run and has not happened, so this ships as a capability rather than a result.

## 0.3.0 — 2026-07-23

Minor bump for one behaviour change in a saved artefact's reload path. Nothing in the
public surface above moved, and a consumer using only that surface can take this as a
patch.

### Added

- **An HTTP embeddings endpoint.** `qfme serve --adapter models/indic-v1` puts a saved
  adapter behind the de facto industry-standard schema — `/health`, `/v1/models`,
  `/v1/embeddings` — so a client migrates by changing a base URL. New `serving` layer,
  new `serve` extra (FastAPI, uvicorn).

  One field is an addition rather than a copy. The standard schema has nowhere to say
  whether a string is a query or a passage, and an E5-family model served on the wrong
  side returns a vector of the right shape and norm, free of NaN, that encodes the wrong
  thing. So an asymmetric model with no configured default answers `400` naming both
  valid values rather than guessing; `--default-input-type` covers the deployment that
  genuinely is single-sided. Every response echoes `prefix_applied`.

  The endpoint has no authentication or rate limiting and binds `127.0.0.1` by default.

- `PretrainedTextEncoder.load` now accepts `normalize`.

### Fixed

- **`load_adapter` was dropping normalization.** `save_adapter` has recorded `normalize`
  in the manifest since `format_version: 1` and the loader never read it, so an encoder
  saved with normalization off reloaded with it on. Cosine scores are unaffected; a
  dot-product index built over those vectors ranks differently, and nothing raised. Any
  adapter saved with `normalize=False` will now reload — and serve — differently than it
  did on 0.2.x, which is what the minor bump is for. Adapters saved with the default are
  unaffected.
- The served model card read `max_length` off the encoder via `getattr` and published
  `0` for a model whose real limit is 256. It reads `adapter.json` now.

## 0.2.1 — 2026-07-23

### Added

- `tests/test_public_api.py` — the public surface above, and the torch-free property,
  asserted rather than documented.

### Fixed

- Two class-scoped fixtures in `tests/tokenizer/test_tokenizer.py` were defined as
  instance methods. pytest builds a fresh instance per test, so `self` inside a
  class-scoped fixture is not the instance any test receives; they are now
  `@classmethod`. No warning is emitted at pytest 8.4.2 even under `-W error`, but the
  instance form is on pytest's removal list and the binding was wrong regardless.

## 0.2.0 — 2026-07-23

First tagged release. `0.1.0` was the version the project carried while it was only ever
installed from a working copy; nothing was ever published under it, so there is no upgrade
path to describe and no changes are listed against it.

### For consumers

The base install is `numpy`, `pandas`, `pyyaml`, `sentencepiece` and `tqdm` — deliberately
no `torch`. The corpus, vocabulary, tokenizer and evaluation layers work from that alone,
so a caller that wants text preparation keeps full control of its own training stack.
Torch and the model hub client sit behind the `neural` and `pretrained` extras.

Requires Python `>=3.12,<3.13`. The ceiling is not arbitrary: torch stopped shipping
x86_64 macOS wheels after 2.2.2, which predates 3.13, so 3.13 leaves an Intel development
machine with no torch at all.

### Added

- **Adapting published checkpoints.** LoRA over a frozen pretrained encoder, behind the
  same `TextEncoder` protocol as the from-scratch models, with `qfme adapt` as a command
  rather than a script.
- **A saved adapter is a real artefact.** `format_version: 1`, a manifest naming the base
  checkpoint, pooling, prefixes and LoRA shape, and `load_adapter` /
  `SemanticSearchPipeline.from_adapter` to reload it with its prefixes applied on the
  right sides.
- **Declared adaptations.** A run states which of six adaptations it claims to measure and
  is checked against its own data before the model loads, so a language-transfer claim
  cannot quietly be a task-adaptation result.
- **Retrieval measurement with confidence intervals**, replacing a falling loss as the
  evidence that anything improved.
- **Corpus provenance fingerprinting**, so which dump produced a corpus stops being a
  matter of recollection.
- **`models/indic-v1/`** — the adapter every published number in this repository was
  measured on, tracked because it predates `qfme adapt` and no committed configuration
  rebuilds it.

### Changed

- The version now has one source of truth. `pyproject.toml` declares
  `dynamic = ["version"]` and hatchling reads
  `src/multilingual_embedding/common/version.py`. Previously both held independent
  literals with nothing asserting they matched — a wheel could publish one number while
  every evaluation report stamped another, making the provenance beside a published metric
  wrong with nothing raising.

### Fixed

- A pair-kind filter that could silently shrink the training set rather than failing.
- A test tokenizer that was not deterministic across processes.
- A reported gain that was within noise, now withdrawn and reported as such.
