# Architecture

---

## The layer diagram

```
pipelines        training and search workflows
    |
evaluation       metrics, scoring, reports
    |
embedding        word2vec, sentence encoders, similarity index
    |
tokenizer        normalizers, pre-tokenizers, SentencePiece
    |
vocabulary       token <-> id mapping, special tokens
    |
corpus           document tree, segmentation, readers, statistics
    |
config           typed, validated configuration
    |
core / utils     exceptions, logging, registry, factory, I/O
    |
common           spans, enums, type aliases, constants
```

## The dependency rule

**Each layer may import only from layers below it. `common` and `core` import nothing
internal at all.**

That is a real constraint with real consequences, not a diagram drawn after the fact:

- `common` holds `Span`, the enums, the type aliases and the constants. It has no
  internal imports, so it can be imported from anywhere without risk of a cycle.
- `core` holds the exception hierarchy, structured logging, the generic `Registry` and
  the config-driven `build_from_config` factory. It depends on nothing internal
  either.
- `corpus` knows nothing about tokenizers or embeddings. A `Sentence` is a span of
  text with metadata; it has no idea a vector will ever be computed from it.
- `evaluation` scores a bare `tokenize: Callable[[str], list[str]]` rather than a
  `Tokenizer` object, which is what lets it score a whitespace-splitting baseline with
  no adapter.
- `pipelines` is the only layer that knows the full order of stages. Every stage below
  it is independently usable.

The one direction that would break this — a lower layer reaching upward — never
happens. `tokenizer/pretokenizer.py` importing `corpus.script` and `corpus.token` is
downward and therefore fine.

---

## `common`

`Span`, `TokenizerModel`, `SpecialToken`, the shared type aliases, the framework
version, and the true constants (`DEFAULT_ENCODING`, `DEFAULT_RANDOM_SEED`,
`DEFAULT_BATCH_SIZE`, `DEFAULT_VOCAB_SIZE`, `DEFAULT_CHARACTER_COVERAGE`).

`Span` is a frozen dataclass over a half-open interval `[start, end)`, validated on
construction (`start >= 0`, `end >= start`). It carries the operations the corpus tree
needs: `slice`, `contains`, `overlaps`, `touches`, `merge`, `shift`. Half-open
intervals compose without off-by-one corrections, which matters a great deal once
spans are nested three deep.

## `core`

**Exceptions.** Everything raised deliberately derives from
`MultilingualEmbeddingError`, so a caller can catch framework failures without also
catching unrelated builtins. Errors carry structured key/value context rather than
pre-formatted strings:

```
error: vocab_size exceeds what the training corpus can support; reduce vocab_size or
supply more text (model_type='unigram', requested_vocab_size=5000, sentences=750, ...)
```

The message stays readable while the individual values remain available to logging
and tests. Subclasses: `ConfigurationError`, `ValidationError`, `RegistryError`,
`SerializationError`, `ResourceNotFoundError`, `NotFittedError`.

**Registry and factory.** See [the registry pattern](#the-registryfactory-pattern)
below.

## `config`

Five dataclasses — `CorpusConfig`, `TokenizerConfig`, `EmbeddingConfig`,
`EvaluationConfig` and the `ExperimentConfig` that binds them — each validating itself
in `__post_init__`.

The decision worth knowing: **dataclasses do not re-run `__post_init__` on mutation**,
so instances are treated as immutable after construction. Deriving a variant goes
through `ExperimentConfig.merged(overrides)`, which round-trips through primitives and
revalidates everything on the way back. Mutating `config.embedding.dimension = -1`
directly would silently bypass every check.

Full field reference in [Configuration](configuration.md).

## `utils`

Validation helpers (`require_positive`, `require_in_range`, `require_one_of`, …) that
each return the validated value so a check inlines into an assignment; content
hashing; filesystem helpers including `atomic_write_path`; YAML/JSON/JSONL I/O with
transparent gzip support; and the `to_primitive`/`from_primitive` pair that reduces
dataclass trees to JSON-compatible structures and rebuilds them with validation.

Atomic writes matter here more than they usually do: a partially written model file is
worse than no model, because the next run will happily load the truncated version.
Artefacts are staged and moved into place only once training has succeeded.

---

## `corpus`

The largest package, and the one where being multilingual changes the implementation
rather than just the documentation.

### The node tree

```
Corpus -> Document -> Paragraph -> Sentence -> Token
```

Every node derives from `TextNode`, which owns exactly three things: the `text` it
covers, the `span` locating that text inside its parent, and its `metadata`. The class
is generic in the metadata type so `sentence.metadata` is statically known to be a
`SentenceMetadata`. `ContainerNode` adds ordered `children`.

`Document` is the unit of provenance — licence, author, source URL attach there — and
the granularity at which corpora are split into train and evaluation sets. Splitting
at sentence level would leak near-duplicate content across the boundary, since
sentences within a document are highly correlated.

### Spans are relative to the immediate parent

This is the single most important thing to know about the tree.

A sentence's span is an offset into **its paragraph's** text, not into the document.
A paragraph's span is an offset into the document. Nothing stores an absolute
position.

```python
from multilingual_embedding.corpus.document import Document
from multilingual_embedding.corpus.offsets import resolve_chain

doc = Document.from_text("First para one. Para one two.\n\nSecond paragraph here.",
                         language="en")

paragraph = doc.paragraphs[1]
sentence = paragraph.sentences[0]

paragraph.span                          # Span(start=31, end=53)
sentence.span                           # Span(start=0, end=22)   <- relative!
resolve_chain([paragraph.span, sentence.span])   # Span(start=31, end=53)
```

**The tradeoff.** Relative spans keep segmentation local and composable: a paragraph
can be re-segmented, or replaced entirely, without renumbering every node that follows
it in the document. Absolute spans would make that a whole-document rewrite, and any
edit anywhere would invalidate offsets everywhere after it.

The cost is that mapping a sentence back to a position in the original source is no
longer a field read — it requires walking the chain of parents. `corpus/offsets.py`
exists for exactly that, and the answer is verifiable:

```python
absolute = resolve_chain([paragraph.span, sentence.span])
absolute.slice(doc.text) == sentence.text     # True
```

The module also provides `to_absolute`, `spans_are_ordered`, `spans_within`,
`merge_overlapping`, and `invert_spans` — the last returning the *gaps* between spans,
which is how the material segmentation discarded can be recovered.

### Container nodes store their own text

A `ContainerNode` keeps both its own `text` and its `children`. It does not derive its
text by joining the children, and this is deliberate: **the material between children
is part of the source and would be lost.**

```python
doc = Document.from_text("One.   Two.\n\nSecond paragraph.", language="en")

doc.text                                              # 'One.   Two.\n\nSecond paragraph.'
"".join(p.text for p in doc.paragraphs)               # 'One.   Two.Second paragraph.'

paragraph = doc.paragraphs[0]
paragraph.text                                        # 'One.   Two.'
"".join(s.text for s in paragraph.sentences)          # 'One.Two.'
[s.span for s in paragraph.sentences]                 # [Span(0, 4), Span(7, 11)]
```

The paragraph break `\n\n` and the three spaces between `One.` and `Two.` exist in the
source and appear in no child. Reconstruct by joining and they are gone — with no
error, no warning, and no way to recover the original formatting. Storing the text
means a document survives a round trip through segmentation unchanged, and the sentence
spans `(0,4)` and `(7,11)` still index correctly into it.

The two views are kept honest by `ContainerNode.verify_children()`, which checks that
every child lies within bounds, that children are ordered and non-overlapping, and
that each child's `text` equals the slice its span designates. `Document.verify()`
applies this recursively down the tree.

### Script-aware segmentation

`corpus/segmentation.py` returns **spans, not strings**, so the caller keeps the
ability to map any unit back to its exact source position.

**The terminator inventory is multilingual.** A period-and-space rule handles English
and silently destroys most other writing systems:

```python
SENTENCE_TERMINATORS = frozenset(".!?।॥۔؟。！？．።…")
```

- `.` `!` `?` — Latin, Cyrillic, Greek and most alphabetic scripts
- `।` `॥` — Devanagari danda and double danda (Hindi, Marathi, Nepali)
- `۔` `؟` — Urdu full stop, Arabic question mark
- `。` `！` `？` `．` — CJK fullwidth forms
- `።` — Ethiopic full stop

A second set matters just as much:

```python
_UNCONDITIONAL_TERMINATORS = frozenset("।॥。！？．።")
```

CJK and Indic scripts do not put whitespace after a terminator, so the usual
heuristic — *terminator, then space, then a capital letter* — never fires for them. If
those terminators were run through the same rule as `.`, Hindi and Japanese would come
back as one sentence per document. They therefore end a sentence unconditionally:

```python
split_sentences("नमस्ते। आप कैसे हैं?")
# [Span(0, 7), Span(8, 20)]

split_sentences("研究者は説明します。学生は教えます。")
# [Span(0, 10), Span(10, 18)]        <- no space between them
```

The Latin path retains the heuristics that Latin needs: repeated terminators (`...`,
`?!`) collapse into one boundary, closing quotes and brackets are absorbed so
`He said "stop!"` keeps its quote, and a known-abbreviation list plus single-letter
initial detection stops `Dr. Smith arrived.` splitting after `Dr.`:

```python
split_sentences("Dr. Smith arrived. He was late.")
# [Span(0, 18), Span(19, 31)]
```

Those Latin checks are skipped entirely for scripts that do not use a bare period as a
terminator, which is a measurable saving on large non-Latin corpora.

**Python's `\w` does not match Unicode combining marks.** This is the other place
where the naive implementation fails silently:

```python
import re
re.findall(r"\w+", "नमस्ते दुनिया")
# ['नमस', 'त', 'द', 'न', 'य']        <- five fragments, no real words
```

Every vowel sign and the virama break the match. The same happens to Arabic, Hebrew
and Thai. So the word splitter builds its own character class from the Unicode
database, scanning the Basic Multilingual Plane for the `Mn`, `Mc` and `Me` general
categories and collapsing them into regex ranges:

```python
split_words("नमस्ते दुनिया")   # two spans, correctly
# ['नमस्ते', 'दुनिया']
```

Two details in that construction:

- The class is **derived from the Unicode database at runtime rather than hardcoded**,
  so it stays correct as the database is updated.
- It is built on **first use, not at import**. Scanning the BMP costs a few hundred
  milliseconds, which is unacceptable to pay on every `import multilingual_embedding`.
  It is `lru_cache`d, so the cost is paid once per process and only if word splitting
  is actually used.

Zero-width joiner (U+200D) and non-joiner (U+200C) are added explicitly. They are
format characters, not marks, so they would not be caught by the category scan — but
they occur word-internally in Devanagari and Arabic and must not split a word.

### Script detection

`corpus/script.py` maps codepoints to ISO 15924 script codes through a table of
ranges, binary-searched. Ranges rather than `unicodedata.name` lookups because they
are an order of magnitude faster over a large corpus and give the same answer for the
scripts targeted here. The table is sorted at import and checked for overlaps at
import, since an overlap would make lookup depend on which row the binary search
landed on.

Punctuation, digits, symbols and whitespace map to `Script.COMMON` and are excluded
from the denominator when computing dominance, so `"hello, world!"` scores 1.0 for
Latin rather than being diluted.

The consequential output is `is_whitespace_delimited`. Han, Hiragana, Katakana and
Thai are flagged as *not* whitespace-delimited, and that flag changes behaviour rather
than just labelling text — see the pre-tokenizer below.

### Readers, filters, statistics

Three readers, each registered by name: `TextFileReader` (one file, one document),
`LineReader` (one line, one single-sentence document — the format most public training
sets ship in, where the framework should not second-guess the source's own
boundaries), and `JsonlReader` (one record, one document, with unrecognised fields
carried through into `metadata.base.attributes`). All are lazy generators.

`SentenceFilter` applies conservative rules — length bounds, a letters-required check,
and a U+FFFD ratio test for encoding damage — and keeps a `FilterReport` tallying why
each rejection happened, so an unexpectedly small training set can be traced to the
rule responsible. Conservative is the deliberate choice: over-aggressive cleaning
silently discards valid non-Latin text, which is a far worse failure here than letting
some noise through.

`StatisticsAccumulator` consumes one document at a time and caps both its word table
and its retained sentence lengths, reporting `truncated_vocabulary` when a cap was
hit. Word frequencies are Zipfian, so an uncapped table over a large corpus fills with
singletons and exhausts memory.

### Streaming

`SentenceStream` is the abstraction the whole training path rests on. It is an
**iterable, not an iterator**: it holds a factory that produces a fresh iterator on
each `__iter__`, so `for _ in stream` can run once per epoch and each run restarts the
underlying reader.

That is what bounds corpus size by disk rather than memory. `Word2Vec.train` documents
the consequence explicitly — the stream is traversed once per epoch plus once more up
front to build the vocabulary, so a one-shot generator is not sufficient.

---

## `vocabulary`

The contract between the tokenizer and the embedding model. The tokenizer produces ids
against it; the embedding matrix is indexed by them.

**Ordering is deterministic**: special tokens first, then remaining tokens by
descending frequency with ties broken on the token string. Two runs over the same
corpus produce byte-identical vocabularies, which is a precondition for a reproducible
training run.

`freeze()` seals a vocabulary once a model has been trained against it. Adding a token
afterwards would create an id with no corresponding embedding row; freezing turns a
subtle numerical bug into a clear error.

`VocabularyBuilder` bounds the counting table, pruning singletons when it exceeds
`max_tracked_tokens` and reporting whether it did. Pruning is lossy — a token that
would have crossed `min_count` later can be undercounted — so the cap defaults high
enough (5,000,000) that ordinary corpora never trigger it.

### Special token ids are fixed

```python
SPECIAL_TOKEN_ORDER = (PAD, UNK, BOS, EOS)

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
```

They are constants, not configuration. Two reasons this ordering specifically:

- **`<pad>` is id 0**, so a zero-filled array is already a valid padded batch and
  padding never has to be written explicitly.
- **`<unk>` is id 1**, so an unknown lookup has a defined answer even against a
  vocabulary built without ever seeing one.

**They must match between SentencePiece and `Vocabulary`, and this is the trap.**
SentencePiece's own defaults are different — no pad at all, `unk=0`, `bos=1`, `eos=2`.
So `SentencePieceTrainerAdapter` pins them at training time:

```python
sentencepiece.SentencePieceTrainer.train(
    ...,
    pad_id=PAD_ID, unk_id=UNK_ID, bos_id=BOS_ID, eos_id=EOS_ID,
    pad_piece=self._special.pad, unk_piece=self._special.unk,
    bos_piece=self._special.bos, eos_piece=self._special.eos,
)
```

Without that, encoding would return ids that index the wrong rows of an embedding
matrix built against our `Vocabulary`. There would be no exception and no NaN — the
model would train without error and be quietly, entirely wrong. Because the ids are
pinned, `SentencePieceTokenizer.to_vocabulary()` can simply append pieces 4..N in
model order and reproduce the id space exactly.

Changing this order would invalidate every previously trained model, since the ids are
baked into the embedding matrix. That is why it is a constant rather than a knob.

---

## `tokenizer`

Four stages: normalize, pre-tokenize, subword-segment, map to ids.

**Normalizers** are small and single-purpose — `nfkc`, `nfc`, `nfd`, `nfkd`,
`lowercase`, `whitespace`, `strip_accents`, `digits` — composed into a
`NormalizerPipeline` by configuration rather than a single monolithic normalizer with
a dozen boolean flags. Order is significant and therefore explicit: case folding
before accent stripping is not the same operation as the reverse. Each must be a pure
function of its input, which is what allows a chain to be described entirely by a
config file and reapplied identically at inference.

Two choices worth noting. `LowercaseNormalizer` uses `casefold()`, not `lower()`,
because `lower()` leaves German `ß` unchanged while `STRASSE` lowercases to `strasse`
— the two spellings of the same word would not unify. `WhitespaceNormalizer` removes
U+200B and U+FEFF but **preserves** U+200C and U+200D, because in Devanagari they
control conjunct rendering and in Indic and Arabic scripts they distinguish genuinely
different words; stripping them silently merges distinct types.

**Pre-tokenizers** decide where a subword model is *allowed* to place a boundary.
Every one emits `Token` objects with spans into the input, upholding
`token.text == text[token.span.start:token.span.end]`.

`ScriptAwarePreTokenizer` is the one that makes the pipeline multilingual. Whitespace
splitting is not a universal rule; it is a property of particular writing systems. It
cuts text into runs of a single script and handles each run by its own rules:

```python
ScriptAwarePreTokenizer().pre_tokenize("Hello 世界")
# ['Hello', '世', '界']
```

Whitespace-delimited runs split on whitespace and punctuation; non-delimited runs emit
one token per character, which gives a subword model a sane starting point without
requiring a language-specific word segmenter. Applied to Japanese, a whitespace
pre-tokenizer would return whole sentences as single tokens — almost all hapaxes, an
exploded vocabulary, and every downstream stage inheriting the damage.

Script-neutral characters (punctuation, digits, whitespace) never start a new run;
they extend the run in progress, so a trailing comma stays with the words it
punctuates.

**Tokenizers.** `SentencePieceTokenizer` is the production path — it handles unseen
words by decomposing them into subwords and needs no language-specific rules, which is
what makes one shared model viable across scripts. `WordTokenizer` is a
dependency-free fallback composed from the framework's own normalizer, pre-tokenizer
and vocabulary; it is what most tests run against and the right choice when word
identity rather than subword units is wanted.

`SentencePieceTokenizer.encode` deliberately returns **no spans**. SentencePiece
pieces carry a synthetic word-boundary marker (`▁`, U+2581) and do not map onto input
characters one for one, so reporting spans would mean fabricating them. `WordTokenizer`
does return spans, into the *normalized* text — normalization can change character
counts, and the normalized form is what the pre-tokenizer actually saw.

**The trainer adapter** exists because SentencePiece is a C++ trainer that reads a
file and writes a file, and reports every failure as a bare `RuntimeError`. The
adapter stages sentences to a temporary file (streamed, so memory stays flat), pins
the special ids, and translates the two `vocab_size` failures into distinct messages —
they are opposite problems and conflating them would send users the wrong way. It
publishes artefacts atomically only after training succeeds.

---

## `embedding`

### Word2Vec: skip-gram with negative sampling, in pure numpy

Predicting each context word from the centre word is the skip-gram objective. Doing
that with a softmax over the whole vocabulary costs one matmul against every row per
token; negative sampling replaces it by scoring the true context word against a
handful of randomly drawn impostors, so an update touches `1 + negatives` rows instead
of all of them.

There is no autograd framework, deliberately. The gradient of a sigmoid on a dot
product is `(label - prediction) * other_vector`. Writing it out is faster than a
graph-building library for a model this small and far easier to reason about when the
vectors come out wrong.

Four details separate an implementation that learns from one that only appears to.

**The 0.75-power noise distribution.** Negatives are drawn from unigram frequency
raised to 0.75:

```python
_NOISE_EXPONENT = 0.75

weights = np.power(np.maximum(frequencies, 0.0), _NOISE_EXPONENT)
weights[:special_count] = 0.0
self._noise_cumulative = np.cumsum(weights / weights.sum())
```

Raw unigram frequencies would make the negatives almost entirely function words, which
teaches the model very little. The 0.75 exponent lifts rare tokens without flattening
the distribution to uniform. Special tokens are zeroed out — they carry no frequency
and never appear as real context, so they must not be drawn as negatives either.

The cumulative table plus `searchsorted` draws a negative in O(log V) from a plain
uniform. `np.random.choice(p=...)` rebuilds its alias structure on every call and is
orders of magnitude slower at the rates this loop needs.

**Subsampling.** Frequent tokens are randomly discarded before they ever become a
training pair, using the classic word2vec formula:

```python
keep = (np.sqrt(relative / threshold) + 1.0) * (threshold / relative)
```

Tokens below `subsample_threshold` are always kept; the discard rate grows with
frequency, so very common tokens stop drowning out informative ones. Setting the
threshold to `0` disables it and lets the training loop skip the draw entirely.

Note that out-of-vocabulary tokens are **dropped rather than folded into `<unk>`**.
Training a single vector on every rare word produces a centroid of unrelated meanings
that then pollutes its neighbours.

**The dynamic window.** The window is redrawn per centre token:

```python
effective = int(self._rng.integers(1, window + 1))
```

A context word at distance 3 is therefore only sampled when the draw is `>= 3`, so
nearer words are effectively weighted more heavily — without any explicit distance
weighting term in the gradient.

**`np.add.at` for colliding negative samples.** Negatives are drawn with replacement,
so the same id can appear more than once in one update, and the positive context word
can be drawn as a negative:

```python
centre_gradient = gradient @ output_weights[targets]
np.add.at(output_weights, targets, gradient[:, None] * hidden[None, :])
input_weights[centre] += centre_gradient
```

Plain fancy-index assignment (`output_weights[targets] += ...`) is **not** an
accumulate. On duplicate indices only one write survives — the update is silently
dropped, and nothing raises. `np.add.at` is the unbuffered version that accumulates
correctly.

The centre-word gradient is accumulated across the positive and all negatives and
applied **once**, after the output update. Updating `W_in` inside the loop over
samples would make later samples see a moved centre vector, silently changing the
objective.

Two supporting choices: `W_in` is initialised uniform in `[-0.5/d, 0.5/d]` and `W_out`
at zero — the small input scale keeps early dot products where the sigmoid gradient is
largest, and a zeroed output matrix means the first updates are driven by labels
rather than noise. And `W_out` is discarded after training: it modelled "is this a
real context word", not word meaning. Only `W_in` leaves the trainer, which is also
why a loaded model can be used for lookup but not resumed.

### `EmbeddingMatrix`

A bare numpy array is not an embedding. Row 4179 means nothing without the vocabulary
that assigned that id, and the two drift apart easily — rebuild the vocabulary with a
different `min_count` and every row now refers to a different token, silently. Pairing
them in one object, validating shapes at construction, and persisting them together is
what prevents that.

All arithmetic is vectorised; `most_similar` uses `argpartition` to find the top-k
boundary in O(n) and sorts only those k.

### Sentence encoders and the similarity index

`MeanPoolingEncoder` averages in-vocabulary token vectors — the obvious baseline and
genuinely hard to beat with anything cheap. `SifEncoder` implements Smooth Inverse
Frequency: token weights of `alpha / (alpha + p(token))` plus removal of the batch's
leading principal direction. The principal-component step makes SIF inherently a
*batch* method — there is no such direction for a single sentence — so an unfitted
encoder returns the weighted average and skips the removal rather than pretending to a
component it cannot estimate.

Both drop out-of-vocabulary tokens rather than mapping them to the unknown row, whose
vector is a training artefact that would drag every sentence containing a rare word
toward the same point.

`SimilarityIndex` stores rows already L2-normalised, so cosine similarity reduces to a
single matmul and a query costs no per-row division. Search is **exact** — the true
nearest neighbour set, not an approximation — at O(n·d) per query. On modern BLAS that
is the right choice up to roughly 10⁵–10⁶ items, where a query still lands in
single-digit milliseconds and the index needs no build step, no tuning and no recall
measurement. Past that an approximate index (HNSW, IVF-PQ) becomes necessary, and it
is deliberately out of scope: an ANN index is a serious piece of software, and
wrapping a poor one would be worse than being honest about the ceiling.

---

## `evaluation`

`TokenizerEvaluator` reports compression (`characters_per_token`), `fertility`
(subword tokens per whitespace word), `unknown_rate` and `vocabulary_utilisation` — and
crucially, does so **per language**. A vocabulary trained on a corpus that is 90%
English encodes English in few tokens and Tamil in many, and a single average hides
that completely. `language_fairness` reduces the spread to a min/max/ratio triple.

It takes a plain `tokenize` callable rather than a `Tokenizer`, so any segmentation
function — including a whitespace baseline — can be scored without an adapter.

`EmbeddingEvaluator` splits metrics into *intrinsic* (similarity correlation, analogy
accuracy — needing labelled data) and *structural* (isotropy, effective dimensions,
mean pairwise similarity, zero-vector count — needing nothing). The structural ones are
the only ones available for a fresh corpus in a language with no benchmark, which is
the normal situation for most languages this framework targets. Intrinsic metrics stay
`None` when no dataset was supplied, so a missing benchmark is never mistaken for a
failing score.

Isotropy is worth watching. Embedding matrices routinely collapse toward a narrow
cone, where every pair looks similar and cosine similarity stops discriminating; a
high `mean_pairwise_similarity` on random tokens is the symptom.

`EvaluationReport` bundles metrics **with the configuration that produced them** and
writes JSON for machines and Markdown for humans. A metric without the settings that
produced it cannot be compared against anything.

---

## `pipelines`

`TrainingPipeline` owns orchestration only — the order of stages, threading artefacts
between them, and recording what happened. Every stage is implemented in its own
package and usable independently.

Two properties are load-bearing. *Streaming*: the corpus is never fully materialised;
every stage pulls from the same re-iterable `SentenceStream`. *Reproducibility*: the
resolved config is written to `config.yaml` before the first stage runs, so even an
interrupted run leaves a record.

One detail worth knowing: the embedding vocabulary is built from the **tokenizer's own
subword pieces**, not from whitespace words. That is what makes the model work for
scripts without whitespace word boundaries, where splitting on spaces would produce
one token per sentence.

`SemanticSearchPipeline` is the inference counterpart. It deliberately loads artefacts
from disk rather than accepting in-memory objects, because that is the path a deployed
service takes and therefore the one that needs to be exercised. It skips sentences
that encode to a zero vector — they would match every query equally poorly and pollute
results — and returns `[]` rather than raising for an unanswerable query, since that
is a normal condition for a search service.

---

## The registry/factory pattern

Every pluggable component family owns a `Registry`, and a config file selects an
implementation by name without importing it.

```python
NORMALIZERS: Registry[Normalizer] = Registry("normalizer")

@NORMALIZERS.register("nfkc")
class NFKCNormalizer(_UnicodeNormalizer):
    ...
```

The families: `NORMALIZERS`, `PRETOKENIZERS`, `TOKENIZERS`, `READERS`,
`SENTENCE_ENCODERS`.

Keys are case-insensitive and stored lowercase, so a config file may use whichever
casing reads best. Re-registering an existing key raises `RegistryError` unless
`override=True` is passed, so an accidental duplicate fails loudly at import time
rather than silently shadowing the earlier implementation. An unknown key produces an
error listing the available keys — which turns a typo in a config file into an obvious
fix rather than a mystery.

`build_from_config` is the bridge from YAML to Python objects. A component
specification is a mapping with a `type` discriminator plus constructor arguments, or
a bare string when no arguments are needed:

```yaml
normalizers:
  - type: nfkc
  - type: whitespace
pretokenizer:
  type: script
```

```python
normalizer = build_from_config(NORMALIZERS, {"type": "nfkc"})
pipeline   = build_all_from_config(NORMALIZERS, [{"type": "nfkc"}, "whitespace"])
```

`build_all_from_config` handles the ordered case, used for normalizer chains where
order is significant.

The detail that makes this usable: **unknown constructor arguments are rejected up
front.** `build_from_config` inspects the target's signature and raises a
`ConfigurationError` naming both the unknown keys and the accepted ones. Without that,
a typo in a YAML key surfaces as an opaque `TypeError` from deep inside a constructor.
Implementations accepting `**kwargs` are skipped, since anything is legal for them.

`overrides` passed as keyword arguments take precedence over the config mapping, which
is how runtime values — a resolved path, a shared vocabulary — get injected into
components without appearing in a static config file.
