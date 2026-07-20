# corpus

> Text representation, multilingual segmentation, script and language identification, streaming readers and writers, cleaning, statistics and auditing — everything between files on disk and a stream of sentences.

## Purpose
Everything above this layer consumes sentences: the tokenizer trains on them, the embedding model treats them as context windows, the evaluators score them. This package is what turns arbitrary files into those sentences without losing the ability to say where any unit came from. It is a layer of its own because segmentation, script detection and cleaning are all script-dependent decisions that must be made in exactly one place — a period-and-space sentence rule silently destroys Hindi and Chinese, and a naive word regex silently destroys every Indic and Arabic word, so these rules cannot be allowed to be reinvented per call site.

## Modules
| Module | Responsibility |
|---|---|
| `corpus.py` | `Corpus` — the in-memory aggregate of documents, with filtering, document-level splitting and JSON Lines persistence. |
| `document.py` | `Document` — the unit of provenance; builds a paragraph/sentence tree from raw text and round-trips to a dict. |
| `paragraph.py` | `Paragraph` — groups sentences; segments its own text on construction unless told not to. |
| `sentence.py` | `Sentence` — the unit the tokenizer trains on; holds surface tokens once a pre-tokenizer has run. |
| `token.py` | `Token` — a surface occurrence with a span, the boundary object between this layer and the tokenizer. |
| `segmentation.py` | `split_paragraphs`, `split_sentences`, `split_words` and the terminator inventory; returns spans, not strings. |
| `script.py` | `Script` enum, `detect_script`, `script_histogram`, `is_whitespace_delimited`, `script_of_character`. |
| `language.py` | Language codes (ISO 639-1, with 639-3 for languages that have no two-letter code), `normalize_language_code`, `expected_script`, and the deliberately conservative `infer_language`. |
| `offsets.py` | Span arithmetic: `resolve_chain`, `to_absolute`, `invert_spans`, `merge_overlapping`, ordering and containment checks. |
| `reader.py` | `CorpusReader` and the registered `TextFileReader` ("text"), `LineReader` ("lines"), `JsonlReader` ("jsonl"), plus `reader_for` and `resolve_reader_type`; all lazy generators. |
| `writer.py` | `CorpusWriter` base, `JsonlCorpusWriter` (full fidelity), `PlainTextCorpusWriter` (one sentence per line) and `write_sentences`; all atomic. |
| `loader.py` | Configuration-driven entry points: `load_corpus` for random access, `stream_documents` and `stream_sentences` for streaming, plus `build_reader` and `build_filter`. |
| `iterator.py` | `SentenceStream` — a re-iterable, not an iterator — plus `batched` and `take`. |
| `statistics.py` | `StatisticsAccumulator` and `CorpusStatistics`; streaming, with bounded word and length tables. |
| `validators.py` | `SentenceFilter`, `DocumentDeduplicator`, `FilterReport`, `validate_document`. |
| `audit.py` | `audit_corpus` and its `CorpusAudit`, `Finding` and `Severity`; judges a corpus rather than describing it. |
| `exceptions.py` | `CorpusError` and its subclasses `CorpusFormatError`, `SegmentationError`, `EmptyCorpusError`. |
| `base/` | Structural base classes for nodes — see `base/README.md`. |
| `metadata/` | Metadata records for each level — see `metadata/README.md`. |

## Key design decisions

### The tree, and why spans are parent-relative

The hierarchy is `Corpus → Document → Paragraph → Sentence → Token`. Every node stores its `span` relative to its **immediate parent's** text, not to the document root. A sentence's `Span(10, 20)` indexes into its paragraph; that paragraph's span indexes into the document.

This is the layer's most consequential structural choice. Its benefit is that segmentation stays local: a paragraph can be re-segmented, or its sentences filtered away, without renumbering every unit that follows it in the document. Under absolute spans, editing one paragraph would be an O(document) operation and a filter that dropped a sentence would invalidate every span after it. Its cost is that recovering a position in the original source requires walking the chain of parents, which is why `offsets.py` exists and why `offsets.resolve_chain` is a documented, tested entry point rather than an implementation detail. That is a real cost paid at every absolute-offset lookup, in exchange for making every local edit free.

### Container nodes store their own text

`Paragraph` has both `text` and `children`, and so does `Document` and `Sentence`. The two are near-redundant — but not redundant, because the material *between* children is covered by the parent's `text` and by no child's span. Whitespace, the punctuation separating one sentence from the next, markup: all of it is part of the source. Deriving a container's text by joining `child_texts()` would discard it, and a document could not survive a round trip through segmentation unchanged. `offsets.invert_spans` recovers exactly that between-material, and can only do so because it was never thrown away.

The price of storing both views is that they can drift apart. `ContainerNode.verify_children()` is the check that they have not; `Document.verify()` composes it down all three levels, and `validators.validate_document` calls it and *reports* rather than raises, so one malformed document does not abort a pass over a large corpus. The three failure modes are enumerated in `base/README.md`.

### Multilingual sentence segmentation

`SENTENCE_TERMINATORS` is `frozenset(".!?।॥۔؟。！？．።…᱾᱿꯫")`. Beyond ASCII that covers the Devanagari danda `।` and double danda `॥` (Hindi, Marathi, Nepali, Sanskrit, Maithili, Konkani, Dogri, Bodo, and used in Bengali, Odia and Gurmukhi too), the Urdu full stop `۔` and Arabic question mark `؟` (also Sindhi and Kashmiri), the CJK fullwidth forms `。！？．`, the Ethiopic full stop `።`, the Ol Chiki mucaad `᱾` and double mucaad `᱿` (Santali), and the Meetei Mayek cheikhei `꯫` (Meitei).

The inventory alone is not enough. The usual heuristic for a bare period — terminator, then whitespace, then something that can begin a sentence — is wrong for most of that list, because CJK and Indic scripts do not require a space after a terminator. A `。` at the end of a Japanese sentence is typically followed immediately by the next character. So `_UNCONDITIONAL_TERMINATORS` (`।॥。！？．።᱾᱿꯫`) bypasses the heuristic entirely and always closes a sentence. Applying the Latin rule uniformly would merge an entire Chinese paragraph into one "sentence"; applying the unconditional rule uniformly would split English at every decimal point.

For the terminators that *are* ambiguous, principally `.`, three guards apply:

- **No following whitespace means internal.** This is what keeps `3.14`, version numbers and URLs intact.
- **Abbreviations.** `_ABBREVIATIONS` holds 28 lowercase Latin forms (`dr`, `mr`, `prof`, `etc`, `e.g`, `i.e`, `vs`, `al`, `pp`, …) that end in a period without ending a sentence.
- **Initials.** `_ends_with_abbreviation` walks backwards over letters and periods, so a single letter before the period — the `J.` in `J. R. R. Tolkien` — is treated as an initial rather than a boundary.
- **Successor case.** A lowercase letter after the whitespace nearly always means the period belonged to an abbreviation this module does not know, so no boundary is placed.

Those Latin checks are skipped entirely for scripts that do not use a bare period, decided by `_should_check_abbreviations` from either an explicit `language` hint or the detected script. That is a correctness no-op and a measurable saving on large non-Latin corpora.

The splitter is rule-based rather than model-based, which is a deliberate ceiling: it is fast, dependency-free and predictable, and it will split on an unknown abbreviation followed by a capitalised proper noun. Callers needing better should segment upstream and feed pre-split sentences, which `LineReader` and `JsonlReader(segment=False)` both support.

### `split_words` builds its own character class from the Unicode database

Python's `\w` does not match Unicode combining marks, and every Indic and Arabic word is written with them. The damage from a naive `re.findall(r"\w+", ...)` is not subtle:

| Input | Naive `\w+` | Correct |
|---|---|---|
| `नमस्ते` | `['नमस', 'त']` | one word |
| `हैं` | `['ह']` | one word |
| `مُحَمَّد` | `['م', 'ح', 'م', 'د']` | one word |
| `กรุงเทพ` | `['กร', 'งเทพ']` | one word |

So `_combining_mark_class()` scans the Unicode database for every codepoint in categories `Mn`, `Mc` and `Me` across the Basic Multilingual Plane, coalesces them into ranges, and emits a regex character class that is added to `\w`. Deriving it from the database rather than hardcoding a table means it stays correct as the database is updated with the Python version — the alternative, a literal range table, would be a silent correctness decay with every Unicode release.

ZWJ (U+200D) and ZWNJ (U+200C) are added to the class as well. They are format characters, not marks, so no category scan would find them, but they occur word-internally in Devanagari — where they control whether a consonant cluster renders as a conjunct — and in Arabic, where they distinguish genuinely different words. Treating them as word boundaries would split words that a reader sees as one. (The tokenizer's `WhitespaceNormalizer` preserves them for the same reason, while removing the genuinely meaningless U+200B and U+FEFF.)

The class is built lazily under `@lru_cache(maxsize=1)`, and so is the compiled pattern. Measured on this machine, the database scan costs on the order of 10–20 ms and finds 1,338 marks; the cached call is a dictionary lookup. The module docstring describes the cost as "a few hundred milliseconds", which overstates it on current CPython, but the reasoning holds regardless of the constant: it is a cost that must not be paid on `import multilingual_embedding` by every process that never splits a word, and paying it once per process on first use is strictly better.

`split_words` returns whole runs, not words, for Han, Hiragana, Katakana and Thai. That is inherent — those scripts have no whitespace word boundaries — and is exactly why the tokenizer layer uses subword models there rather than this function.

### `script.py`: sorted, disjoint, and `COMMON` excluded from the denominator

Detection uses an explicit codepoint range table rather than `unicodedata.name` string matching, which is roughly an order of magnitude faster over a large corpus and gives the same answer for the scripts targeted here.

Two invariants are enforced at import rather than trusted:

- The table is written grouped by script for readability and then **sorted by start codepoint programmatically** into `_SCRIPT_RANGES`. Lookup is a binary search, so a manually misplaced row would not raise anything — it would silently misclassify an entire script.
- `_assert_ranges_disjoint()` runs at import and raises `RuntimeError` on any overlap. An overlap would make the answer depend on which row the binary search happened to land on, turning a structural error into an intermittent misclassification.

`Script.COMMON` covers punctuation, digits, symbols, separators and control characters — everything shared across writing systems. `detect_script` excludes both `COMMON` and `UNKNOWN` from the confidence *denominator*, so `"hello, world!"` scores `1.0` for Latin rather than being diluted by the comma, the space and the exclamation mark. Including them would make confidence a function of punctuation density, and `ScriptProfile.is_mixed` (confidence below 0.8) would fire on ordinary punctuated prose.

`is_whitespace_delimited` is the single place that knows Han, Hiragana, Katakana and Thai are written without spaces. The tokenizer's `ScriptAwarePreTokenizer` branches on it, which is how a script fact stays in the script module instead of being duplicated in the tokenizer layer.

### `language.py` refuses to guess

`infer_language` consults `_UNAMBIGUOUS_SCRIPTS`, which holds only the scripts served by exactly one language or by one that dominates usage by an order of magnitude: Tamil to `ta`, Telugu to `te`, Kannada to `kn`, Malayalam to `ml`, Gujarati to `gu`, Gurmukhi to `pa`, Odia to `or`, Ol Chiki to `sat`, Meetei Mayek to `mni`, Hebrew to `he`, Greek to `el`, Hangul to `ko`, Thai to `th`, Ethiopic to `am`, Hiragana and Katakana to `ja`. Two entries are dominance judgements rather than one-to-one facts and should be read as such: Devanagari maps to `hi` although `_LANGUAGES` lists eight Devanagari languages, and Bengali maps to `bn` although Assamese shares the script. For Latin, Arabic, Cyrillic and Han — each shared by many languages with no comparable dominance — it returns `None`. It also returns `None` for mixed-script text (`ScriptProfile.is_mixed`) and for text with no script evidence.

Returning `None` rather than the most populous language of a script is the whole point. A plausible wrong answer propagates: it is written into `DocumentMetadata`, it selects the abbreviation rules in `split_sentences`, it may select a per-language normalizer, and nothing downstream has any way to notice it was a guess. `None` is honest, and callers are documented to fall back to a *configured* default rather than to English. Statistical language identification would give real answers here and is deliberately out of scope for this layer; a caller that needs it sets the language explicitly.

### Three registered formats, and all three are reachable

`READERS` holds three entries, and each answers a different shape of source:

| Name | Reader | One document is |
|---|---|---|
| `text` | `TextFileReader` | one whole file, segmented into paragraphs and sentences |
| `lines` | `LineReader` | one non-blank line, kept as a single sentence |
| `jsonl` | `JsonlReader` | one JSON Lines record, segmented unless `segment=False` |

`lines` is the reader for corpora already segmented one sentence per line, which is how most public training sets are distributed. It skips segmentation entirely rather than second-guessing boundaries the source already committed to — running the segmenter over such a file can only merge or split what was already correct.

All three names are accepted by `CorpusConfig.format` and by the `--format` option on the CLI subcommands. That is worth stating because it was briefly untrue: `lines` was registered in `reader.py` while neither the config validator nor the CLI would accept it, so the only way to reach it was to import the class. A registry entry that no configuration can name is not a feature, and the drift is the kind that no test of `reader.py` alone would catch.

`resolve_reader_type` covers the fourth accepted value, `auto`, which chooses by extension — `.jsonl` and `.ndjson` give `JsonlReader`, everything else `TextFileReader`, with a directory inspected by its first matching file. Note that `auto` never selects `lines`, because a `.txt` file gives no clue whether its lines are sentences or prose layout; a sentence-per-line corpus must say `lines` explicitly.

### Readers stream, and `SentenceStream` is re-iterable

`CorpusReader.iter_documents` is a generator, and so is every `read_file` implementation, so a corpus larger than memory flows through the pipeline one document at a time. `paths()` sorts, so two runs over the same directory produce the same document sequence — reproducibility that matters because document order feeds `Corpus.split`'s shuffle and the vocabulary's tie-breaking.

`SentenceStream` is the subtler piece. It implements `Iterable[str]`, **not** `Iterator[str]`: it holds a `factory` callable and calls it afresh in every `__iter__`. Training makes several passes over the sentences, and a plain generator is exhausted after one — the natural workaround, materialising the sentences into a list, bounds corpus size by RAM and defeats the streaming readers entirely. Because each pass restarts the reader, multi-epoch training over a corpus larger than memory works with no special handling at the call site. `map` and `limited` return new streams rather than mutating, so a stream can be shared. `count()` is documented as costing a full pass precisely because the type invites forgetting that.

### `Corpus.split()` splits documents, not sentences

Sentences within one document are strongly correlated — shared topic, shared vocabulary, often near-duplicate phrasing. Dividing them across a train/evaluation boundary lets the model see near-duplicates of what it is scored on, and the evaluation number that results is measuring memorisation. Splitting at document level is what keeps it honest.

The shuffle is seeded (`seed=42` by default) so a split is reproducible, and the pivot is clamped with `max(1, min(len(shuffled) - 1, ...))` so that neither side is empty for any corpus of two or more documents — an extreme `train_fraction` should not produce an empty evaluation set that silently scores as perfect.

### Statistics: streaming, with bounded tables that admit it

`StatisticsAccumulator` folds one document at a time, so a report can be produced for a corpus far larger than memory. Two tables inside it would otherwise grow without bound:

- **The word frequency table**, capped at `max_tracked_words=1_000_000`. Word frequencies are Zipfian: an uncapped table over a large corpus is dominated by singletons that contribute almost nothing to any statistic and can exhaust memory on their own. Once the cap is reached, *new* words are ignored while existing counts keep incrementing.
- **The sentence length reservoir**, capped at `max_tracked_lengths=500_000`. Percentiles need the actual values, not a running mean, but half a million samples is ample for a p99.

Crucially, truncation is reported rather than hidden: `CorpusStatistics.truncated_vocabulary` is set, and it is documented that `unique_words` then understates the true figure. A capped statistic that does not say it was capped is worse than no statistic, because it will be compared against an uncapped one from a smaller run.

The reported distributions are percentile-based for a reason given in `LengthSummary`'s docstring: a corpus with a handful of enormous unsegmented sentences shows a perfectly reasonable mean and a p99 that reveals the problem. `_percentile` uses linear interpolation matching numpy's default method, so figures here agree with any downstream analysis. `word_count` is documented as not comparable across scripts; `character_count` is the measure to use when comparing a Chinese corpus against an English one.

### `validators.py`: conservative rules, exact deduplication

Filtering happens before the tokenizer ever sees the text, because training on boilerplate and duplicates wastes model capacity and skews the vocabulary toward noise. The rules are deliberately conservative — a rule fires only on unambiguous evidence — because over-aggressive cleaning silently discards valid non-Latin text, which is a far worse failure for this framework than letting some noise through. `SentenceFilter.expected_script` is left unset by default for exactly that reason: mixed-script text is normal in a genuinely multilingual corpus.

`FilterReport` counts rejections per rule, not just in total, so an unexpectedly small training set can be traced to the rule responsible rather than guessed at.

`DocumentDeduplicator` is **exact-match** on NFC-normalised, whitespace-collapsed text (`_canonical`), storing only content hashes so memory grows by a fixed number of bytes per distinct document. Near-duplicate detection via MinHash or SimHash is a deliberate omission, not an oversight: it carries a false-positive risk that exact matching does not, and a false positive here means silently deleting legitimate training text — in a multilingual corpus, most likely the text in the least-represented language, where near-duplicates are hardest to distinguish from the genuine repetition of a small corpus.

`SentenceFilter.apply` prunes a document's segmentation in place but deliberately leaves the document's own `text` unchanged, so the original source stays recoverable.

### `audit.py`: statistics describe a corpus, an audit judges it

`compute_statistics` will tell you a corpus has 4.2 million sentences and a p99 length of 190 characters. It will not tell you that a third of them still carry `[[wikilinks]]`, that the language column was never populated, that the extractor guessed the wrong encoding and left replacement characters throughout, or that the same article was ingested twice under different ids. Those are the failure modes an extraction pipeline actually has, and their defining property is that **none of them raise**. The corpus loads, training completes, and the model is worse than it should be for reasons that are invisible by the time anyone looks at the metrics.

`audit_corpus(documents)` returns a `CorpusAudit`: volume seen, documents per declared language and detected script, and a list of `Finding` objects. Each finding carries a `Severity`, a stable machine-readable `code`, the `count` and `share` of documents affected, a few example identifiers to go and look at, and a `remedy` naming what to do. The examples matter more than they look — a share of 0.31 tells you the scale of a problem, but the three identifiers are what let you open a file and see it.

Severity is the part a caller can act on, and the three levels are defined by what they imply rather than by how alarming they sound:

- **`ERROR` — unusable as it stands.** Empty documents, surviving wiki or HTML markup, replacement characters from a wrong encoding guess. Markup trains the tokenizer on syntax instead of language; replacement characters mean the bytes were decoded wrongly and the text is not recoverable by any downstream cleaning.
- **`WARNING` — it will train, but something was probably lost upstream.** Documents in no recognised script, duplicates of earlier documents, documents declaring no language, documents shorter than 40 characters. An unpopulated language column is a warning rather than an error because inference can sometimes recover it — but not for Latin, Arabic, Cyrillic or Han, which is precisely why the finding exists rather than being silently patched by `infer_language`.
- **`INFO` — context worth reading before trusting a result.** Chiefly documents holding a single sentence, which is exactly right for a sentence-per-line corpus and suspicious for one that is supposed to hold articles. The same observation means opposite things depending on what the source was meant to be, so the audit reports it and declines to judge.

Findings are ordered most severe first and, within a severity, by descending count, so the first line of the report is the thing most worth fixing.

The audit streams. It consumes the iterable once and holds only counters and content hashes — the same bounded-memory discipline as `DocumentDeduplicator`, and for the same reason — so it runs over a corpus far larger than memory. Duplicate detection is exact-match on whitespace-collapsed text, inheriting that module's argument for exactness over near-duplicate matching.

It is exposed as `qfme validate`, which prints the report and **exits non-zero when the corpus is unusable**, so a data pipeline can gate on it rather than discovering the problem partway through a training run. `--strict` extends that to warnings, for a pipeline that would rather stop than train on text with an unpopulated language column. `--output` writes the whole audit as JSON for a build system to keep. The CLI passes `deduplicate=False` to `stream_documents` deliberately: the loader's own deduplication would remove the duplicates before the audit could count them, and reporting "no duplicates" because they were silently dropped upstream is the exact failure this module exists to prevent.

## Usage

```python
from multilingual_embedding.corpus import (
    Corpus, Document, detect_script, infer_language,
    split_sentences, split_words, compute_statistics,
)

document = Document.from_text(
    "Dr. Smith paid 3.14 dollars. He left.\n\nनमस्ते। आप कैसे हैं?",
    identifier="doc-1",
)

print("paragraphs:", document.paragraph_count)
print("sentences:", [s.text for s in document.sentences()])
print("language:", document.metadata.base.language)
print("script:", document.metadata.base.script)

document.verify()
print("verify: ok")

print("danda split:", split_sentences("नमस्ते। आप कैसे हैं?"))
print("word spans:", split_words("नमस्ते"))
print("confidence:", detect_script("hello, world!").confidence)
print("infer latin:", infer_language("hello"))

corpus = Corpus.from_texts(["one. two.", "three. four.", "five. six."], name="demo")
train, evaluation = corpus.split(train_fraction=0.7)
print("split:", len(train), len(evaluation))

stats = compute_statistics(corpus)
print("stats:", stats.sentence_count, stats.unique_words, stats.truncated_vocabulary)
```

Output:

```
paragraphs: 2
sentences: ['Dr. Smith paid 3.14 dollars.', 'He left.', 'नमस्ते।', 'आप कैसे हैं?']
language: None
script: Latn
verify: ok
danda split: [Span(start=0, end=7), Span(start=8, end=20)]
word spans: [Span(start=0, end=6)]
confidence: 1.0
infer latin: None
split: 2 1
stats: 3 6 False
```

Several decisions are visible at once. `Dr.` and `3.14` did not split, but the danda did, without needing a following space. `split_words("नमस्ते")` returns one span of six codepoints rather than the two fragments a naive `\w+` produces. `"hello, world!"` scores exactly 1.0 for Latin because punctuation is out of the denominator. And the document's language is `None` — the text is mixed-script, so `infer_language` declined to guess rather than answering "en".

## Dependencies
May import from `common` (`Span`, `SpecialToken`, constants), `core` (exceptions, `Registry`, logging), `utils` (`io`, `filesystem`, `hashing`) and `config` (`loader.py` takes a `CorpusConfig`). Within itself, `base/` and `metadata/` sit beneath the concrete node modules.

Must **not** import `vocabulary`, `tokenizer`, `embedding`, `evaluation` or `pipelines`. Enforced by `tests/test_architecture.py`.

Consumed by `tokenizer` (`pretokenizer.py` imports `Script`, `is_whitespace_delimited`, `script_of_character` and `Token`), `evaluation/tokenizer_eval.py` (`detect_script`), `pipelines/training.py` (`SentenceStream`, the loader and statistics) and `cli.py` (the loader, statistics and `audit_corpus`).

## Tests
Package total: **395 tests** across `tests/corpus`, of which 389 cover this package's own modules and 6 the `base/` and `metadata/` subpackages.

- `tests/corpus/test_indian_languages.py` — 166 tests (per-language segmentation, script detection and inference across the scheduled languages)
- `tests/corpus/test_analysis.py` — 74 tests (script, language, offsets, statistics, validators)
- `tests/corpus/test_script.py` — 33 tests
- `tests/corpus/test_io.py` — 29 tests (readers, writers, loader, round-tripping)
- `tests/corpus/test_segmentation.py` — 25 tests
- `tests/corpus/test_nodes.py` — 22 tests
- `tests/corpus/test_audit.py` — 21 tests
- `tests/corpus/test_corpus.py` — 19 tests
- `tests/corpus/base/test_text_node.py` — 5 tests
- `tests/corpus/metadata/test_base.py` — 1 test

Layering is separately enforced by `tests/test_architecture.py`, and the whole layer is exercised end to end by `tests/integration/test_end_to_end.py`.
