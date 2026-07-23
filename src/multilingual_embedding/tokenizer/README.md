# tokenizer

> Text in, model input ids out: a four-stage pipeline of normaliser, pre-tokeniser, tokenizer and `Encoding`, plus the adapter that trains the subword model.

## Purpose
The embedding model consumes integer ids and nothing else. Getting from a string to those ids involves a series of decisions — which surface forms collapse together, where a boundary may be placed, how unseen words are handled — each of which is script-dependent and none of which can be corrected downstream. This package makes those decisions explicit, configurable by name, and persistable, so that the exact mapping used at training time is reproduced at inference. It sits above `vocabulary` because it produces ids against one, and above `corpus` because it consumes `Token` and `Script`.

## Modules
| Module | Responsibility |
|---|---|
| `normalizer.py` | `Normalizer` base, the `NORMALIZERS` registry, eight single-purpose normalisers, and `NormalizerPipeline`. |
| `pretokenizer.py` | `PreTokenizer` base, the `PRETOKENIZERS` registry, and the whitespace, character, punctuation and script-aware implementations. |
| `tokenizer.py` | `Tokenizer` base, the `TOKENIZERS` registry, `SentencePieceTokenizer` and `WordTokenizer`, with save/load. |
| `encoding.py` | `Encoding` — ids, surface pieces, spans and attention mask, with `truncate` and `pad_to`. |
| `trainer.py` | `SentencePieceTrainerAdapter` — stages a corpus, pins the special ids, translates SentencePiece failures — plus `trainer_for`, which builds one from plain settings. |

## The four stages

```
text
  -> NormalizerPipeline      unify surface forms
  -> PreTokenizer            propose boundaries, with spans
  -> Tokenizer               map to vocabulary ids
  -> Encoding                ids, pieces, spans, attention mask
```

Each stage is registered by name, so a YAML file can select it without importing anything.

The two tokenizers use different amounts of the pipeline, and the difference is not arbitrary. `WordTokenizer` runs all four stages. `SentencePieceTokenizer` runs the normaliser and then hands the result to the model, because SentencePiece consumes a raw character stream by design — that is exactly what lets one model serve scripts with no whitespace word boundaries, and there is no point at which a framework pre-tokeniser could be inserted without destroying the property. `SentencePieceTrainerAdapter._warn_if_pretokenizer_is_inapplicable` logs a warning when `config.pretokenizer` is set to anything but the inert `whitespace` default, on the principle that silently dropping a setting the user wrote is the one unacceptable response.

## Key design decisions

### NFKC is the default first step, and why it matters multilingually

Unicode offers several encodings of what a reader treats as one character. A tokenizer that does not unify them trains a separate vector for each, splitting the evidence for one word across several vocabulary rows.

- **Fullwidth Latin.** `Ａ` (U+FF21) is pervasive in Japanese and Chinese text, where Latin letters are commonly typed in the fullwidth forms that match ideographic metrics. NFKC maps it to plain `A`. Without it, an English loanword appearing in a CJK corpus is a different type from the same word in an English one.
- **Arabic presentation forms.** The U+FB50 and U+FE70 blocks encode contextual ligatures and the isolated, initial, medial and final glyph variants of letters that are one letter. Text extracted from PDFs is full of them. NFKC folds them back to their base letters.
- **Devanagari nukta consonants.** Some are encodable both as a precomposed character and as base plus combining nukta; composition makes those identical.

NFKC is lossy — superscripts, ligatures and fullwidth forms do not survive it — which is acceptable here because the goal is to maximise evidence per vocabulary entry rather than to preserve typography. `NFCNormalizer` is registered for the cases where that folding would destroy the object of study, and `NFD`/`NFKD` exist for pipelines that strip combining marks immediately afterwards.

### `casefold()`, not `.lower()`

`LowercaseNormalizer` calls `text.casefold()`. The difference is not cosmetic:

- German `ß` is left unchanged by `.lower()`, while `STRASSE` lowercases to `strasse`. The two spellings of the same word would not unify. `casefold()` maps `ß` to `ss`, so both reach `strasse`.
- Greek final sigma `ς` and medial `σ` are the same letter in different positions. `.lower()` leaves them distinct; `casefold()` maps `ς` onto `σ`, so `ΣΟΦΟΣ` folds to `σοφοσ` and matches the medial spelling.

Case folding is a no-op for the uncased scripts — Devanagari, Arabic, Han, Thai — so it is safe to leave enabled in a multilingual pipeline rather than gating it per language.

### The whitespace normaliser must not strip ZWJ and ZWNJ

`WhitespaceNormalizer` collapses whitespace runs and removes invisible characters — but selectively. Its `_REMOVED` tuple contains exactly two: U+200B zero-width space and U+FEFF byte order mark, both formatting artefacts with no linguistic content.

U+200C zero-width non-joiner and U+200D zero-width joiner are **deliberately preserved**. In Devanagari they control whether a consonant cluster renders as a conjunct; in Arabic and other Indic scripts they distinguish genuinely different words. Stripping them — which the obvious "remove all zero-width characters" implementation does — silently merges distinct types, and does so invisibly, because the characters have no visual form to reveal what was lost. `corpus.segmentation` treats them as word-internal for the same reason, so the two layers agree.

The whitespace collapsing itself uses `" ".join(cleaned.split())`. Bare `str.split()` splits on any Unicode whitespace, so U+00A0 no-break space (ubiquitous in HTML), U+3000 ideographic space (CJK typesetting) and the U+2000 block are all handled without an explicit table, and empty fields are discarded so runs collapse and the ends are trimmed in one operation.

### Normalisers are small and composed, not one class with flags

The registry holds eight single-purpose normalisers — `nfkc`, `nfc`, `nfd`, `nfkd`, `lowercase`, `whitespace`, `strip_accents` and `digits` — assembled into a `NormalizerPipeline` by configuration. The alternative — one monolithic normaliser with a dozen booleans — would hide the fact that **order is significant**. Case folding before accent stripping is not the same operation as the reverse. Making the chain an ordered list forces that order to be stated rather than inherited from the order the flags happen to be checked in.

`StripAccentsNormalizer` carries a matching warning: it drops every `Mn` codepoint, which is desirable for European accents and Arabic harakat but destructive for Indic vowel signs and the virama, since those share the category. It is meant to be enabled per language, not globally.

`DigitNormalizer` restricts itself to category `Nd`, because the other numeric categories cover fractions, Roman numerals and circled forms that have no single-digit ASCII equivalent.

### `ScriptAwarePreTokenizer` emits per-character tokens for non-delimited scripts

This is the pre-tokeniser that makes the pipeline multilingual rather than Latin-with-exceptions. Whitespace splitting is not a universal rule; it is a property of particular writing systems. Japanese, Chinese and Thai are written without spaces between words, so a whitespace pre-tokeniser applied to Japanese returns **whole sentences as single tokens**. Those tokens are almost all hapaxes, the vocabulary explodes with them, and every subsequent stage inherits the damage.

So the text is first cut into runs of a single script, and each run is handled according to `corpus.script.is_whitespace_delimited`:

- delimited runs (Latin, Devanagari, Arabic, Cyrillic, Hangul, …) are split on whitespace and punctuation;
- non-delimited runs (Han, Hiragana, Katakana, Thai) emit one token per character, which gives a subword model a sane starting point without requiring a language-specific word segmenter.

Punctuation, digits and whitespace carry no script evidence and never start a new run — `_script_runs` skips `COMMON` and `UNKNOWN` characters entirely — so a trailing space or comma stays with the words it punctuates rather than forming a run of its own.

Every pre-tokeniser upholds one invariant, stated in the module docstring and worth restating: `token.text == text[token.span.start:token.span.end]`. Without exact spans a prediction over token positions cannot be mapped back onto the characters that produced it, and debugging becomes guesswork.

### The same normaliser chain must be applied at training time and at encode time

This is the subtlest coupling in the package. SentencePiece learns its pieces from whatever bytes reach the staged corpus file, so `SentencePieceTrainerAdapter._stage_corpus` runs each sentence through `NormalizerPipeline.from_config(config.normalizers)` before writing it. Normalising *only* at encode time would be worse than not normalising at all, because the text presented to the model would then differ from the text its pieces were learned from — every fullwidth-Latin or presentation-form character would decompose into pieces the model never saw.

The consequence is that the chain is not a property of the tokenizer alone; it is a property of the trained model, and the two must agree. Three mechanisms enforce that:

- `SentencePieceTokenizer` takes the same `normalizers` argument and re-applies the chain in `encode` and `tokenize`.
- `SentencePieceTokenizer.save` writes the chain alongside the model, into `SENTENCEPIECE_CONFIG_FILENAME` — deliberately `sentencepiece.json`, **not** `tokenizer.json`, because a `WordTokenizer` saved into the same directory writes that second name and the two payloads are not interchangeable. Without persisting the chain, a reloaded tokenizer would normalise differently from the one that produced the embeddings, and the mismatch would surface only as quietly degraded results.
- `SentencePieceTokenizer.load` reads it back when present, and falls back to *no* normalisation when the config file is absent — which is correct, because a directory holding only a `.model` file is one the trainer produced before `save` ran, and such a model was trained without a chain. The default for the constructor is likewise no normalisation, matching that case rather than guessing at a chain.

Note the ordering inside `_stage_corpus`: normalise first, flatten whitespace second. A normaliser may emit a newline, and the staging format is line-delimited, so flattening first would let a stray newline silently split a training example in half.

`TokenizerConfig.normalizers` defaults to `[{"type": "nfkc"}, {"type": "whitespace"}]`, which is the chain both paths get unless configured otherwise.

### The trainer pins the special ids, and this is the sharpest edge in the package

`SentencePieceTrainerAdapter._run_trainer` passes:

```python
pad_id=PAD_ID, unk_id=UNK_ID, bos_id=BOS_ID, eos_id=EOS_ID,
```

imported directly from `vocabulary/special_tokens.py`. **SentencePiece's own defaults differ** — no pad at all, `unk=0`, `bos=1`, `eos=2`. Leaving them at their defaults produces no error, no warning and a model that trains successfully. Encoding would simply return ids that index the wrong rows of an embedding matrix built against this framework's `Vocabulary`. The result is a model that is quietly wrong, which is why the ids are imported from the vocabulary layer rather than repeated as literals here: there is exactly one definition, and it cannot drift.

`SentencePieceTokenizer.to_vocabulary` depends on the same pinning. It can append pieces 4 onwards in model order and reproduce the id space exactly, precisely because pieces 0–3 are already pad/unk/bos/eos.

### The trainer distinguishes the two opposite `vocab_size` failures

SentencePiece reports configuration problems as a bare `RuntimeError` with a message. Two of them are the ones users actually hit, and they are **opposite problems**:

| Markers | Meaning | Advice given |
|---|---|---|
| `_VOCAB_TOO_HIGH_MARKERS`: `"vocabulary size too high"`, `"please set it to a value"` | `vocab_size` exceeds the distinct pieces the corpus can support | reduce `vocab_size`, or supply more text |
| `_VOCAB_TOO_LOW_MARKERS`: `"smaller than required_chars"`, `"increase vocab_size"` | `vocab_size` cannot cover the corpus's character inventory | raise `vocab_size`, or lower `character_coverage` |

A single generic "vocab_size is wrong" message would be actively harmful. The first case is the common one on small corpora, so a generic message would be read as "shrink it" — and a multilingual corpus hits the second case easily, because a corpus spanning Devanagari, Han and Latin has a character inventory of thousands before a single subword is learned. Telling that user to shrink `vocab_size` sends them in exactly the wrong direction, and the error will recur with a different message they will read the same way. Matching is case-insensitive on substrings, with an unmatched `RuntimeError` falling through to a generic `ConfigurationError` that still reports the requested size, the sentence count and the underlying message.

### Artefacts are published only after success

Training runs into a `tempfile.TemporaryDirectory`, and `_publish` moves the `.model` and `.vocab` files out through `atomic_write_path` only once training has returned. An interrupted run therefore never leaves a half-written model where the next run will load it. `_publish` also uses string concatenation rather than `Path.with_suffix`, which would replace an existing dotted component of a prefix such as `tokenizer.v2`.

### `trainer_for` exists because a public class took a non-public argument

`SentencePieceTrainerAdapter` is named in the published surface. The only type its constructor accepted was `TokenizerConfig`, which lives in `config` — deliberately internal, and free to change in a patch release. A consumer pinning this package could therefore reach the public class only by importing a private type, which means a change this repo would call internal was able to break their build without being breaking on this repo's own terms. That is a hole rather than a design, and a sibling repo found it.

`trainer_for(vocab_size=..., model_type=..., normalizers=...)` closes it, in the same shape as `corpus.reader_for` — the pattern that had already solved the same problem on the corpus side. Three properties make its signature reachable with no private import: `TokenizerModel` is a `StrEnum`, so `model_type: str` accepts both the string and the member; `SpecialTokenSet` is already public; and everything else is a builtin or a `Mapping` of them. `tests/test_public_api.py` names the function, and `TestTrainerFor` asserts that no annotation in its signature mentions `TokenizerConfig` or `config` — the guarantee is the point, so it is tested rather than reviewed.

Every argument defaults to `None` meaning *keep the framework default*, rather than restating the defaults in the signature. Two copies of a default drift, and the copy in a signature drifts silently; there is exactly one place the values live, and the test compares against `TokenizerConfig()` rather than against a literal.

The read-back path is closed the same way: `adapter.vocab_size` and `adapter.model_type` return an `int` and a `str`, so inspecting a trainer no longer requires touching `.config` either. `.config` remains, and its docstring now says what it returns and what to use instead.

### `Encoding` carries pieces and spans, and returns new instances

`Encoding` deliberately carries the surface pieces and their spans alongside the ids. Without them a prediction over token positions cannot be mapped back onto the source characters.

Two related choices in the same class:

- **`spans` is `None` rather than fabricated** when the tokenizer cannot supply them. `SentencePieceTokenizer.encode` returns no spans, because SentencePiece pieces carry a synthetic word-boundary marker and do not map onto input characters one for one — and the text it encodes has been normalised first, so even a one-for-one mapping would not index the caller's string. `pad_to` likewise drops spans entirely: a padding token has no position in the source, and inventing one would let a downstream offset lookup return a plausible but wrong answer.
- **`truncate` and `pad_to` return new instances** rather than mutating, because the same encoding is frequently reused across batches with different length budgets.

`pad_to` always produces an attention mask, since padding is exactly the situation that makes one necessary, and it raises rather than silently truncating when asked to pad to a length shorter than the encoding.

`WordTokenizer.encode` documents that its spans index the **normalised** text, not the input — normalisation can change character counts, and the normalised form is what the pre-tokeniser actually saw.

### Two tokenizers, for two real situations

`SentencePieceTokenizer` is the production path: subword decomposition handles unseen words and needs no language-specific rules, which is what makes one shared model viable across scripts. It is constructible without a model so it can be built from configuration before training exists, and every operation raises `NotFittedError` until `load_model` has run. It carries a normaliser chain but no pre-tokeniser, for the reason given above.

`WordTokenizer` is dependency-free, trains in a single pass, and produces directly readable tokens. It is what the tests run against and the right choice where word identity matters more than subword units. It persists its component *specifications* rather than instances, so the exact configuration reloads.

`WordTokenizer.save` resolves `self.vocabulary` before touching the filesystem, so an unfitted tokenizer raises rather than leaving an empty directory behind. `SentencePieceTokenizer.load` falls back to the single `.model` file present when the canonical filename is absent, so a model trained under a custom `model_prefix` loads without renaming.

## Usage

```python
from multilingual_embedding.tokenizer import (
    LowercaseNormalizer, NormalizerPipeline,
    ScriptAwarePreTokenizer, WhitespacePreTokenizer, WordTokenizer,
)

pipeline = NormalizerPipeline.from_config([{"type": "nfkc"}, "lowercase", "whitespace"])
print("pipeline:", pipeline)
print("normalize:", pipeline.normalize("  ＨＥＬＬＯ　STRASSE  "))
print("casefold vs lower:", "STRASSE".casefold(), LowercaseNormalizer().normalize("ΣΟΦΟΣ"))

pretokenizer = ScriptAwarePreTokenizer()
print("script-aware:", [(t.text, t.span.start, t.span.end)
                        for t in pretokenizer.pre_tokenize("Hello 世界")])
print("whitespace:", [t.text for t in WhitespacePreTokenizer().pre_tokenize("こんにちは世界です")])
print("script-aware ja:", [t.text for t in pretokenizer.pre_tokenize("こんにちは世界です")])

tokenizer = WordTokenizer(pretokenizer={"type": "script"}, min_count=1)
tokenizer.train(["Hello 世界", "hello world"])
encoding = tokenizer.encode("Hello 世界")
print("vocabulary_size:", tokenizer.vocabulary_size)
print("ids:", encoding.ids)
print("tokens:", encoding.tokens)
print("decode:", tokenizer.decode(encoding.ids))
padded = encoding.pad_to(6, pad_id=0)
print("padded ids:", padded.ids)
print("attention_mask:", padded.attention_mask, "spans:", padded.spans)
```

Output:

```
pipeline: NormalizerPipeline(steps=['NFKCNormalizer', 'LowercaseNormalizer', 'WhitespaceNormalizer'])
normalize: hello strasse
casefold vs lower: strasse σοφοσ
script-aware: [('Hello', 0, 5), ('世', 6, 7), ('界', 7, 8)]
whitespace: ['こんにちは世界です']
script-aware ja: ['こ', 'ん', 'に', 'ち', 'は', '世', '界', 'で', 'す']
vocabulary_size: 9
ids: [4, 7, 8]
tokens: ['Hello', '世', '界']
decode: Hello 世 界
padded ids: [4, 7, 8, 0, 0, 0]
attention_mask: [1, 1, 1, 0, 0, 0] spans: None
```

The two middle lines are the argument for the script-aware pre-tokeniser stated as a single comparison: the whitespace pre-tokeniser returns the entire Japanese sentence as one token, while the script-aware one returns nine. Note also that `ΣΟΦΟΣ` folds to `σοφοσ` — final sigma unified — and that the padded encoding's `spans` is `None` rather than a list containing invented positions.

`decode` returns `Hello 世 界` with spaces between the ideographs. That is inherent to word-level tokenization, not a defect: whitespace is the only separator available at that level, which is one more reason `SentencePieceTokenizer` is the production path.

## Dependencies
May import from `common` (`Span`, constants), `core` (exceptions, `Registry`, `build_from_config`, logging), `utils` (`filesystem`, `io`, `validation`), `config` (`TokenizerConfig`), `corpus` (`Script`, `is_whitespace_delimited`, `script_of_character`, `Token`) and `vocabulary` (`Vocabulary`, `VocabularyBuilder`, `SpecialTokenSet` and the four id constants).

Must **not** import `embedding`, `evaluation` or `pipelines`. Enforced by `tests/test_architecture.py`.

Consumed by `pipelines/training.py`, `pipelines/search.py` and `cli.py`.

## Tests
Package total: **196 tests** across `tests/tokenizer`.

- `tests/tokenizer/test_pretokenizer.py` — 57 tests
- `tests/tokenizer/test_normalizer.py` — 44 tests
- `tests/tokenizer/test_tokenizer.py` — 40 tests
- `tests/tokenizer/test_encoding.py` — 23 tests
- `tests/tokenizer/test_trainer.py` — 32 tests
- `tests/tokenizer/conftest.py` — shared fixtures, no tests of its own

The full pipeline is additionally exercised by `tests/integration/test_end_to_end.py`, and the tokenizer's own quality metrics live in `evaluation/tokenizer_eval.py` with tests in `tests/evaluation/test_evaluators.py`.
