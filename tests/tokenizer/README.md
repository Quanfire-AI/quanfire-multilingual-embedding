# tests/tokenizer

> Tests for [`multilingual_embedding.tokenizer`](../../src/multilingual_embedding/tokenizer/README.md) — normalizers, pre-tokenizers, SentencePiece.

**187 tests.** Run with `pytest tests/tokenizer -q`.

## Files

| File | Covers |
|---|---|
| `conftest.py` | Shared fixtures, including a SentencePiece model trained once per module |
| `test_pretokenizer.py` | Each pre-tokenizer and the registry, with span correctness across scripts |
| `test_normalizer.py` | Each normalizer, the registry, and the composed pipeline |
| `test_tokenizer.py` | SentencePiece and word tokenizers, encode-time normalisation, round trips, persistence |
| `test_encoding.py` | The `Encoding` container: construction, `to_dict`, truncation, padding |
| `test_trainer.py` | Training, staging and artefact publication, the two opposite `vocab_size` failures, normalizer application, the pre-tokenizer inapplicability report |

## What matters here

**Span correctness is asserted for every pre-tokenizer.** Each test checks
`token.text == text[token.span.start:token.span.end]`. Spans are what let a token be
traced back to its exact position in the source, and an off-by-one would be invisible
until something downstream tried to use it.

**Script-aware pre-tokenization is tested behaviourally.** Japanese must produce
per-character tokens while English produces word tokens, because splitting Japanese on
whitespace yields one token per sentence — the single most common way a "multilingual"
tokenizer turns out not to be.

**ZWJ and ZWNJ must survive normalisation.** They are format characters that a naive
whitespace normalizer would strip, but they are meaningful in Devanagari and Arabic and
removing them changes the text. Tested explicitly on Devanagari input.

**Case folding uses `casefold()`, not `lower()`.** Tested against German ß and Greek
final sigma, where the two differ.

**Digit normalisation unifies numeral systems**, so Devanagari and Arabic-Indic digits
map to ASCII rather than fragmenting the vocabulary across scripts.

**Round trips are exact across all five scripts.** `decode(encode(text))` recovers the
input for Latin, Devanagari, Japanese, Arabic and Han text.

**Both `vocab_size` failure directions are tested.** SentencePiece fails when the target
is too high for the corpus *and* when it is smaller than the corpus's required character
set — opposite problems with opposite fixes. A single generic message would tell a user
to shrink the vocabulary when they needed to grow it, so each is translated into a
distinct `ConfigurationError`.

**Normalizers must apply at training *and* encode time.** Applying them only at training
would be worse than not applying them at all: the model learns pieces of normalised text
while encode feeds it raw text, and the mismatch never raises — it surfaces as quietly
degraded results. `TestSentencePieceNormalization` trains on a mixed-case corpus and
asserts both halves.

**An unhonourable setting must be reported, not dropped.** SentencePiece consumes a raw
character stream by design, which is the property that lets one model serve scripts with
no whitespace word boundaries — so there is nowhere to insert a framework pre-tokenizer.
The setting is still legitimate for `WordTokenizer`, so the trainer cannot reject it; the
unacceptable outcome is discarding it in silence. `TestPretokenizerIsReportedAsInapplicable`
asserts a warning is logged, that it names the specific setting being ignored, that a
bare-string spec (`"script"` rather than a mapping) is understood, and that the default
whitespace pre-tokenizer stays silent — re-joining whitespace tokens on spaces is the
identity here, so warning about it would train users to ignore the warning.

**Staging must not reshape the corpus.** SentencePiece reads a line-per-sentence file, so
a sentence containing an embedded newline would silently become two training examples.
`TestStaging` asserts newlines are folded to spaces rather than split on, that blank
sentences are skipped, and that the staging directory does not survive training — a
leftover temporary corpus in the artefact directory would be shipped alongside the model.

## Speed

The SentencePiece model is trained once per module via a fixture rather than per test,
which keeps the whole group around one second despite exercising real model training.
