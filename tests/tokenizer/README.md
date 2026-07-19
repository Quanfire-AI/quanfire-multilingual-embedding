# tests/tokenizer

> Tests for [`multilingual_embedding.tokenizer`](../../src/multilingual_embedding/tokenizer/README.md) — normalizers, pre-tokenizers, SentencePiece.

**172 tests.** Run with `pytest tests/tokenizer -q`.

## Files

| File | Covers |
|---|---|
| `conftest.py` | Shared fixtures, including a SentencePiece model trained once per module |
| `test_normalizer.py` | Each normalizer and the composed pipeline |
| `test_pretokenizer.py` | Each pre-tokenizer, with span correctness across scripts |
| `test_encoding.py` | The `Encoding` container: length invariants, truncation, padding |
| `test_tokenizer.py` | SentencePiece and word tokenizers, encode/decode round trips, persistence |
| `test_trainer.py` | Training, artefact publication, and the two opposite `vocab_size` failures |

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

## Speed

The SentencePiece model is trained once per module via a fixture rather than per test,
which keeps the whole group under two seconds despite exercising real model training.
