# tests/common

> Tests for [`multilingual_embedding.common`](../../src/multilingual_embedding/common/README.md) — spans, enums, type aliases, constants.

**21 tests.** Run with `pytest tests/common -q`.

## Files

| File | Covers |
|---|---|
| `test_span.py` | `Span` arithmetic: length, slicing, containment, overlap, touch, merge, shift |
| `test_enums.py` | `TokenizerModel` and `SpecialToken` member values |
| `test_types.py` | The text-hierarchy type aliases hold the shapes they claim |
| `test_constants.py` | Framework constants are present and within sane bounds |
| `test_version.py` | `__version__` is a non-empty string, and the installed distribution reports the same one |

## What matters here

`Span` is the foundation of the whole corpus layer — every node's position is one — so
its arithmetic is tested exhaustively despite being simple code. A bug here would
surface far away, as text that slices to the wrong characters.

The half-open interval convention `[start, end)` is what makes `touches` and `overlaps`
distinguishable: adjacent spans share a boundary value without overlapping. The merge
and touch tests pin that behaviour down.

Constant tests assert ranges rather than exact values (`DEFAULT_VOCAB_SIZE >= 1000`,
`0 < DEFAULT_CHARACTER_COVERAGE <= 1`) so that tuning a default does not break the
suite, while a nonsensical value still does.
