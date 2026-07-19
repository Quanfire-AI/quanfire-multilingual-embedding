# tests/utils

> Tests for [`multilingual_embedding.utils`](../../src/multilingual_embedding/utils/README.md) — validation, hashing, filesystem, I/O, serialization.

**61 tests.** Run with `pytest tests/utils -q`.

## Files

| File | Covers |
|---|---|
| `test_validation.py` | Each precondition helper accepts valid input, rejects invalid input, and attaches structured context |
| `test_hashing.py` | Stability across calls, Unicode normalisation, digest sizes, file and object hashing, collision resistance |
| `test_filesystem.py` | Directory creation, existence checks, sorted recursive listing, atomic write semantics |
| `test_io.py` | Text/JSON/JSON Lines/YAML round trips, gzip transparency, error reporting, safe YAML loading |
| `test_serialization.py` | Primitive reduction for every supported type, dataclass round trips, unknown-field rejection |

## What matters here

**Atomic writes must survive failure.** `test_atomic_write_leaves_original_on_failure`
writes a file, then interrupts a rewrite, and asserts the original content is intact —
because a partially written model is worse than no model, the next run loads it happily.
A companion test asserts no temporary file is left behind.

**Hashing must be stable across processes and Unicode forms.** Python's builtin `hash()`
is randomised per interpreter run, so identifiers would change between runs. And the
same characters in NFC and NFD must hash identically — a real concern for Devanagari and
Hangul, where one document arriving in two normalisation forms would otherwise become
two corpus entries.

**`hash_iterable` must length-prefix.** `test_hash_iterable_avoids_boundary_collision`
asserts `["ab", "c"]` and `["a", "bc"]` differ, which they would not under naive
concatenation.

**YAML must refuse to build objects.** `test_yaml_rejects_arbitrary_objects` feeds a
`!!python/object/apply` tag and asserts it raises. A configuration file must not be able
to decide which code runs.

**Enum ordering in `to_primitive` is load-bearing.** The framework's enums subclass
`str`, so a scalar check placed before the enum check returns the member itself rather
than its value — which YAML then cannot represent. This was a real bug; the round-trip
tests guard it.

**Unknown fields must be rejected, not dropped.** A renamed configuration key silently
losing its value is precisely the failure that makes a persisted config untrustworthy.
