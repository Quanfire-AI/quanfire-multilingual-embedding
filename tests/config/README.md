# tests/config

> Tests for [`multilingual_embedding.config`](../../src/multilingual_embedding/config/README.md) — typed configuration and loading.

**30 tests.** Run with `pytest tests/config -q`.

## Files

| File | Covers |
|---|---|
| `test_config.py` | Per-section validation, derived directories, dict round trip, deep merge, file loading, environment overrides, `--set` parsing, persistence |

## What matters here

**Validation is the point of this layer.** Each config section validates itself in
`__post_init__` so that a bad setting fails at load rather than an hour into a training
run. The tests assert the specific rules: inverted length bounds, an out-of-range
character coverage, a `min_learning_rate` above `learning_rate`, an unknown tokenizer
model, an empty experiment name.

**Merging must be deep.** `test_merge_is_deep` asserts that overriding
`embedding.dimension` leaves `embedding.window` alone. A shallow merge would silently
reset every sibling field to its default — the kind of bug that produces a model that
trains fine and is quietly wrong.

**Merging must revalidate.** An override is just as capable of being invalid as a file,
so `merged()` runs the same checks rather than trusting its caller.

**Environment values must arrive typed.** `QFME_EMBEDDING__DIMENSION=64` has to become
the integer `64`, not the string `"64"`, or it fails validation for the wrong reason.
The double-underscore nesting convention is tested alongside it.

**Round tripping must be lossless.** `to_dict` → `from_dict` → `to_dict` compares equal.
This is what makes the resolved config persisted next to a model trustworthy as a record
of what produced it; a lossy round trip would make that record a lie.
