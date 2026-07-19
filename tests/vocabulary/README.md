# tests/vocabulary

> Tests for [`multilingual_embedding.vocabulary`](../../src/multilingual_embedding/vocabulary/README.md) — the token/id mapping.

**36 tests.** Run with `pytest tests/vocabulary -q`.

## Files

| File | Covers |
|---|---|
| `test_vocabulary.py` | Special token ids, construction and ordering, lookup, mutation, freezing, persistence, streaming builder |

## What matters here

**The special token ids are asserted as literals.** `test_ids_are_fixed` checks
`(pad, unk, bos, eos) == (0, 1, 2, 3)`. These are baked into every trained embedding
matrix, so a change would silently invalidate every existing model rather than fail
loudly. Padding being id 0 is what makes a zero-filled array a valid padded batch.

**Ordering must not depend on input iteration order.**
`test_ordering_is_deterministic` builds a vocabulary from a mapping and from its reverse
and asserts the results are identical. Ties break on the token string, which is what
makes two runs over the same corpus produce a byte-identical vocabulary — the basis of
the framework's reproducibility claim.

**Lookup asymmetry is deliberate and tested.** `id_of` returns the unknown id for an
out-of-vocabulary token and never raises, because OOV is an expected condition at
inference. `token_of` *does* raise for an out-of-range id, because that means the
caller's model and vocabulary disagree, which is a genuine defect rather than a normal
occurrence.

**Freezing must be enforced.** Adding a token after training would create an id with no
corresponding embedding row — a numerical bug that would surface far away as garbage
vectors. `freeze()` turns it into an immediate, clear error.

**Persistence must reject malformed payloads.** Version mismatches, mismatched token and
frequency lengths, and a token list that does not begin with its special tokens are all
rejected, so a corrupt file fails at load rather than producing a subtly wrong mapping.

**The builder must admit when counts are approximate.** `VocabularyBuilder` prunes
singletons when it exceeds its tracking cap, which is lossy; `test_pruning_reports_itself`
asserts `pruned_count` reflects that rather than the loss being silent.
