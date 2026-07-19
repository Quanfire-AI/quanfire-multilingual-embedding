# vocabulary

> The bidirectional token/id mapping that is the contract between the tokenizer and the embedding matrix.

## Purpose
The tokenizer emits integer ids; the embedding matrix is indexed by them. Those two components must agree on every id, not merely on the token set, because an id that shifts between training and inference does not fail — it indexes the wrong row and produces a model that is quietly wrong. This package owns that mapping, the fixed reserved ids at the bottom of it, and the deterministic ordering that makes it reproducible. It is its own layer, below the tokenizer, because the embedding layer needs it without needing a tokenizer.

## Modules
| Module | Responsibility |
|---|---|
| `special_tokens.py` | The fixed reserved ids `PAD_ID`/`UNK_ID`/`BOS_ID`/`EOS_ID`, their order, and `SpecialTokenSet` holding their surface forms. |
| `vocabulary.py` | `Vocabulary` — lookup in both directions, frequencies, coverage, freezing, and JSON persistence. |
| `builder.py` | `VocabularyBuilder` — streaming token counting with a bounded table, producing a `Vocabulary`. |

## Key design decisions

### Special token ids are fixed, and not configurable

```python
SPECIAL_TOKEN_ORDER = (SpecialToken.PAD, SpecialToken.UNK, SpecialToken.BOS, SpecialToken.EOS)

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
```

The *surface forms* are configurable through `SpecialTokenSet` — a model trained with SentencePiece may use different strings — but the ids are not. There is no constructor argument, no config key and no environment variable that moves them.

The reason is that these ids are baked into every trained embedding matrix. Changing the order would not raise anything anywhere; it would silently invalidate every previously trained model, because row 0 of the matrix would now be indexed by a different token. That is the worst class of defect this framework can produce, and the mitigation is to remove the knob rather than to document it.

The specific assignments earn their places:

- **`pad = 0`** means a zero-filled array is already a valid padded batch. Padding never has to be written explicitly, and the common bug of forgetting to fill a buffer produces correct padding instead of a batch of whatever token happened to be id 0.
- **`unk = 1`** means an unknown lookup has a defined answer even against a vocabulary built without ever encountering an out-of-vocabulary token, because the id is reserved at construction rather than allocated on first use.

`SpecialTokenSet` is `frozen=True` and exposes `as_tuple()` in id order, `as_mapping()` (surface form to id), and `count`. Every place that needs to know how many ids are reserved reads `count` rather than the literal `4`, so `Vocabulary.most_common` and `decode(skip_special=True)` stay correct against a set of a different size. The one place the literal appears is `Vocabulary.from_dict`, which validates that a persisted payload defines exactly four — a deliberate assertion about the on-disk format version, not an assumption about the runtime object.

### Ordering is deterministic

`Vocabulary.from_counter` sorts candidates with `key=lambda entry: (-entry[1], entry[0])` — descending frequency, ties broken on the token string. The sort key is not "descending frequency" alone, and the difference matters: a `Counter` built from a corpus has an iteration order that depends on insertion order and hash seeding, so equal-frequency tokens would otherwise land in arbitrary relative positions and the same corpus would yield a different id space on every run. With the tie-break, two runs over the same corpus produce a byte-identical vocabulary. That is what makes a training run reproducible at all, and what makes a saved vocabulary comparable against a re-derived one.

`most_common` is then a slice rather than a sort, because the ordering is already an invariant of construction.

`min_count` filtering happens before the cap, not after. Rare tokens receive too few gradient updates to acquire a useful vector, and each one costs a full embedding row — the filter is about model quality, not only memory.

### `id_of` never raises; `token_of` does

```python
def id_of(self, token: str) -> int:
    return self._ids.get(token, UNK_ID)
```

An out-of-vocabulary token is an expected condition at inference time. Every real deployment sees words the corpus did not contain; that is what `<unk>` is for. Raising here would force every call site into a try/except that can only convert the exception back into `UNK_ID`.

```python
def token_of(self, token_id: int) -> str:
    if not 0 <= token_id < len(self._tokens):
        raise ValidationError("Token id is out of range", token_id=token_id, size=len(self._tokens))
```

An out-of-range id is the opposite kind of event. It cannot arise from unusual input — it means the caller's model and vocabulary disagree about the size of the id space, which is a real defect, almost always a model paired with the wrong vocabulary file. Returning `<unk>` here would be the more "forgiving" choice and would let a mispaired model run to completion producing garbage. The asymmetry between these two methods is deliberate and is the package's clearest statement about which failures are expected and which are bugs.

### `freeze()` seals the mapping after training

`freeze()` sets a flag; `add()` raises `ValidationError` when it is set. It is called once a model has been trained against the vocabulary. Adding a token afterwards would allocate an id one past the end of the embedding matrix — the next lookup at that id is an out-of-bounds index into the weights, or worse, silently valid if the matrix was over-allocated. `freeze()` converts a subtle numerical bug into a clear error at the point of the mistake. `SentencePieceTokenizer.to_vocabulary` returns a frozen vocabulary for exactly this reason: a vocabulary mirroring a trained model's piece table must never grow.

`freeze()` returns `self`, so it chains.

### `VocabularyBuilder` bounds the counting table and says when it did

Counting every token of a large corpus in a plain dictionary is the usual way vocabulary building runs out of memory. Word frequencies are Zipfian, so the table fills with singletons that `min_count` will discard anyway.

The builder caps distinct tracked tokens at `max_tracked_tokens=5_000_000` and, when exceeded, prunes the entries seen exactly once — overwhelmingly the ones destined for removal. Pruning is lossy: a token that would have crossed `min_count` later can be undercounted, because its earlier occurrences were discarded. Two mitigations follow from admitting that rather than hiding it. The cap defaults high enough that ordinary corpora never reach it. And when pruning has occurred, `build()` logs a warning and `pruned_count` stays non-zero, documented as meaning counts are now approximate and that `max_tracked_tokens` should be raised if exactness matters.

`_prune_singletons` also declines to act when there are no singletons, leaving the table over its cap rather than discarding tokens that have real evidence behind them. Honouring the cap is less important than not throwing away counted data.

### `coverage()` is the number worth looking at

`coverage(tokens)` returns the fraction of a token stream the vocabulary can represent. Its complement is the unknown rate, which is the single most useful figure for judging whether a vocabulary fits a corpus — more informative than size, which says nothing about fit.

### Persistence carries a format version

`to_dict` writes `format_version`, and `from_dict` refuses any other value rather than attempting a best-effort parse. It additionally validates that tokens and frequencies are the same length, that exactly four special tokens are declared, and that the token list *begins* with those special tokens — a file whose head does not match would produce a vocabulary with the reserved ids pointing somewhere else, which is precisely the failure the fixed ids exist to prevent.

## Usage

```python
from multilingual_embedding.core.exceptions import ValidationError
from multilingual_embedding.vocabulary import (
    PAD_ID, UNK_ID, BOS_ID, EOS_ID, Vocabulary, VocabularyBuilder,
)

builder = VocabularyBuilder(min_count=1)
builder.add_all([["नमस्ते", "दुनिया"], ["नमस्ते", "world"], ["hello", "world"]])
print("distinct:", builder.distinct_tokens, "total:", builder.total_tokens,
      "pruned:", builder.pruned_count)

vocabulary = builder.build()
print("size:", len(vocabulary))
print("tokens:", vocabulary.tokens())
print("special ids:", PAD_ID, UNK_ID, BOS_ID, EOS_ID)
print("most_common:", vocabulary.most_common(3))

print("id_of known:", vocabulary.id_of("world"))
print("id_of unseen:", vocabulary.id_of("абв"))
print("encode:", vocabulary.encode(["hello", "абв", "world"]))
print("coverage:", vocabulary.coverage(["hello", "абв", "world"]))

try:
    vocabulary.token_of(999)
except ValidationError as error:
    print("ValidationError:", error)

vocabulary.freeze()
try:
    vocabulary.add("late")
except ValidationError as error:
    print("ValidationError:", error)
```

Output:

```
distinct: 4 total: 6 pruned: 0
size: 8
tokens: ['<pad>', '<unk>', '<bos>', '<eos>', 'world', 'नमस्ते', 'hello', 'दुनिया']
special ids: 0 1 2 3
most_common: [('world', 2), ('नमस्ते', 2), ('hello', 1)]
id_of known: 4
id_of unseen: 1
encode: [6, 1, 4]
coverage: 0.6666666666666666
ValidationError: Token id is out of range (size=8, token_id=999)
ValidationError: Cannot modify a frozen vocabulary (token='late')
```

The tie-break is visible: `world` and `नमस्ते` both occur twice, and `world` takes the lower id because `"world" < "नमस्ते"` as strings. Every rerun produces that same order. The unseen Cyrillic token encodes to `1`, `UNK_ID`, without an exception — while an id past the end of the vocabulary raises.

## Dependencies
May import from `common` (`SpecialToken`), `core` (`ValidationError`, logging) and `utils` (`read_json`, `write_json`). It imports nothing from `corpus`, and must not: a vocabulary is a mapping of strings to integers and has no need of the document tree.

Must **not** import `tokenizer`, `embedding`, `evaluation` or `pipelines`. Enforced by `tests/test_architecture.py`.

Consumed by `tokenizer/tokenizer.py` and `tokenizer/trainer.py` (which pins the SentencePiece special ids to the constants here), and by `embedding/matrix.py` and `embedding/word2vec.py`.

## Tests
- `tests/vocabulary/test_vocabulary.py` — **36 tests**, covering both `Vocabulary` and `VocabularyBuilder`.

The pairing between these ids and the tokenizer is separately covered by `tests/tokenizer/test_trainer.py` (15 tests) and `tests/tokenizer/test_tokenizer.py` (33 tests), and end to end by `tests/integration/test_end_to_end.py`.
