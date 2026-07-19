# utils

> Cross-cutting helpers used everywhere and owned by no domain: precondition checks, stable content hashing, filesystem safety, transparent I/O and primitive serialisation.

## Purpose

Several problems recur in every domain package — writing a file without risking a truncated result, hashing text so the identifier is the same next week, reducing an object to something YAML can hold, checking an argument and reporting it usefully. Each has a correct answer that is easy to get subtly wrong, and a wrong answer that only shows up much later. Solving them once here means the domain layers above never reimplement them, and never reimplement them differently.

## Modules

| Module | Responsibility |
|---|---|
| `filesystem.py` | Path handling: `ensure_directory`, `require_file`, `require_directory`, deterministic `iter_files`, the `atomic_write_path` context manager, and `human_readable_size`. |
| `hashing.py` | Reproducible content identifiers: `hash_text`, `hash_bytes`, `hash_file`, `hash_object`, `hash_iterable`, and `DEFAULT_DIGEST_SIZE`. |
| `io.py` | Text, JSON, JSON Lines and YAML reading and writing, all gzip-aware and all atomic on write. `open_text` and `count_lines` sit alongside the `read_*` / `write_*` pairs. |
| `serialization.py` | `to_primitive` and `from_primitive`, the conversion between framework objects and plain data, plus the `is_dataclass_type` guard. |
| `validation.py` | The `require_*` precondition helpers, each returning the validated value and raising `ValidationError` with structured context. |

## Key design decisions

### `atomic_write_path`: a partial file is worse than no file

If a training run is killed midway through writing a model, a truncated file is left on disk. That is worse than nothing, because the next run finds a file where it expects one and loads it — producing garbage results rather than an obvious "not found" error.

`atomic_write_path` yields a temporary sibling created by `tempfile.mkstemp` in the target's own directory, and calls `os.replace` only after the block completes. The rename is atomic on POSIX filesystems, so a concurrent reader observes either the complete old contents or the complete new ones, never a half-written state. On any exception the temporary is unlinked and the original is left untouched. The sibling directory matters: a temporary in `/tmp` could sit on a different filesystem, where the rename would degrade into a copy and lose atomicity.

Every writer in `io.py` routes through it, so the guarantee is not something callers have to remember.

### Hashing normalises to NFC, and uses SHA-256

Python's builtin `hash()` is randomised per interpreter run for `str`, so an identifier derived from it changes between processes. Anything cached, compared across runs or written to a manifest must therefore use a cryptographic digest instead. `hash_text` takes SHA-256 over UTF-8 bytes and truncates to `DEFAULT_DIGEST_SIZE` (16) hex characters — short enough to read in a log, ample for identifying documents within a corpus.

Before hashing, text is normalised to Unicode NFC. The same visible characters can be encoded as a precomposed code point or as a base plus combining marks, and the two byte sequences are not equal. For the scripts this framework targets this is a live concern: `"한국어"` is 3 code points in NFC and 8 in NFD. Without normalisation the same document from two sources would receive two identifiers and be counted twice.

### `hash_iterable` length-prefixes its elements

Concatenating elements before hashing makes `["ab", "c"]` and `["a", "bc"]` produce identical bytes and therefore identical digests, even though they are different token sequences. `hash_iterable` writes the byte length, a colon, then the encoded element, which makes the boundaries unambiguous and the collision impossible. It is order sensitive by construction, which is what a token sequence requires — unlike `hash_object`, which sorts mapping keys so that two equal configurations hash identically regardless of insertion order.

### I/O handles gzip by suffix and loads YAML safely

`open_text` checks for a `.gz` suffix and routes through `gzip.open` in text mode. Corpora are routinely distributed compressed, and making that invisible at the call site means no reader in the framework branches on compression. The writers apply the same rule.

`read_yaml` uses `yaml.safe_load`, never `yaml.load`. The unsafe loader can instantiate arbitrary Python objects named in the document, which would let a configuration file — the sort of thing that gets copied between machines and pasted from issue trackers — execute code merely by being read.

`write_json` and `write_yaml` both disable ASCII escaping, so a Hindi or Arabic corpus file stays readable to a human inspecting it rather than becoming a wall of `\uXXXX`.

### `to_primitive` checks `Enum` before scalars

This ordering is load-bearing and was an actual bug. The framework's enums are `StrEnum`, so they subclass `str` — and the scalar branch tests `isinstance(value, (str, bool, int, float))`. Put the scalar check first and an enum matches it, returning the member itself rather than its `value`. The result is an object YAML cannot represent, and the failure surfaces at dump time rather than at the line that produced it. The `Enum` check therefore comes first, and there is a comment in the source saying so.

Unknown types raise `SerializationError` rather than being stringified. A silent `str()` fallback would produce a file that writes cleanly and cannot be read back — the gap discovered on load, long after the run that created it.

### `from_primitive` rejects unknown fields

Deserialisation compares the incoming keys against the target's fields and raises `SerializationError` on any extra, listing both the unknown keys and the accepted ones. Ignoring them is the tempting default and the wrong one: a renamed or misspelt configuration key would then be dropped silently, the field would fall back to its default, and the run would complete with settings nobody chose. Missing keys *are* allowed, falling back to field defaults — that is what makes a partial config file legal.

Deserialisation is also deliberately narrow. `from_primitive` populates a known dataclass type given as an argument; it never reads a type name out of the data and imports it. A corpus file does not get to decide which code runs.

### Validation helpers return the value they checked

`require_positive(window, name="window")` returns `window`, so the check inlines into an assignment: `self.window = require_positive(window, name="window")`. The alternative — a check on one line and the assignment on the next — makes it possible to add a field and forget the check, with nothing to notice the omission. Every helper raises `ValidationError` carrying the name and the offending value as structured context.

## Usage

```python
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from multilingual_embedding.utils import (
    atomic_write_path, hash_text, hash_iterable, hash_object,
    human_readable_size, read_jsonl, write_jsonl, read_yaml, write_yaml,
    from_primitive, to_primitive, require_positive,
)
from multilingual_embedding.common.enums import TokenizerModel
from multilingual_embedding.core.exceptions import SerializationError, ValidationError

work = Path("utilsdemo"); work.mkdir(exist_ok=True)

nfc = unicodedata.normalize("NFC", "한국어")
nfd = unicodedata.normalize("NFD", "한국어")
print("distinct code point sequences:", nfc != nfd, len(nfc), len(nfd))
print("one digest:", hash_text(nfc) == hash_text(nfd), hash_text(nfc))

print('["ab","c"] vs ["a","bc"]:', hash_iterable(["ab", "c"]), hash_iterable(["a", "bc"]))
print("order-independent object hash:", hash_object({"a": 1, "b": 2}) == hash_object({"b": 2, "a": 1}))

with atomic_write_path(work / "model.txt") as temporary:
    temporary.write_text("weights", encoding="utf-8")
print("atomic write landed:", (work / "model.txt").read_text(encoding="utf-8"))

try:
    with atomic_write_path(work / "model.txt") as temporary:
        temporary.write_text("half", encoding="utf-8")
        raise RuntimeError("training crashed")
except RuntimeError:
    pass
print("original intact after failure:", (work / "model.txt").read_text(encoding="utf-8"))

print("records written:", write_jsonl(work / "corpus.jsonl.gz", [{"text": "नमस्ते"}, {"text": "hello"}]))
print("read back through gzip:", list(read_jsonl(work / "corpus.jsonl.gz")))

@dataclass(slots=True)
class Settings:
    model_type: TokenizerModel = TokenizerModel.BPE
    vocab_size: int = 8000
    output: Path = Path("artifacts")

print("to_primitive:", to_primitive(Settings()))
write_yaml(work / "settings.yaml", to_primitive(Settings()))
print("round trip:", from_primitive(Settings, read_yaml(work / "settings.yaml")))

try:
    from_primitive(Settings, {"vocab_sizes": 8000})
except SerializationError as error:
    print("unknown field:", error)

try:
    require_positive(-1, name="dimension")
except ValidationError as error:
    print("validation:", error, "| context:", error.context)

print("size:", human_readable_size(1536), human_readable_size(512))
```

Output:

```
distinct code point sequences: True 3 8
one digest: True ea3252281bc3bcec
["ab","c"] vs ["a","bc"]: 430fb1b4ac43316e 5310a58788781ab2
order-independent object hash: True
atomic write landed: weights
original intact after failure: weights
records written: 2
read back through gzip: [{'text': 'नमस्ते'}, {'text': 'hello'}]
to_primitive: {'model_type': 'bpe', 'vocab_size': 8000, 'output': 'artifacts'}
round trip: Settings(model_type=<TokenizerModel.BPE: 'bpe'>, vocab_size=8000, output=PosixPath('artifacts'))
unknown field: Unknown fields for target type (known=['model_type', 'output', 'vocab_size'], target='Settings', unknown=['vocab_sizes'])
validation: dimension must be > 0 (name='dimension', value=-1) | context: {'name': 'dimension', 'value': -1}
size: 1.5 KB 512 B
```

The failed write leaving `"weights"` rather than `"half"` is the atomic guarantee; `to_primitive` emitting `'bpe'` rather than a `TokenizerModel` member is the `Enum`-before-scalar ordering.

## Dependencies

`utils` is the third layer. It may import from `common` and `core` only, and does — `common.constants` for `DEFAULT_ENCODING`, and `core.exceptions` for `ValidationError`, `SerializationError` and `ResourceNotFoundError`. It imports no logging and configures none.

It **must not** import from `config`, `corpus`, `vocabulary`, `tokenizer`, `embedding`, `evaluation` or `pipelines`. All of those may import it, and `config` in particular depends on it heavily — every config validation rule is a `require_*` call, and `to_dict` / `from_dict` are `to_primitive` / `from_primitive`. Adding a domain import here would invert that relationship and create a cycle.

One caveat when importing: `utils/__init__.py` re-exports every public function, but **not** `DEFAULT_DIGEST_SIZE`. Import it from `multilingual_embedding.utils.hashing` if you need it by name.

Two internal dependencies within the package are worth knowing: `io.py` imports from `filesystem.py` for `atomic_write_path` and `require_file`, and `hashing.py` imports `DEFAULT_ENCODING`. `serialization.py` imports numpy lazily inside `_numpy_to_primitive`, so the module stays usable where the array stack is not needed.

`tests/test_architecture.py` enforces the layer rule by parsing every module.

## Tests

Tests live in `tests/utils/`, 61 in total — the largest suite of the four foundation packages, which reflects how much of the framework's correctness rests here:

| File | Tests | Coverage |
|---|---|---|
| `tests/utils/test_hashing.py` | 13 | Stability across calls, sensitivity to input change, Unicode normalisation, digest size handling and its rejection at 0, -1 and 65, `hash_bytes`, `hash_file`, key-order independence of `hash_object`, and both the order sensitivity and the boundary-collision avoidance of `hash_iterable`. |
| `tests/utils/test_filesystem.py` | 12 | Directory creation and idempotency, the `require_*` rejections, `iter_files` sorting, recursion, non-recursion and pattern de-duplication, and all three `atomic_write_path` paths — moving into place, leaving the original on failure, and cleaning up the temporary. |
| `tests/utils/test_io.py` | 12 | Round trips for text, JSON, JSON Lines and YAML, each plain and — for text and JSON Lines — gzipped; Unicode preservation; malformed JSON; the line number in a bad JSON Lines record; blank-line skipping; `safe_load` rejecting arbitrary objects; `count_lines`; and parent directory creation on write. |
| `tests/utils/test_serialization.py` | 12 | `to_primitive` over scalars, enums, datetimes, paths, sets and numpy, and its rejection of unknown types; the dataclass round trip; defaults for missing fields; unknown-field and invalid-enum rejection; non-mapping input; optional `None`; and `is_dataclass_type`. |
| `tests/utils/test_validation.py` | 12 | Each helper's accept and reject paths, including four parametrised `require_non_empty_string` cases, inclusive versus exclusive ranges, and a test asserting the structured context on the raised error. |

Run them with `.venv/bin/python -m pytest tests/utils -q`.
