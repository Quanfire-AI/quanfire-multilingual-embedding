# common

> The vocabulary layer: shared constants, enumerations, type aliases and the `Span` value object that every other package speaks in terms of.

## Purpose

Packages higher in the stack need to agree on a handful of primitives — what a token identifier is, what the reserved special tokens are called, what a character range means. If each package defined its own, the definitions would drift and conversions between layers would become lossy. This package holds those definitions and nothing else: it has no behaviour beyond `Span`'s arithmetic, no I/O, and no dependency on any other framework package.

## Modules

| Module | Responsibility |
|---|---|
| `constants.py` | Framework-wide immutable values: `DEFAULT_ENCODING`, `DEFAULT_RANDOM_SEED`, `DEFAULT_BATCH_SIZE`, `DEFAULT_VOCAB_SIZE`, `DEFAULT_CHARACTER_COVERAGE`. |
| `enums.py` | `TokenizerModel` (`UNIGRAM`, `BPE`, `WORD`, `CHAR`) and `SpecialToken` (`PAD`, `UNK`, `BOS`, `EOS`), both `StrEnum`. |
| `span.py` | `Span`, a frozen slots dataclass modelling a half-open character interval, with `length`, `slice`, `contains`, `overlaps`, `touches`, `merge` and `shift`. |
| `types.py` | Type aliases for the text hierarchy (`SentenceText`, `ParagraphText`, `DocumentText`, `CorpusText`) and token identifiers (`TokenId`, `TokenIds`). |
| `version.py` | `__version__`, the single source of the framework version string. |

## Key design decisions

### `Span` is half-open, `[start, end)`

`Span(0, 5)` covers indices 0 to 4; index 5 belongs to the next span. Two consequences follow, and both are why the convention was chosen.

Adjacent spans touch without overlapping. `Span(0, 5)` and `Span(5, 10)` partition `"Hello world"[0:10]` between them with no shared index, so `overlaps` returns `False` while `touches` returns `True`. Under a closed interval `[start, end]` the two would have to be written `Span(0, 4)` and `Span(5, 9)`, and every adjacency test would need an off-by-one adjustment that is easy to get wrong in one place out of ten.

Length is `end - start`, with no `+ 1`. This matters because `Span.slice` delegates directly to Python's own slicing, which is half-open: `text[self.start : self.end]`. A closed-interval `Span` would need to add one at every slicing site, and the framework would carry two conventions — its own and Python's — that must be reconciled by hand each time a span crosses a package boundary.

Validation happens in `__post_init__`: `start` must be non-negative and `end` must be at least `start`. Zero-length spans are therefore legal, which is deliberate — an empty match at a position is a meaningful thing for a tokenizer to report.

`merge` refuses spans that neither overlap nor touch, raising `ValueError`. Merging a gap would silently invent coverage over characters neither span described.

### Enums are `StrEnum`

`TokenizerModel.UNIGRAM == "unigram"` is `True`, so a value read from a YAML file compares equal to the enum member without an explicit conversion, and the member serialises to a readable string rather than an opaque integer. The cost is real and appears elsewhere in the framework: because these enums subclass `str`, any code that dispatches on type must check `Enum` before checking for scalars. `utils.serialization.to_primitive` does exactly that, and the ordering there is load-bearing.

### Type aliases use PEP 695 `type` statements

`type ParagraphText = list[SentenceText]` creates a lazily evaluated alias. The text hierarchy is defined bottom-up — sentence, paragraph, document, corpus — and each level is transparently the list type it aliases, so no runtime wrapper is involved and no conversion is needed at layer boundaries.

`TokenId` is the exception: it is a `NewType`, not an alias. A token identifier is an `int` at runtime but must not be interchangeable with an arbitrary `int` under the type checker, because passing a vocabulary index where a count was expected is a bug a plain alias would not catch.

### Constants here are true constants

`DEFAULT_VOCAB_SIZE` and friends are named starting points, not settings. Anything a user should be able to change for a run belongs in `config`, which is where these values are consumed — `config.base` imports them as dataclass field defaults. Keeping the two apart means a tunable value has exactly one home and there is no ambiguity about which of two definitions wins.

## Usage

```python
from multilingual_embedding.common.span import Span
from multilingual_embedding.common.enums import SpecialToken, TokenizerModel
from multilingual_embedding.common.constants import DEFAULT_VOCAB_SIZE

text = "Hello world"
hello, world = Span(0, 5), Span(6, 11)

print(hello.slice(text), world.slice(text))
print("length:", hello.length)
print("contains(4):", hello.contains(4), "contains(5):", hello.contains(5))
print("overlaps:", hello.overlaps(Span(5, 11)))
print("touches:", hello.touches(Span(5, 11)))
print("merge:", hello.merge(Span(5, 11)))
print("shift:", world.shift(-6))

print("model:", TokenizerModel.UNIGRAM, TokenizerModel.UNIGRAM == "unigram")
print("pad:", SpecialToken.PAD.value)
print("default vocab size:", DEFAULT_VOCAB_SIZE)

try:
    Span(5, 2)
except ValueError as error:
    print("error:", error)
```

Output:

```
Hello world
length: 5
contains(4): True contains(5): False
overlaps: False
touches: True
merge: Span(start=0, end=11)
shift: Span(start=0, end=5)
model: unigram True
pad: <pad>
default vocab size: 32000
error: end must be >= start
```

Note `overlaps` and `touches` disagreeing on the same pair: that is the half-open convention doing its job.

## Dependencies

`common` sits at the bottom of the layer order and **imports nothing from inside the framework** — only the standard library. Every other package may import it.

`tests/test_architecture.py::test_foundation_layers_have_no_internal_dependencies` asserts this directly, by parsing the source and requiring that the set of framework imports made from `common` is empty.

`common/__init__.py` re-exports everything public — the constants, both enums, the type aliases, `Span` and `__version__` — so `from multilingual_embedding.common import Span` and the fully qualified `multilingual_embedding.common.span` both work.

## Tests

Tests live in `tests/common/`, 20 in total:

| File | Tests | Coverage |
|---|---|---|
| `tests/common/test_span.py` | 7 | One test per operation: `length`, `slice`, `contains`, `overlap`, `touch`, `merge` and `shift`. The half-open boundary behaviour is asserted inside `test_contains`, `test_overlap` and `test_touch`. |
| `tests/common/test_constants.py` | 5 | Constant values and their types. |
| `tests/common/test_types.py` | 4 | Alias definitions and `TokenId` behaviour. |
| `tests/common/test_enums.py` | 2 | Enum members and their string equality. |
| `tests/common/test_version.py` | 2 | Version string presence and format. |

Run them with `.venv/bin/python -m pytest tests/common -q`.
