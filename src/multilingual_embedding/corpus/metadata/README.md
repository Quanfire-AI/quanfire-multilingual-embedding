# corpus.metadata

> Descriptive records attached to every corpus node, carrying identity, language, provenance and level-specific annotations.

## Purpose
Every node in the corpus tree needs somewhere to record what it is and where it came from, but the useful fields differ by level: a document has a licence and an author, a sentence has a language confidence, a token has a lemma. This package holds one metadata class per level, plus the `BaseMetadata` record of fields that every level shares. Keeping metadata in its own package rather than inline on the node classes means the node classes stay concerned with text and spans, and metadata can be extended without touching the tree.

## Modules
| Module | Responsibility |
|---|---|
| `base.py` | `BaseMetadata` — the identity, language, script, provenance and timestamp fields shared by every level. |
| `corpus.py` | `CorpusMetadata` — dataset name, version and description. |
| `document.py` | `DocumentMetadata` — title, author, URL and licence. |
| `paragraph.py` | `ParagraphMetadata` — zero-based index within the parent document. |
| `sentence.py` | `SentenceMetadata` — sentiment and language confidence. |
| `token.py` | `TokenMetadata` — lemma, part of speech, corpus frequency and stopword flag. |

## Key design decisions

**Composition, not inheritance.** `DocumentMetadata`, `SentenceMetadata` and the rest do *not* subclass `BaseMetadata`. Each holds one under a field named `base`:

```python
@dataclass(slots=True)
class DocumentMetadata:
    base: BaseMetadata = field(default_factory=BaseMetadata)
    title: str | None = None
    ...
```

Inheritance was the obvious alternative and would have given `metadata.language` directly instead of `metadata.base.language`. It was rejected for two reasons. First, these are all `@dataclass(slots=True)`, and inheriting from a slotted dataclass while adding fields with defaults constrains field ordering across the hierarchy in ways that surface as confusing `TypeError`s at class definition time. Second, and more importantly, composition makes the shared block addressable as a unit: `metadata.base` can be copied, compared or serialised wholesale, and a caller reading provenance does not have to know which concrete metadata class it holds. The cost is one extra attribute hop in every access, paid at every call site.

`TextNode.base_metadata` (in `corpus/base`) exists precisely to absorb that cost for generic code — it resolves either shape, a `BaseMetadata` held directly or one wrapped under `.base`, so a caller reading common fields does not need to know which it has.

**`base` is default-constructible.** Every metadata class declares `base: BaseMetadata = field(default_factory=BaseMetadata)`, and every field on `BaseMetadata` itself has a default. The consequence is that `DocumentMetadata()` with no arguments is always valid, which is what allows the node classes to declare `metadata: DocumentMetadata = field(default_factory=DocumentMetadata)` and lets a document be constructed from nothing but text. Metadata is therefore always present and never `None`, so no consumer needs a null check before reading it. The alternative — required fields — would have forced every construction site, including every test, to supply provenance it does not have.

**`created_at` and `updated_at` are timezone-aware UTC.** Both use `field(default_factory=lambda: datetime.now(UTC))`. Naive datetimes compare and serialise ambiguously across machines; a corpus assembled from several sources on several hosts would otherwise carry timestamps that cannot be ordered reliably.

**`license` on `DocumentMetadata` is not decoration.** A trained embedding model inherits the licensing constraints of the text it was trained on. If the licence is not carried on the document, the pipeline has no way to answer the question "may this model be redistributed?" after the fact, because by the time text has been segmented, filtered and streamed into a tokenizer trainer, the connection back to its source is gone. This is why the document, not the corpus, is the level that carries it: a corpus assembled from several sources has several licences, and a single dataset-level field would be a lie. `Document.to_dict` and `Document.from_dict` both round-trip `license` explicitly for the same reason.

**`SentenceMetadata.language_confidence` distinguishes declared from inferred.** `None` means the language was given by the source, not guessed. Any float means it was inferred and carries that much confidence. Without the distinction, a downstream filter cannot tell an authoritative language tag from a script-based guess, and would have to treat both as equally trustworthy.

**`BaseMetadata.attributes` is an open dictionary.** `JsonlReader` copies any record field it does not recognise into it, so dataset-specific columns survive into the pipeline without the framework needing to know about them. The tradeoff is that nothing in `attributes` is type-checked; it is an escape hatch, and anything the framework itself depends on gets a real field.

## Usage

```python
from multilingual_embedding.corpus.metadata import (
    BaseMetadata, DocumentMetadata, SentenceMetadata, TokenMetadata,
)

metadata = DocumentMetadata()
print("base default:", isinstance(metadata.base, BaseMetadata))
print("license:", metadata.license, "title:", metadata.title)

metadata.base.language = "hi"
metadata.base.script = "Deva"
metadata.license = "CC-BY-SA-4.0"
print("language:", metadata.base.language, "script:", metadata.base.script)
print("license:", metadata.license)

print("inherits BaseMetadata:", issubclass(DocumentMetadata, BaseMetadata))

sentence_metadata = SentenceMetadata()
print("language_confidence:", sentence_metadata.language_confidence)
print("token defaults:", TokenMetadata().is_stopword, TokenMetadata().frequency)
```

Output:

```
base default: True
license: None title: None
language: hi script: Deva
license: CC-BY-SA-4.0
inherits BaseMetadata: False
language_confidence: None
token defaults: False None
```

Note the third-to-last line: the composition choice is observable. `DocumentMetadata` is not a `BaseMetadata`.

## Dependencies
This package imports nothing from the framework at all — only `dataclasses`, `datetime` and `typing` from the standard library. It is the leaf of the corpus layer and deliberately has no edges into `common`, `core` or `utils`.

It is imported by `corpus/base` (`TextNode` imports `BaseMetadata` for its `base_metadata` property) and by every concrete node module in `corpus`. Nothing below the corpus layer may import it, and nothing above it should construct these classes directly rather than going through the node constructors.

## Tests
- `tests/corpus/metadata/test_base.py` — 1 test.

The metadata classes are otherwise exercised indirectly and extensively through the node and corpus tests, principally `tests/corpus/test_nodes.py` (22 tests) and `tests/corpus/test_io.py` (22 tests), which cover the round-tripping of `license`, `author`, `url` and `attributes` through `Document.to_dict` and `Document.from_dict`.
