# corpus.base

> The structural base classes every corpus node derives from: text plus a span plus typed metadata, and the container variant that adds ordered children.

## Purpose
`Token`, `Sentence`, `Paragraph` and `Document` are four levels of the same idea — a stretch of text that knows where it sits inside its parent and carries metadata. This package holds that common structure once, so the concrete node modules in `corpus` contain only what is specific to their level. It also holds the invariant that makes the tree trustworthy: `verify_children()`, which checks that a container's stored text and its children's spans still describe the same thing.

## Modules
| Module | Responsibility |
|---|---|
| `text_node.py` | `TextNode` — the generic base for every textual object: `text`, `span`, `metadata`, and the string-like conveniences over them. |
| `container_node.py` | `ContainerNode` — a `TextNode` with ordered children, plus `verify_children()`. |
| `node.py` | `Spanned` and `Composite` — runtime-checkable protocols describing shape rather than inheritance. |

## Key design decisions

**`TextNode` is a PEP 695 generic over its metadata type.** The declaration is `class TextNode[MetadataT](ABC)`, and `Sentence` fixes it as `ContainerNode[SentenceMetadata, Token]`. The payoff is that `sentence.metadata.language_confidence` type-checks, because `sentence.metadata` is known statically to be a `SentenceMetadata` rather than a bare `BaseMetadata` or an `Any`. The alternative — a single untyped `metadata` attribute — would have compiled equally well and moved every metadata field access into the class of errors that only appear at runtime, on the level where the field happens not to exist.

`ContainerNode` is generic over *two* parameters, `[MetadataT, ChildT: TextNode[Any]]`, so the child type is known too: `paragraph[0]` is a `Sentence`, and `paragraph.children` is a `list[Sentence]`. The bound on `ChildT` is what enforces that only text nodes can be children.

**Spans are relative to the immediate parent, not the document root.** A sentence's span indexes into its paragraph's text; the paragraph's span indexes into the document's. This keeps segmentation local and composable — a paragraph can be re-segmented without renumbering every unit that follows it in the document — at the cost that recovering an absolute position requires walking the chain of parents. `multilingual_embedding.corpus.offsets.resolve_chain` exists to do exactly that walk. Absolute spans everywhere would have made that lookup free and made every local edit an O(document) renumbering.

**Container nodes store their own `text` rather than deriving it from their children.** `ContainerNode` has both `text` (inherited from `TextNode`) and `children`, and the two are redundant views of the same content — almost. The material *between* children is not: whitespace, punctuation and markup separating one sentence from the next are part of the source and are covered by the parent's `text` but by no child's span. Deriving `text` by joining `child_texts()` would silently discard it, and a document could not survive a round trip through segmentation unchanged. `offsets.invert_spans` recovers that between-material precisely because it is still there to recover. The cost of storing both is that they can drift apart, which is what the next decision addresses.

**`verify_children()` checks the two views agree.** It walks the children in order against the parent's `text` and raises `CorpusError` on the first inconsistency. Every one of the three carries the offending `child_index` and the `node` type it was found on, because the same message on a `Sentence` and on a `Document` points at different bugs. The three failure modes each have their own message and their own meaning:

- *"Child span falls outside parent text"* — a child's span starts below zero or ends past `len(self.text)`. This almost always means a child was built with an absolute offset where a relative one was expected, or the parent's text was replaced with a shorter string after segmentation.
- *"Child spans overlap or are out of order"* — a child starts before the previous child ended. A segmenter emitted units that are not a partition, or children were appended out of document order. `offsets.spans_are_ordered` tests the same property on a bare span list.
- *"Child text does not match the slice its span designates"* — the span is in range and correctly ordered, but `self.text[child.span.start:child.span.end]` is not `child.text`. This is the drift case: usually an off-by-one in a span, or a text mutation applied at one level of the tree and not another. The error reports the first 40 characters of both the expected and actual strings, because the difference is often a single leading space and is otherwise invisible in a log.

Verification is a method rather than an assertion in `add()`, because a tree is legitimately inconsistent while it is being built. `Document.verify()` composes it across all three levels, and `validators.validate_document` calls it and reports rather than raises, so a bad document does not abort a pass over a large corpus.

**`node.py` defines protocols, not base classes.** `Spanned` and `Composite` are `@runtime_checkable` `Protocol`s. They let a function accept "anything with `text` and a `span`" without importing the concrete classes, which is what keeps utility code from acquiring a dependency on the whole node hierarchy. They are structural, so a caller's own type satisfies them without inheriting anything.

**`base_metadata` resolves both metadata shapes.** The concrete metadata classes wrap a `BaseMetadata` under `.base` (see `corpus/metadata`), but a plain node may hold one directly. `TextNode.base_metadata` returns whichever it finds and `None` if neither, so `TextNode.language` works on either shape without the caller branching.

**All node classes are `@dataclass(slots=True)`.** A corpus holds a great many of these objects and `__slots__` removes a per-instance `__dict__`. One consequence is worth knowing: because the concrete subclasses are themselves dataclasses, they regenerate `__repr__`, so `TextNode.__repr__` — which produces the short `ClassName(length=..., text=...)` preview — is only seen on a node type that does not redeclare the decorator.

## Usage

```python
from multilingual_embedding.common.span import Span
from multilingual_embedding.corpus import Sentence, Token
from multilingual_embedding.corpus.exceptions import CorpusError

sentence = Sentence.create("नमस्ते दुनिया", start=0, language="hi")
sentence.extend([Token.create("नमस्ते", start=0), Token.create("दुनिया", start=7)])

print("character_count:", sentence.character_count, "is_blank:", sentence.is_blank)
print("language:", sentence.language)
print("child_texts:", sentence.child_texts())

sentence.verify_children()
print("verify_children: ok")

sentence.children[1].span = Span(6, 12)   # off by one
try:
    sentence.verify_children()
except CorpusError as error:
    print("CorpusError:", error)
```

Output:

```
character_count: 13 is_blank: False
language: hi
child_texts: ['नमस्ते', 'दुनिया']
verify_children: ok
CorpusError: Child text does not match the slice its span designates (actual='दुनिया', child_index=1, expected=' दुनिय', node='Sentence')
```

The expected slice begins with the separating space — exactly the kind of one-character drift the diagnostic exists to make visible. Note also that `character_count` is 13 for two six-character words: the Devanagari text counts as Unicode codepoints, combining marks included.

## Dependencies
May import from `common` (`Span`) and from `corpus.metadata` (`BaseMetadata`) and `corpus.exceptions` (`CorpusError`) within its own layer. It imports nothing from `core`, `utils` or `config`.

Nothing below the corpus layer may import it. Within the corpus layer it is imported by `token.py`, `sentence.py`, `paragraph.py` and `document.py`. Above the corpus layer, code should work with the concrete node classes exported from `multilingual_embedding.corpus` rather than with these bases — the one intended exception being the `Spanned` and `Composite` protocols, which exist to be depended on structurally.

## Tests
- `tests/corpus/base/test_text_node.py` — 5 tests.

`ContainerNode.verify_children` and the node hierarchy as a whole are covered more heavily through the concrete classes, chiefly in `tests/corpus/test_nodes.py` (22 tests) and `tests/corpus/test_corpus.py` (19 tests). The whole `tests/corpus` tree is 462 tests.
