# docs

> The MkDocs documentation source.

## Purpose

Long-form documentation that does not belong in the root README: the full getting-started
walkthrough, the architecture explanation, and the complete configuration field
reference.

The division of labour across the project's documentation is deliberate:

| Where | Answers |
|---|---|
| Root `README.md` | What is this, why does it exist, how do I run it, how do I deploy it |
| Package `README.md` files | What does this layer own, and why is it built this way |
| `docs/` | Extended walkthroughs and complete reference material |
| Docstrings | What does this specific function do, and why |

## Pages

| Page | Contents |
|---|---|
| `index.md` | Overview, design principles, scope — including what is explicitly not implemented |
| `getting-started.md` | Install, inspect a corpus, train, search, evaluate, Python API, running tests |
| `architecture.md` | Layer diagram and dependency rule, package walkthrough, the non-obvious design decisions |
| `configuration.md` | Every configuration field with type, default, meaning and validation rule; the precedence chain; annotated example configs |

`mkdocs.yml` at the repository root defines the navigation, and its `nav` must list
exactly the pages present here — `mkdocs build --strict` fails on a mismatch, and CI
runs it on every push.

## Building

```bash
mkdocs serve                # live reload at http://127.0.0.1:8000
mkdocs build --strict       # what CI runs; fails on broken links or missing pages
```

The theme is Material for MkDocs, with light and dark palettes and a `watch` entry on
the source tree so docstring changes trigger a rebuild during `serve`.

## Conventions

**Every code example must have been executed.** Command output shown in these pages is
copied from a real run, not written by hand. This is not a stylistic preference: the
project's own documentation review found six genuine bugs precisely because the examples
were run rather than assumed.

**Do not document what does not exist.** There is no transformer, no fastText, no
contrastive learning and no PyTorch in this project. Where future work is mentioned it
is labelled as not implemented.

**Prefer an honest caveat to a confident claim.** Stating that exact search is the right
choice up to roughly 10⁵–10⁶ vectors and needs replacing beyond that is more useful than
calling it fast.

**Keep the field reference synchronised with `config/base.py`.** That module is the
source of truth for defaults and validation rules; `configuration.md` restates them for
readers and will drift if changed in only one place.
