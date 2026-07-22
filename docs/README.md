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

Listed in navigation order, which is also reading order.

| Page | Contents |
|---|---|
| `index.md` | Overview, design principles, scope — including what is explicitly not implemented |
| `installing.md` | The use-it/develop-it split: `uv tool install` for running `qfme` anywhere, `uv sync` in a clone for changing it, and which extras each job needs |
| `getting-started.md` | Install, inspect a corpus, train, search, evaluate, Python API, running tests |
| `architecture.md` | Layer diagram and dependency rule, package walkthrough, the non-obvious design decisions |
| `configuration.md` | Every configuration field with type, default, meaning and validation rule; the precedence chain; annotated example configs |
| `data-format.md` | The exact JSON Lines contract an extraction pipeline must produce, the rules it must satisfy, how to check an extraction with `qfme validate`, and notes specific to Wikipedia dumps |
| `compute-profiles.md` | The machine/experiment split: what a `--profile` carries, why it is configuration rather than a branch per machine, and which single setting is not result-neutral |
| `reading-results.md` | How to tell a real retrieval result from a flattering one — the disqualifying checks, in the order that stops you interpreting an invalid number |

`reading-results.md` is the page to send someone who has just been handed an adaptation
report. It is ordered so that each step can disqualify everything below it: the baseline
before the gain, the candidate pool before the recall, the overlap bands before the
headline. A +40% recall@1 against no baseline, or against 50 candidates, means nothing, and
the report contains everything needed to notice that.

`mkdocs.yml` at the repository root defines the navigation, and its `nav` must list
exactly the pages present here — `mkdocs build --strict` fails on a mismatch, and CI
runs it on every push. This file is the one exception: `exclude_docs` drops `README.md`
from the build, because MkDocs would otherwise treat it as a second index page and
collide with `index.md`. It exists for anyone browsing the directory on GitHub.

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
project's own documentation pass found nine genuine defects in code that was already
passing its tests, precisely because the examples were run rather than assumed. They are
listed in part 11 of the handbook.

**Do not document what does not exist.** Describe what is in the tree, and say plainly
when something is not there — an honest gap with a pointer to `ROADMAP.md` is worth more
than an impressive-sounding capability a reader will go looking for and fail to find.

**Re-check the scope claims when the tree changes.** This is the convention these pages
have most often broken. Statements of the form "there is no X here" age badly and, unlike
a wrong command, nothing fails when they go stale — `embedding/neural/` now holds a
transformer encoder, contrastive training, LoRA and gradient caching, and torch is a real
dependency under the optional `neural` extra, so any page still denying those is wrong
rather than merely dated. A negative claim needs verifying against the source on the same
terms as a positive one.

**Prefer an honest caveat to a confident claim.** Stating that exact search is the right
choice up to roughly 10⁵–10⁶ vectors and needs replacing beyond that is more useful than
calling it fast.

**Keep the field reference synchronised with `config/base.py`.** That module is the
source of truth for defaults and validation rules; `configuration.md` restates them for
readers and will drift if changed in only one place.
