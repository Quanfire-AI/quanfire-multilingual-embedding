# tests

> The test suite. 1204 tests, 94% statement coverage, mirroring the source layout.

## Purpose

Tests here serve three distinct jobs, and it is worth knowing which is which when a
failure appears:

1. **Unit tests** verify one module against its contract. They live in a directory
   named after the source package.
2. **Integration tests** (`integration/`) run the real pipeline — real SentencePiece
   training, real word2vec training, real persistence and search — to catch interfaces
   drifting between two layers that each pass their own unit tests.
3. **Architecture tests** (`test_architecture.py`) parse the source and enforce the
   layering rule. Nothing else in the suite would notice a convenience import that
   quietly introduces a cycle. Two of its cases also enforce the encoder contract
   structurally: every registered encoder must satisfy `TextEncoder`, and the search
   pipeline must build without an embedding matrix. Those are architectural rather than
   behavioural, because a contextual encoder has no per-token table to hand over.

## Layout

| Directory | Tests | Covers |
|---|---|---|
| [`common/`](common/README.md) | 20 | Spans, enums, type aliases, constants, version |
| [`core/`](core/README.md) | 27 | Registry, factory, logging, exception context |
| [`config/`](config/README.md) | 87 | Config validation, merging, precedence, compute profiles, persistence |
| [`utils/`](utils/README.md) | 61 | Validation, hashing, filesystem, I/O, serialization |
| [`corpus/`](corpus/README.md) | 424 | Scripts, segmentation, nodes, readers, statistics, auditing, the 22 scheduled languages |
| [`vocabulary/`](vocabulary/README.md) | 36 | Token/id mapping, special tokens, builder, persistence |
| [`tokenizer/`](tokenizer/README.md) | 187 | Normalizers, pre-tokenizers, encoding, training, round trips |
| [`embedding/`](embedding/README.md) | 148 | Matrix, word2vec, sentence encoders, index, contextual encoder, LoRA, gradient caching |
| [`evaluation/`](evaluation/README.md) | 67 | Metric primitives, evaluators, report rendering |
| [`pipelines/`](pipelines/README.md) | 18 | Query/passage prefixes, batched indexing, building a pipeline from a saved adapter |
| [`integration/`](integration/README.md) | 29 | End-to-end training, search and CLI |
| `test_architecture.py` | 16 | Layering rule, acyclic import graph, `py.typed` marker, encoder contract |

`conftest.py` holds shared fixtures, including the multilingual sample texts used
throughout. Those are real sentences in each script rather than placeholder text,
because several code paths — segmentation, script detection, combining-mark handling —
behave differently on genuine text.

## Running

```bash
pytest                      # everything
pytest -m "not slow"        # skip the model-training integration tests (1175 tests)
pytest --cov                # with coverage report
pytest tests/corpus -q      # one package
pytest -k segmentation      # by name
```

Integration tests are marked `slow` because they train models. They still complete in
about five seconds, so there is rarely a reason to skip them locally; the marker
exists so a fast inner loop is available when iterating on a single module. `slow` is
the only marker that deselects anything — all 29 integration tests carry it, and nothing
else does.

## Conventions

**Test names state the expected behaviour, not the method under test.**
`test_unknown_token_maps_to_unk` rather than `test_id_of`. When one fails, the name
alone should tell you what broke.

**Docstrings explain why a case matters** when that is not obvious from the assertion.
For example, `test_split_is_document_level_and_exhaustive` documents that splitting at
sentence level would leak near-duplicates across the train/eval boundary — the
assertion cannot say that on its own.

**Multilingual coverage is deliberate, not decorative.** Cases run against Latin,
Devanagari, Tamil, Han, Kana, Arabic, Hebrew, Greek, Thai and Bengali text because
several real bugs were only visible in non-Latin scripts. One example is preserved as
a regression test: `test_devanagari_words_counted_correctly` guards the fix for
Python's `\w` not matching Unicode combining marks, which fragmented the phrase into
five pieces and silently discarded the marks themselves.

**Failure paths are tested as carefully as success paths.** Over 140 cases assert that
invalid input raises the right typed exception carrying the right context, because a
framework that fails obscurely is hard to operate. The CLI's failure paths are asserted
on exit status rather than exception type, since a shell script can only see the former.

**Tests are fast and hermetic.** Every test that touches disk uses pytest's `tmp_path`.
Nothing reaches the network. Model-training tests use deliberately small dimensions and
few epochs.

**Optional dependencies skip rather than fail.** The contextual-encoder tests in
`embedding/` and the adapter test in `pipelines/` require torch, which lives behind the
`neural` extra. They use `pytest.importorskip`, so a checkout with only the core
dependencies installed still runs green — it simply runs fewer tests. The counts in the
table above are for a full install; without the extra, those two directories collect
fewer.

## Adding tests

Put unit tests in the directory matching the source package. If a change spans layers,
add an integration test as well — that is the only place a cross-layer interface break
will surface. New packages must respect the layering rule or `test_architecture.py`
will fail the build.
