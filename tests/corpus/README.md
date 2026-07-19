# tests/corpus

> Tests for [`multilingual_embedding.corpus`](../../src/multilingual_embedding/corpus/README.md) — the document tree, segmentation, readers and statistics.

**196 tests**, the largest group in the suite. Run with `pytest tests/corpus -q`.

## Files

| File | Covers |
|---|---|
| `test_script.py` | Script detection across 13 scripts, range-table invariants, whitespace-delimitation flags |
| `test_segmentation.py` | Sentence, paragraph and word splitting across scripts; abbreviations, initials, decimals, quotes |
| `test_nodes.py` | `Token`/`Sentence`/`Paragraph`/`Document` construction, span consistency, round trips |
| `test_corpus.py` | Counts, filtering, language selection, document-level splitting, persistence |
| `test_io.py` | Readers, writers and the config-driven loader, including gzip and malformed input |
| `test_analysis.py` | Language utilities, offset arithmetic, streaming iteration, statistics, filters, deduplication |

## What matters here

This package is where "multilingual" is either true or merely claimed, so the tests are
weighted accordingly.

**Segmentation is tested per script, not per feature.** Devanagari splits on the danda,
Chinese on `。` with no following space, Arabic on `؟`. English must *not* split on
`Dr.`, on the initials in `J. R. R. Tolkien`, or on the decimal point in `3.14`, and
must keep a closing quote with the sentence it ends.

**Combining marks have a dedicated regression test.**
`test_devanagari_words_counted_correctly` asserts `नमस्ते दुनिया` is two words. Python's
`\w` does not match combining marks, so a naive regex counted five fragments
(`['नमस', 'त', 'द', 'न', 'य']`) and silently discarded the marks themselves — a bug that
corrupted word statistics for every Indic, Arabic, Hebrew and Thai corpus.

**The script range table's invariants are asserted.** Lookup binary-searches it, so
`test_range_table_is_sorted_and_disjoint` checks sortedness and non-overlap. A misplaced
row would misclassify an entire script, and the ordering was in fact wrong once.

**Span consistency is verified structurally.** `verify()` walks the tree checking that
every child's text matches the slice its span designates; tests assert it passes on
well-formed documents and catches tampering, out-of-bounds children and overlaps.

**Splitting is document-level.** `test_split_is_document_level_and_exhaustive` and its
reproducibility companions exist because sentences within a document are correlated —
splitting at sentence level would leak near-duplicates into the evaluation set.

**Streams must be re-iterable.** `test_stream_sentences_is_reiterable` asserts two
passes give the same result, because multi-epoch training depends on the source
restarting rather than being exhausted after one epoch.

**Filters must be Unicode-aware.** Letter detection uses Unicode categories, so
`नमस्ते` and `こんにちは` pass a filter that rejects digit-and-punctuation-only strings.
