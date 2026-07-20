# tests/corpus

> Tests for [`multilingual_embedding.corpus`](../../src/multilingual_embedding/corpus/README.md) — the document tree, segmentation, readers, statistics and auditing.

**395 tests**, the largest group in the suite by a wide margin. Run with
`pytest tests/corpus -q`; the whole group takes under a second.

## Files

| File | Covers |
|---|---|
| `test_indian_languages.py` | The 22 scheduled languages plus English: script detection, terminator, word count, code normalisation and naming, and the shared-script cases |
| `test_analysis.py` | Language utilities, offset arithmetic, streaming iteration, statistics, length summaries, filters, deduplication, document validation |
| `test_script.py` | Script detection, range-table invariants, mixed-script flagging, whitespace-delimitation flags |
| `test_segmentation.py` | Sentence, paragraph and word splitting across scripts; abbreviations, initials, decimals, quotes; span integrity |
| `test_io.py` | Readers, writers and the config-driven loader, including gzip and malformed input, plus the format-name agreement check |
| `test_nodes.py` | `Token`/`Sentence`/`Paragraph`/`Document` construction, span consistency, round trips |
| `test_audit.py` | Corpus auditing: extraction failures, quality warnings, finding shape, `qfme validate` |
| `test_corpus.py` | Document-level splitting and persistence |
| `base/test_text_node.py` | The `TextNode` base: length and character counting |
| `metadata/test_base.py` | `BaseMetadata` defaults |

## What matters here

This package is where "multilingual" is either true or merely claimed, so the tests are
weighted accordingly.

**The 22 scheduled languages get their own module, and it is the largest here.**
`test_indian_languages.py` runs each of them plus English through the full chain: the
script is recognised, the sample segments on its own terminator, sentence spans slice
back to the same text, words survive combining marks, and the language code is both
accepted and named. Parametrising one language set across those assertions is what turns
support from a claim into something a failing test can contradict.

Three properties of that set are why it needs the coverage. Eight of the languages share
Devanagari, two share Bengali and three share Perso-Arabic, so script detection alone
cannot identify them — `TestSharedScripts` pins exactly which groups collide, bounding
what script-based inference is allowed to claim. Only Ol Chiki and Meetei Mayek are used
by a single language each, and those two are the only ones `infer_language` is tested to
resolve. Six languages — Maithili, Santali, Konkani, Dogri, Meitei and Bodo — have no ISO
639-1 two-letter code at all, so a validator that accepted only two-letter codes would
make them impossible to label.

**Terminators are per script, not per language.** Most Indic text ends a sentence with
the danda, Perso-Arabic with the Urdu full stop, Ol Chiki with the mucaad, Meetei Mayek
with the cheikhei. A period-and-space rule finds one sentence in nearly all of these
samples, which is why each case asserts two.

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

**Format names must agree in three places.** `TestFormatNamesAgree` compares the reader
registry against the config validator and the CLI's `--format` choices. The three lists
are written out separately and drifted once: `lines` was registered and usable from
Python, yet rejected by both the config and the CLI, so a sentence-per-line corpus could
not be read through either entry point. The CLI half is checked per subcommand rather
than unioned across them, because a union passes while one subcommand alone is missing a
format — which is the shape the drift actually took.

## Auditing

`test_audit.py` covers `corpus.audit` and the `qfme validate` subcommand. The audit
exists for extraction pipelines, so the tests mirror the ways an extraction actually
fails rather than the ways an API can be misused. None of these conditions raise on their
own: each produces a corpus that loads, trains, and yields a worse model. That is exactly
why they need naming.

**Surviving markup is an error, not a warning.** `'''Mumbai''' is a city in
[[Maharashtra]]` is the characteristic failure of a Wikipedia extraction, and HTML
`<ref>` tags are tested alongside it. Markup left in body text trains the tokenizer on
syntax instead of language, and it is invisible in aggregate statistics — document
counts, sentence counts and length summaries all look healthy.

**Replacement characters mean the encoding guess was wrong**, and no amount of training
recovers from that, so it is an error too.

**A missing language is a warning, and the test asserts the corpus stays usable.**
Blocking on it would be wrong: an unlabelled corpus still trains. The distinction between
error and warning is the whole point of the severity field, so it is asserted rather than
assumed.

**Duplicates are matched after whitespace normalisation.** Two copies of the same article
differing only in runs of spaces are still two copies, and an exact-string check would
miss the most common form the duplication takes.

**Findings are ordered by severity and carry counts, shares, examples and a remedy.** A
finding that only says something is wrong makes the operator go looking; the share tells
them whether it affects two documents or the whole corpus, and the examples tell them
what to grep for. Example lists are capped so a corpus that is wrong everywhere does not
produce an unreadable report.

**The CLI must be gateable.** A clean corpus exits `0`, a broken one exits `1`, and
`--strict` promotes warnings to failures. That triple is what lets a data pipeline refuse
to proceed. `--output` writes the audit as JSON, and the test parses it back rather than
merely checking the file exists.
