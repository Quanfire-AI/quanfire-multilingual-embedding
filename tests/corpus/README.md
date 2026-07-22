# tests/corpus

> Tests for [`multilingual_embedding.corpus`](../../src/multilingual_embedding/corpus/README.md) — the document tree, segmentation, readers, statistics, auditing, Wikipedia extraction and pair mining.

**493 tests**, the largest group in the suite by a wide margin. Run with
`pytest tests/corpus -q`; the whole group takes under a second.

## Files

| File | Tests | Covers |
|---|---:|---|
| `test_indian_languages.py` | 166 | The 22 scheduled languages plus English: script detection, terminator, word count, code normalisation and naming, and the shared-script cases |
| `test_analysis.py` | 74 | Language utilities, offset arithmetic, streaming iteration, statistics, length summaries, filters, deduplication, document validation |
| `test_script.py` | 33 | Script detection, range-table invariants, mixed-script flagging, whitespace-delimitation flags |
| `test_io.py` | 31 | Readers, writers and the config-driven loader, including gzip and malformed input, plus the format-name agreement check |
| `test_pairs.py` | 30 | Pair mining: the three kinds, lexical overlap, the bigram path for non-spaced scripts, rejection accounting, streaming |
| `test_wikipedia.py` | 30 | Dump extraction: markup stripping, namespace and redirect filtering, boilerplate headings, deduplication, section preservation |
| `test_segmentation.py` | 25 | Sentence, paragraph and word splitting across scripts; abbreviations, initials, decimals, quotes; span integrity |
| `test_audit.py` | 25 | Corpus auditing: extraction failures, quality warnings, finding shape, `qfme validate` |
| `test_nodes.py` | 22 | `Token`/`Sentence`/`Paragraph`/`Document` construction, span consistency, round trips |
| `test_corpus.py` | 19 | Document-level splitting and persistence |
| `test_domain_pairs.py` | 16 | The miner against a corpus that is not Wikipedia: contracts, transcripts and support tickets carry structure of their own, and the three kinds must find it without a Wikipedia-shaped assumption |
| `test_pair_io.py` | 15 | Reading a pair file back: what a record may leave out, and that reservoir sampling can reach the end of the file — a head-window sampler reported a two-language run as monolingual |
| `base/test_text_node.py` | 5 | The `TextNode` base: length and character counting |
| `metadata/test_base.py` | 2 | `BaseMetadata` defaults |

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

## Wikipedia extraction

`test_wikipedia.py` covers `corpus/wikipedia.py`, the front door for real data. It needs
`mwparserfromhell` (the `wikipedia` extra) and calls `pytest.importorskip`, so a core-only
checkout skips the module rather than erroring.

**The suite's own audit is the acceptance criterion.**
`TestTheOutputPassesOurOwnAudit::test_extraction_produces_a_corpus_with_no_errors` runs an
extraction and feeds the result straight into `corpus.audit`. That closes the loop
deliberately: `_MARKUP_MARKERS` in the extractor matches the marker set `test_audit.py`
grades as an `ERROR`, so a leak fails at the stage that caused it rather than surfacing
three stages later as a mysterious quality finding.

**`strip_code` is not sufficient on its own, and eight parametrised cases prove it.**
`test_no_marker_survives` runs templates, nested templates, `<ref>`, tables, HTML blocks,
comments, entities, and bold-plus-links, each embedded in real Meetei Mayek prose.
`test_table_contents_do_not_leak_into_prose` is separate because a stripped table does not
vanish — its cell contents reappear as a sentence-shaped fragment. When the markup cannot
be repaired the article is dropped rather than emitted dirty.

**Filtering is tested by what it removes.** Redirects
(`test_redirects_are_skipped`), non-article namespaces
(`test_non_article_namespaces_are_skipped`), articles under `--minimum-characters`
(`test_stubs_are_skipped`) and boilerplate headings
(`test_boilerplate_sections_are_dropped`).

**Deduplication has three cases, and it is not hypothetical.** Repeated boilerplate is
dropped by default, `--keep-duplicates` turns that off, and whitespace variants still count
as duplicates — a real Meetei Mayek wiki yielded 118 articles that were all the same stub
differing in spacing.

**Sections must survive.** `test_sections_are_kept_for_pair_mining` is the load-bearing
one, and its name says why: flatten the article and `heading_section` mining becomes
impossible one stage later, with nothing about the flattened corpus looking wrong.

**The identifier is the page id, not the title.** Titles are neither stable nor unique
across a dump; `test_the_identifier_is_the_page_id_not_the_title` pins that. The same code
path carries the ElementTree fix where a leaf whose text is `"0"` is falsy and was once
read as absent.

**One article must not be able to stall a 227 MB dump.**
`TestExtractionCannotBeStalledByOneArticle` pins the `_MEDIA_LINK` catastrophic-backtracking
fix — an unclosed media link with many pipes took 8.5 seconds and now takes microseconds —
with two companion cases asserting the faster regex still removes media links and still
keeps ordinary wikilink text.

**File handling is asserted end to end.** Plain XML as well as bzip2, `--limit` stopping
early, a missing dump and a truncated dump each reported clearly rather than as a parser
traceback, gzipped JSON Lines written, and the written records read back by the ordinary
corpus reader — which is the only assertion that proves the two halves agree on a format.

## Pair mining

`test_pairs.py` covers `corpus/pairs.py`, which manufactures supervision out of article
structure. Pure Python and numpy, no optional dependency, so it always runs.

**Each kind is tested for what it is good for.** `title_lead`, `heading_section` and
`adjacent` each get a case, `--kinds` restriction gets another, and
`test_a_corpus_without_structure_still_yields_pairs` pins the property that makes
`adjacent` worth having: it needs no headings, no title, no structure at all, so it works
on prose from any source.

**Provenance is recorded on every pair.** `test_provenance_is_recorded` asserts the
`document` identifier survives, which is what lets training avoid treating two pairs from
the same article as negatives of each other.

**Lexical overlap is the control, so it is tested as one.** `TestTokenOverlap` covers full
containment, disjoint text, proportionality, case, and an empty anchor not dividing by
zero. `test_it_works_without_whitespace_word_boundaries` is the regression that matters:
the word-split version returned 0.0 for Japanese, silently disabling the leakage filter for
exactly the scripts that most needed it. The bigram path is now chosen by script rather
than assumed.

**Leakage must be visible, filterable, and reported per kind.** Overlap is recorded on
every pair; a leaky pair can be rejected and a clean one survives the same threshold; mean
overlap is reported per kind; and a kind whose mean is high enough to be suspicious is
warned about rather than silently mined.

**Quality filters are tested by their asymmetry.** Short anchors and short positives are
rejected, but a long positive is *truncated rather than dropped* — throwing away the
article's best passage because it ran long would be the wrong trade. Identical pairs are
deduplicated and whitespace is normalised first.

**Bad configuration fails at construction, not mid-run.** Six parametrised settings each
raise when `PairConfig` is built. A mining run over a 163,768-article corpus takes half an
hour; discovering an invalid threshold at the end of it is not acceptable.

**Statistics and pairs both serialise.** `test_a_pair_serialises_to_the_trainer_format`
pins the JSON shape `scripts/adapt_pretrained.py` reads, and is the only thing standing
between the miner and the trainer disagreeing silently.
