# Data format

What an extraction pipeline must produce for this framework to consume it, and how to
check that it did.

---

## The format

**JSON Lines. One JSON object per line, one document per object, UTF-8.**

```json
{"id": "hi-mumbai", "language": "hi", "text": "मुंबई एक शहर है। यह बड़ा है।", "title": "मुंबई", "source": "wikipedia", "license": "CC BY-SA 4.0"}
```

| Field | Required | Meaning |
|---|---|---|
| `text` | **yes** | The document body. Plain text, no markup. |
| `id` | recommended | Stable identifier. Derived from a content hash when absent, which makes it unstable across extractions. |
| `language` | recommended | ISO 639-1, or 639-2/3 where no two-letter code exists (`mai`, `sat`, `kok`, `doi`, `mni`, `brx`). |
| `title` | optional | Carried into metadata; also a useful pair source, see below. |
| `license` | optional | **Record it.** A model inherits the licensing constraints of its training text. |
| anything else | optional | Carried through into `document.metadata.base.attributes` untouched |

Accepted extensions: `.jsonl`, `.jsonl.gz`, `.ndjson`. **Keep it gzipped** — the readers
decompress transparently and the corpora are large.

Plain text is also accepted (`.txt`, `.txt.gz`): one file becomes one document. Pass
`--format lines` when the file is already one sentence per line — extension sniffing
cannot tell that from prose, so it has to be named.

An empty `text` is skipped by the JSON Lines reader rather than reported, so a pipeline
that silently emits blank records loses them without a count. Check the document total
against what the extraction claims it wrote.

## Rules the extraction must satisfy

**Strip all markup.** Wiki syntax and HTML must be gone. This is the failure mode the
audit exists to catch, because markup left in body text trains the tokenizer on syntax
rather than on language and is invisible in aggregate statistics.

**Emit real UTF-8.** A wrong encoding guess produces replacement characters that no
amount of training recovers from.

**Do not pre-segment.** The framework segments, and it does so correctly across scripts —
the Devanagari danda, the CJK full stop, the Ol Chiki mucaad. Feeding it text that has
already been split on periods discards that. Pass whole documents.

**Populate `language` where you know it.** Script detection cannot recover it: eight
scheduled Indian languages share Devanagari, two share Bengali, three share Perso-Arabic.
The framework declines to guess rather than guessing wrongly.

**Deduplicate, or expect the audit to complain.** Scraped corpora routinely contain the
same article several times, which inflates the apparent frequency of whatever it holds.

## Checking an extraction

```bash
qfme validate --source data/wikipedia/hi.jsonl.gz
```

Reports document and sentence counts, the language and script distribution, and every
problem it can identify — each with an example and a remedy. It exits non-zero when the
corpus is unusable, so a data pipeline can gate on it:

```bash
qfme validate --source "$OUTPUT" || exit 1        # blocks on errors
qfme validate --source "$OUTPUT" --strict || exit 1   # blocks on warnings too
qfme validate --source "$OUTPUT" --output audit.json  # machine-readable
```

| Severity | Meaning |
|---|---|
| `ERROR` | Unusable as it stands — markup, encoding damage, empty documents |
| `WARNING` | Will train, but something was probably lost — duplicates, missing language, fragments |
| `INFO` | Context worth knowing before trusting a result |

## Extracting Wikipedia

```bash
uv sync --extra wikipedia
curl -O https://dumps.wikimedia.org/hiwiki/latest/hiwiki-latest-pages-articles.xml.bz2

qfme extract --dump hiwiki-latest-pages-articles.xml.bz2 \
             --output data/wikipedia/hi.jsonl.gz --language hi
qfme validate --source data/wikipedia/hi.jsonl.gz
```

Streaming, so a multi-gigabyte dump runs on a laptop within one article's memory. Try a
dump before committing to it with `--limit 500`.

What it does, and why each part is not optional:

| Step | Reason |
|---|---|
| Keeps namespace 0, drops redirects | Talk pages and templates are project machinery, not prose |
| Removes tables and block HTML *before* parsing | A parser keeps a table's cells, so a statistics table arrives as bare numbers that read like prose |
| Drops boilerplate sections | A references list teaches citation formatting |
| Skips articles under 200 characters | Wikipedia is full of one-line stubs that add vocabulary noise |
| Drops articles whose markup is malformed | ~1% of articles; the source is broken and no parser can fix it |
| Deduplicates | Template-generated stubs repeat verbatim — the Meetei Mayek wiki has **118 country articles sharing one sentence** |
| Keeps sections as `(heading, body)` | A heading and its section are a query and a passage, which Phase C needs |

Every drop is counted and logged. Nothing is removed silently.

Measured end to end on the Meetei Mayek wiki — a 5 MB dump, about 16 seconds:

| | Pages |
|---|---:|
| Seen in the dump | 15,514 |
| Redirects and non-article namespaces | −4,348 |
| Shorter than 200 characters | −8,547 |
| Malformed markup, dropped | −24 |
| Duplicates, dropped | −151 |
| **Written** | **2,444** |

The result passes `qfme validate` with no errors. Note how much of a dump is not article
prose: **84% of pages were discarded**, most of them stubs. Plan corpus size from what
survives, not from the dump's page count.

## Notes on Wikipedia specifically

**Dumps.** `dumps.wikimedia.org`, one archive per language, `*-pages-articles.xml.bz2`.
Sizes differ by orders of magnitude between languages — verify the current figures before
planning storage, as they move.

**Article counts fall away steeply across the scheduled languages.** Hindi, Bengali and
Tamil have substantial Wikipedias; Santali, Bodo, Dogri and Meitei have very little.
Expect the low-resource languages to remain low-resource after extraction, and plan the
language tier accordingly rather than assuming coverage follows from the dump existing.

**Interlanguage links are the most valuable thing Wikipedia offers**, because an article
linked across languages gives *aligned* content — the Hindi and Tamil articles on the same
subject. That is the parallel signal cross-lingual alignment needs; without it a
multilingual model puts each language in its own region of the space and retrieval across
languages does not work.

**They are not in the article dump.** An earlier version of this page said to extract them
from the wikitext, which was wrong and would have cost you an afternoon. Inline `[[xx:…]]`
links were how it worked before Wikidata; they have since moved out. Measured on the
Meetei Mayek wiki: **0 of 2,260 articles** carry one.

They live in a separate file, as SQL rows of `(page_id, language, foreign_title)`:

```
https://dumps.wikimedia.org/<wiki>/latest/<wiki>-latest-langlinks.sql.gz
```

So a cross-lingual pair set is a **join**, not a scrape: article text from
`pages-articles.xml.bz2`, keyed by page id, against `langlinks.sql.gz`. Budget for that
rather than expecting the links to fall out of the text.

**Natural training pairs fall out of the structure**, which matters because Phase C of
the roadmap needs pairs and none are labelled:

| Wikipedia structure | Pair |
|---|---|
| Title ↔ lead paragraph | query ↔ passage |
| Section heading ↔ section body | query ↔ passage |
| Article ↔ its interlanguage counterpart | cross-lingual alignment |
| Consecutive paragraphs | weak positive |

**The licence needs a decision, not a footnote.** Wikipedia text is CC BY-SA 4.0, which
carries attribution and share-alike terms. Whether model weights trained on such text are
a derivative work is legally unsettled. For a research artefact this is a low risk; for a
commercial product sold to enterprises it is worth a considered answer before the corpus
is baked into a model. Record the licence per document — the field exists for this — so
the question stays answerable later.
