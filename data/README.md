# data

> Corpus data. Only the small sample corpus is tracked in version control.

## Purpose

This directory is where corpora live. Everything in it is gitignored **except**
`sample/`, which is small, synthetic and depended upon by the examples, the
documentation and the getting-started walkthrough.

Real corpora are large, frequently licensed, and reproducible from their source, so
they do not belong in the repository. Point the framework at them by path instead:

```bash
qfme train --source /data/corpora/my-corpus.jsonl
```

## Contents

| Path | Tracked | Description |
|---|---|---|
| `sample/corpus.jsonl` | yes | 150 documents, 750 sentences, six languages |
| anything else | no | Gitignored — put working corpora here freely |

## The sample corpus

`sample/corpus.jsonl` is JSON Lines, one document per line:

```json
{"id": "sample-001", "language": "en", "text": "A student reviews the results. ...", "source": "synthetic-sample"}
```

| Field | Meaning |
|---|---|
| `id` | Stable document identifier |
| `language` | ISO 639-1 code |
| `text` | Document text, several sentences |
| `source` | Provenance marker |

It covers six languages across six scripts:

| Language | Code | Script | Documents |
|---|---|---|---|
| English | `en` | Latin | 25 |
| Hindi | `hi` | Devanagari | 25 |
| Tamil | `ta` | Tamil | 25 |
| Japanese | `ja` | Han + Kana | 25 |
| Arabic | `ar` | Arabic | 25 |
| French | `fr` | Latin | 25 |

The languages were chosen to exercise genuinely different segmentation behaviour, not
for variety: Hindi ends sentences with the danda (`।`), Japanese with `。` and no
following space, Arabic is right-to-left with its own question mark (`؟`), and Tamil
has heavy use of combining marks. A corpus of six European languages would have
exercised one code path.

### It is synthetic, and that matters

Sentences are generated from per-language subject/verb/object templates. This makes the
corpus balanced, licence-free and small enough to commit, and it means a tokenizer
trained on it produces clean, inspectable output.

It also means the corpus has almost no lexical diversity — a type/token ratio around
0.07 — so models trained on it have a collapsed embedding space and near-identical
similarity scores. **Use it to verify the pipeline runs, never to judge model quality.**

### Regenerating

The corpus is committed rather than generated at test time so that results are stable
across machines and runs. If you change it, re-run the examples and the getting-started
documentation, since both quote real output from it.

## Bringing your own corpus

Two formats are supported, selected automatically by file extension:

**Plain text** (`.txt`, `.txt.gz`) — one file becomes one document; paragraphs come
from blank lines and sentences from the segmenter.

**JSON Lines** (`.jsonl`, `.jsonl.gz`, `.ndjson`) — one record becomes one document.
Only a text field is required; `id` and `language` are used when present, and any other
keys are carried through into document metadata.

Point `--source` at a directory to read every matching file, visited in sorted order so
runs are reproducible. Gzip is handled transparently. If your corpus is already one
sentence per line, use `--format lines` so the framework does not re-segment text that
is already segmented.
