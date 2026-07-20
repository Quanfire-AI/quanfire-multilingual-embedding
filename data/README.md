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

It also means the corpus has almost no lexical diversity: 3,050 whitespace-separated
tokens draw on only 137 distinct types. Models trained on it therefore have a collapsed
embedding space and near-identical similarity scores — the example's top hits all score
above 0.99. **Use it to verify the pipeline runs, never to judge model quality.**

### Regenerating

The corpus is committed rather than generated at test time so that results are stable
across machines and runs. If you change it, re-run the examples and the getting-started
documentation, since both quote real output from it.

## Bringing your own corpus

Three readers are available, named by `--format`. The default, `auto`, picks one by file
extension:

**Plain text** (`.txt`, `.text`, `.txt.gz`) — one file becomes one document; paragraphs
come from blank lines and sentences from the segmenter.

**JSON Lines** (`.jsonl`, `.jsonl.gz`, `.ndjson`) — one record becomes one document.
Only a text field is required; `id` and `language` are used when present, and any other
keys are carried through into document metadata.

**One sentence per line** — `--format lines`. This one has to be named. Extension
sniffing cannot tell a sentence-per-line file from prose, and reading it as prose would
re-segment text that is already segmented.

Point `--source` at a directory to read every matching file, visited in sorted order so
runs are reproducible. Gzip is handled transparently.

[`docs/data-format.md`](../docs/data-format.md) is the full contract: every field, the
rules an extraction must satisfy, and notes on Wikipedia dumps specifically.

## Audit before you train

```bash
qfme validate --source /data/corpora/my-corpus.jsonl.gz
```

`qfme validate` reads a corpus and reports the problems it can identify — leftover
markup, encoding damage, empty or duplicated documents, missing language codes,
suspicious fragments — each with an example and a remedy, alongside the document and
sentence counts and the language and script distribution.

It is worth running because the failures it catches are the ones that do not announce
themselves. A corpus with wiki markup still in the body text trains a tokenizer on syntax
rather than on language, and nothing about the aggregate statistics looks wrong; you find
out from a poor model weeks later. Empty records are skipped by the JSON Lines reader
without a count, so a pipeline that silently emits blanks loses them invisibly.

Findings are graded `ERROR` (unusable as it stands), `WARNING` (will train, but something
was probably lost upstream) and `INFO` (context worth having). The command exits non-zero
on errors, so a data pipeline can gate on it:

```bash
qfme validate --source "$OUTPUT" || exit 1              # blocks on errors
qfme validate --source "$OUTPUT" --strict || exit 1     # blocks on warnings too
qfme validate --source "$OUTPUT" --output audit.json    # machine-readable
```

`qfme stats` remains the lighter option when you only want counts and distributions and
not a judgement.
