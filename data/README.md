# data

> Where corpora live: the tiny committed sample, the Wikipedia dumps you fetch, the
> corpora extracted from them, and the mined pairs that train an adapter. Only the sample
> is in version control.

This directory is the **input side of the whole factory**. Every model this project has
produced started as a file under here.

- [Purpose](#purpose)
- [Layout](#layout)
- [What is tracked, and why so little](#what-is-tracked-and-why-so-little)
- [The sample corpus](#the-sample-corpus)
- [The three file formats](#the-three-file-formats)
- [The Wikipedia path, end to end](#the-wikipedia-path-end-to-end)
- [When can multilingual Wikipedia training start?](#when-can-multilingual-wikipedia-training-start)
- [Bringing your own corpus](#bringing-your-own-corpus)
- [Audit before you train](#audit-before-you-train)
- [Pros and cons of this arrangement](#pros-and-cons-of-this-arrangement)
- [Done and pending](#done-and-pending)

---

## Purpose

**What** — a working directory for corpus data at every stage: compressed dumps, extracted
corpora, mined training pairs, plus one small synthetic corpus that is committed.

**Why** — real corpora are large, often licensed, and reproducible from their source. A
227 MB Hindi dump and the 143 MB corpus extracted from it do not belong in a git history;
the four commands that regenerate them do. The sample is the deliberate exception, because
the examples, the docs and the getting-started walkthrough all quote real output from it and
would otherwise be untestable.

**Where** — this directory on a development machine, or any path you like; nothing in the
code assumes `data/`. Every command takes `--source` / `--dump` / `--output` explicitly.

**When** — before anything else. `qfme train`, `qfme mine-pairs` and
`scripts/adapt_pretrained.py` all begin by reading a file from here.

**Who reads it** — [`corpus/reader.py`](../src/multilingual_embedding/corpus/reader.py) and
[`corpus/loader.py`](../src/multilingual_embedding/corpus/loader.py) for corpora,
[`corpus/wikipedia.py`](../src/multilingual_embedding/corpus/wikipedia.py) for dumps,
[`corpus/pairs.py`](../src/multilingual_embedding/corpus/pairs.py) for mining. All three
stream; none loads a file into memory.

## Layout

| Path | Tracked | Contains | Produced by |
|---|---|---|---|
| `sample/corpus.jsonl` | **yes** | 150 documents, 750 sentences, six languages | committed by hand |
| `sample/domain-corpus.jsonl` | **yes** | 10 structured non-Wikipedia documents; the pair-mining contract a domain export must satisfy | committed by hand |
| `dumps/` | no | `*wiki-latest-pages-articles.xml.bz2` as downloaded | `curl` from dumps.wikimedia.org |
| `corpora/` | no | extracted corpus JSON Lines, one file per language | `qfme extract` |
| `pairs/` | no | mined contrastive pairs, one file per language | `qfme mine-pairs` |
| anything else | no | yours to use freely | — |

`corpora/` and `pairs/` are conventions used throughout the documentation, not directories
the code creates. Make them or don't; pass whatever path you want.

Sibling directories worth knowing, both gitignored:

| Path | Contains |
|---|---|
| [`../verify-output/`](../verify-output) | what `scripts/verify_e2e.py` produced on the real dumps — corpora, pair files, and `verification-report.txt` |
| `../models/` | saved adapters, e.g. `models/indic-v1` (3.4 MB) |

## What is tracked, and why so little

`.gitignore` carries three lines that say the whole policy:

```gitignore
data/*
!data/sample/
!data/README.md
```

Ignore everything, then re-admit exactly two things. Written this way round because the
alternative — listing each thing to ignore — fails open: a new `data/bengali-dump/` would be
committed by accident, and a 250 MB blob in a git history is not something you undo.

The dumps currently sitting here on the development machine are a good illustration of why:

| File | Size |
|---|---:|
| `dumps/hiwiki-latest-pages-articles.xml.bz2` | 227 MB |
| `dumps/tawiki-latest-pages-articles.xml.bz2` | 258 MB |

Half a gigabyte of input that Wikimedia already hosts, versions, and serves faster than
GitHub would.

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

The languages were chosen to exercise genuinely different segmentation behaviour, not for
variety: Hindi ends sentences with the danda (`।`), Japanese with `。` and no following
space, Arabic is right-to-left with its own question mark (`؟`), and Tamil has heavy use of
combining marks. A corpus of six European languages would have exercised one code path.

### It is synthetic, and that matters

Sentences are generated from per-language subject/verb/object templates. This makes the
corpus balanced, licence-free and small enough to commit, and it means a tokenizer trained
on it produces clean, inspectable output.

It also means the corpus has almost no lexical diversity: 3,050 whitespace-separated tokens
draw on only 137 distinct types. Models trained on it therefore have a collapsed embedding
space and near-identical similarity scores — the example's top hits all score above 0.99.
**Use it to verify the pipeline runs, never to judge model quality.** Every number this
project actually claims came from Wikipedia, not from here.

### Regenerating

The corpus is committed rather than generated at test time so that results are stable across
machines and runs. If you change it, re-run the examples and the getting-started
documentation, since both quote real output from it.

## The three file formats

The pipeline is three files in sequence. Knowing what each looks like is most of knowing how
to debug it.

### 1. Dump — `*-pages-articles.xml.bz2`

MediaWiki XML, bzip2-compressed, as published. Read with `iterparse` and cleared element by
element, so peak memory is one article regardless of dump size. Never decompress it first;
nothing needs you to.

### 2. Corpus — one JSON object per line

What `qfme extract` writes and every trainer reads. A real record, abbreviated:

```json
{
  "id": "10",
  "language": "hi",
  "title": "हम होंगे कामयाब",
  "text": "हम होंगे कामयाब ... प्रकाशित किया गया था।\n\nHum Honge Kamyab Lyrics",
  "source": "wikipedia",
  "license": "CC BY-SA 4.0",
  "sections": [{"heading": "सन्दर्भ", "text": "Hum Honge Kamyab Lyrics"}]
}
```

| Field | Required | Why it is there |
|---|---|---|
| `text` | **yes** | the only field a reader insists on |
| `id` | no | stable identity; used to detect false negatives during training |
| `language` | no | drives per-language fairness reporting |
| `title` | no | becomes the anchor of a `title_lead` pair |
| `sections` | no | `heading`/`text` pairs; **this is what makes `heading_section` mining possible** |
| `source`, `license` | no | provenance, carried through into metadata |

`sections` is the field worth defending. Flattening an article to a single blob would be
simpler and would destroy the structure that manufactures supervision one stage later.

### 3. Pairs — one JSON object per line

What `qfme mine-pairs` writes and `scripts/adapt_pretrained.py` reads:

```json
{"anchor": "हम होंगे कामयाब",
 "positive": "हम होंगे कामयाब ( का गिरिजा कुमार माथुर द्वारा ...) एक प्रतिरोध गीत है। ...",
 "kind": "title_lead", "document": "10", "language": "hi", "overlap": 1.0}
```

| Field | Meaning |
|---|---|
| `anchor` | the query side |
| `positive` | the passage side |
| `kind` | `title_lead`, `heading_section` or `adjacent` |
| `document` | source document id — lets training avoid treating a sibling pair as a negative |
| `language` | ISO code, carried from the corpus |
| `overlap` | share of anchor tokens already present in the positive; the leakage measure |

`overlap` is recorded **per pair**, not just aggregated, which is what makes the
`by_overlap` breakdown in an adaptation report possible — and that breakdown is the control
that shows adaptation is not learning to match strings.

Three kinds, three trade-offs:

| Kind | Anchor | Positive | Character |
|---|---|---|---|
| `title_lead` | article title | first paragraph | large and reliable, and the leakiest (mean overlap ≈ 0.98) |
| `heading_section` | section heading | section body | closest to a real query/passage pair (≈ 0.77) |
| `adjacent` | one paragraph | the next | works on any prose, needs no structure (≈ 0.48) |

## The Wikipedia path, end to end

Four commands take a URL to a trainable pair file. All of them stream; none needs a GPU.

```bash
# 1. Fetch a dump. ~200-300 MB for a mid-sized Indic wiki.
curl -o data/dumps/hiwiki-latest-pages-articles.xml.bz2 \
     https://dumps.wikimedia.org/hiwiki/latest/hiwiki-latest-pages-articles.xml.bz2

# 2. Extract. Namespace-0 articles only, redirects and stubs dropped.
qfme extract --dump data/dumps/hiwiki-latest-pages-articles.xml.bz2 \
             --output data/corpora/hi.jsonl.gz --language hi

# 3. Gate on quality before spending GPU time.
qfme validate --source data/corpora/hi.jsonl.gz --output reports/hi-audit.json

# 4. Mine pairs — all three kinds, leakage capped and reported per kind.
qfme mine-pairs --source data/corpora/hi.jsonl.gz \
                --output data/pairs/hi.jsonl.gz \
                --max-overlap 0.9 --report reports/hi-pairs.json
```

Input and output at each step:

| Step | In | Out | Notable flags |
|---|---|---|---|
| `extract` | `.xml.bz2` dump | corpus JSON Lines | `--minimum-characters` (200), `--limit`, `--keep-duplicates` |
| `validate` | corpus | findings + counts, non-zero exit on `ERROR` | `--strict`, `--output` |
| `mine-pairs` | corpus | pair JSON Lines | `--kinds`, `--max-overlap`, `--report` |

Output ending in `.gz` is gzipped automatically; every reader decompresses transparently, so
you never have to decide twice.

`qfme extract` needs the `wikipedia` extra (`mwparserfromhell`). Without it the command
raises a message naming the fix rather than an `ImportError` traceback.

Then, on the GPU box:

```bash
python scripts/adapt_pretrained.py \
    --checkpoint intfloat/multilingual-e5-small \
    --pairs data/pairs/hi.jsonl.gz \
    --query-prefix "query: " --passage-prefix "passage: " \
    --rank 32 --epochs 2 --batch-size 64 \
    --sample-pairs 120000 --train-pairs 20000 --eval-pairs 2000 \
    --output reports/hi-v1.json --save-adapter models/hi-v1
```

## When can multilingual Wikipedia training start?

**It already has.** Hindi and Tamil are complete end to end — dumps fetched, extracted,
audited, mined, adapted, measured, and the adapter saved as `models/indic-v1`. Nothing
blocks a third language, or a twentieth. The only prerequisites are a dump URL and disk.

### What it cost, measured

Steps 2–4, on an **Intel MacBook with no GPU** — the numbers come from
[`../verify-output/verification-report.txt`](../verify-output), a 9/9-stage run of 1h 30m 24s:

| Step | Hindi | Tamil |
|---|---:|---:|
| `extract` | 7.4s → 118,571 articles (143 MB) | 8.2s → 163,768 articles (134 MB) |
| `validate` | 11m 02s → 2,235,798 sentences | 12m 15s → 2,677,328 sentences |
| `mine-pairs` | 25m 04s → **642,536 pairs** | 28m 46s → **893,523 pairs** |
| Peak resident memory | **< 197 MB** | **< 201 MB** |
| Disk (corpus + pairs) | ~288 MB | ~276 MB |

**Roughly 40 minutes and 300 MB per language, on a laptop, unattended.** The GPU step is
minutes at 20,000 pairs. Data preparation dominates by an order of magnitude — which is good
news for a 22-language programme, because preparation parallelises across machines and needs
no scarce hardware.

Mined pair mix, and the leakage each kind carries:

| | Hindi pairs | mean overlap | Tamil pairs | mean overlap |
|---|---:|---:|---:|---:|
| `adjacent` | 414,166 | 0.50 | 507,058 | 0.47 |
| `heading_section` | 130,243 | 0.77 | 237,049 | 0.76 |
| `title_lead` | 98,127 | 0.98 | 149,416 | 0.98 |

### Train jointly, not one adapter per language

Concatenate the pair files. This was settled by a controlled experiment, not assumed: joint
training is numerically best on both languages, never worse than either specialist, and
produces one artefact instead of two.

```bash
cat data/pairs/hi.jsonl.gz data/pairs/ta.jsonl.gz > data/pairs/indic.jsonl.gz
```

Gzip members concatenate — `cat` on two `.gz` files is a valid `.gz` file — and sampling is a
reservoir over the whole file, so a mixed file yields a mixed sample without any
interleaving step.

### Which languages to add, and in what order

The controlled task/language experiment changes the obvious plan. Adaptation turned out to be
**language-general** (+95% of the achievable gain survives a Hindi→Tamil switch) and
**task-specific** (only −17% survives a pair-shape switch). Consequences:

1. **Mine where the text is cleanest and most abundant first.** Hindi, Tamil, Bengali,
   Telugu, Marathi have wikis large enough to matter. The smallest — Santali (Ol Chiki),
   Meitei (Meetei Mayek), Dogri — yield few pairs and, on this evidence, would have been
   largely covered by the larger languages anyway.
2. **Always mine all three kinds.** `--kinds` defaults to all three for this reason. A
   single-shape adapter forfeited 17% of the achievable gain when the shape was wrong.
3. **Cap the leakiest kind rather than dropping it.** `title_lead` averages 0.98 overlap and
   is still the second-largest source; `--max-overlap 0.9` keeps the volume and removes the
   pairs a string matcher would solve outright.
4. **Pin the evaluation set with `--eval-pairs-file`** whenever comparing runs, or the
   held-out split moves with the training filter and the comparison measures the wrong thing.
5. **Set `--sample-pairs` to several times `--train-pairs`** whenever a facet filter is used.
   Filters run after reservoir sampling, so without it a run naming a minority kind silently
   trains on less data. This actually happened — 25,000 pairs against 7,000.

### What is *not* yet possible from this directory

- **A non-Wikipedia corpus axis is untested.** Every comparison so far has Wikipedia on both
  sides. `--adaptation domain` exists for it and wants a pair file mined from real QuanFire
  documents — that is the run that would justify "this will help on our contracts".
- **No hard negatives.** Negatives are in-batch only; nothing here mines them against a base
  encoder. Most likely next source of gain.
- **No cross-lingual pairs.** Nothing mines translation pairs, so cross-lingual retrieval
  works only to the extent the corpus held comparable content.
- **No dump fetcher.** `curl` is the tool; there is no `qfme fetch`, deliberately — dump
  mirrors, resume and bandwidth policy are better handled by something that already does them.

## Bringing your own corpus

Three readers are available, named by `--format`. The default, `auto`, picks one by file
extension:

**Plain text** (`.txt`, `.text`, `.txt.gz`) — one file becomes one document; paragraphs come
from blank lines and sentences from the segmenter.

**JSON Lines** (`.jsonl`, `.jsonl.gz`, `.ndjson`) — one record becomes one document. Only a
text field is required; `id` and `language` are used when present, and any other keys are
carried through into document metadata.

**One sentence per line** — `--format lines`. This one has to be named. Extension sniffing
cannot tell a sentence-per-line file from prose, and reading it as prose would re-segment
text that is already segmented.

Point `--source` at a directory to read every matching file, visited in sorted order so runs
are reproducible. Gzip is handled transparently.

To mine pairs from your own documents, supply `sections` on each record and all three kinds
become available; supply only `text` and you get `adjacent` pairs, which need no structure at
all.

[`docs/data-format.md`](../docs/data-format.md) is the full contract: every field, the rules
an extraction must satisfy, and notes on Wikipedia dumps specifically.

### The domain path — mining pairs from text that is not Wikipedia

Every published result in this repository came from a MediaWiki dump, which leaves a fair
question open: is the miner coupled to Wikipedia, or was it merely pointed at one?

**It is not coupled.** [`corpus/pairs.py`](../src/multilingual_embedding/corpus/pairs.py)
reads three things and nothing else — a `title`, a list of `sections`, and blank-line
paragraphs in `text`. A matter file with a subject line and headed sections, an invoice with
its schedule of services, a policy with numbered clauses: each already *is* a title plus
sections plus paragraphs. Producing a domain corpus is a **format conversion, not a new
capability**, and that is the single most useful thing to know before planning Phase C.

[`data/sample/domain-corpus.jsonl`](sample/domain-corpus.jsonl) is a committed ten-document
example in exactly that shape — professional-services text, English and Hindi, no Wikipedia
anywhere in it. Mine it:

```bash
qfme mine-pairs --source data/sample/domain-corpus.jsonl \
    --output /tmp/domain-pairs.jsonl --report /tmp/domain-report.json
```

```
kind                  pairs   mean overlap
adjacent                 19           0.21
heading_section          30           0.22
title_lead                7           0.23
```

Ten documents, **56 pairs**, all three kinds. Five more were rejected and named as such
(one short anchor, four short positives), because a pair that vanishes without a reason is
a pair you cannot reason about.

**The trap: `sections` goes at the top level of the record.** Nested under an `attributes`
key it is silently invisible — `JsonlReader` flattens unrecognised *top-level* fields into
metadata, so a nested one never arrives. Mining still succeeds and still reports success; it
simply produces zero `heading_section` pairs, which on a large corpus is easy to miss
entirely. `tests/corpus/test_domain_pairs.py` demonstrates that failure rather than
describing it, because it has no error message of its own.

**A result worth checking, and not yet a result.** The overlap figures above sit near 0.22
across all three kinds. On Hindi Wikipedia the same miner returns **0.977** for `title_lead`
— an encyclopedia lead restates its title by convention, and lexical leakage was the
dominant difficulty in every experiment run so far. Business prose has no such convention:
an invoice's payment-terms section does not open by repeating the words "payment terms".

If that survives contact with a real export, the hardest problem in the Wikipedia work is
substantially smaller on domain text. **But this corpus is synthetic and was written by
someone who knew overlap would be measured**, so the figures are a hypothesis, not evidence.
The way to settle it costs one command: run `qfme mine-pairs --report` over a real export and
read `mean_overlap_by_kind`. Do that before assuming either answer.

**What the fixture does not do.** It does not train anything and cannot: Phase C's exit
criterion is a model trained on mined domain pairs beating the base checkpoint *on that
domain*, and that needs a real corpus and a GPU. What is settled is everything upstream of
the training run — the record shape, the reader, the miner, and the accounting.

## Audit before you train

```bash
qfme validate --source data/corpora/hi.jsonl.gz
```

`qfme validate` reads a corpus and reports the problems it can identify — leftover markup,
encoding damage, empty or duplicated documents, missing language codes, suspicious fragments
— each with an example and a remedy, alongside the document and sentence counts and the
language and script distribution.

It is worth running because the failures it catches are the ones that do not announce
themselves. A corpus with wiki markup still in the body text trains a tokenizer on syntax
rather than on language, and nothing about the aggregate statistics looks wrong; you find out
from a poor model weeks later. Empty records are skipped by the JSON Lines reader without a
count, so a pipeline that silently emits blanks loses them invisibly.

Findings are graded `ERROR` (unusable as it stands), `WARNING` (will train, but something was
probably lost upstream) and `INFO` (context worth having). The command exits non-zero on
errors, so a data pipeline can gate on it:

```bash
qfme validate --source "$OUTPUT" || exit 1              # blocks on errors
qfme validate --source "$OUTPUT" --strict || exit 1     # blocks on warnings too
qfme validate --source "$OUTPUT" --output audit.json    # machine-readable
```

It does find things. The Hindi extraction was flagged `usable: False` on
`encoding_damage` — a real finding on real data, from a corpus that looked fine by every
aggregate count. Tamil came back clean.

`qfme stats` remains the lighter option when you only want counts and distributions and not a
judgement.

## Pros and cons of this arrangement

**Pros**

- Nothing large is ever committed; a clone is small and a history stays small.
- Every non-sample file is reproducible from a URL and four commands, so "which data was
  this trained on" has an answer that fits in a paragraph.
- Streaming throughout: the largest thing processed here was a 258 MB compressed dump, at
  under 201 MB of resident memory.
- One committed corpus keeps examples, docs and the walkthrough executable in CI without any
  network access.
- Formats are plain JSON Lines. `gzcat | head -1 | jq` is a complete debugging toolkit.

**Cons**

- The sample corpus is synthetic and lexically collapsed. It proves the pipeline runs and
  nothing about model quality — a trap for anyone who reads the 0.99 similarity scores as a
  result.
- No dataset versioning or checksums. "latest" dumps change under you, and nothing here
  records which day's dump produced a corpus.
- No fetcher, no resume, no mirror selection; `curl` and a stable connection.
- `corpora/` and `pairs/` are conventions rather than enforced structure, so two people can
  end up with different layouts and neither is wrong.
- Disk grows fast — a corpus plus its pairs is roughly 1.3× the compressed dump, and the
  pair file is the larger half.
- Nothing prunes. Old corpora and pair files stay until deleted by hand.

## Done and pending

| | Status |
|---|---|
| Committed sample corpus, six languages, six scripts | ✅ done |
| Streaming readers: text, JSON Lines, sentence-per-line, gzip, directories | ✅ done |
| Wikipedia dump → corpus (`qfme extract`) | ✅ done, on 2 real dumps |
| Corpus audit with graded findings (`qfme validate`) | ✅ done |
| Corpus → contrastive pairs, three kinds, leakage measured (`qfme mine-pairs`) | ✅ done, 1.53M pairs mined |
| Joint multilingual pair files by concatenation | ✅ done, and validated by experiment |
| Domain (non-Wikipedia) pair mining | ⬜ pending — needs a real document set |
| Hard-negative mining | ⬜ pending — Phase C's remaining piece |
| Cross-lingual / translation pairs | ⬜ pending |
| Dataset versioning, checksums, dump-date provenance | ⬜ pending |
| `qfme fetch` for dumps | ❌ not planned — `curl` does this better |

Related reading: [`../src/multilingual_embedding/corpus/README.md`](../src/multilingual_embedding/corpus/README.md)
for how extraction and mining actually work,
[`../scripts/README.md`](../scripts/README.md) for the adaptation experiment,
and [`../ROADMAP.md`](../ROADMAP.md) for every measured result quoted above.
