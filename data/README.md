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
- [The legal-domain path: two front doors and a wall](#the-legal-domain-path-two-front-doors-and-a-wall)
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
| `corpora/judgments/` | no | public court-judgment PDFs, the legal **training** front door | downloaded, then `qfme extract-judgments` |
| `corpora/milpac/` | no | MILPaC `.xlsx` workbooks, the legal **evaluation** front door | downloaded, then `qfme prepare-eval` |
| `pairs/` | no | mined contrastive pairs, one file per language | `qfme mine-pairs` |
| `eval/` | no | held-out evaluation pair files | `qfme prepare-eval` |
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

### The rule, stated: no customer text, anywhere

**No customer or client text is committed to this repository, and none is used to train
anything in it.** Every model this project has published was adapted on Wikipedia. The two
corpora tracked here are generated from templates. Nothing else has ever been in the
history.

Size is the reason the `.gitignore` rule was written; this is the reason it matters. A
corpus deleted in a later commit is still in the history, so there is no version of this
mistake that gets corrected afterwards — which makes it one of the few decisions here worth
enforcing rather than remembering.

[`tests/test_data_policy.py`](../tests/test_data_policy.py) asserts the part of that a
machine can check: every JSON Lines file git tracks, anywhere in the repository, declares a
`source` on every record, and every one of those begins `synthetic-`. A sample added with
no `source` field fails, which is deliberate — the file most likely to be real is the one
whose provenance nobody thought to write down.

Two things it cannot check, and that therefore stay a matter of judgement:

- **Text that is not in a corpus file.** A client sentence pasted into a Python fixture, a
  docstring or a markdown example is invisible to it. Nothing here reads prose looking for
  a name.
- **The weights.** A model adapted on customer text carries that text, and an adapter is
  opaque to every check above. `models/indic-v1/adapter.json` records `trained_on` as a
  *path*, which says nothing about provenance — so a second field now does. Every saved
  adapter carries a required `data_provenance` of `public`, `synthetic` or `licensed`, with
  no default and no entry for customer data: an adapter cannot be written without a human
  stating where its training text came from, and there is deliberately no value that says
  "customer", because that text must not reach training in the first place. It does not
  prove the weights are clean — nothing can read that off an adapter — but it turns the one
  question that matters from something a path leaves blank into something the save refuses
  to skip.

If a domain-adapted model is ever wanted for real QuanFire documents, that is the decision
to take first and explicitly: public domain-specific text, synthetic text written to the
same shape (which is what `sample/domain-corpus.jsonl` demonstrates), or a per-tenant
arrangement with a contractual basis. It is not a decision to arrive at by default because
a pair file happened to be on the training machine.

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
    --output reports/hi-v1.json --save-adapter models/hi-v1 \
    --data-provenance public
```

`--data-provenance` is required the moment `--save-adapter` is set: the saved adapter
records where its training text came from as a fact about the model, and the run is refused
before it starts if the declaration is missing. `public` here because this is Wikipedia.

## The legal-domain path: two front doors and a wall

Wikipedia is a general-domain source. The domain this project actually targets is Indic
legal text, and it enters through **two** front doors that are deliberately kept apart —
one for training, one for evaluation — with a wall between them that is structural, not a
matter of discipline.

| | Training front door | Evaluation front door |
|---|---|---|
| Module | [`corpus/judgments.py`](../src/multilingual_embedding/corpus/judgments.py) | [`corpus/milpac.py`](../src/multilingual_embedding/corpus/milpac.py) |
| Source | public court judgments (official court portals) | MILPaC parallel legal corpus |
| Shape | PDF, full text, one file per case | `.xlsx`, English aligned to nine Indic languages |
| Licence | **statutory public domain** — Copyright Act §52(1)(q) | **CC BY-NC-SA 4.0** — non-commercial |
| Command | `qfme extract-judgments` | `qfme prepare-eval` |
| Extra | `judgments` (`pypdf`) | `milpac` (`openpyxl`) |
| Feeds | `qfme mine-pairs`, then `--pairs` | `qfme adapt --eval-pairs-file`, never `--pairs` |

**Why two doors and not one split.** A model scored on text from the same source it trained
on is scored on its own memorisation. Holding the evaluation set out *by origin* — a
different corpus, a different register, a different licence — is stronger than a random
split of one corpus, because a random split still shares vocabulary and formatting across
the boundary. So judgments train and MILPaC scores, and the two never mix.

**Why the wall is also a licence.** MILPaC is non-commercial. Scoring a shipped adapter
against it is fair use of a benchmark; *training* that adapter on it is not, because the
adapter is a commercial artefact and the NC term travels into it. `corpus/milpac.py`
therefore produces an evaluation file and nothing else, stamps `license` on every record so
the restriction is legible in the raw file, and does **not** route through the pair miner —
there is no code path by which MILPaC can reach the training side. Judgments, public domain
by statute (§52(1)(q)) and drawn only from official court portals, carry no such restriction,
which is exactly why they are the training source.

```bash
# TRAINING side — public judgments, statutory public domain (§52(1)(q)),
# official court portals only, commercial use permitted.
# 1. Download a judgment collection into data/corpora/judgments/ (PDFs).
# 2. Extract to corpus JSON Lines. Scanned PDFs with no text layer are
#    dropped and counted — this does not OCR.
qfme extract-judgments --source data/corpora/judgments/ \
                       --output data/corpora/judgments.jsonl.gz --language en

# 3. Gate on quality, exactly as for Wikipedia. PDF extraction is the step
#    whose fidelity cannot be trusted until it is checked.
qfme validate --source data/corpora/judgments.jsonl.gz --output reports/judgments-audit.json

# 4. Mine pairs to train on.
qfme mine-pairs --source data/corpora/judgments.jsonl.gz \
                --output data/pairs/judgments.jsonl.gz --report reports/judgments-pairs.json

# EVALUATION side — MILPaC, CC BY-NC-SA, scoring only.
# 5. Download the MILPaC workbooks into data/corpora/milpac/ (.xlsx), then
#    build the held-out pair file. hi and ta only, the pair this project adapted for.
qfme prepare-eval --source data/corpora/milpac/ --output data/eval/milpac-hi-ta.jsonl.gz

# 6. Adapt on judgments, score on MILPaC. The eval file goes to
#    --eval-pairs-file, never --pairs; the origin wall depends on it.
python scripts/adapt_pretrained.py \
    --checkpoint intfloat/multilingual-e5-small \
    --pairs data/pairs/judgments.jsonl.gz \
    --eval-pairs-file data/eval/milpac-hi-ta.jsonl.gz \
    --query-prefix "query: " --passage-prefix "passage: " \
    --save-adapter models/legal-v1 --data-provenance public
```

**Status: done in code, unrun on real data.** Both readers are built and tested against
fixtures — a MILPaC workbook and a fake PDF reader — with no network and no downloaded
corpus. What a fixture cannot tell you is whether a *real* judgment collection extracts to
clean prose or to image-only PDFs that yield nothing, and whether MILPaC's columns match
what its distribution actually ships. Those are properties of the downloaded data, checkable
only once it is in `data/corpora/` and run through `qfme validate` — the same honesty the
Wikipedia path holds itself to, said out loud rather than assumed. The judgment reader in
particular does **not** OCR: a scanned collection is dropped and counted, not silently
turned into empty records.

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
| Cross-lingual / translation pairs | ✅ done — MILPaC parallel units, held out for evaluation |
| Legal-domain training front door: judgment PDFs → corpus (`qfme extract-judgments`) | ✅ done in code — unrun until a real collection is downloaded |
| Legal-domain evaluation front door: MILPaC → held-out pairs (`qfme prepare-eval`) | ✅ done in code |
| Dataset versioning, checksums, dump-date provenance | ⬜ pending |
| No-customer-text policy, asserted on every tracked corpus file | ✅ done — `tests/test_data_policy.py` |
| Training-data provenance recorded in a saved adapter | ✅ done — `data_provenance` is a required field on every saved adapter |
| `qfme fetch` for dumps | ❌ not planned — `curl` does this better |

Related reading: [`../src/multilingual_embedding/corpus/README.md`](../src/multilingual_embedding/corpus/README.md)
for how extraction and mining actually work,
[`../scripts/README.md`](../scripts/README.md) for the adaptation experiment,
and [`../ROADMAP.md`](../ROADMAP.md) for every measured result quoted above.
