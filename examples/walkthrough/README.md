# Walkthrough — seeing the project work

Nine steps, about twenty minutes, entirely on this machine. Every command and every
output below was executed on an Intel MacBook with no GPU; nothing here is predicted.

The point is not that commands run. It is that you can **check the claims yourself** —
that the vectors mean something, that the model actually learns, and that the parts which
do not yet work say so.

---

## 1. Install, and confirm it

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/) — nothing else. If `uv` is
missing: `curl -LsSf https://astral.sh/uv/install.sh | sh`

```bash
cd quanfire-multilingual-embedding
uv sync --extra neural --extra wikipedia
source .venv/bin/activate
```

About a minute the first time, mostly PyTorch. Then:

```bash
qfme --version
python -c "import torch; print('torch', torch.__version__)"
```

```
qfme 0.5.0
torch 2.2.2
```

**Two things that trip people up.**

`qfme` is not a system-wide command — it lives at `.venv/bin/qfme`. A terminal that has
not activated the environment says `command not found: qfme`, which is expected. Either
activate as above, or prefix every command with `uv run` (`uv run qfme --version`), or
call it by path (`.venv/bin/qfme --version`).

And **pass both extras**. Plain `uv sync` does not merely skip them, it uninstalls them —
you would lose the contextual encoder used in step 7 and the extractor used in step 9, with no error explaining why.

Steps 2–6 work without either extra, so a torch-free install is a valid way to start.

---

## 2. Look at a corpus before touching it

```bash
qfme stats --source data/sample/corpus.jsonl
```

```json
{
  "document_count": 150,
  "sentence_count": 750,
  "unique_words": 235,
  "languages": { "ar": 25, "en": 25, "fr": 25, "hi": 25, "ja": 25, "ta": 25 },
  "scripts":   { "Arab": 25, "Deva": 25, "Hani": 15, "Hira": 10, "Latn": 50, "Taml": 25 }
}
```

Six languages, six scripts, 25 documents each. Note `Hani` and `Hira` both appear for
Japanese — script detection is per character, not per document.

---

## 3. Make the corpus prove itself

This is the step most pipelines skip, and the reason models quietly underperform.

```bash
qfme validate --source data/sample/corpus.jsonl
```

```
documents  150
sentences  750
languages  ar, en, fr, hi, ja, ta
scripts    Arab, Deva, Hani, Hira, Latn, Taml

No problems found.
```

Now run it against a deliberately damaged extraction, shipped here as
`broken-extraction.jsonl` — six records carrying the failures a real Wikipedia extraction
produces:

```bash
qfme validate --source examples/walkthrough/broken-extraction.jsonl
```

```
ERROR   1 documents still contain wiki or HTML markup.
        e.g. w1
        -> Finish the extraction. Markup trains the tokenizer on syntax rather than language.
ERROR   1 documents contain replacement characters.
        e.g. w2
        -> The source was decoded with the wrong encoding. Re-extract as UTF-8.
WARNING 1 documents duplicate earlier ones.
WARNING 1 documents declare no language.
WARNING 1 documents are shorter than 40 characters.
```

**Check the exit code, because that is what a pipeline gates on:**

```bash
qfme validate --source examples/walkthrough/broken-extraction.jsonl; echo "exit $?"
```

| Corpus | Flag | Exit |
|---|---|---|
| clean | — | `0` |
| errors present | — | `1` |
| warnings only | — | `0` |
| warnings only | `--strict` | `1` |

Not one of those six records would have raised an exception during training.

---

## 4. Train a model, then search it

```bash
qfme train --config examples/walkthrough/experiment.yaml
```

```json
{
  "name": "demo",
  "documents": 150,
  "sentences": 750,
  "vocabulary_size": 226,
  "dimension": 64,
  "characters_per_token": 2.922,
  "unknown_rate": 0.0,
  "experiment_directory": "artifacts/demo"
}
```

Under two seconds. SentencePiece writes a wall of progress to stderr — that is its own
logging, not this project's; append `2>/dev/null` to silence it.

**Try setting `vocab_size: 500` in the config.** It fails, and the failure is the point:

```
error: vocab_size exceeds what the training corpus can support; reduce vocab_size or
supply more text (... 'Vocabulary size too high (500). Please set it to a value <= 327.')
```

It tells you the number to use. That is the intended behaviour of configuration errors
throughout — fail at load, say what to do.

Now query it, in three scripts, against a model that was never told they were different
languages:

```bash
qfme search --experiment artifacts/demo --source data/sample/corpus.jsonl --query "machine learning" --top-k 3
qfme search --experiment artifacts/demo --source data/sample/corpus.jsonl --query "मशीन लर्निंग"      --top-k 3
qfme search --experiment artifacts/demo --source data/sample/corpus.jsonl --query "இயந்திர கற்றல்"    --top-k 3
```

```
 1. [0.9713] The teacher studies machine learning.
 1. [0.9844] अभियंता देखता है मशीन लर्निंग।
 1. [0.9851] ஆய்வாளர் பார்க்கிறார் இயந்திர கற்றல்.
```

**Read those scores sceptically.** 0.97+ looks excellent and mostly reflects a small
templated sample corpus, not model quality. Retrieval *within* a language works here;
querying in Hindi does **not** return English results, because nothing in this corpus
aligns the languages. That limit is real and is discussed in step 8.

---

## 5. See the per-language fairness

```bash
qfme evaluate --experiment artifacts/demo --source data/sample/corpus.jsonl
python -c "import json;print(json.load(open('reports/demo/report.json'))['tokenizer_by_language'])"
```

| Language | chars/token | fertility |
|---|---:|---:|
| Tamil | 4.07 | 2.05 |
| English | 3.98 | 1.67 |
| French | 3.64 | 1.96 |
| Arabic | 2.86 | 2.20 |
| Hindi | 2.75 | 2.00 |
| Japanese | **1.07** | **13.32** |

Japanese costs roughly **3.8× more tokens per character** than Tamil. An average would
have hidden that. This is the number to watch when you add the scheduled Indian
languages — a language that tokenises badly will underperform for reasons that have
nothing to do with the embedding model.

Note `similarity_correlation` and `analogy_accuracy` report `None`, not `0.0`. No
benchmark dataset was supplied, so no score is claimed.

---

## 6. See why the static model is not enough

```bash
python examples/walkthrough/03_static_limit.py
```

```
Nearest neighbours of 'bank':
   approved   0.9800
   loan       0.9746
sim(river, savings)   = 0.4222
occurrences of 'bank' = 320
rows in the matrix    = 36
```

The corpus uses `bank` in two senses — *river bank* and *savings bank* — in equal
measure. The model has **one row** for it, so the finance sense wins outright and the
river sense is unrepresentable. No amount of extra data fixes this; it is the shape of
the model.

That is the entire reason the next step exists.

---

## 7. Prove the contextual encoder actually learns

A transformer that emits well-formed vectors containing no information passes every
shape and contract test. Only a retrieval test catches it, so that is what this does.

```bash
python examples/walkthrough/04_contextual_learns.py
```

```
parameters: 28,288   device: cpu

BEFORE training: within-topic +0.568  cross-topic +0.422  margin 0.146
AFTER  training: within-topic +0.790  cross-topic -0.703  margin 1.492

loss 0.921 -> 0.458  over 118 steps
improved: True
```

Two topics with no shared vocabulary. Before training the model barely separates them —
everything looks vaguely similar to everything. After 118 steps, **cross-topic similarity
has gone negative**: the model has learned that these are actively different things. The
margin went up roughly tenfold, in about four seconds on CPU.

Those exact numbers reproduce on every run. That is deliberate — the tokenizer uses
blake2b rather than Python's built-in `hash`, which is randomised per process and would
otherwise make each run print something different.

Change `epochs=24` to `epochs=1` and the margin drops to **0.342** with cross-topic
similarity still positive at +0.283 — better than the untrained 0.146, but nowhere near
separated. Training longer is doing the work; the number is not just decoration.

---

## 8. See what domain adaptation costs

```bash
python examples/walkthrough/05_lora_economics.py
```

```
rank                              16
total parameters         109,737,984
trainable parameters         884,736
trainable share                0.81%
full model on disk         415.2 MB
adapter on disk              3.4 MB
Adam state, full            0.81 GB
Adam state, LoRA             6.8 MB
```

**The rank is the whole story** — set `RANK = 8` and every figure halves. A LoRA number
quoted without its rank is meaningless.

This is what makes the domain-specific plan affordable: one frozen base model plus a
3.4 MB adapter per domain, rather than a 415 MB model per domain.

---

## 9. Extract real text, and see what a dump is really like

Everything above ran on 150 synthetic templated documents. This is how you get real text.

Start with a small wiki so the whole loop takes a minute. Meetei Mayek is 5 MB:

```bash
curl -O https://dumps.wikimedia.org/mniwiki/latest/mniwiki-latest-pages-articles.xml.bz2

qfme extract --dump mniwiki-latest-pages-articles.xml.bz2 \
             --output data/wikipedia/mni.jsonl.gz --language mni
```

```
WARNING  Dropped duplicate articles, usually template-generated stubs
WARNING  Dropped articles whose source markup was malformed beyond repair
Wrote 2444 articles to data/wikipedia/mni.jsonl.gz
Next: qfme validate --source data/wikipedia/mni.jsonl.gz
```

About sixteen seconds. Then do what it tells you:

```bash
qfme validate --source data/wikipedia/mni.jsonl.gz
```

```
documents  2444
sentences  45059
languages  mni
scripts    Beng, Deva, Latn, Mtei
```

No errors — the extractor is held to the standard the audit sets.

**Look at what was thrown away**, because it is most of the file:

| | Pages |
|---|---:|
| Seen in the dump | 15,514 |
| Redirects and non-article namespaces | −4,348 |
| Shorter than 200 characters | −8,547 |
| Malformed markup | −24 |
| Duplicates | −151 |
| **Written** | **2,444** |

**84% of a dump is not article prose.** Size a corpus from what survives, not from the
page count. Those 151 duplicates are one boilerplate sentence repeated across 118 country
stubs — left in, they would inflate every token in that sentence by two orders of
magnitude.

Use `--limit 500` to try a dump before committing to it. For a real corpus, Hindi and
Tamil are 227 MB and 258 MB, and take a few minutes each.

Note the four scripts in a Meetei Mayek wiki: real text is messier than the sample corpus,
and the audit is how you find that out rather than discovering it in a model.

---

## 10. Turn that corpus into training pairs

An extracted corpus is not yet supervision. `qfme mine-pairs` manufactures it out of
article structure:

```bash
qfme mine-pairs --source data/wikipedia/mni.jsonl.gz \
                --output data/pairs/mni.jsonl.gz \
                --max-overlap 0.9 --report reports/mni-pairs.json
```

Each line is an anchor, a positive, the kind it came from, the source document, and the
lexical overlap between the two:

```json
{"anchor": "...", "positive": "...", "kind": "title_lead",
 "document": "10", "language": "mni", "overlap": 1.0}
```

Three kinds are mined by default, and the mean overlap of each is reported because it is
the number that says how much of the task a string matcher could solve on its own. On the
real Hindi run:

| Kind | Pairs | Mean overlap |
|---|---:|---:|
| `adjacent` | 414,166 | 0.50 |
| `heading_section` | 130,243 | 0.77 |
| `title_lead` | 98,127 | 0.98 |

`title_lead` at 0.98 is the reason `--max-overlap` exists. It is still worth mining — it is
the second-largest source — but capping it removes the pairs that are trivially solvable.

**Cost, measured on an Intel MacBook with no GPU:** Hindi took 25m 04s to mine 642,536
pairs from 118,571 articles; Tamil 28m 46s for 893,523 from 163,768. Peak resident memory
under 201 MB in both cases, because nothing is held in memory that does not have to be.

## 11. Adapt a published model, and measure whether it helped

This is the step the project exists for, and it needs a GPU:

```bash
qfme adapt --config examples/adaptation.yaml --profile configs/gpu.yaml \
    --set adaptation.pairs=data/pairs/hi.jsonl.gz \
    --set compute.batch_size=64 \
    --rank 32 --epochs 2 \
    --save-adapter models/hi-v1 --output reports/hi-v1.json
```

It scores the checkpoint first, then trains, then scores again on the same held-out pairs.
The before/after comparison is the whole point — beating chance proves nothing, and
fine-tuning a well-pretrained model on a narrow corpus can easily make it worse. The
command exits non-zero if the adapted model lost, so this is safe to put in a script.

`compute.batch_size` is set explicitly here because `gpu.yaml` raises it to 256 and the
figures below came from a run at 64. Batch size is the one compute setting that changes the
result as well as the memory use — it is also the number of in-batch negatives — so a
comparison has to pin it. The flag-driven command that produced those figures still works
unchanged:

```bash
python scripts/adapt_pretrained.py \
    --checkpoint intfloat/multilingual-e5-small \
    --pairs data/pairs/hi.jsonl.gz \
    --query-prefix "query: " --passage-prefix "passage: " \
    --rank 32 --epochs 2 --batch-size 64 \
    --sample-pairs 120000 --train-pairs 20000 --eval-pairs 2000 \
    --output reports/hi-v1.json --save-adapter models/hi-v1
```

Measured on an RTX 4070 Ti SUPER, against ~2,000 held-out pairs, training **0.50% of
parameters**:

| | Hindi base | Hindi adapted | Tamil base | Tamil adapted |
|---|---:|---:|---:|---:|
| recall@1 | 0.4238 | **0.5451** (+28.6%) | 0.3219 | **0.4535** (+40.9%) |
| MRR | 0.5136 | 0.6364 (+23.9%) | 0.3931 | 0.5397 (+37.3%) |

The weaker language gained more, which is the argument for doing this at all. The output
is a 3.4 MB adapter, and serving it is one call:

```python
from multilingual_embedding.pipelines import SemanticSearchPipeline

pipeline = SemanticSearchPipeline.from_adapter("models/hi-v1")
```

`from_adapter` restores the query and passage prefixes recorded in the artefact. Serving an
E5 model without them produces plausible vectors encoding the wrong thing, with nothing
raising — which is why they are stored rather than left to whoever loads it.

[`docs/reading-results.md`](../../docs/reading-results.md) is how to read the report
without fooling yourself.

---

## What this does and does not show

**Shown, end to end:** installation, corpus preparation, auditing that catches real
extraction damage, tokenizer and vocabulary training, three families of embedding model,
retrieval in three scripts, per-language fairness reporting, evidence the contextual model
learns, the economics of domain adaptation, extraction of a real Wikipedia dump, mining it
into contrastive pairs, and adapting a published checkpoint with a measured before/after.

Steps 1–9 run on the development machine with no GPU. Steps 10 and 11 split: mining runs
anywhere, adaptation needs CUDA.

**Not shown, because it does not work yet:**

- **A transformer trained from scratch has no CLI path.** `qfme train` trains the static
  model and `qfme adapt` runs the adaptation experiment shown in step 11, but training the
  contextual encoder in this repository from nothing is still Python-API only. The three
  families share the `TextEncoder` contract; two of the three also share the command line.
- **Cross-lingual retrieval.** A Hindi query does not find English passages. That needs
  aligned training pairs, and nothing here mines them.
- **Hard negatives.** Contrastive training uses in-batch negatives only.
- **Loading external weights into *this* transformer.** The encoder here is pre-norm and
  most published ones are post-norm, and the shapes match — so a cross-load would succeed
  and be silently wrong. Published checkpoints are therefore adapted through their own
  library instead, which is what step 11 does; what remains impossible is the specific act
  of initialising this project's architecture from someone else's file.
- **CUDA verified by a test.** The GPU results in step 11 were produced by hand on a 4070
  Ti SUPER. No automated test exercises a CUDA path, because development happens on a
  machine without one. Expect device-specific problems to appear first on the training box.

**A note on the sample corpus.** It is small and templated, which makes it ideal for
seeing the mechanics and useless for judging quality. Real judgement needs real text —
see [`docs/data-format.md`](../../docs/data-format.md) for the format to target and
`qfme validate` for checking what you extract.
