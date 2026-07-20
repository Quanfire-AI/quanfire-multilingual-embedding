# Walkthrough — seeing the project work

Eight steps, about fifteen minutes, entirely on this machine. Every command and every
output below was executed on an Intel MacBook with no GPU; nothing here is predicted.

The point is not that commands run. It is that you can **check the claims yourself** —
that the vectors mean something, that the model actually learns, and that the parts which
do not yet work say so.

```bash
cd /path/to/quanfire-multilingual-embedding
uv sync --extra neural        # the extra is needed from step 5 onward
source .venv/bin/activate
```

---

## 1. Confirm the install

```bash
qfme --version
python -c "import torch; print('torch', torch.__version__)"
```

```
qfme 0.1.0
torch 2.2.2
```

If `torch` fails to import, steps 1–4 still work. Only the contextual model needs it.

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

## What this does and does not show

**Shown, end to end:** corpus preparation, auditing that catches real extraction damage,
tokenizer and vocabulary training, two families of embedding model, retrieval in three
scripts, per-language fairness reporting, evidence the contextual model learns, and the
economics of domain adaptation.

**Not shown, because it does not work yet:**

- **The contextual encoder has no CLI path.** `qfme train` trains the static model only.
  The transformer is Python-API only — the two share the `TextEncoder` contract, not the
  command line.
- **Cross-lingual retrieval.** A Hindi query does not find English passages. That needs
  aligned training pairs, which this corpus does not contain.
- **External pretrained checkpoints.** This encoder is pre-norm; most published ones are
  post-norm, so their weights do not load directly.
- **Anything on a GPU.** No CUDA hardware here, so those paths are unexercised. Expect
  device-specific problems to appear first on the training box, not on this machine.

**A note on the sample corpus.** It is small and templated, which makes it ideal for
seeing the mechanics and useless for judging quality. Real judgement needs real text —
see [`docs/data-format.md`](../../docs/data-format.md) for the format to target and
`qfme validate` for checking what you extract.
