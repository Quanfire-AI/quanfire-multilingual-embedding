---
license: apache-2.0
base_model: intfloat/multilingual-e5-small
library_name: transformers
pipeline_tag: text-ranking
tags:
  - reranker
  - cross-encoder
  - text-ranking
  - legal
  - statute
  - retrieval
  - e5
language:
  - en
---

# Quanfire Statute Reranker — `rerank-statute-en` (cross-encoder, English central statutes)

> ### ✅ Measured — 2026-08-22 · **+47.2 % Recall@1** on a document-isolated split
>
> This model has a valid published number again. Read the next sentence before you quote it.
>
> **It is not a corrected +63.8 %.** The +63.8 % this card once carried was withdrawn on
> 2026-08-21 and stays withdrawn — permanently, as a measurement of nothing. This is a
> *different run* on a *different* split, at *different* volume, under *different* checkpoint
> discipline. It stands on its own or not at all, exactly as this card pre-committed it would.
>
> | Stage | Recall@1 | 95 % CI (paired bootstrap, B=2000) |
> |---|---|---|
> | bi-encoder (`multilingual-e5-small`) retrieve only | 0.0723 | [0.0589, 0.0857] |
> | **+ this cross-encoder rerank** | **0.1064** | [0.0910, 0.1218] |
>
> **Delta = +0.0341 → +47.2 %**, paired 95 % CI **[+0.0207, +0.0482]** — **excludes 0**.
> **1,494** held-out queries against a pool of **1,205** passages, and **no held-out query
> comes from an Act that appears anywhere in training**. Recall@100 ceiling **0.7557**; of the
> queries whose gold passage is recoverable at all, the reranker puts it at #1 for **14.1 %**.
>
> **Two caveats travel with this number.**
>
> 1. **Attribution.** Two things changed at once between the null re-measure and this run —
>    **2.93×** the training volume, and **best-checkpoint selection** instead of last-step.
>    They landed in a single run and **cannot be separated**. +47.2 % is the effect of the
>    pair; neither ablation was run.
> 2. **Volume.** This run trained on **77.1 %** of the volume the withdrawn run used
>    (26,252 rows against 34,052). So no
>    arithmetic between the two figures means anything, and in particular **nothing here is
>    "the cost of the leak"** — the comparison is not volume-matched and never will be.
>
> The absolute numbers are low because the task is hard and the pool is document-isolated: the
> first-stage bi-encoder gets 7.2 % at rank 1 here. Judge the reranker by the delta and its
> interval, not by 0.1064.

<details>
<summary><b>The record — the withdrawal (2026-08-21) and the null re-measure that preceded this.</b> Retained in full; the figures inside it must not be quoted.</summary>

> ### 🛑 Number withdrawn — 2026-08-21
>
> **The +63.8 % improvement this card reported is withdrawn. It was measured on a
> contaminated evaluation split and no replacement number exists yet.**
>
> This reranker's training script built its split with a plain shuffle —
> `random.shuffle(rows)`, first 1,200 rows to eval, the rest to train — with **no
> exclusion of any kind** between the two sides. Replaying that exact split shows:
>
> | held-out queries (1,200) | share |
> |---|---|
> | whose **positive passage text** also appears in training | 928 (77.3 %) |
> | whose **query text** also appears in training | 829 (69.1 %) |
> | drawn from a **source Act** that also appears in training | 1,200 (100 %) |
> | genuinely unseen on both text and document | **0 (0 %)** |
>
> Every held-out query came from an Act the model had trained on, and three in
> four had their answer passage verbatim in the training set. A reranker measured
> that way is being asked to recognise text it has already seen.
>
> **What is *not* withdrawn.** The weights are unchanged and the model is not
> being unpublished. The build lesson below — that form-separable negatives make
> a cross-encoder learn a query-independent shortcut, and that negatives must
> match the candidate distribution seen at inference — was diagnosed from a
> training collapse, not from this eval, and stands on its own.
>
> **What replaces it — nothing, so far. Updated 2026-08-23.** The clean re-measure
> has now run, on a split that holds out whole Acts. **It came back null:**
>
> | 1,494 queries, pool of 1,205 | Recall@1 | 95 % CI |
> |---|---|---|
> | bi-encoder alone | 0.0723 | [0.0589, 0.0857] |
> | + this reranker | 0.0823 | [0.0689, 0.0971] |
> | **delta** | **+0.0100 (+13.9 %)** | **paired [−0.0027, +0.0228] — spans zero** |
>
> The paired interval includes zero, so on an honest split **we cannot distinguish
> this reranker from no reranker at all.** It does not clear our own bar, which is
> that a shipped claim carries a paired CI excluding zero. **This model still has
> no published effectiveness number, and now has a measured failure to produce one.
> Do not quote +63.8 % anywhere.**
>
> **We are not claiming the leak caused this, and the earlier promise on this card
> that the two runs would be "comparable" was not kept.** The pool sizes match
> (1,205 vs 1,182), but the training volumes do not: the clean run trained on
> **8,966 rows against the withdrawn run's ~34,052 — 26.3 %**. Three separate
> things all push the clean number down and none of them can be separated from the
> leak or from each other:
>
> 1. The trainer filtered the corpus to rows carrying mined negatives — cutting it
>    by 73 % — and then never used those negatives, because negatives are drawn
>    from other rows' positives. The filter was made obsolete by the very fix that
>    rescued the first training run, and nothing caught it.
> 2. The saved weights came from the run's final step, with no best-checkpoint
>    tracking, and the last 1,000 steps ran at roughly 1.7× the epoch-mean loss. We
>    measured a checkpoint we know was not the best one the run produced.
> 3. Learning rate was constant with an effective batch of one query.
>
> So "the model is ineffective" and "we starved the model" both fit this evidence,
> and this card will not pretend to know which. A corrected run — full training
> volume and checkpoint selection on a validation slice carved from *training*
> documents, never the evaluation set — is in flight. **If it produces a
> significant number, that number will not be presented as a replacement for
> +63.8 %.** Different volume, different checkpoint discipline; it would stand on
> its own or not at all.
>
> A correction of record: an earlier internal notice attributed this to the
> shared fine-tuning pipeline's split filter. That was wrong — this script never
> used that pipeline. The defect here was its own, and simpler.

</details>


A **cross-encoder reranker** for English **central-statutory** (bare-Act) text. It reads a
query and a candidate passage **together** and returns a single relevance score, used to
**reorder the top-k of a bi-encoder retriever** — the second stage of a retrieve-then-rerank
pipeline. It is a fine-tune of
[`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)
(MIT) with a scalar relevance head (`AutoModelForSequenceClassification`, `num_labels=1`),
`max_length` 256.

This is Quanfire's **first reranker**. Where a bi-encoder embeds query and passage
independently and ranks by cosine, a cross-encoder attends across the pair — slower, but able
to catch relevance a bi-encoder misses. It is a companion to
[`quanfire-ai/embed-statute-en`](https://huggingface.co/quanfire-ai/embed-statute-en): that
model retrieves, this one re-ranks what it retrieves.

- **Framework & code:** [github.com/quanfire-ai/quanfire-multilingual-embedding](https://github.com/quanfire-ai/quanfire-multilingual-embedding) (Apache-2.0)
- **Weights licence:** Apache-2.0. The model is ours; the training text is bare-Act statutory
  content, train-safe under Copyright Act §52(1)(q)(ii) for a **non-reconstructive** model
  (it emits a *score*, never the source text). The corpus is **not** redistributed.
- **Internal run (these weights):** `statute-reranker-v2-fulldata` · base e5-small + scalar
  head · listwise cross-entropy over 1 positive + 6 form-matched negatives · lr 2e-5, 2 epochs,
  `max_length` 256, seed 0 (bf16, CUDA) · 26,252 training rows · checkpoint `e1s25748` selected
  on a 600-row validation slice carved from *training* documents (val loss 0.2891 vs the final
  step's 0.3052). The earlier, unmeasured weights remain pinned at tag **`v1.0.0`**.

## What it is for

**Re-ranking a retriever's shortlist on English central-statutory text.** Given a query and
the top-k passages a first-stage retriever returned, it scores each `(query, passage)` pair
and sorts by score. It does **not** retrieve on its own — it needs a candidate set (typically
a bi-encoder's top-50 or top-100). Its job is to lift the *right* section from somewhere in
that shortlist up to rank 1.

## Scope — read this before you use it

| | Validated? |
|---|---|
| **Re-ranking** first-stage results over English central-statutory (bare-Act) text | ✅ **Yes** — significant, measured below |
| Court **judgments / case law** | ❌ **Not this model** — judgments are [`embed-legal-en`](https://huggingface.co/quanfire-ai/embed-legal-en)'s domain; a reranker for them is not trained |
| State legislation, rules, notifications, and non-English statute | ❌ **Not validated** — central bare-Act English only |
| Use as a **retriever** (no candidate set) | ❌ **Wrong tool** — a cross-encoder cannot score a whole corpus economically; pair it with a bi-encoder |

## How to use

Rerank a first-stage retriever's shortlist. The model scores each pair; sort descending.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

name = "quanfire-ai/rerank-statute-en"
tok = AutoTokenizer.from_pretrained(name)
ce = AutoModelForSequenceClassification.from_pretrained(name).eval()

query = "What is the presumptive taxation rate for eligible small businesses?"
candidates = [                      # e.g. the top-k a bi-encoder returned
    "Section 44AD. ... a sum equal to eight per cent of the total turnover ...",
    "Section 44AE. ... in respect of each goods carriage ...",
    "Section 80C. ... deduction in respect of life insurance premia ...",
]

with torch.no_grad():
    enc = tok([query] * len(candidates), candidates, padding=True,
              truncation=True, max_length=256, return_tensors="pt")
    scores = ce(**enc).logits.squeeze(-1)          # higher = more relevant

ranked = [c for _, c in sorted(zip(scores.tolist(), candidates), reverse=True)]
```

It is also loadable as a `sentence_transformers.CrossEncoder(name)` if you prefer that API.

## Results

Evaluated the **production way**: a bi-encoder (`multilingual-e5-small`) retrieves the top-100
for each query, this cross-encoder rescores those 100, and we compare **Recall@1**. 95 % CIs
are by **paired** bootstrap (B=2000) — both arms see the same queries, so the delta is
bootstrapped paired and its interval excluding 0 *is* the significance test.

**The split is document-isolated.** Held-out queries are excluded by *Act*, not by query
string: an Act is either wholly in training or wholly in evaluation, never split across both.
This is the thing the withdrawn run got wrong, and it is why these numbers are lower.

- **1,494** held-out queries · pool of **1,205** unique statutory passages
- Training volume **26,252** rows (2.93× the 8,966-row first clean re-measure)
- Checkpoint chosen by held-out loss, not last step

| Stage | Recall@1 | 95 % CI |
|---|---|---|
| bi-encoder (e5-small) retrieve only | 0.0723 | [0.0589, 0.0857] |
| **+ cross-encoder rerank** | **0.1064** | [0.0910, 0.1218] |

- **Delta = +0.0341 (+47.2 %)**, paired 95 % CI **[+0.0207, +0.0482]** — **excludes 0**, a
  statistically significant improvement.
- **Recall@100 ceiling = 0.7557** — the fraction of queries whose gold passage the first-stage
  retriever surfaces at all. Everything above that is out of the reranker's reach.
- Of the **recoverable** queries (gold in the top-100), the reranker puts it at **#1 for
  14.1 %**.

**Read this before comparing to anything.** Both caveats in the notice at the top of this card
apply here: the +47.2 % is the joint effect of **more data *and* best-checkpoint selection**
(not separable — no ablation was run), and this run used **77.8 %** of the withdrawn run's
volume, so it is **not volume-matched** against any earlier figure. The withdrawn +63.8 % is
kept in the collapsed record above for provenance only and must not be quoted, differenced, or
described as having been "corrected" to this.

## How it was built (honest — the negatives lesson)

The first training run **collapsed to random** at evaluation. It trained on *mined hard
negatives* (schedule stubs, repealed-section markers, OCR-garbled fragments) which are
**form-separable** from clean section bodies — so the cross-encoder learned a
query-independent shortcut ("does this look like a clean positive?") that crushed the training
loss and carried **zero** discriminative power at eval, where every candidate in the
retriever's shortlist is already a clean body.

The fix (this release) draws **form-matched negatives**: each training negative is another
record's real positive — a clean statutory body, form-identical to the answer. Now the only
way to pick the right passage among several clean passages is to *read the query and judge
relevance*, so the training distribution matches the evaluation distribution and no shortcut
exists. The general lesson: **a reranker's negatives must match the candidate distribution it
will see at inference, or it optimises a proxy.**

**The second lesson — how the first clean re-measure came back null.** After the split defect
was found, the re-measure ran on a document-isolated split and produced *nothing*: a delta
whose interval spanned zero. That null was real, but it was measured through two suppressors
of its own. A vestigial filter left over from the abandoned mined-negatives recipe was silently
dropping **68.9 %** of the training pairs — 8,152 rows survived where 26,252 were
available — and the checkpoint scored was the last step rather
than the best one. Removing the filter and scoring the best checkpoint is the whole difference
between that null and the **+47.2 %** at the top of this card — which is exactly why the two
changes cannot be attributed apart, and why this card says so instead of picking one.
The general lesson: **a null is a claim too, and deserves the same audit you would give a
positive result before you publish it.**

## Licence & provenance

- **Base:** `intfloat/multilingual-e5-small` (MIT).
- **Training data:** English **central bare-Act** text — 858 Central Acts sourced from a
  public Zenodo dataset (record 5088102, CC-BY-4.0) — mined into section-level query/passage
  pairs, with form-matched negatives drawn from the same pool. Under Indian Copyright Act
  **§52(1)(q)(ii)**, the text of a Central Act (bare Act, no third-party headnotes or
  annotations) is not an infringement to reproduce; only bare statutory text is used.
- **Weights:** Apache-2.0. The reranker is **non-reconstructive** — it emits a relevance
  score and never reproduces the source text — so it is train-safe on §52-clean statutory
  text and the weights are cleanly licensable. **The corpus itself is not redistributed.**

## Intended use & limits

Use it as the **second stage** behind a statute retriever (ideally `embed-statute-en`) to
re-rank English central-statutory shortlists. Do **not** use it as a retriever, on court
judgments (that is `embed-legal-en`), on state/subordinate legislation, or on non-English
text — none of those are validated.

**What the number does and does not license you to assume.** The +47.2 % is measured on
English central bare-Act text, with queries held out by whole Act, against a 1,205-passage
pool. It is a **relative** gain over a weak first stage — absolute Recall@1 is 0.1064, and the
retriever's own ceiling is 0.7557, so most of the headroom is upstream in retrieval, not here.
Transfer to other corpora is **not** measured: this model showed nothing about judgments,
state legislation, or other languages, and a sibling Quanfire retriever has already
demonstrated that in-domain legal gains can go flat out-of-origin. Measure it on your own data
before depending on it.
