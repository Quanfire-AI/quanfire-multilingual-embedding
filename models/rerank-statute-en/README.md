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
- **Internal run:** `statute-reranker-v0` · base e5-small + scalar head · listwise
  cross-entropy over 1 positive + 6 form-matched negatives · lr 2e-5, 2 epochs, `max_length`
  256, seed 0 (bf16, CUDA).

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

## Results (held-out, scored on CUDA)

Evaluated the **production way**: a bi-encoder (`multilingual-e5-small`) retrieves the top-100
for each query, the cross-encoder rescores those 100, and we compare **Recall@1** — over
**1,200 held-out queries** against a pool of **1,182** unique statutory passages. 95% CIs are
by **paired** bootstrap (B=2000): the two arms see the same queries, so the delta is
bootstrapped paired, and its interval excluding 0 is the significance test.

| Stage | Recall@1 | 95% CI |
|---|---|---|
| bi-encoder (e5-small) retrieve only | 0.2050 | [0.1817, 0.2292] |
| **+ cross-encoder rerank** | **0.3358** | [0.3100, 0.3625] |

- **Delta = +0.1308 (+63.8%)**, paired 95% CI **[+0.1058, +0.1567]** — **excludes 0**, a
  statistically significant improvement.
- **Recall@100 ceiling** (the gold section is *somewhere* in the retriever's top-100) =
  **0.7183**. That is the reranker's headroom — it can only promote what the retriever surfaced.
- Of the **recoverable** queries (gold in the top-k), the reranker puts it at **#1 for 46.8%**.

**Read this honestly:** the gain over retrieve-only is large and significant, but absolute
Recall@1 is 0.336 against a 0.718 ceiling — for most queries where the right section is already
in the shortlist, it is *not yet* lifted to the top. This is a strong first reranker with real
headroom left, not a solved problem. See below.

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
text — none of those are validated. Treat the 0.336-vs-0.718 gap as honest headroom: this is
a proven, significant first reranker, and a later version will spend that headroom.
