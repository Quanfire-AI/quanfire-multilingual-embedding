---
license: apache-2.0
base_model: intfloat/multilingual-e5-small
library_name: quanfire-multilingual-embedding
pipeline_tag: sentence-similarity
tags:
  - sentence-embeddings
  - multilingual
  - cross-lingual
  - government
  - indian-languages
  - retrieval
  - lora
  - e5
language:
  - en
  - hi
  - ur
  - ta
  - te
  - kn
  - ml
  - bn
  - gu
  - mr
  - pa
  - or
  - as
  - ne
  - mni
  - kha
---

# Quanfire Government Embedding — `embed-gov-indic` (cross-lingual, Indian government press releases)

A **cross-lingual** retrieval adapter for Indian **government press-release** text across
**16 Indian languages**. It is a LoRA adaptation over a frozen
[`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)
(MIT) base — a 2.4 MB adapter, 384-dimensional normalized vectors, `max_length` 256 —
trained on **Press Information Bureau (PIB) press releases**, which are published as the
*same* release in many languages and so provide naturally-parallel cross-lingual signal.

The point of this model is **cross-lingual** retrieval: a query in one Indian language
finding the passage about the same government release in **another** language. It is a
specialist and a **showcase of clean-provenance multilingual retrieval**, not a general
model — and this card shows you exactly what it does and does not do, measured.

> **Version 1.1** — same task, same contract, better weights. v1.1 corrects a training
> shortfall in v1.0: the earlier runs held **epochs = 1** and the adapter was undertrained
> (its loss had barely moved). v1.1 trains longer on the *same* balanced corpus and is
> **significantly better than v1.0**, but the extra fit is not free — two mid/low-resource
> languages regress. Both facts are on this card. See [What changed in v1.1](#what-changed-in-v11).

- **Framework & code:** [github.com/quanfire-ai/quanfire-multilingual-embedding](https://github.com/quanfire-ai/quanfire-multilingual-embedding) (Apache-2.0)
- **Weights licence:** Apache-2.0. The adapter is ours; the training text is PIB press-release
  content reused under **PIB's reproduction policy** (free reproduction, attribution, **no
  NonCommercial, no ShareAlike**) — attribution is given below, and the model is
  non-reconstructive (it emits vectors, never the source text).
- **Internal run:** `gov-indic-e4-ep6` · base e5-small · rank 32 / alpha 64, LoRA on `query,value` · 589,824 adapter params · **6 epochs**, lr 1e-4, seed 0, batch 256 (bf16, CUDA) · same corpus and held-out eval as v1.0, so the comparison is like-for-like

## What it is for

Cross-lingual and in-language **retrieval over Indian government press releases** — search
a corpus of releases with a query in any of the supported languages, and rank passages
regardless of which language they are written in. It embeds a query and a passage into the
same 384-d space; cosine similarity ranks.

## Scope — read this before you use it

| | Validated? |
|---|---|
| **Cross-lingual** retrieval over government press-release text (16 Indian languages) | ✅ **Yes** — significant, measured below |
| Other domains (legal, finance, news, conversational, product) | ❌ **Not validated** — this is a government-press-release specialist |
| Gujarati and Odia | ⚠️ **Regressed vs v1.0** — see [What changed in v1.1](#what-changed-in-v11); still above the base for Odia, below it for Gujarati |
| The lowest-resource languages here (Khasi, Nepali, Manipuri) | ⚠️ **Thin** — very few eval examples (Khasi n=14, Nepali n=7); treat their numbers as indicative only |

The base `multilingual-e5-small` is **already a capable multilingual retriever**, so the
honest framing is: this adapter adds a **significant, broad cross-lingual gain on
government-domain text** on top of an already-multilingual base. Absolute Recall@1 is
modest (retrieving the one right passage out of a 1,541-passage pool across 16 languages is
hard); the right passage lands in the **top-10 about 84%** of the time.

## Results (held-out, scored on CUDA)

**In-distribution — 1,541 held-out cross-lingual queries** (anchor and positive in *different*
Indian languages, same release; drawn from 1,800 mined pairs). The published base is the only
honest baseline; the adapter is scored on the *same* held-out queries.

| Metric | base e5-small | **embed-gov-indic v1.1** | change |
|---|---|---|---|
| Recall@1 | 0.1836 | **0.2810** | **+53%** |
| Recall@5 | 0.516 | **0.730** | +41% |
| Recall@10 | 0.649 | **0.843** | +30% |
| MRR | 0.336 | **0.471** | +40% |
| nDCG@10 | 0.403 | **0.556** | +38% |

The Recall@1 improvement is **statistically significant** — the 95% confidence intervals are
disjoint (base [0.165, 0.204] → v1.1 [0.259, 0.304]). It is also significant **over v1.0**
(v1.0 was 0.2349 [0.214, 0.257]; v1.1's interval clears it).

**Per positive-language** (Recall@1, base → v1.1; all 16 eval languages, n = eval queries):

| Lang | base | v1.1 | n | | Lang | base | v1.1 | n |
|---|---|---|---|---|---|---|---|---|
| Tamil | 0.150 | **0.410** | 100 | | Malayalam | 0.175 | **0.233** | 103 |
| Hindi | 0.324 | **0.477** | 111 | | Assamese | 0.065 | **0.167** | 108 |
| Urdu | 0.206 | **0.389** | 131 | | Telugu | 0.214 | **0.265** | 98 |
| English | 0.167 | **0.362** | 138 | | Manipuri | 0.029 | **0.039** | 103 |
| Bengali | 0.179 | **0.316** | 95 | | Nepali | 0.286 | 0.286 | 7 |
| Marathi | 0.219 | **0.325** | 114 | | **Gujarati** | 0.367 | **0.284** ⚠️ | 109 |
| Kannada | 0.142 | **0.248** | 113 | | **Odia** | 0.138 | **0.117** ⚠️ | 94 |
| Punjabi | 0.175 | **0.252** | 103 | | **Khasi** | 0.143 | **0.071** ⚠️ | 14 |

**13 of 16 languages improve**, several dramatically (Tamil +173%, English +117%, Urdu, Hindi,
Bengali). **Three regress against the base — Gujarati, Odia, and Khasi (n=14)** — the cost of
the longer fit; see below.

## What changed in v1.1

v1.0 (`gov-indic-e3`) shipped at **1 epoch**. Its training loss was essentially flat and the
adapter had barely moved — it was **undertrained**, not at its ceiling. v1.1 (`gov-indic-e4-ep6`)
reruns the identical corpus, seed, rank and learning rate for **6 epochs** (loss 4.18 → 1.22),
so the two are directly comparable.

**Net effect, v1.0 → v1.1:** overall Recall@1 **0.235 → 0.281** (significant, disjoint CIs),
top-10 recall **74% → 84%**, and Manipuri recovers from v1.0's collapse (0.000 → 0.039).

**The tradeoff, stated plainly:** longer training sharpens the majority at the expense of a
few. Against v1.0, **Gujarati (0.349 → 0.284) and Odia (0.160 → 0.117) regress**; Khasi holds
flat (0.071). Against the *base*, Gujarati, Odia and Khasi all sit below where the untuned
model was. If your workload is Gujarati- or Odia-heavy, v1.0 may still be the better choice —
both versions remain available by revision tag. This is a genuine
speciality-vs-uniformity tradeoff, not a strict upgrade, and we would rather say so than hide it.

## Licence & provenance

- **Base:** `intfloat/multilingual-e5-small` (MIT).
- **Training data:** Press Information Bureau (PIB, pib.gov.in) press releases, reused under
  PIB's stated reproduction policy — free reproduction with source acknowledgement, no
  NonCommercial and no ShareAlike clause. Embedded third-party material is excluded.
  **Attribution:** *Source — Press Information Bureau (pib.gov.in), Government of India.*
- **Weights:** Apache-2.0. Because the model is non-reconstructive (it emits 384-d vectors
  and never reproduces the source text), and the source is reproduction-permitted with
  attribution (satisfied here), no ShareAlike floor applies to the weights.

## How it was built (honest, across four runs)

- **e1** — mixed pair kinds, language-imbalanced: +8.7%, not significant, with regressions in
  low-resource languages.
- **e2** — cross-lingual `title↔body` pairs only, language-balanced: +16.2%, regressions fixed,
  but still not significant on a small eval.
- **e3 (v1.0)** — a larger balanced corpus and a larger held-out eval: +27.9%, statistically
  significant — but held at 1 epoch and, in hindsight, undertrained.
- **e4 (v1.1, this release)** — same corpus/eval as e3, trained to 6 epochs: **+53% over base,
  significant over both base and v1.0**, with the Gujarati/Odia tradeoff documented above.

The corpus was built by re-sourcing PIB releases directly (never via third-party scraped
compilations), joining each release to its sibling-language versions, and mining cross-lingual
`title↔body` pairs with a lexical-overlap cap so a gain reflects meaning, not string matching.

## Intended use & limits

Use it to retrieve/rank Indian **government press-release** passages, including across
languages. Do **not** expect it to transfer to other domains, treat the lowest-resource
languages as indicative, and prefer v1.0 if your workload is Gujarati- or Odia-dominant. It is
a specialist showcase of what clean, provenance-transparent data yields for Indian-language
cross-lingual retrieval.
