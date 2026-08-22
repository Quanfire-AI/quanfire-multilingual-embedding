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

> ### ⚠️ Measurement correction — 2026-08-21
>
> **The headline this card used to carry (+53 % Recall@1) was measured on a
> contaminated evaluation split. The corrected figure is +37.1 %.**
>
> The bug: the training filter dropped a pair only when its *positive* was in the
> held-out set. This corpus is bidirectional, so the **reverse** of a held-out
> pair was still eligible for training, and pairs from held-out source documents
> were never excluded at all. The model was partly trained on the text it was
> scored on.
>
> The adapter was retrained and rescored on a split that isolates at the
> **source-document** level. **Every number in *Results* below is the clean
> measurement.** The gain is smaller and it is still statistically significant
> (disjoint 95 % CIs).
>
> Three things a reader deserves to know:
>
> 1. **The clean figures are for a retrain (`gov-indic-e4c`), not for the weight
>    file currently in this repo.** The published weights saw the evaluation
>    documents during training, so no honest score for *them* exists — not a
>    lower one either. Publishing the clean adapter as a new revision is pending.
> 2. **The runs are not volume-matched.** The clean run trained on **2,115**
>    pairs against the withdrawn run's 7,270, because one press release yields
>    many same-document pairs across 16 languages and document isolation removes
>    most of the pool. Do not read the difference as "the leak was worth 16
>    points"; the honest finding is that this corpus is too document-poor to
>    support a clean high-volume train.
> 3. **v1.0 was never re-measured**, so the v1.0-vs-v1.1 comparison this card
>    used to make has been withdrawn rather than corrected.
>
> Fix: `without_held_out()` in
> [quanfire-multilingual-embedding](https://github.com/quanfire-ai/quanfire-multilingual-embedding)
> (commits `66470fe`, `6fe7e6b`, `3deaf8d`).


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

> **Version 1.1** — same task, same contract, longer training. v1.1 addresses a training
> shortfall in v1.0: the earlier runs held **epochs = 1** and the adapter was undertrained
> (its loss had barely moved). v1.1 trains longer on the *same* balanced corpus. **The claim
> that it is significantly better than v1.0 has been withdrawn** — that comparison was made on
> a contaminated split and v1.0 has not been re-measured, so there is no valid comparison
> between the two. The extra fit is not free: three mid/low-resource languages sit below the
> base. See [What changed in v1.1](#what-changed-in-v11).

- **Framework & code:** [github.com/quanfire-ai/quanfire-multilingual-embedding](https://github.com/quanfire-ai/quanfire-multilingual-embedding) (Apache-2.0)
- **Weights licence:** Apache-2.0. The adapter is ours; the training text is PIB press-release
  content reused under **PIB's reproduction policy** (free reproduction, attribution, **no
  NonCommercial, no ShareAlike**) — attribution is given below, and the model is
  non-reconstructive (it emits vectors, never the source text).
- **Internal run:** `gov-indic-e4-ep6` · base e5-small · rank 32 / alpha 64, LoRA on `query,value` · 589,824 adapter params · **6 epochs**, lr 1e-4, seed 0, batch 256 (bf16, CUDA) · the clean re-measure reported below is `gov-indic-e4c`: identical configuration, retrained on a document-isolated split

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
| Gujarati, Manipuri and Khasi | ⚠️ **Below the base** on the clean split (Gujarati 0.367 → 0.330, Manipuri 0.029 → 0.000, Khasi 0.143 → 0.071 at n=14) — see the per-language table; the older "regressed vs v1.0" claim is withdrawn because v1.0 has not been re-measured |
| The lowest-resource languages here (Khasi, Nepali, Manipuri) | ⚠️ **Thin** — very few eval examples (Khasi n=14, Nepali n=7); treat their numbers as indicative only |

The base `multilingual-e5-small` is **already a capable multilingual retriever**, so the
honest framing is: this adapter adds a **significant, broad cross-lingual gain on
government-domain text** on top of an already-multilingual base. Absolute Recall@1 is
modest (retrieving the one right passage out of a 1,541-passage pool across 16 languages is
hard); on the clean split the right passage lands in the **top-10 about 75%** of the time.

## Results (held-out, scored on CUDA)

**In-distribution — 1,541 held-out cross-lingual queries** (anchor and positive in *different*
Indian languages, same release; drawn from 1,800 mined pairs). The published base is the only
honest baseline; the adapter is scored on the *same* held-out queries.

| Metric | base e5-small | **clean retrain (`gov-indic-e4c`)** | change | withdrawn figure |
|---|---|---|---|---|
| Recall@1 | 0.1836 | **0.2518** | **+37.1%** | ~~0.2810 / +53%~~ |
| Recall@5 | 0.5159 | **0.6470** | +25.4% | ~~0.730~~ |
| Recall@10 | 0.6489 | **0.7521** | +15.9% | ~~0.843~~ |
| MRR | 0.3362 | **0.4216** | +25.4% | ~~0.471~~ |
| nDCG@10 | 0.4029 | **0.4955** | +23.0% | ~~0.556~~ |

The base column is unchanged — the contamination only ever inflated the adapter, so the
correction is confined to one column.

The Recall@1 improvement is **statistically significant** — the 95% confidence intervals are
disjoint (base [0.1651, 0.2038] → clean retrain [0.2307, 0.2741]). **No claim of significance
over v1.0 is made any more:** v1.0's number came from the same contaminated split and has not
been re-measured.

**Per positive-language** (Recall@1, base → v1.1; all 16 eval languages, n = eval queries):

| Lang | base | clean | n | | Lang | base | clean | n |
|---|---|---|---|---|---|---|---|---|
| Tamil | 0.150 | **0.300** | 100 | | Malayalam | 0.175 | **0.223** | 103 |
| Hindi | 0.324 | **0.441** | 111 | | Assamese | 0.065 | **0.102** | 108 |
| Urdu | 0.206 | **0.336** | 131 | | Telugu | 0.214 | **0.255** | 98 |
| English | 0.167 | **0.348** | 138 | | **Manipuri** | 0.029 | **0.000** ⚠️ | 103 |
| Bengali | 0.179 | **0.221** | 95 | | Nepali | 0.286 | 0.286 | 7 |
| Marathi | 0.219 | **0.272** | 114 | | **Gujarati** | 0.367 | **0.330** ⚠️ | 109 |
| Kannada | 0.142 | **0.248** | 113 | | Odia | 0.138 | **0.181** | 94 |
| Punjabi | 0.175 | **0.214** | 103 | | **Khasi** | 0.143 | **0.071** ⚠️ | 14 |

**12 of 16 languages improve**, Nepali (n=7) is flat, and **three regress against the base —
Manipuri, Gujarati and Khasi (n=14)**.

The clean split changes which languages are in that regression list, so the old version of this
paragraph was wrong in both directions: **Odia now improves** (0.138 → 0.181) where the
contaminated run showed it regressing, and **Manipuri now falls to 0.000** where the
contaminated run claimed it recovered to 0.039. On a 2,115-pair train, the lowest-resource
languages get too little signal to move, and Manipuri gets none.

## What changed in v1.1

v1.0 (`gov-indic-e3`) shipped at **1 epoch**. Its training loss was essentially flat and the
adapter had barely moved — it was **undertrained**, not at its ceiling. v1.1 (`gov-indic-e4-ep6`)
reruns the identical corpus, seed, rank and learning rate for **6 epochs** (loss 4.18 → 1.22),
so the two are directly comparable.

**The v1.0 → v1.1 comparison has been withdrawn.** It read "overall Recall@1 0.235 → 0.281,
top-10 recall 74% → 84%, Manipuri recovers from v1.0's collapse". Both sides of that comparison
were measured on the contaminated split, and only v1.1 has been re-measured, so there is
nothing left to compare against. v1.0 remains available by revision tag; we no longer make a
claim about which of the two is stronger.

**The tradeoff that survives, stated plainly:** longer training sharpens the majority at the
expense of a few. Against the *base*, on the clean split, **Gujarati (0.367 → 0.330), Manipuri
(0.029 → 0.000) and Khasi (0.143 → 0.071, n=14) sit below where the untuned model was.** If
your workload is Gujarati-dominant, measure before switching. This is a genuine
speciality-vs-uniformity tradeoff, not a strict upgrade, and we would rather say so than hide it.

## Which weights should I use?

Two revisions are published. They share the recipe and differ in **what can be said about
them**:

| Revision | Trained on | Has a valid score? |
|---|---|---|
| `main` (default) | 7,270 pairs — the full mined pool | ❌ **No.** It trained on the evaluation documents, so no clean held-out set exists for it *within this corpus* |
| `clean-2026-08-21` | 2,115 pairs — document-isolated split | ✅ **Yes** — every figure in *Results* above is this adapter |

```bash
hf download quanfire-ai/embed-gov-indic --revision clean-2026-08-21 --local-dir embed-gov-indic-clean
```

**Which one to take.** If you need a number you can cite or audit, take
`clean-2026-08-21` — it is the one the Results section describes. `main` saw
3.4x the training data and may well be the stronger retriever in practice, but "may
well be" is precisely the kind of claim this card no longer makes.

**We have not swapped the default**, and the reason is worth stating: doing so would trade a
plausibly-stronger model for a measurable one with no evidence that the trade is good.
Settling it properly needs a head-to-head of the two adapters on a corpus *neither* of them
trained on. That is planned, and until it runs, both revisions stay up and this section stays
honest about which is which.

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
- **e4 (v1.1, this release)** — same corpus/eval as e3, trained to 6 epochs. Originally
  reported as +53% over base; on a clean, document-isolated split this is **+37.1% over base,
  significant against the base only** (the comparison against v1.0 is withdrawn — see the
  correction notice at the top), with the Gujarati/Manipuri/Khasi tradeoff documented above.
- **Every figure in this lineage above e4 was produced by the same contaminated split rule and
  none of them has been re-measured.** Read them as a record of how the work progressed, not as
  comparable numbers.

The corpus was built by re-sourcing PIB releases directly (never via third-party scraped
compilations), joining each release to its sibling-language versions, and mining cross-lingual
`title↔body` pairs with a lexical-overlap cap so a gain reflects meaning, not string matching.

## Intended use & limits

Use it to retrieve/rank Indian **government press-release** passages, including across
languages. Do **not** expect it to transfer to other domains, treat the lowest-resource
languages as indicative, and prefer v1.0 if your workload is Gujarati- or Odia-dominant. It is
a specialist showcase of what clean, provenance-transparent data yields for Indian-language
cross-lingual retrieval.
