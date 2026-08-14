---
license: apache-2.0
base_model: intfloat/multilingual-e5-small
library_name: quanfire-multilingual-embedding
pipeline_tag: sentence-similarity
tags:
  - sentence-embeddings
  - legal
  - indian-law
  - statutes
  - legislation
  - retrieval
  - lora
  - e5
language:
  - en
---

# Quanfire Statute Embedding — `embed-statute-en` (Central Acts / bare statutory text)

A retrieval adapter for **English-language Indian central statutory text** — the
sections, sub-sections and marginal-note headings of Acts enacted by Parliament. It
is a LoRA adaptation over a frozen
[`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)
(MIT) base — a ~2.4 MB adapter, 384-dimensional normalized vectors, `max_length` 256.

This is a **sibling** to [`embed-legal-en`](https://huggingface.co/quanfire-ai/embed-legal-en),
built to cover the register that model measured itself *flat* on. `embed-legal-en` is a
judgment specialist; on **statutory / bare-act** text its transfer was honestly zero.
`embed-statute-en` is the purpose-built statute retriever for exactly that text — and,
like its sibling, this card shows you where it helps and where it does not, measured.

- **Framework & code:** [github.com/quanfire-ai/quanfire-multilingual-embedding](https://github.com/quanfire-ai/quanfire-multilingual-embedding) (Apache-2.0)
- **PyPI:** `pip install quanfire-multilingual-embedding`
- **Weights licence:** Apache-2.0 (see *Licence & provenance* — attribution to the source dataset is required)
- **Internal run:** `statute-en-e2` · base e5-small · rank 32 / alpha 64, LoRA on `query,value` · mean pooling · 2 epochs, lr 1e-4 (bf16, CUDA) · adapter-mined hard negatives (4/pair, positive-margin 0.05)

## What it is for

Retrieving and ranking **passages of English central statutory text** — statute search,
section-to-section similarity, "find the provision that says X" over an Act corpus. It
embeds a query and a passage into the same 384-d space; cosine similarity ranks.

## Scope — read this before you use it

| | Validated? |
|---|---|
| English **central statutory** (bare-Act) retrieval | ✅ **Yes** — +48 % Recall@1 overall, **+131 % on the un-gameable low-overlap slice**; see Results |
| **Judgment / case-law** text | ❌ **No** — that is [`embed-legal-en`](https://huggingface.co/quanfire-ai/embed-legal-en)'s register, not this one |
| **State legislation, rules, notifications, contracts** | ❌ **Not measured** — bring your own evaluation |
| **Non-English** statutory text (Hindi, Tamil, …) | ❌ **No** — the model and its training data are English-only |

The gain this adapter provides is **statute-specific**. For judgments use the sibling
model; for other registers, measure before you trust it.

## Results (held-out, scored on CUDA)

**In-distribution — 1,978 held-out central-statute pairs** (2,000 sampled, 22 duplicate-positive
queries dropped). The published base is the only honest baseline; the adapter is scored on the
*same* held-out pairs, which were **never** seen in training. All figures are single-run,
measured on CUDA (never MPS).

| Metric | base e5-small | **embed-statute-en** | change |
|---|---|---|---|
| Recall@1 | 0.182 | **0.269** | **+48 %** |
| Recall@5 | 0.346 | **0.488** | +41 % |
| Recall@10 | 0.411 | **0.575** | +40 % |
| MRR | 0.262 | **0.375** | +43 % |
| nDCG@10 | 0.290 | **0.415** | +43 % |

The Recall@1 gain clears sampling noise with **disjoint 95 % confidence intervals**: base
**[0.165, 0.199]** vs adapter **[0.250, 0.289]** — the intervals do not touch.

**The un-gameable slice.** Statute marginal-note headings often restate the section body, so
a string-matcher can win on high-overlap pairs without learning meaning. The honest readout is
the **low-lexical-overlap bucket** (`<0.3` token overlap — pure semantics), and it is where the
adapter helps most:

| Lexical-overlap slice | base Recall@1 | **embed-statute-en** | change |
|---|---|---|---|
| **low `<0.3`** (n = 874, un-gameable) | 0.077 | **0.177** | **+131 %** |
| mid `0.3–0.7` (n = 1,104) | 0.265 | **0.342** | +29 % |

On the pairs a string-matcher *cannot* solve, the adapter **more than doubles** Recall@1
(and Recall@10 rises 0.245 → 0.462) — evidence it learned statutory semantics, not surface
overlap. By pair kind, the gain holds across both dominant types: adjacent-section
0.185 → 0.287 (n = 1,488) and heading↔section 0.172 → 0.215 (n = 489).

## Usage

Pull the adapter and run it through the Quanfire framework, which applies the LoRA over
the frozen base and produces normalized embeddings:

```bash
pip install 'quanfire-multilingual-embedding[neural]'

hf download quanfire-ai/embed-statute-en --local-dir embed-statute-en
```

**As an HTTP embeddings service** (OpenAI-compatible `POST /v1/embeddings`):

```bash
qfme serve --adapter embed-statute-en --port 8000

curl -s localhost:8000/v1/embeddings \
  -H 'content-type: application/json' \
  -d '{"input": ["What is the punishment for criminal breach of trust by a public servant?"]}'
```

**In-process, as a search pipeline:**

```python
from multilingual_embedding.pipelines.search import SemanticSearchPipeline

pipe = SemanticSearchPipeline.from_adapter("embed-statute-en")
pipe.index([
    "Whoever, being in any manner entrusted with property, dishonestly misappropriates it, commits criminal breach of trust.",
    "Every Act shall come into operation on the day it receives the assent of the President, unless otherwise provided.",
    "This Act extends to the whole of India.",
])
for hit in pipe.search("when does a statute take effect", top_k=3):
    print(hit.rank, round(hit.score, 3), hit.text)
```

Vectors are L2-normalized `float32` (dimension 384). The model is symmetric (empty
prefixes), so `input_type` is not required. Exact (brute-force cosine) search is the
intended regime up to ~10⁵–10⁶ vectors; add an ANN index beyond that.

## Training corpus

Built from **858 Central Acts** enacted by the Indian Parliament, mined into contrastive
section-level pairs. Two mining fixes distinguish this corpus from a naive Wikipedia-style
pass, and both were **proven by an oracle-diff at build time** (naive vs tuned mine),
not assumed:

1. **Short-provision recovery.** Statute provisions (extent, commencement, short-title)
   are far terser than encyclopedic prose. The default 100-character positive floor
   dropped 2,039 of them; lowering it to 40 recovers ~2,015 genuine short provisions
   (only 24 remain rejected).
2. **Lexical-leakage removal.** Marginal-note headings frequently restate the section
   body verbatim (overlap ≈ 1.0 — pure string leakage a model can "solve" without
   learning meaning). Capping pair overlap at 0.5 removes 34,294 such leaky pairs,
   dropping the heading↔section mean overlap **0.68 → 0.26**.

Result: **35,252 clean pairs** (adjacent 26,214 + heading↔section 9,026 + title-lead 12),
every pair kind under 0.4 mean overlap.

**Hard negatives (the e2 step).** On top of in-batch negatives, four hard negatives per pair
were mined against the first-run (e1) adapter — its own confusions — then **filtered at
positive-margin 0.05**. This filter matters: an unfiltered mine was 78.8 % *false* negatives,
because generic statutory headings ("Short title", "Definitions") legitimately match sections
across every Act; the margin drops any candidate scoring within 0.05 of the pair's own
positive, keeping only genuinely harder negatives. Training for 2 epochs on this set is what
produced the numbers above.

## Licence & provenance

**Weights: Apache-2.0** — usable and redistributable commercially, **with attribution**
to the source dataset (below). There is **no share-alike obligation**: the source
dataset is CC-BY-4.0 (attribution, not copyleft), and — critically — this is a
**non-reconstructive** model. It emits 384-d vectors; it does not store or reproduce
statutory text, so training on bare-Act text is sound and the weights carry no text.

| Source | Role | Licence / status |
|---|---|---|
| *An annotated dataset of Central Acts enacted by the Indian Parliament* (Zenodo `5088102`) | training corpus (section text) | **CC-BY-4.0** — attribution required |
| `intfloat/multilingual-e5-small` | frozen base checkpoint | MIT |

**On the bare-Act text itself.** Under §52(1)(q)(ii) of the Copyright Act 1957, the text
of an Act may be reproduced only together with commentary. That condition governs
*republishing the text*; it does **not** constrain a non-reconstructive embedder, which
learns a vector geometry and cannot emit the source text. Accordingly Quanfire ships the
**weights, not the corpus** — the bare-Act training corpus is not redistributed. Please
retain the Zenodo CC-BY-4.0 attribution when you use these weights.

The framework source code is Apache-2.0 (separate from these weights).

## Limitations

- A LoRA adapter over a published checkpoint — not an independently pretrained model.
- **English only.** It does not embed Hindi, Tamil or other Indic statutory text.
- **Central Acts only.** Trained on central parliamentary Acts; state legislation, rules,
  regulations, notifications and contracts are out of distribution and unmeasured.
- **Statute-specific.** For judgment/case-law text use [`embed-legal-en`](https://huggingface.co/quanfire-ai/embed-legal-en);
  transfer between the two registers is not assumed in either direction.
- Not legal advice, and not a substitute for authoritative sources — retrieval surfaces
  candidate provisions; verify against the official text (India Code / the Gazette).
- Exact cosine search is the intended regime up to ~10⁵–10⁶ vectors.

## Citation

```
Quanfire Statute Embedding — embed-statute-en (internal run statute-en-e2).
Quanfire, 2026. https://github.com/quanfire-ai/quanfire-multilingual-embedding

Training corpus: "An annotated dataset of Central Acts enacted by the Indian
Parliament", Zenodo 5088102 (CC-BY-4.0).
```
