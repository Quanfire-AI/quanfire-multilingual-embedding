---
license: apache-2.0
base_model: intfloat/multilingual-e5-small
library_name: quanfire-multilingual-embedding
pipeline_tag: sentence-similarity
tags:
  - sentence-embeddings
  - legal
  - indian-law
  - judgments
  - retrieval
  - lora
  - e5
language:
  - en
---

# Quanfire Legal Embedding — `embed-legal-en` (Supreme Court judgments)

A retrieval adapter for **English-language Indian Supreme Court judgment text**. It
is a LoRA adaptation over a frozen
[`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)
(MIT) base — a 2.4 MB adapter, 384-dimensional normalized vectors, `max_length` 256 —
trained **only on statutory public-domain judgment text**.

This is **not** a from-scratch model, and it is **not** a general "Indian legal"
model. It is a specialist: on judgment-to-judgment retrieval it is markedly stronger
than the base; on other legal registers (statutes) it is not — and this card shows
you both, measured.

- **Framework & code:** [github.com/quanfire-ai/quanfire-multilingual-embedding](https://github.com/quanfire-ai/quanfire-multilingual-embedding) (Apache-2.0)
- **PyPI:** `pip install quanfire-multilingual-embedding`
- **Weights licence:** Apache-2.0 (see *Licence & provenance* — the training text is statutory public domain, so no share-alike floor applies)
- **Internal run:** `legal-indic-e1` · base e5-small · rank 32 / alpha 64, LoRA on `query,value` · 589,824 adapter params · 1 epoch, lr 1e-4, batch 256 (bf16, CUDA)

## What it is for

Retrieving and ranking **passages of English Supreme Court judgments** — case-law
search, judgment-to-judgment similarity, semantic retrieval over a judgment corpus.
It embeds a query and a passage into the same 384-d space; cosine similarity ranks.

## Scope — read this before you use it

| | Validated? |
|---|---|
| English Supreme Court **judgment** retrieval | ✅ **Yes** — strong, measured below |
| **Statutory / bare-act** text, FAQs, other legal registers | ❌ **No** — transfer tested, came back flat (see below) |
| **Non-English** legal text (Hindi, Tamil, …) | ❌ **No** — the model and its training data are English-only |

The gain this adapter provides is **judgment-specific**. If your text is statutes,
contracts, or non-English legal material, use the base model or a purpose-built model
— this one will not help there, and we measured that rather than assuming it.

## Results (held-out, scored on CUDA)

**In-distribution — 2,000 held-out Supreme Court judgment pairs.** The published base
is the only honest baseline; the adapter is scored on the *same* held-out pairs.

| Metric | base e5-small | **embed-legal-en** | change |
|---|---|---|---|
| Recall@1 | 0.309 | **0.545** | **+76.2 %** |
| Recall@5 | 0.494 | **0.772** | +56.3 % |
| Recall@10 | 0.567 | **0.829** | +46.2 % |
| MRR | 0.399 | **0.647** | +62.2 % |
| nDCG@10 | 0.433 | **0.687** | +58.7 % |

The Recall@1 95 % confidence intervals are disjoint (base `[0.289, 0.330]` →
adapter `[0.523, 0.566]`), so the gain is not sampling noise. It holds where it is
hardest: on the **low-lexical-overlap** bucket (`<0.3`, pure-semantic matches, no
shared words to lean on) Recall@1 more than doubles, **0.156 → 0.325**.

**Out-of-origin transfer — 1,578 English *statutory* adjacency pairs** (a different
legal register: bare-act sections and regulatory FAQs, origin-walled from the
judgment training corpus). This is the honest generalization test, and it is
**flat**:

| Metric | base e5-small | embed-legal-en |
|---|---|---|
| Recall@1 | 0.036 | 0.036 (−1.8 %) |
| nDCG@10 | 0.219 | 0.208 |

The instrument is informative, not degenerate — the base model has real signal on it
(high-overlap Recall@5 0.76). The adapter simply does not improve statutory
retrieval. **We publish this row on purpose:** the +76 % is a judgment-domain result,
not a "legal English" result, and the difference is exactly what a buyer needs to
know.

## Usage

Pull the adapter and run it through the Quanfire framework, which applies the LoRA
over the frozen base and produces normalized embeddings:

```bash
pip install 'quanfire-multilingual-embedding[neural]'

hf download quanfire-ai/embed-legal-en --local-dir embed-legal-en
```

**As an HTTP embeddings service** (OpenAI-compatible `POST /v1/embeddings`):

```bash
qfme serve --adapter embed-legal-en --port 8000

curl -s localhost:8000/v1/embeddings \
  -H 'content-type: application/json' \
  -d '{"input": ["Whether the appellant was denied a fair hearing under Article 21."]}'
```

**In-process, as a search pipeline:**

```python
from multilingual_embedding.pipelines.search import SemanticSearchPipeline

pipe = SemanticSearchPipeline.from_adapter("embed-legal-en")
pipe.index([
    "The conviction under Section 302 is set aside for want of corroboration.",
    "Bail is granted subject to the appellant surrendering the passport.",
    "The writ petition challenges the vires of the impugned notification.",
])
for hit in pipe.search("appeal against a murder conviction", top_k=3):
    print(hit.rank, round(hit.score, 3), hit.text)
```

Vectors are L2-normalized `float32` (dimension 384). The model is symmetric (empty
prefixes), so `input_type` is not required. Exact (brute-force cosine) search is the
intended regime up to ~10⁵–10⁶ vectors; add an ANN index beyond that.

## Licence & provenance

**Weights: Apache-2.0.** Use them commercially and redistribute them freely, with
attribution. There is **no share-alike obligation**, because — unlike a
Wikipedia-derived model — every input to this adapter is public-domain text:

| Source | Role | Licence / status |
|---|---|---|
| Indian Supreme Court judgment text (official court portals) | training corpus | **Public domain** — Copyright Act 1957, §52(1)(q) places judgment text outside copyright |
| `intfloat/multilingual-e5-small` | frozen base checkpoint | MIT |

**What was removed, and why it matters.** §52(1)(q) frees the *text of the judgment*
— it does **not** free the reporter-written **headnote / syllabus**, which is
separately copyrightable editorial matter. The training corpus was built by
extracting each judgment with a layout-faithful parser, dropping page furniture, and
**excising the headnote span** between the coram line and the start of the reported
judgment body. Only the statutory-public-domain judgment text was trained on. The
base checkpoint is frozen and unmodified (MIT); the adapter is a separate set of
weights over it.

The framework source code is Apache-2.0 (separate from these weights).

## Limitations

- A LoRA adapter over a published checkpoint — not an independently pretrained model.
- **English only.** It does not embed Hindi, Tamil or other Indic legal text; a
  cross-lingual Indian-legal model is separate work requiring parallel legal data.
- **Judgment-specific.** Transfer to statutory / bare-act text was tested and is flat
  (table above). Do not rely on it outside judgment-style text without your own
  evaluation. It is evaluated only on held-out Indian Supreme Court judgment pairs
  from the same distribution as its training data; transfer to other jurisdictions or
  legal systems is not measured and should not be assumed.
- Exact cosine search is the intended regime up to ~10⁵–10⁶ vectors.

## Citation

```
Quanfire Legal Embedding — embed-legal-en (internal run legal-indic-e1).
Quanfire, 2026. https://github.com/quanfire-ai/quanfire-multilingual-embedding
```
