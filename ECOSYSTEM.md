# Quanfire AI Ecosystem — Architecture and Plan

> How the modalities fit together, what each one actually requires, and which are
> worth building rather than integrating.

**Status:** planning document. Nothing here is built except the text embedding
pipeline in this repository.

---

## 0. What this ecosystem is, and what it is not

**Infrastructure, data and compute are three separate problems.** Conflating them is the
main way this plan can go wrong.

| | What it is | How it is obtained |
|---|---|---|
| **Infrastructure** | The code that trains, evaluates, versions and serves a model | Engineering — within our control |
| **Data** | Corpora, pairs, labelled examples | Collection and money |
| **Compute** | GPU hours | Money |

Model *quality* is mostly a function of data and compute. Pipeline *correctness* is a
function of engineering. The second does not wait on the first: the same code that trains
a toy model correctly trains a production model correctly, differing only in
configuration and volume.

**The objective is therefore the infrastructure**, proven to work, ready to scale when
data and compute arrive. Not a competitive model — a machine that makes models.

### What "proven" means here

A modality counts as supported when, at deliberately small scale:

1. The model **demonstrably learns** — loss falls and a held-out metric improves against
   an untrained baseline.
2. The run is **reproducible** — the same seed gives the same result.
3. The artefact **round-trips** — saves, loads, and serves identically.
4. It is **the same code path** a full-scale run would take, differing only in config.

Point 4 is what makes the proof meaningful. A separate "toy mode" proves nothing.

This bar costs no data budget and no GPU rental. The existing word2vec implementation
already meets it: a synthetic two-topic corpus, with a test asserting within-topic
similarity exceeds cross-topic similarity by a clear margin.

### Design constraints

- **Minimal.** Build the training and serving spine; integrate anything that is a
  commodity. The spine is what must be owned.
- **Independent.** No vendor lock-in, no mandatory external service, runs on-premise.
- **Manageable.** Small enough for a small team to hold in their heads. Every component
  earns its place or is removed.

### The spine, and what plugs into it

Almost everything is shared across modalities. Only four things are not.

| Shared — belongs in core | Modality-specific — a plugin |
|---|---|
| Configuration, validation, precedence | Model architecture |
| Structured logging | Loss function |
| Registry and factory | Data preprocessing |
| Streaming data abstractions | Evaluation metrics |
| Training loop — checkpointing, resumption, mixed precision, accumulation | |
| Artefact persistence and versioning | |
| Evaluation and reporting harness | |
| Serving — batching, versioning, standard schema | |

A modality is therefore a thin plugin over a common spine, not a separate system. That is
what keeps the ecosystem manageable at minimum size.

### Data collection is a separate concern

Corpus collection, licensing, cleaning and storage have a different lifecycle from
training code and should live in their own repository. The training side consumes a
prepared dataset; it does not scrape.

---

## 1. Token embeddings and retrieval embeddings

An earlier draft of this document split these into "token embeddings, elsewhere" and
"retrieval embeddings, here". **That was wrong and misleading.** This project builds
both, and the second is made out of the first.

There are three objects, not two.

### 1. The token embedding table — built here

A matrix of shape `vocabulary × dimension`: one vector per token id. The project produces
these on both paths.

| Path | The table |
|---|---|
| Static | `EmbeddingMatrix` — the trained word2vec output, one row per token id |
| Contextual | `TransformerEncoderModel.token_embedding` — an `nn.Embedding` of the same shape, the model's first layer |

```python
matrix.vector_for("rain")        # by token
matrix.vector_for_id(4)          # by id
model.token_embedding.weight     # (vocabulary, dimension)
```

### 2. The retrieval embedding — built here, out of the first

One vector per piece of text, obtained by combining the token vectors of its tokens. On
the static path that combination is a mean; on the contextual path the transformer
contextualises each token before pooling, so the same token contributes differently
depending on its neighbours.

The relationship is literal, not analogical:

```python
encoder.encode("rain storm")
    == (matrix.vector_for("rain") + matrix.vector_for("storm")) / 2      # True
```

**Token embeddings feeding retrieval embeddings is exactly what this project does.**

### 3. A generative model's own token table — belongs to the LLM project

Structurally the same object as (1) — `vocabulary × dimension` — but learned jointly with
all the other weights of a specific generative model, against a next-token objective.
That makes it inseparable from the model that produced it: its rows are meaningful only
in the coordinate system the rest of those weights define.

This is the only sense in which token embeddings belong elsewhere, and it is a narrow
one.

> **Could a table trained here initialise an LLM's embedding layer?**
> Technically yes, if the vocabularies match exactly. In practice it is rarely worth it.
> Initialising from word2vec was standard before transformers and is now uncommon,
> because a transformer learns a better table jointly than it inherits. Treat it as an
> option to measure, never as an assumed benefit.

### What this means for retrieval-augmented generation

The retrieval vector is not fed into the language model. It selects *which text* enters
the prompt; the model then applies its own token table to that text.

```
query ──► retrieval embedding ──► nearest documents ──► document TEXT into the prompt
                                                             │
                                       LLM tokenises that text
                                                             │
                                       LLM's own token table ──► its layers
```

The exceptions — vision projectors in multimodal models, soft prompts, adapter layers —
are architectural features of those models rather than a general mechanism for feeding
external vectors to a language model.

---

## 2. Repository architecture

### The repository map

All backend, all Python, all under `python-projects/`. No frontend at this stage.

| Repository | Owns | Depends on | Status |
|---|---|---|---|
| `quanfire-ml-core` | Config, logging, registry, artefact versioning, training loop, serving base, text preparation | nothing ML | proposed |
| `quanfire-datasets` | Corpus acquisition, licensing, cleaning, versioned dataset publication | core | proposed |
| **`quanfire-multilingual-embedding`** | **Text → vectors: tokenizer, vocabulary, encoders, training, evaluation, serving** | core, torch | **exists** |
| **`quanfire-llm`** | Text → text: fine-tuning, inference, serving | core, torch | **exists** — already consumes this repo as a pinned dependency |
| `quanfire-vision` | Image ↔ text, image generation | core, torch | proposed |
| `quanfire-speech` | Text ↔ speech | core, torch | proposed |

Three Quanfire repositories already exist alongside this one — `quanfire-ai-backend`,
`quanfire-ai-models` and `quanfire-mcp`. Their contents have not been examined, and the
map above may overlap with them. **Reconcile before creating anything**, particularly
`quanfire-ai-models`, whose name suggests it may already occupy part of this space.

### Scope of this repository

Stated tightly, because scope creep across a modality boundary is how these become
unmaintainable.

**In scope**

- Tokenizer and vocabulary training
- Embedding model architectures — static and contextual
- Training: from scratch, and adaptation from a pretrained checkpoint
- Pair mining for contrastive objectives
- Evaluation of embedding quality, per language and per domain
- Serving vectors

**Out of scope**

| Not here | Where |
|---|---|
| A *generative model's own* token table | `quanfire-llm` — learned jointly with that model's other weights, so not suppliable from outside. Token tables for retrieval **are** built here; see section 1. |
| Corpus acquisition, scraping, licensing | `quanfire-datasets` |
| Generation of any kind | the modality repository that owns it |
| Authentication, quotas, billing, routing | the API gateway |
| Vector storage and indexing at scale | a vector database; this repository provides exact search only |

The last one is worth stating plainly: this repository produces vectors and can search
them exactly, which is correct up to roughly 10⁶. It is not, and should not become, a
vector database.

### How a request flows

```
client ──► api.quanfire.ai ──► gateway: auth, quota, routing
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
        /v1/embeddings      /v1/completions       /v1/audio/speech
              │                    │                    │
        embedding svc          llm svc             speech svc
              │                    │                    │
              └──────────── quanfire-ml-core ───────────┘
                        artefact loading, batching,
                        versioning, structured logging
```

Each modality repository ships a serving adapter; the gateway owns auth, quotas and
routing and is the only public surface. Adopting the de facto industry request and
response schemas means existing clients migrate by changing a base URL.

The gateway probably belongs with the existing platform rather than being a new
repository — another reason to reconcile with `quanfire-ai-backend` first.

### Recommendation: separate repositories, split by team boundary

This was argued both ways before landing here, and the reasoning is recorded because the
decision is close and may need revisiting.

**A monorepo is technically viable.** The usual objection — that dependency sets conflict
— is false, and was tested rather than assumed. A uv workspace isolates dependencies per
package:

```
$ uv sync --package qf-core
 + pyyaml==6.0.3          # only what this package declares
numpy : False            # sibling's dependency, correctly absent
```

**It is nevertheless the wrong choice here**, for three reasons that outweigh it.

**The lockfile is a shared mutable file.** A workspace has one `uv.lock`. This repository's
is 779 lines for a single package with five dependencies; a workspace spanning six
packages with a training stack, a diffusion stack and an audio stack would run to many
thousands. Every dependency change by any team rewrites it, and it is machine-generated,
so conflicts are frequent and unpleasant to resolve.

**Access control is repository-level.** Ownership files govern review, not permission.
Separating repositories is the only straightforward way to bound what a team can change.

**The existing convention is already multi-repo.** The product repositories are separate
and work. Consistency has real value: the same deployment patterns, the same CI shape, no
second mental model.

**And the cost of separation is lower than it appears.** `uv` resolves dependencies
directly from a git tag, so there is no package index to run and no publish step:

```toml
[tool.uv.sources]
quanfire-ml-core = { git = "https://github.com/…/quanfire-ml-core", tag = "v0.2.0" }
```

That removes the main friction usually cited against splitting.

```
quanfire-ml-core        config, logging, registry, artefacts, text preparation
        ▲               no ML dependencies; stable; changes rarely
        │ git tag
        ├── quanfire-multilingual-embedding    ← this repository
        ├── quanfire-llm
        ├── quanfire-vision
        └── quanfire-speech
```

### Split coarsely, and only when a package exists

Do not create empty repositories in advance. The first split is the only one justified
today:

| Repository | Contents | When |
|---|---|---|
| `quanfire-ml-core` | `common`, `core`, `utils`, `config`, plus `corpus`, `tokenizer`, `vocabulary` | When a second consumer needs it |
| this repository | `embedding`, `evaluation`, `pipelines` | Already exists |
| `quanfire-llm` | — | Exists already; consumes this repo as a pinned dependency |
| `quanfire-vision`, `quanfire-speech` | — | When that work actually starts |

Text preparation sits in core rather than in this repository, because the LLM and speech
work both need correct segmentation and script handling and should not depend on a
training stack to get it.

### The split points already exist

The enforced layer graph maps onto repository boundaries without redesign, which is the
payoff from having tested it:

| Destination | Current layers |
|---|---|
| `quanfire-ml-core` | `common`, `core`, `utils`, `config`, `corpus`, `tokenizer`, `vocabulary` |
| this repository | `embedding`, `evaluation`, `pipelines` |

The architecture test forbidding upward imports is what guarantees these cut cleanly. A
`git filter-repo` extraction preserves history for the moved paths.

### When to do it

**Not yet.** Extracting core before a second consumer exists creates a versioning
relationship with nothing on the other end of it. Do it at the point the LLM or speech
work begins, which is when the second consumer appears.

### What belongs in the shared core

Most of it already exists here and would be extracted rather than written:

| Component | State |
|---|---|
| Typed configuration with validation and precedence | Built |
| Structured logging | Built |
| Registry and factory for config-driven components | Built |
| Atomic artefact persistence and versioning | Built |
| Evaluation report scaffolding | Built |
| Corpus and text preparation | Built — text modalities only |
| Serving base — batching, versioning, standard schema | To build |

Text preparation belongs in core rather than in this repository, because the LLM and
speech work both need correct segmentation and script handling and should not depend on a
training stack to get it.

---

## 3. What one GPU can actually do

A single RTX 4070 Ti SUPER with 16 GB VRAM. This table is the constraint that shapes
everything below.

| Task | Fine-tune | Train from scratch |
|---|---|---|
| Text embedding, 568M | ✅ LoRA, ~4 GB | ✅ ~30 days for 20B tokens |
| LLM, 7B | ✅ QLoRA, ~10 GB | ❌ needs 100B+ tokens |
| LLM, 13B | ⚠️ QLoRA, very tight | ❌ |
| Text-to-image, SD1.5 class | ✅ LoRA, ~9 GB | ❌ needs 100M+ image pairs |
| Text-to-image, SDXL class | ⚠️ LoRA, ~13 GB | ❌ |
| Image-to-text (VLM), 2–7B | ⚠️ QLoRA, ~12 GB | ❌ |
| Text-to-speech | ✅ ~8 GB | ⚠️ needs 100s–1000s of hours |
| Image-to-video | ❌ | ❌ |

**Fine-tuning is broadly available. Training from scratch is available for embeddings and
essentially nothing else.**

That is not a reason to abandon the ecosystem. It is a reason to be deliberate about
which parts are trained in-house and which are integrated.

---

## 4. Prioritise by business proximity, not by modality

The instinct is to work through the modality list in order. That is the wrong axis. The
right one is **how close each capability sits to data Quanfire already owns**, because
that is the only place an in-house model beats a free one.

| Capability | Relevance to Quanfire | Build or integrate |
|---|---|---|
| **Text embedding, domain-tuned** | **High** — DocPro, BillAI, MindMap corpora | **Build** (this repo) |
| **Document understanding (VLM)** | **High** — OCR, layout, invoices, contracts | **Fine-tune** |
| **LLM for document tasks** | **High** — extraction, summarisation, Q&A | **Fine-tune** |
| Text-to-speech | Medium — listening to documents and summaries | **Integrate**, fine-tune a voice later |
| Speech-to-text | Medium — dictation, meeting capture | **Integrate** |
| Text-to-image | Low — commodity, no data advantage | **Integrate** |
| Image-to-image | Low | **Integrate** |
| Image-to-video | Lowest — cannot train, expensive to serve | **Integrate**, or defer entirely |

**Document understanding is the highest-value modality after embeddings**, and it is not
obvious from the list. A vision-language model fine-tuned on invoices and contracts —
reading layout, tables, stamps, handwriting — is directly DocPro and BillAI, and the
training data is the documents already flowing through them. Nobody else has that corpus.

General image generation, by contrast, has no data advantage. Integrating an open model
gives the same result for a fraction of the effort.

---

## 5. Per-modality reference

### 5.1 Text embedding — *this repository*

**Architecture.** Transformer **encoder**, bidirectional attention, pooled to one vector.
Trained contrastively: InfoNCE, in-batch negatives, mined hard negatives. Matryoshka
representation learning allows truncating dimensions at inference.

```
text → tokenizer → encoder layers → pooling (mean/CLS) → L2 normalise → vector
```

**Data.** Pairs, not raw text. 100k–1M pairs for a domain adaptation. Where labelled
pairs do not exist, they are mined from document structure — see the roadmap's Phase C.

**Compute.** LoRA over a 568M base, ~4 GB, ~3.7 days for 1M pairs over 3 epochs.

**Verdict: build.** Detailed in [ROADMAP.md](ROADMAP.md).

### 5.2 LLM (text-to-text)

**Architecture.** Transformer **decoder**, causal masking. Current practice: rotary
position embeddings, grouped-query attention, SwiGLU activations, RMSNorm, pre-norm
residuals.

```
token ids → token embeddings → N × (attention + feed-forward) → output projection → next-token logits
```

The token embedding table lives here, sized `vocab × hidden_dim`, and is learned with
everything else.

**Data.**

| Stage | Volume |
|---|---|
| Pretraining | 100B+ tokens — **out of reach** |
| Continued pretraining (domain) | 1–10B tokens — marginal |
| Instruction fine-tuning | 10k–100k examples — **feasible** |
| Preference tuning (DPO) | 5k–50k preference pairs — feasible |

**Compute.** QLoRA on a 7B base fits in ~10 GB. Full fine-tuning caps around 1B.

**Verdict: fine-tune, never pretrain.** Start with instruction tuning on document tasks —
extraction, summarisation, structured output — where Quanfire's data is the advantage.

### 5.3 Image-to-text (vision-language) — *the underrated one*

**Architecture.** Three parts:

```
image → vision encoder (ViT/SigLIP) → projector (MLP) → LLM decoder → text
                                            ↑
                    maps image features into the LLM's token embedding space
```

The projector is the interesting piece: it is *literally* the case where external vectors
enter an LLM, by being mapped into the same space as its token embeddings. Usually the
vision encoder is frozen, the projector trained, and the LLM adapted with LoRA.

**Data.** Image–text pairs. 100k+ for general capability; 10k task-specific pairs for a
narrow domain like invoice extraction. **Quanfire generates this data as a by-product of
DocPro**, provided the documents and their extracted output are retained together.

**Compute.** A 2–7B VLM under QLoRA lands around 12 GB — tight but workable.

**Verdict: fine-tune, and prioritise it.** Highest-value non-embedding modality.

### 5.4 Text-to-image and image-to-image

**Architecture.** Latent diffusion. Two families, and the distinction matters:

| | **U-Net** | **DiT / MMDiT** |
|---|---|---|
| Used by | SD 1.5, SDXL | SD3, Flux and newer |
| Shape | Convolutional encoder–decoder with skip connections | Transformer over latent patches |
| Status | Mature, abundant tooling, lighter | Current direction, scales better |

You asked specifically about U-Net. It is the classic diffusion backbone, but **new work
has largely moved to diffusion transformers**. If building rather than fine-tuning, DiT is
the better target; if fine-tuning, follow whatever the base model uses.

```
text → text encoder → conditioning
                          ↓
noise → [U-Net or DiT] × T denoising steps → latent → VAE decoder → image
```

Image-to-image is the same stack with an image-derived starting latent instead of noise.

**Data.** LoRA on a style or subject: 20–1000 images. From scratch: 100M+ pairs, out of
reach by several orders of magnitude.

**Compute.** SD1.5-class LoRA ~9 GB; SDXL-class ~13 GB; larger models need quantisation.

**Verdict: integrate.** LoRA only for a specific style the product needs. No data
advantage here.

### 5.5 Text-to-speech

**Architecture.** Either a pipeline — text → normalisation → phonemes → acoustic model →
vocoder — or an end-to-end model. Modern small models are end-to-end.

**Text normalisation is the part worth owning**, and it belongs in this repository's
corpus layer: expanding numbers, dates and currency to spoken form. Indian numbering is
lakh/crore grouped, so `₹1,23,456` must read as *"one lakh twenty-three thousand four
hundred fifty-six"*. Generic engines get this wrong, and on invoices it is immediately
audible.

**Data.** Voice cloning from seconds of reference audio. Full training needs hundreds to
thousands of hours.

**Compute.** Inference runs on CPU for small models. Fine-tuning ~8 GB.

**Verdict: integrate**, and check licences carefully — several well-regarded open TTS
models are non-commercial and unusable in a product.

### 5.6 Image-to-video

**Architecture.** Spatiotemporal diffusion — a DiT extended with temporal attention,
operating on video latents.

**Data.** Millions of captioned clips.

**Compute.** Training is out of reach entirely. Even inference on small video models is
tight at 16 GB and slow.

**Verdict: integrate, or defer.** No path to training this, and no data advantage. It is
the last thing to add and the first thing to drop.

---

## 6. Sequencing

Each step should produce something usable before the next begins.

| Order | Work | Repository |
|---|---|---|
| 1 | Finish the embedding factory — Phases A–D | this repository |
| 2 | Extract core when a second consumer appears | `quanfire-ml-core` |
| 3 | Integrate TTS behind a service, with Indian text normalisation | `quanfire-speech` |
| 4 | Fine-tune an LLM for document extraction and summarisation | `quanfire-llm` |
| 5 | Fine-tune a VLM for document understanding | `quanfire-vision` |
| 6 | Integrate image generation | `quanfire-vision` |
| 7 | Integrate video, if still wanted | `quanfire-vision` |

Steps 4 and 5 depend on **retaining paired training data from DocPro now** — documents
alongside their extracted output. That data is being generated already; if it is not being
kept in a form suitable for training, that is the cheapest high-value action available
today, and it costs nothing but storage.

---

## 7. The honest constraint

**One GPU and a small team cannot train eight modalities.** Nothing in this plan pretends
otherwise.

What is achievable is an ecosystem where Quanfire *offers* all these capabilities, trains
in-house only where its own data makes the model better, and integrates open models
everywhere else. The differentiation is not "we trained everything" — it is the document
corpus, the domain adaptation, and the fact that the whole stack can run on-premise for
customers whose data cannot leave the country.

Ranked by defensibility:

1. **Domain-adapted text embeddings** — proprietary corpus, cheap to train, direct product value
2. **Document understanding VLM** — training data is a by-product of the existing product
3. **Document-task LLM fine-tunes** — same argument, more competition
4. **On-premise deployment of everything** — an operational moat, not a model one
5. Everything else — integrate, and compete on product rather than on weights
