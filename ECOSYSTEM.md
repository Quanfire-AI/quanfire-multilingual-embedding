# QuanFire AI Ecosystem — Architecture and Plan

> How the modalities fit together, what each one actually requires, and which are
> worth building rather than integrating.

**Status:** planning document. Nothing here is built except the text embedding
pipeline in this repository.

---

## 1. Two different things called "embedding"

This distinction determines what belongs in which project, so it comes first.

| | **Retrieval embedding** | **Token embedding** |
|---|---|---|
| Unit | One vector per sentence or document | One vector per token id in a vocabulary |
| Purpose | Measure similarity; find relevant text | First layer of a transformer |
| Trained | Separately, contrastively, on pairs | Jointly with every other weight of the LLM |
| Reusable | Yes — any model can consume the vectors | No — meaningless outside its own model |
| Where it lives | **This project** | **Inside the LLM project** |

A retrieval embedding is never fed into an LLM. The pipeline is:

```
query ──► retrieval embedding ──► nearest documents ──► document TEXT into the prompt
                                                             │
                                       LLM tokenises that text
                                                             │
                                       LLM's own token embeddings ──► transformer layers
```

The retrieval vector selects *what text goes in the prompt*. It never crosses into the
model. The exceptions — vision projectors in multimodal models, soft prompts, adapter
layers — are architectural features of those models, not a general mechanism for feeding
external vectors to an LLM.

**Consequence:** token embeddings cannot be produced by this project. They are created
when an LLM is trained, and a separate project owns that.

---

## 2. Repository architecture

### Recommendation: one repository, as a uv workspace

An earlier draft argued for separate repositories on the grounds that the dependency
sets conflict and that a caller wanting text segmentation should not be made to install
video libraries. **That objection is wrong, and was tested rather than assumed.**

A uv workspace gives each package its own dependencies inside a single repository.
Syncing one package installs only what that package declares:

```
$ uv sync --package qf-core
 + pyyaml==6.0.3
 + qf-core==0

$ uv run --package qf-core python -c "..."
pyyaml: True
numpy : False        # declared by a sibling package, correctly absent
```

With the main objection removed, the balance favours a monorepo decisively for a small
team:

| | Monorepo | Separate repositories |
|---|---|---|
| Change core and every consumer | One commit, one CI run | Coordinated pull requests across repos |
| Internal versioning | None needed | Publish and pin every package |
| Dependency isolation | ✅ per-package via workspace | ✅ inherent |
| Refactor across a boundary | Trivial | Painful |
| Independent open-sourcing | Harder | Easy |
| Independent access control | Harder | Easy |

Only the last two favour separation, and neither applies yet.

**The decisive argument is asymmetry.** Splitting a monorepo later is a mechanical
history-preserving operation. Merging separate repositories later loses history or
requires surgery. Start together; split when there is a concrete reason, such as
open-sourcing one component or handing it to a different team.

```
quanfire-ai/                        one repository, uv workspace
├── pyproject.toml                  workspace root
├── uv.lock                         single lockfile, all packages
└── packages/
    ├── core/       config, logging, registry, artefacts   no ML dependencies
    ├── text/       corpus, tokenizer, vocabulary          no ML dependencies
    ├── embedding/  encoders, training, evaluation         torch
    ├── llm/        text-to-text                           torch, transformers
    ├── vision/     image generation and understanding     torch, diffusers
    └── speech/     text-to-speech, speech-to-text         torch, audio stack
```

### The split points already exist

This repository's layer graph maps onto the package boundaries without redesign, which
is the payoff from having enforced it:

| Package | Current layers |
|---|---|
| `core` | `common`, `core`, `utils`, `config` |
| `text` | `corpus`, `tokenizer`, `vocabulary` |
| `embedding` | `embedding`, `evaluation`, `pipelines` |

The architecture test that forbids upward imports is what guarantees these cut cleanly.

### When to do it

**Not yet.** A workspace containing one package is ceremony. Restructure when the second
package is created — most likely when `core` is extracted for the LLM or speech work.
Doing it then costs the same as doing it now and avoids speculative churn.

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

Text preparation is its own package rather than part of `embedding`, because the LLM and
speech work both need correct segmentation and script handling and should not pull in a
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
right one is **how close each capability sits to data QuanFire already owns**, because
that is the only place an in-house model beats a free one.

| Capability | Relevance to QuanFire | Build or integrate |
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
extraction, summarisation, structured output — where QuanFire's data is the advantage.

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
narrow domain like invoice extraction. **QuanFire generates this data as a by-product of
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

| Order | Work | Package |
|---|---|---|
| 1 | Finish the embedding factory — Phases A–D | `packages/embedding` |
| 2 | Restructure into a workspace; extract `core` and `text` | `packages/core`, `packages/text` |
| 3 | Integrate TTS behind a service, with Indian text normalisation | `packages/speech` |
| 4 | Fine-tune an LLM for document extraction and summarisation | `packages/llm` |
| 5 | Fine-tune a VLM for document understanding | `packages/vision` |
| 6 | Integrate image generation | `packages/vision` |
| 7 | Integrate video, if still wanted | `packages/vision` |

Steps 4 and 5 depend on **retaining paired training data from DocPro now** — documents
alongside their extracted output. That data is being generated already; if it is not being
kept in a form suitable for training, that is the cheapest high-value action available
today, and it costs nothing but storage.

---

## 7. The honest constraint

**One GPU and a small team cannot train eight modalities.** Nothing in this plan pretends
otherwise.

What is achievable is an ecosystem where QuanFire *offers* all these capabilities, trains
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
