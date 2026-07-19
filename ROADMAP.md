# Roadmap

> From a word2vec baseline to a served, Indic-first text embedding model.

**Status:** Stage 0 not started. Stages 1–5 are planned, not built.
Everything described past Stage 0 is a proposal, not a commitment.

---

## Positioning

The objective is an embedding model in the class of OpenAI's
`text-embedding-3-*`, exposed over an API.

The strategy is **not** to match those models on English. That is not winnable:
Qwen3-Embedding-8B, Llama-Embed-Nemotron-8B and KaLM-Embedding-Gemma3-12B already
outperform them on open leaderboards, backed by Alibaba, NVIDIA and Tencent compute
budgets. Entering that race means losing it slowly.

The strategy is to **own Indian languages**. OpenAI's models are mediocre on Indic text.
AI4Bharat's IndicBERT covers 12 languages; Google's MuRIL covers 17. Nine of the 22
scheduled languages — Santali, Bodo, Dogri, Maithili, Meitei, Konkani, Kashmiri, Sindhi
and Sanskrit — are served badly or not at all by anyone.

This repository already handles the text of all 22 correctly. That is the foundation of
a defensible position: *the best embedding model for Indian languages*, rather than a
worse version of a general-purpose one.

---

## What already exists, and what does not

The existing framework is roughly 40% of the work, and it is the 40% that teams usually
get wrong.

| Component | Reusable? | Notes |
|---|---|---|
| Corpus layer | **Yes, entirely** | Script-aware segmentation, combining-mark handling, readers, deduplication, statistics. Correct across 22 scheduled languages. |
| Tokenizer | **Yes** | Still required. May be replaced by the base model's own tokenizer in Stage 2. |
| Vocabulary | Partly | Needed for the baseline; a transformer brings its own. |
| Evaluation harness | **Yes, critical** | Per-language fairness reporting is exactly the evidence needed to substantiate the Indic claim. |
| Config, artifacts, reproducibility | **Yes** | Directly reusable. |
| CLI and pipelines | **Yes, with extension** | Structure holds; a new encoder path is added. |
| CI, packaging, docs | **Yes** | Directly reusable. |
| `embedding/word2vec.py` | **No** | Retained as a baseline to measure against, not as a path forward. |

### The one architectural blocker

`EmbeddingModel.train()` returns an `EmbeddingMatrix`, which is a `vocabulary × dimension`
lookup table. A transformer has no such table: it computes a vector per *input text* at
call time. `SentenceEncoder` and `SemanticSearchPipeline` are both built on
`EmbeddingMatrix` too.

So the current contract **cannot** accommodate a contextual model. An earlier version of
the handbook claimed a transformer "would slot into the existing pipeline without changes
elsewhere". That was wrong, and it is corrected. Fixing it is Stage 0.

---

## Stages

Each stage has an exit criterion. A stage is not finished when the code works; it is
finished when the criterion is met.

### Stage 0 — Make the architecture admit contextual models

**Why first:** everything downstream depends on it, and it is small.

Introduce a `TextEncoder` abstraction — `encode(texts: list[str]) -> np.ndarray` — that
both static and contextual models implement. Re-point `SemanticSearchPipeline` at
`TextEncoder` rather than `EmbeddingMatrix`. The word2vec path becomes one implementation;
nothing about its behaviour changes.

- **Deliverables:** `TextEncoder` protocol; `StaticTextEncoder` wrapping the current
  matrix; search pipeline decoupled; architecture test extended to cover the new layer.
- **Exit criterion:** the existing 963 tests still pass, and a stub encoder returning
  random vectors can be served through the pipeline end to end.
- **Effort:** ~1 week. No GPU. No new dependencies.

### Stage 1 — Build the evaluation benchmark, before any model

**Why before the model:** you cannot claim "best for Indic" without evidence, and there is
no adequate public benchmark for the nine low-resource languages. Building it is both the
proof and a moat — a credible public Indic retrieval benchmark is itself a contribution
that attracts users.

Assemble held-out retrieval and similarity sets per language. Sources: MIRACL (Hindi,
Bengali, Telugu), FLORES-200 (all 22, translation pairs usable for bitext retrieval),
Samanantar, IndicGLUE, translated mMARCO. For low-resource languages, bootstrap from
parallel corpora and human-verify a small gold set.

Score the word2vec baseline, BGE-M3, and `text-embedding-3-large` on it. **Publish the
numbers even where we lose** — a benchmark that only flatters its author convinces nobody.

- **Deliverables:** per-language eval sets; a `qfme benchmark` command; a public
  leaderboard table; baseline numbers for three models.
- **Exit criterion:** a reproducible number for every one of the 22 languages, and a
  documented, defensible gap where OpenAI is weak.
- **Effort:** 3–4 weeks. Mostly data work. No GPU.
- **Risk:** for Santali, Bodo, Dogri and Meitei there may be too little text to build a
  meaningful set. If so, say so publicly rather than fabricating one.

### Stage 2 — Fine-tune an open base model

Contrastively fine-tune **BGE-M3** (568M, XLM-RoBERTa, MIT licence, 100+ languages,
8192 context) on Indic pairs. Qwen3-Embedding-0.6B is the alternative if licence terms
suit better.

Technique: InfoNCE with in-batch negatives plus mined hard negatives. Large effective
batch matters more than almost anything else for contrastive training — use GradCache if
GPU memory is the constraint. Add Matryoshka representation learning so callers can
truncate dimensions, matching the `text-embedding-3` interface.

- **Deliverables:** training pipeline; `TransformerEncoder` implementing `TextEncoder`;
  a trained checkpoint; Stage 1 benchmark numbers for it.
- **Exit criterion:** **beats `text-embedding-3-large` on the Indic benchmark** for at
  least the 12 well-resourced languages, and does not regress badly on English.
- **Effort:** 4–6 weeks.
- **Compute:** runs locally on the 16 GB RTX 4070 Ti SUPER using LoRA plus GradCache —
  see the compute profile below. Expect 1.5–3 days per full run over ~1M pairs, so
  validate the pipeline on a smaller base model first. **No rental cost for this stage.**

### Stage 3 — Serve it

FastAPI over the artifact-loading pattern that already exists. Use an **OpenAI-compatible
request and response schema** so it is a drop-in replacement — that removes the main
adoption barrier at essentially zero design cost.

- **Deliverables:** `POST /v1/embeddings`; batching; model versioning; dimension
  truncation; ONNX export and quantization; container image; latency and throughput
  benchmarks; auth and rate limiting.
- **Exit criterion:** p95 latency under 100 ms for a single short input on CPU, and a
  client can switch from OpenAI by changing only the base URL.
- **Effort:** 3–4 weeks.
- **Note:** serving cost, not training cost, will dominate the budget from here on.

### Stage 4 — Deeper training *(conditional)*

Only if Stage 2 demonstrates real demand. Continued pretraining on IndicCorp v2 (~20.9B
tokens, 24 languages) followed by contrastive training, rather than fine-tuning alone.

- **Exit criterion:** a material gain over the Stage 2 checkpoint on the same benchmark.
- **Effort:** 3–6 months.
- **Compute:** beyond the local workstation. 8× A100 for weeks — on the order of
  **$10,000–25,000** rented. Do not begin this without Stage 2 numbers justifying it.

### Stage 5 — Multimodal *(parked)*

Text-to-image, image-to-text and speech are each separate multi-year projects with
different data, architectures and evaluation. They share almost nothing with this codebase
beyond config and serving scaffolding.

**Deliberately not planned in detail.** Revisit only once the text model is serving real
traffic. Sequencing these earlier would starve the one thing with a defensible position.

---

## Compute profile

Training happens on a dedicated workstation, separate from the development machine:

| | |
|---|---|
| CPU | Intel i7-14700K, 20 cores / 28 threads |
| RAM | 32 GB |
| GPU | NVIDIA RTX 4070 Ti SUPER, **16 GB VRAM** |
| Storage | ~720 GB free |
| OS | Windows (x64) |

**This is sufficient for Stages 0 through 3.** Only Stage 4 requires rented hardware.

### What fits in 16 GB

For BGE-M3 (568M parameters), measured against the usual training-memory formula:

| Configuration | Model + optimizer | Left for activations |
|---|---|---|
| Full fine-tune, bf16 + fp32 Adam | 8.5 GB | 7.5 GB |
| Full fine-tune, bf16 + 8-bit Adam | 5.3 GB | 10.7 GB |
| **LoRA (r=16), bf16 frozen base** | **1.1 GB** | **14.9 GB** |

**Use LoRA.** Activations are where contrastive training actually needs room, and LoRA
leaves nearly the whole card for them. A full fine-tune fits only with 8-bit Adam and a
batch size too small to be useful.

### The technique that makes this work

Contrastive training quality depends heavily on **effective batch size**, because
in-batch examples serve as each other's negatives. A 16 GB card fits perhaps 8–16
sequences at 512 tokens — far below the 512–2048 that good results need.

**GradCache** solves this: representations are computed in chunks without gradients, then
recomputed chunk-by-chunk during the backward pass. It decouples effective batch size from
VRAM at the cost of roughly a second forward pass. With it, effective batches of 1024+ are
reachable on this card. Without it, this hardware cannot train a competitive contrastive
model, regardless of how long it runs.

### Expected wall-clock

The 4070 Ti SUPER is roughly a third of an A100 for this workload (~44 vs ~156 dense
bf16 TFLOPS, 672 vs 1555 GB/s bandwidth). A LoRA run over ~1M pairs that takes 12–24h on
an A100 should be expected to take **1.5–3 days** here.

That makes full runs a weekend activity, not an interactive one — so **iterate on a
smaller base model first**. `multilingual-e5-small` (118M) trains several times faster and
validates the entire pipeline; move to BGE-M3 only once the data, loss and evaluation are
known-good. Reserve the long runs for configurations already proven at small scale.

### Practical notes

- **Use WSL2, not native Windows.** CUDA support is full, performance is near-native, and
  the ML tooling — `bitsandbytes`, GradCache implementations, most training scripts — is
  Linux-first. Native Windows will cost time on dependency problems that have nothing to
  do with the model.
- **Storage needs planning.** IndicCorp v2 is ~20.9B tokens; the full set will not sit
  comfortably alongside checkpoints in 720 GB. Keep corpora gzipped — the framework's
  readers already handle `.gz` transparently — and subset by language tier.
- **32 GB system RAM is adequate** because the corpus layer streams rather than
  materialising. That design choice, made for a different reason, pays off here.
- **PyTorch becomes a dependency at Stage 2.** This is the first real change to the
  project's dependency posture and should be a deliberate, recorded decision. Keep it
  confined to the new encoder package so the existing NumPy pipeline stays installable
  without it.

---

## What will actually be hard

Ordered by how likely each is to sink the project.

**Data, not compute.** Contrastive training needs *pairs*, not raw text. For the nine
low-resource languages, retrieval pairs essentially do not exist. Bootstrapping via
translation and transliteration is possible but introduces bias that the evaluation must
be honest about. **Start sourcing data now** — it gates Stage 2 and it is the long pole.

**Benchmark credibility.** A self-published benchmark by the model's author is easy to
dismiss. Mitigate by publishing the construction method, releasing the eval sets, scoring
competitors fairly, and reporting losses.

**Evaluation for languages with no ground truth.** For Santali or Bodo there may be no
one to verify a gold set cheaply. Budget for native-speaker review, or scope those
languages as best-effort and label them as such.

**Licence compatibility.** BGE-M3 is MIT and safe. Verify any base model's terms before
building on it, and record the training data's licence — a model inherits the constraints
of its corpus.

**Serving economics.** A 568M model on CPU is viable; on GPU it is not free. Model the
per-request cost before committing to a pricing structure.

---

## Immediate next steps

1. **Stage 0** — the encoder abstraction. Small, unblocks everything, no external
   dependency. Can start immediately.
2. **Data sourcing in parallel** — audit what exists per language, starting with
   Samanantar, IndicCorp v2, MIRACL and FLORES-200. This is the long pole; begin it
   before it is needed.
3. **Decide the language tier** — all 22 at best-effort, or 6–8 with genuine depth. This
   determines the shape of Stages 1 and 2.

## Open decisions

| Decision | Why it matters | Status |
|---|---|---|
| GPU access — owned or rented? | Determines Stage 2 iteration speed. | **Resolved** — local RTX 4070 Ti SUPER (16 GB) covers Stages 0–3; only Stage 4 needs rental |
| Language tier — all 22, or a focused subset? | Drives benchmark and data effort. | Open |
| Base model — BGE-M3 or Qwen3-Embedding? | Licence and size trade-off. BGE-M3 is the safer default. | Leaning BGE-M3 |
| Open weights or closed? | Affects adoption, benchmark credibility, and commercial model. | Open |
| Target dimensions | Matching 1536/3072 eases migration from OpenAI. | Leaning Matryoshka, truncatable |

---

## Principles carried forward

The existing framework's discipline should survive this expansion:

- **Measure before claiming.** The benchmark precedes the model deliberately.
- **Report per-language, never only an average.** An average is how a model that fails
  half its languages looks acceptable.
- **Publish limitations.** The current documentation states plainly where the framework
  stops; that should remain true as capability grows.
- **Reproducible artefacts.** Seeded runs and configuration persisted alongside every
  model.
