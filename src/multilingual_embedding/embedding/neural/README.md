# embedding/neural

> The contextual half of the framework: a transformer encoder, contrastive training, LoRA, gradient caching, adaptation of published checkpoints, and the 3.4 MB artefact a run produces.

**This is where the models that are actually worth serving come from.** Everything else in
`embedding/` is either the static baseline they are measured against or the contract they
are served through.

---

## Purpose

### What

Seven modules, splitting along one line — tensors on one side, text on the other:

| Module | Owns | In | Out |
|---|---|---|---|
| `architecture.py` | `TransformerEncoderModel`, `EncoderConfig` | `(ids, mask)` tensors | pooled vectors |
| `encoder.py` | `NeuralTextEncoder`, `resolve_device`, `autocast_for` | text | numpy vectors |
| `pretrained.py` | `PretrainedTextEncoder`, `POOLING_STRATEGIES` | a HuggingFace checkpoint name or path | the same contract, someone else's weights |
| `training.py` | `ContrastiveTrainer`, `ContrastiveConfig`, `TextPair`, `TrainingReport` | pairs of texts | a fitted model and a loss curve |
| `lora.py` | `LoRALinear`, `apply_lora`, `merge_lora`, `lora_state_dict` | an `nn.Module` | the same module, 99.5% frozen |
| `gradcache.py` | `cached_contrastive_backward`, `suggest_chunk_size` | an encode function and a batch | exact gradients, chunk-sized memory |
| `adapter.py` | `save_adapter`, `load_adapter`, `AdapterMetadata` | a trained encoder | a 3.4 MB directory that reloads identically |

### Why it is separate from `embedding/`

**torch is optional, and this subpackage is the only thing in the framework that imports
it.** The corpus, tokenizer, vocabulary, evaluation and static-embedding layers stay
installable without a training stack — which matters because other QuanFire work needs
text preparation and nothing else. `embedding/__init__.py` does not import `neural`;
nothing is pulled in transitively. Import it explicitly:

```bash
uv sync --extra neural        # adds torch, transformers
```

```python
from multilingual_embedding.embedding.neural import NeuralTextEncoder
```

`neural/__init__.py` catches a missing torch and re-raises it naming that command, rather
than letting an `ImportError` on a transitive module reach the caller.

### Where it sits

Inside the `embedding` layer, so it may import `common`, `core`, `utils`, `config`,
`corpus`, `vocabulary` and `tokenizer`, and nothing above. It is consumed by
`pipelines/search.py` (through `SemanticSearchPipeline.from_adapter`) and by
`scripts/adapt_pretrained.py`. It is **not** consumed by `pipelines/training.py`, which
still trains the static model only — see *What is not here*.

### When to reach for it

| Situation | Use |
|---|---|
| You need retrieval quality on your own text | `PretrainedTextEncoder` + LoRA + `ContrastiveTrainer` — i.e. `scripts/adapt_pretrained.py` |
| You want a floor to measure that against | the static `Word2Vec` path in the parent package |
| You need a model with no upstream licence | `TransformerEncoderModel` from scratch — capable, but see the honest note below |
| You have a trained adapter and want to query it | `SemanticSearchPipeline.from_adapter` |
| Your batch does not fit in VRAM | `cached_contrastive_backward` |

---

## The two transformers, and which one to use

There are two, and confusing them wastes weeks.

**`architecture.py` — ours.** A complete pre-norm transformer encoder written out in this
repository: fused QKV attention, GELU, learned positions, masked mean pooling. It trains
from scratch and it works.

**`pretrained.py` — theirs.** A thin `nn.Module` wrapper around a published checkpoint
loaded through `transformers`, presenting the same `(ids, mask) -> vectors` shape.

Writing ours first was the right way to build a training loop that can be trusted: a
borrowed checkpoint would have masked a broken loop, and every bug in the objective, the
sampler and the evaluation surfaced against a model we controlled end to end. It is the
wrong way to get a model worth serving. **What a published encoder has is pretraining
scale, and that is exactly what one consumer GPU cannot reproduce.** Roughly 20B tokens
over 30 days locally, per ROADMAP Phase E.

So the production path is: their architecture and their pretraining, your corpus, your
mined pairs, your adaptation, your evaluation. The last four are what a domain-specific
model is actually made of.

**Why the two are not interchangeable at the weight level.** Ours is pre-norm — the layer
norm sits inside the residual branch — and most published encoders are post-norm. The
parameter *shapes* match, so loading their weights into ours **succeeds** and produces a
model that is structurally valid and numerically wrong, degrading quietly rather than
failing. Going through the original library avoids inventing that failure mode. This is
the single most important thing to know about this subpackage.

---

## The failure mode this code is organised around

Every module here has at least one guard against the same class of defect: **a
configuration that produces plausible vectors encoding the wrong thing, with nothing
raising.**

| Mistake | What you see | Guard |
|---|---|---|
| Trained mean-pooled, served CLS-pooled | mediocre retrieval | pooling recorded in `adapter.json` |
| E5 model served without `query: ` / `passage: ` | mediocre retrieval | prefixes recorded, applied by the pipeline |
| LoRA applied where nothing matched | loss barely moves | `ValidationError` listing available layer names |
| `B` initialised randomly | first steps undo the pretraining | `B` is zero-initialised; asserted by test |
| Gradient caching with dropout on | wrong gradients, by a wide margin | RNG state captured and restored per chunk |
| fp16 without a `GradScaler` | small gradients flush to zero | fp16 is **rejected**; bf16 or fp32 only |
| Duplicate passage in a contrastive batch | model punished for a correct answer | sampler de-duplicates |
| Mined negative that is a correct answer | **loss improves**, retrieval degrades | identity, provenance and ceiling guards in `negatives.py`; audit sample for the rest |

The last one is the exception that proves the rule: it is the only entry whose symptom is a
*better*-looking number rather than a mediocre one, and the only one no threshold can close
completely — see the hard-negative section in the [layer README](../README.md).

None of these raise on their own. All of them look like "the model is not very good",
which is indistinguishable from the model not being very good. That is why the checks are
structural rather than left to discipline.

---

## Key design decisions

### Contrastive training is memory-hungry for a reason

The objective is InfoNCE over in-batch negatives: every *other* passage in the batch is a
negative for each query. So **batch size is a quality parameter, not a speed one.** A batch
of 16 asks the model to pick the right passage from 16 candidates; a batch of 1024 makes it
pick from 1024. Published sentence encoders train at 1024 and above.

A 16 GB card fits perhaps 8–16 sequences of 512 tokens with activations retained. That gap
is memory, not time, and no amount of patience closes it.

**Mined hard negatives buy back some of that gap.** A `TextPair` may carry a `negatives`
tuple; those texts become extra candidate columns, so a batch of 16 with 4 mined negatives
each poses a harder task than a batch of 16 alone — and a harder one than a random batch of
64, because the extra candidates were chosen for being confusable rather than drawn at
random. The columns are shared and deduplicated across the batch. A pair set without
negatives yields zero extra columns and trains exactly as before.

Temperature scales the logits before the softmax; 0.05 is the default and the usual
starting point. Weight decay applies to weight matrices only — decaying a LayerNorm gain
pulls it toward zero and scales down that layer's entire output.

### LoRA: frozen base, low-rank update, zero-initialised up-projection

`LoRALinear` learns `B @ A` beside a frozen base layer and adds `scaling * B(A(x))`, where
`scaling = alpha / rank` so changing the rank does not change the effective learning rate.

Measured at BERT-base shape (30,522 tokens, width 768, 12 layers, rank 16 on the attention
projections):

| | Full fine-tune | LoRA |
|---|---|---|
| Trainable parameters | 109.7M (100%) | 884,736 (**0.81%**) |
| Checkpoint | 419 MB | **3.4 MB** (adapters only) |
| Adam moment state | 0.82 GB | **6.8 MB** |

The small checkpoint is the reason to prefer LoRA even where memory is not the constraint:
**many domain adaptations of one base cost less to store than one model.** That is the
whole shape of the product — one base, several adapters, swapped per tenant or per domain.

`apply_lora` freezes everything first and *then* attaches adapters, so anything untargeted
stays frozen rather than relying on the caller. `lora_state_dict` selects by walking for
`LoRALinear` instances rather than by name — the feed-forward block legitimately contains
layers called `up` and `down`, and a name filter would pull base weights into a supposedly
adapter-only file. `merge_lora` folds each adapter into its base weight exactly; the
adapter was only ever an additive term, so this is not an approximation.

### Gradient caching is exact, not an approximation

`cached_contrastive_backward` in three steps: encode every chunk under `no_grad` keeping
only the vectors; compute the loss over *all* vectors at once and take its gradient with
respect to them (a small `batch × dimension` tensor holding everything the chunks need to
know about each other); re-encode each chunk with the graph and backpropagate the cached
vector gradient.

Measured on an RTX 4070 Ti SUPER, batch 256, a 5.3M-parameter encoder over 4,000 pairs —
all four cells of one experiment:

| precision | no caching | chunk size 32 |
|---|---|---|
| fp32 | 4.89 GB / 4.3 s | 0.40 GB / 4.7 s |
| bf16 | 2.99 GB / 2.7 s | 0.29 GB / 4.7 s |

**Caching is what buys the memory** — 12.2× on its own against 1.6× for bf16, and 16.9×
together. Final losses across all four cells spanned 0.51%, which is the exactness claim
holding on real hardware rather than in a unit test. Wall-clock cost is 1.09× against fp32
and 1.74× against bf16.

`chunk_size` sets peak memory; the largest value that fits is the right one.
`suggest_chunk_size` is a starting point from measured bytes-per-example, not an answer.

### Mixed precision is bf16 or nothing

`autocast_for` accepts `"fp32"` (returning a null context, so the ordinary path carries no
autocast machinery at all) and `"bf16"`, and raises on anything else. bf16 shares fp32's
exponent range, trading mantissa bits instead, which training tolerates well. It is ignored
with a **warning** on Apple Metal and on CUDA hardware reporting no bf16 support — silently
training in a precision the caller did not ask for is worse than refusing audibly.

The forward pass runs under autocast; the backward pass deliberately does not.

### The artefact records how the model must be used

`save_adapter` writes `adapter.pt` (the low-rank tensors) and `adapter.json`. The JSON
names the base checkpoint rather than copying it — hundreds of megabytes, already cached,
and unchanged, because LoRA freezes it — and records **pooling, max length, normalisation,
the LoRA settings, and the query and passage prefixes**, plus free-form `notes` carrying
what it trained on and what it scored.

Those fields are not metadata for humans. Pooling and prefixes belong to the model as
firmly as its weights do, and `load_adapter` returns `(encoder, metadata)` as a pair
specifically so the prefixes cannot be forgotten. `SemanticSearchPipeline.from_adapter`
consumes both.

The reload is verified rather than assumed: the round-trip test asserts byte-identical
vectors. A saved model that scores differently on reload is worse than no saved model,
because it is trusted.

---

## Input and output

```
                     data/dumps/*.xml.bz2
                              |  qfme extract
                              v
                     corpus JSON Lines           <- corpus layer
                              |  qfme mine-pairs
                              v
                     pairs.jsonl.gz              <- corpus/pairs.py
                              |
     +------------------------+------------------------+
     |                        |                        |
  TextPair              PretrainedTextEncoder      LoRAConfig
     |                        |                        |
     +--------> ContrastiveTrainer.train() <-----------+
                              |
              +---------------+---------------+
              |                               |
      TrainingReport (loss curve)      save_adapter()
                                              |
                                              v
                                       models/<name>/
                                       adapter.pt + adapter.json   (3.4 MB)
                                              |
                                              v
                            SemanticSearchPipeline.from_adapter()
```

| Stage | Input | Output |
|---|---|---|
| `PretrainedTextEncoder.load` | checkpoint name/path, pooling, device, max length | an encoder satisfying `TextEncoder` |
| `apply_lora` | the encoder's `nn.Module`, `LoRAConfig` | same module, ~0.5–0.8% trainable |
| `ContrastiveTrainer.train` | `list[TextPair]`, `ContrastiveConfig` | `TrainingReport` — per-epoch losses |
| `save_adapter` | encoder, `LoRAConfig`, prefixes, notes | a directory, ~3.4 MB |
| `load_adapter` | that directory | `(PretrainedTextEncoder, AdapterMetadata)` |

---

## Usage

### Adapt a published checkpoint (the production path)

In practice, run `scripts/adapt_pretrained.py` — it wraps everything below plus sampling,
the held-out split, the facet checks and the retrieval report. See
[`scripts/README.md`](../../../../scripts/README.md).

```python
from multilingual_embedding.embedding.neural import (
    ContrastiveConfig,
    ContrastiveTrainer,
    LoRAConfig,
    PretrainedTextEncoder,
    TextPair,
    apply_lora,
    save_adapter,
)

encoder = PretrainedTextEncoder.load(
    "intfloat/multilingual-e5-small",
    pooling="mean",
    device="cuda",
    max_length=256,
)

apply_lora(encoder.train_mode(), LoRAConfig(rank=32, alpha=64, targets=("query", "value")))

pairs = [TextPair(anchor="query: " + q, positive="passage: " + p) for q, p in mined]

report = ContrastiveTrainer(encoder, ContrastiveConfig(epochs=2, batch_size=64)).train(pairs)

save_adapter(
    encoder,
    "models/indic-v1",
    lora=LoRAConfig(rank=32, alpha=64, targets=("query", "value")),
    query_prefix="query: ",
    passage_prefix="passage: ",
    notes={"pairs": len(pairs), "final_loss": report.losses[-1]},
)
```

### Serve it

```python
from multilingual_embedding.pipelines.search import SemanticSearchPipeline

pipeline = SemanticSearchPipeline.from_adapter("models/indic-v1")

pipeline.index(passages)                    # passage prefix applied here
pipeline.search("संविधान में मौलिक अधिकार")  # query prefix applied here
```

### Train from scratch (capability, not the default)

```python
from multilingual_embedding.embedding.neural import (
    EncoderConfig, NeuralTextEncoder, TransformerEncoderModel,
)

model = TransformerEncoderModel(EncoderConfig(vocabulary_size=32000, dimension=384, layers=6))
encoder = NeuralTextEncoder(model, tokenizer)
```

Useful for languages no published checkpoint serves, and for a model with no upstream
licence. Not competitive with an adapted checkpoint at this compute budget, and the
honest reason is pretraining scale, not implementation quality.

### Fit a batch that does not fit

```python
from multilingual_embedding.embedding.neural import cached_contrastive_backward, suggest_chunk_size

cached_contrastive_backward(encode, anchors, positives, chunk_size=suggest_chunk_size(...))
```

---

## What this has achieved

Four controlled experiments on Hindi and Tamil Wikipedia, each varying exactly one facet
against a pinned evaluation set. The full write-up is in
[`ROADMAP.md`](../../../../ROADMAP.md); the result that matters here:

| varied | held fixed | transfer captured |
|---|---|---|
| task shape (`adjacent` → `heading_section`) | language, corpus | **−17%** (none) |
| language (hi → ta) | task shape, corpus | **+95%** (essentially complete) |

**The adaptation is language-general and task-specific** — the reverse of how this work was
framed for months. In-distribution gains reached +40.9% recall@1 over the published
checkpoint, concentrated in the *low* lexical-overlap band (+126.7%) where string matching
cannot help, which is why the headline is believable rather than merely large.

One 40,000-pair mixed adapter (`indic-v1`, 3.4 MB) delivered +38.0% on `heading_section`
and +40.8% on `adjacent` simultaneously. A mixture containing the shape works; a single
different shape does not.

---

## Pros and cons

**Strengths**

- One frozen base plus several 3.4 MB adapters is the whole multi-domain story, and it is
  cheap enough to be routine.
- Gradient caching makes batch size a free parameter, and it is exact rather than
  approximate — verified gradient-for-gradient and invariant to chunk size.
- Every quiet-degradation path listed above is structurally guarded, not documented.
- The artefact carries how it must be used, and reloads byte-identically.
- Runs on one 16 GB consumer card, including a 568M base under LoRA.

**Limitations**

- **Single-process.** No data-loader parallelism, no distributed training. A run is bounded
  by one process on one device.
- **CUDA is unverified by the test suite.** Development is on an Intel Mac with no NVIDIA
  GPU; the CUDA paths are exercised by hand on the training box. Device-specific bugs
  surface there first.
- **No CLI path.** `qfme train` produces the static model. The contextual model is trained
  through `scripts/adapt_pretrained.py` and the Python API.
- **From-scratch pretraining is a capability, not a competitive option** at this compute
  budget.
- **First use reaches the network.** `PretrainedTextEncoder` downloads and caches weights.
  This is opt-in behind an extra; `local_files_only=True` refuses the network entirely,
  which is what a reproducible experiment should set.
- **Hard negatives are mined but their false-negative rate is unmeasured.** `qfme
  mine-negatives` produces them and the trainer consumes them; what no code here can tell
  you is how many of them are correct answers in disguise. `--audit` writes the hardest
  sample for a person to label, and until someone does, the run reports the population that
  rate is drawn from and refuses to name a rate.
- **The lever is unproven on this corpus.** Mining is implemented and tested; whether it
  improves `models/indic-v1` is a GPU experiment that has not been run.
- **No ONNX export or quantisation**, so serving runs the torch stack.

---

## What is not here

**No orchestration of the from-scratch path.** `pipelines/training.py` runs the static
model only, and that is deliberate rather than an omission: contrastive training consumes
pairs, and until `corpus/pairs.py` existed there was nothing to feed a stage with.

Half of that gap is now closed. Adapting a *published* checkpoint is orchestrated by
`pipelines/adaptation.py` and reachable as `qfme adapt`, which resolves a config file and a
compute profile into an experiment and runs it. What is still missing is the from-scratch
side: training the encoder in this package from nothing means driving `ContrastiveTrainer`
directly, and a neural stage in `TrainingPipeline` is tracked in `ROADMAP.md`.

**No pair mining.** That lives in [`corpus/pairs.py`](../../corpus/README.md), one layer
down, because manufacturing pairs is a property of document structure rather than of
models.

**No serving API.** `SemanticSearchPipeline.from_adapter` is the local path. The HTTP
endpoint, versioning, dimension truncation, ONNX and the container image are Phase D.

---

## Tests

| File | Tests | Covers |
|---|---|---|
| `tests/embedding/test_neural.py` | 49 | Architecture, contract conformance, retrieval quality, persistence, pipeline service, precision, the candidate columns |
| `tests/embedding/test_lora_gradcache.py` | — | LoRA init/freeze/learn/merge, adapter checkpoints, exact gradient caching, chunk sizing |
| `tests/embedding/test_pretrained.py` | — | Checkpoint loading, pooling strategies, failure messages |
| `tests/embedding/test_adapter.py` | — | Byte-identical round trip, prefixes surviving, mismatch failing loudly |
| `tests/integration/test_hard_negatives.py` | 5 | Documents → pairs → mined negatives → a model that trains, across three layers |
| `tests/pipelines/test_search_adapter.py` | 6 | `from_adapter`, and prefixes reaching the encoder |

All require the `neural` extra and use `pytest.importorskip`, so a core-only checkout runs
green with fewer tests. The mining algorithm itself does not — it takes the `TextEncoder`
contract, so `tests/embedding/test_negatives.py` (35) runs on a base install and this file's
job is only to show that a torch-backed encoder satisfies that contract in practice. Checkpoints in tests are **built, not downloaded** — a 1,000-token
vocabulary and a two-layer 64-dimension BERT written to `tmp_path`. Nothing reaches the
network.

The decisive tests are the ones asserting a *property* rather than a value: that `B @ A` is
exactly zero at step one, that cached gradients equal uncached ones, that a reloaded
adapter produces byte-identical vectors, and that the model actually learns — mean
within-topic cosine similarity must exceed cross-topic by a clear margin, without which
every other test would pass against an implementation returning noise.
