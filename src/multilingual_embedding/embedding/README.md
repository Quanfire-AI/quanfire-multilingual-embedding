# embedding

> Turns tokens into vectors — a static word2vec baseline, a transformer encoder trained contrastively, a published checkpoint adapted with LoRA, and exact similarity search over whichever produced them.

## Purpose

This layer is where text stops being symbols and becomes geometry. It holds three routes
to a vector and one contract that hides which route was taken.

The **static** route is `Word2Vec`, which fits skip-gram vectors with negative sampling
in pure numpy and hands back an `EmbeddingMatrix` — a `vocabulary × dimension` lookup
table binding those vectors to the vocabulary that indexes them. The sentence encoders
compose word vectors into sentence vectors, and `SimilarityIndex` answers
nearest-neighbour queries over a set of those.

The **contextual** route is `neural/architecture.py`: a transformer encoder written out in
this repository, fitted with an InfoNCE contrastive objective, with LoRA and gradient
caching so it trains on the hardware actually available. It computes a vector for the whole
input at call time.

The **adapted** route is `neural/pretrained.py`: a published encoder — `multilingual-e5-small`
and its kind — loaded through its own library and put behind the same contract, then
adapted with LoRA on mined pairs and saved by `neural/adapter.py` as a few megabytes.
**This is the route that produces the models this project actually ships.** Writing the
architecture out was the right way to build a training loop worth trusting; it is the wrong
way to get pretraining scale, which a single consumer GPU cannot reproduce. See
[`neural/README.md`](neural/README.md).

`encoder.py` defines the contract all three satisfy: `TextEncoder`, a `Protocol` of
`dimension`, `encode` and `encode_batch`. This is the layer's most important boundary and
the reason it exists in this shape — see below.

It is its own layer because everything below deals in text and ids while everything here
deals in float arrays, and because that contract lets the pipeline layer swap the model
without knowing which one it holds.

## Modules

| Module | Responsibility |
|---|---|
| `base.py` | `EmbeddingModel` ABC: `train`, `matrix`, `save`, `load`. The contract for a *static* model. Deliberately thin — anything one model needs and the others do not lives on that model. |
| `encoder.py` | `TextEncoder` Protocol — text in, vectors out — and `encoder_dimension`, which checks a declared dimension against real output. |
| `word2vec.py` | `Word2Vec`, the skip-gram negative-sampling trainer, its noise and subsampling tables, and the linear learning-rate decay. |
| `matrix.py` | `EmbeddingMatrix`: vectors paired with their vocabulary, plus `similarity`, `most_similar`, `analogy`, `normalized`, and versioned save/load. |
| `sentence.py` | `SentenceEncoder` ABC, `MeanPoolingEncoder`, `SifEncoder`, and the `SENTENCE_ENCODERS` registry (`"mean"`, `"sif"`). |
| `index.py` | `SimilarityIndex` and `SearchResult`: exact brute-force cosine search over labelled vectors, with persistence. |
| `neural/architecture.py` | `EncoderConfig` and `TransformerEncoderModel` — pre-norm blocks, fused attention, GELU, learned positions, masked mean pooling. Tensors only. |
| `neural/encoder.py` | `NeuralTextEncoder` (tokenise, pad, batch, device, normalise, save/load), plus `resolve_device` and `autocast_for`. |
| `neural/pretrained.py` | `PretrainedTextEncoder`, `POOLING_STRATEGIES`, `PretrainedEncoderError` — a published checkpoint loaded through its own library, wearing `(ids, mask) → pooled vectors`. |
| `neural/adapter.py` | `save_adapter`, `load_adapter`, `AdapterMetadata` — a ~3.4 MB artefact recording the adapter *and how the encoder must be used*. |
| `neural/training.py` | `ContrastiveTrainer`, `ContrastiveConfig`, `TextPair`, `TrainingReport` — InfoNCE over in-batch negatives, with warmup, decay and clipping. |
| `neural/lora.py` | `LoRALinear`, `LoRAConfig`, `apply_lora`, `merge_lora`, adapter-only checkpoints, `parameter_summary`. |
| `neural/gradcache.py` | `cached_contrastive_backward` and `suggest_chunk_size` — chunked encoding with a cached vector gradient. |

`neural/` has its own [README](neural/README.md) covering the training stack in full: the
two transformers and why weights do not cross-load, the seven quiet-degradation failure
modes and their structural guards, the measured LoRA and GradCache numbers, and the
dump → corpus → pairs → adapter → pipeline flow.

## The two contracts, and why there are two

`EmbeddingModel` returns an `EmbeddingMatrix`. That shape is intrinsic to a static model:
every token has one vector, decided at training time and fixed thereafter. It is also the
model's structural limit. In a word2vec matrix, "river bank" and "savings bank" contain
the *same row* for `bank` — the two phrases differ only in their other tokens, and the
sense that distinguishes them is nowhere in the vectors. That is precisely the failure a
contextual encoder fixes, and it is worth being explicit that no amount of training data
fixes it for a static model.

A transformer has no such table. It computes a vector for the whole input at call time,
so there is nothing to look up and nothing to store per token. A pipeline written against
`EmbeddingMatrix` therefore cannot accept a contextual model at all — not as a matter of
quality, but of shape.

`TextEncoder` is the narrower contract both *can* satisfy. It is a `Protocol` rather than
an ABC so the existing `SentenceEncoder` satisfies it without changing its inheritance and
`NeuralTextEncoder` satisfies it without importing anything from `encoder.py`. The contract
is the shape, not the ancestry. It carries three guarantees callers cannot cheaply check:
one-dimensional output of length `dimension`, batch output in the order given, and — the
one that matters — **unencodable input yields a zero vector rather than NaN**. A caller
detects the degenerate case with a norm check; a NaN silently poisons every similarity
computed against it.

`SemanticSearchPipeline` depends on `TextEncoder`, which is what lets it serve either kind
unchanged.

## Key design decisions — the contextual encoder

**The architecture is written out rather than imported from a model library.** A borrowed
checkpoint can hide a broken training loop: a good model trains adequately in spite of a
bug, and the bug surfaces only when the checkpoint is replaced. A model defined here
cannot hide it. That loop is now verified, and external checkpoints land on it —
`neural/pretrained.py` is the smaller, later step, and it has shipped.

**Pre-norm residuals, and this has a consequence worth knowing.** `EncoderBlock` normalises
*before* each sub-layer and adds the residual to the un-normalised input, so gradients reach
early layers through an unobstructed path. Post-norm — the 2017 arrangement, and what most
published encoders including BERT still use — needs a learning-rate warmup to train stably
at depth, which is an extra thing to get right when the training budget allows few attempts.
The consequence: **external weights do not transfer into this model directly.** A BERT-shaped
checkpoint is post-norm, and loading its tensors into pre-norm blocks produces a model that
is structurally valid and numerically wrong.

**Fused scaled dot-product attention.** `F.scaled_dot_product_attention` dispatches to a
fused kernel where one exists rather than materialising the full `(batch, heads, length,
length)` attention matrix. The memory saved there is what makes a usable sequence length
fit on a 16 GB card, so this is a capability decision rather than a speed one.

**Positions are learned, not sinusoidal.** At the few-hundred-token lengths an embedding
model sees, learned positions cost little and train slightly better. The cost is that the
model cannot generalise past `max_length`, and `forward` raises `ValidationError` rather
than silently indexing off the table. `sinusoidal_positions` is kept in `architecture.py`
as the drop-in alternative for when that generalisation is needed; rotary embeddings would
be the choice for a long-context decoder, not here.

**Padding is excluded from attention and from pooling, separately.** The key-padding mask
is broadcast over heads and query positions so padding is never attended to; without it a
sequence encodes differently depending on what it happened to be batched with. Pooling then
averages over the true mask, floored at a count of one. There is one subtlety: a row of pure
padding — an empty string — would be masked at every position, and a softmax over an
all-`-inf` row is NaN which contaminates the whole batch through the shared weights. Such
rows are allowed to attend to position 0 purely to keep the kernel finite, while pooling
still uses the untouched mask, so they come out as zeros. That is what the `TextEncoder`
contract promises for unencodable input.

**Vectors are L2-normalised on the way out.** Cosine similarity then reduces to a dot
product, which is what `SimilarityIndex` assumes, and vectors from differently-sized models
become directly comparable. The norm is clamped at `1e-12` so a zero row stays zero instead
of becoming NaN.

## Key design decisions — contrastive training

**The objective is InfoNCE over in-batch negatives.** Each example is a `TextPair` of texts
that should encode close together — a question and the passage answering it, a heading and
its section, a sentence and its translation. Every *other* positive in the batch serves as a
negative for that anchor, so the targets are simply the diagonal of the
`anchor @ positive.T` similarity matrix.

**Batch size is a quality parameter, not just a speed one.** A batch of 16 asks the model
to pick the right passage from 16 candidates; a batch of 1024 makes it pick from 1024,
which is a far harder and more useful task. This is why contrastive training is
memory-hungry in a way supervised training is not, why `ComputeConfig.batch_size` is the
one machine setting that legitimately changes a result, and why it is recorded alongside
the artefacts rather than treated as an implementation detail.

**Duplicate positives within a batch are poison, so the sampler drops them.** If the same
passage appears twice, the loss actively punishes the model for matching a correct answer,
because the second copy is labelled a negative for the first anchor. Dropping the duplicate
costs one example and avoids training against a contradiction. A trailing batch of one is
dropped for the related reason that it has no negatives at all: its loss is identically
zero, which looks like success and teaches nothing. `train` refuses fewer than two pairs
outright for the same reason.

**Anchors and positives are encoded in one forward pass.** `_step` concatenates both sides
and chunks the result. Beyond halving launch overhead, it guarantees both sides see
identical dropout conditions, which they would not if encoded separately.

**Temperature divides the logits before the softmax.** Lower values sharpen the distribution
and push harder on the hardest negatives; `0.05` is the usual starting point for sentence
encoders and the default here.

**Warmup is not optional in practice.** Contrastive training is unstable in its opening
steps because the encoder's outputs are near-random and the loss gradient is correspondingly
large. `_learning_rate` gives linear warmup over `warmup_ratio` of total steps, then linear
decay.

**Weight decay applies to weight matrices only.** `_parameter_groups` excludes biases and
anything with fewer than two dimensions. This matters more than it sounds: decaying a
LayerNorm gain pulls it toward zero and scales down that layer's entire output.

## Key design decisions — the adapted route

**A published checkpoint is loaded through its own library, not into our transformer.**
The shapes match, so a cross-load *succeeds*: it returns a model that is structurally valid
and numerically wrong, because ours is pre-norm and most published encoders are post-norm.
That is a failure with no exception and no NaN attached to it — the only symptom is
mediocre retrieval, which is indistinguishable from a mediocre model. Going through
`transformers` avoids inventing that failure. `_PooledTransformer` then wraps the result
in this project's forward signature — `(ids, mask) → pooled, normalised vectors` — which
is the same shape `TransformerEncoderModel` produces, so `ContrastiveTrainer`, `apply_lora`,
`cached_contrastive_backward` and `evaluate_retrieval` all apply unchanged. The wrapper is
the whole integration; there is no parallel training path.

**Pooling is a required decision, and `POOLING_STRATEGIES` is `("mean", "cls")`.** A model
trained mean-pooled and served CLS-pooled produces plausible vectors encoding the wrong
thing, and nothing raises. Sentence-transformers models are usually mean-pooled. The value
is stored in the adapter artefact rather than left to be remembered.

**The base install downloads nothing at runtime, and this route breaks that.** Weights are
fetched on first use and cached. It is deliberate and opt-in behind an extra, and
`local_files_only=True` makes a run refuse to reach the network at all — which is what a
reproducible experiment wants, since an upstream repository can change what sits behind a
name.

**The adapter artefact records how the encoder must be used, not only what it weighs.**
`save_adapter` writes the low-rank update — about 3.4 MB at rank 32 on a 118M-parameter
encoder — plus `AdapterMetadata`: the base checkpoint *name* (not a copy; it is hundreds of
megabytes, already cached, and LoRA did not change it), the `LoRAConfig`, the pooling, and
the query and passage prefixes. Those prefixes belong to the model as firmly as its weights
do: an E5 model served without `query: ` degrades quietly. `load_adapter` returns the
encoder **and** its metadata as a tuple, deliberately, because returning the encoder alone
would invite the prefixes being dropped. `SemanticSearchPipeline.from_adapter` consumes
both and applies them automatically — see [`pipelines/README.md`](../pipelines/README.md).

`AdapterMetadata` is a plain mapping rather than a dataclass, so unknown keys survive a
round trip between versions of this code that disagree on fields. The reload is verified
rather than assumed: a round-trip test asserts byte-identical vectors, because a saved model
that scores differently on reload is worse than no saved model — it is trusted.

## Key design decisions — fitting the hardware

**LoRA: frozen base, low-rank update, zero-initialised up-projection.** Instead of learning
the full `ΔW`, `LoRALinear` learns `B @ A` beside a frozen base layer and adds
`scaling * B(A(x))` to its output, where `scaling = alpha / rank` so that changing the rank
does not change the adapter's effective learning rate. The base weights need no gradients
and no optimizer state.

The initialisation is load-bearing. `A` is Kaiming-uniform and `B` is **zero**, so `B @ A`
is exactly zero at step one and the adapted model is numerically identical to the base it
wraps. Were both random, training would begin from a corrupted version of the pretrained
weights and the first optimizer steps would be spent undoing that damage. There is a test
asserting the property directly.

Measured on this code at BERT-base shape — 30,522 tokens, width 768, 12 layers, 12 heads,
rank 16 on the attention projections:

| | Full fine-tune | LoRA |
|---|---|---|
| Trainable parameters | 109.7M (100%) | 884,736 (**0.81%**) |
| Checkpoint | 419 MB | **3.4 MB** (adapters only) |
| Adam moment state | 0.82 GB | **6.8 MB** |

The small checkpoint is the reason to prefer LoRA even where memory is not the constraint:
many domain adaptations of one base can be stored and swapped for the cost of one model.
`apply_lora` freezes every parameter first and *then* attaches adapters, so anything not
explicitly targeted stays frozen rather than relying on the caller. It raises
`ValidationError` listing the available linear layer names when no target matched, because
the alternative presentation of that mistake is a model that silently refuses to learn.
`lora_state_dict` selects adapters by walking for `LoRALinear` instances rather than by
matching names — the feed-forward block legitimately contains layers called `up` and `down`,
and a name filter would pull those base weights into a supposedly adapter-only file.
`merge_lora` folds each adapter into its base weight exactly; the adapter was only ever an
additive term, so this is not an approximation.

**GradCache decouples batch size from memory, and is exact.** A 16 GB card fits perhaps 8
to 16 sequences of 512 tokens with activations retained for the backward pass. Published
sentence encoders train at 1024 and above. That gap is memory, not time, so no amount of
patience closes it. `cached_contrastive_backward` closes it in three steps: encode every
chunk under `no_grad` keeping only the vectors, so activations are discarded as each chunk
finishes; compute the loss over *all* vectors at once and take its gradient with respect to
them, which is a small `batch × dimension` tensor holding everything the chunks need to know
about each other; then re-encode each chunk with the graph and backpropagate the cached
vector gradient as the incoming signal.

The gradients this produces are mathematically identical to a single large backward pass —
`tests/embedding/test_lora_gradcache.py` verifies this gradient-for-gradient and separately
verifies the result is invariant to chunk size. The cost is close to one extra forward pass,
so roughly 1.5 to 1.7 times the wall-clock of an unconstrained batch of the same size, which
would not fit at all. `chunk_size` sets peak memory and the largest value that fits is the
right one; `suggest_chunk_size` is a starting point from measured bytes-per-example, not an
answer.

**Mixed precision is bf16 or nothing.** `autocast_for` accepts `"fp32"` — returning a null
context, so the ordinary path carries no autocast machinery at all — and `"bf16"`, and
raises `ValidationError` on anything else. **fp16 is deliberately rejected.** It needs loss
scaling to stop small gradients flushing to zero, this trainer implements no `GradScaler`,
and offering fp16 without one would produce silently degraded training. bf16 shares fp32's
exponent range so it has no such failure; it trades mantissa bits instead, which training
tolerates well. `ContrastiveConfig.__post_init__` checks the same set, so a typo in a compute
profile fails when the config is built rather than after the model has been placed on a
device.

**bf16 is ignored with a warning on Apple Metal.** MPS autocast support has been incomplete
across torch versions. Silently training in a precision the caller did not ask for is worse
than refusing the request audibly, so `autocast_for` logs a warning and falls back to fp32.
The same happens on a CUDA device whose hardware reports no bf16 support.

**`resolve_device` prefers CUDA, then Apple Metal, then CPU, and understands `"auto"`.** An
explicit preference is honoured without an availability check, so a caller can force `"cpu"`
for a reproducibility run — GPU reductions are not bit-deterministic across runs. `"auto"` is
spelled out because that is the string a configuration file carries; it means the same as no
preference.

**The forward pass runs under autocast; the backward pass deliberately does not.** Autograd
replays each operation in the dtype autocast chose for it, so wrapping the backward would be
redundant at best, and torch documents it as unsupported.

## Key design decisions — the static baseline

**Negative sampling from the unigram distribution raised to 0.75.** Raw unigram frequency
would make almost every negative a function word, which teaches the model very little; a
uniform distribution would make negatives too easy. The `_NOISE_EXPONENT = 0.75` dampening
in `word2vec.py` lifts rare tokens without flattening the distribution, and is the value from
the original paper.

**The noise table is a cumulative distribution sampled with `np.searchsorted`, not
`np.random.choice(p=...)`.** `_prepare_tables` computes `np.cumsum(weights / total)` once;
`_sample_negatives` then draws in O(log V) from a plain uniform. `np.random.choice` with a
`p=` argument rebuilds its internal structure on every call and is orders of magnitude too
slow at the rate this loop needs — one draw per negative per context pair per epoch.
Special-token weights are zeroed before the cumsum so pad and unk are never drawn as
impostors.

**Subsampling and a dynamic window.** `_subsample_probabilities` applies the classic word2vec
keep formula against `config.subsample_threshold` (default `1e-3`), so very frequent tokens
are randomly discarded before pairs are formed. Independently, `_train_sentence` redraws the
window per centre token with `self._rng.integers(1, window + 1)`. A context word at distance 3
is therefore only sampled when the draw is at least 3, which weights nearer context more
heavily without any explicit distance term in the gradient. The tradeoff is that the effective
window is smaller than the configured one; that is intended.

**`W_in` uniform in ±0.5/dimension, `W_out` zeroed, and `W_out` discarded after training.**
The small input scale keeps early dot products near zero where the sigmoid gradient is
largest; a zeroed output matrix means the first updates are driven by labels rather than by
initialisation noise. `train` sets `self._output_weights = None` before building the matrix,
because `W_out` modelled "is this a real context word", not word meaning. The consequence is
that a loaded model can be used for lookup but not resumed for further training, and `load`
says so.

**The centre-word gradient is accumulated once.** In `_update_pair`, `centre_gradient =
gradient @ output_weights[targets]` sums the contribution of the positive and all negatives,
and `input_weights[centre] += centre_gradient` is applied a single time. Updating `W_in`
inside a loop over samples would let later samples see a moved centre vector, which silently
changes the objective rather than raising anything.

**`np.add.at` for the `W_out` update — a correctness fix, not a micro-optimisation.**
Negatives are drawn independently, so a sampled negative can collide with the context token
or with another negative, and `targets` then contains a repeated index. Fancy-index `+=`
evaluates the right-hand side once and keeps only the last write for a duplicated index,
dropping the other updates without any error. `np.add.at(output_weights, targets, ...)`
accumulates every one. It is slower than buffered fancy indexing; that is the price of the
update being right.

**Out-of-vocabulary tokens are dropped during training rather than folded into `<unk>`.**
`_select_ids` skips any token whose id falls below `vocabulary.special_tokens.count`. Routing
every rare word through one shared row would train a single vector that is the centroid of
unrelated meanings, and that vector then appears as a plausible neighbour to everything it
absorbed. The same reasoning drives `SentenceEncoder._known_ids`, which drops unknown tokens
rather than mapping them to the unknown row.

**`EmbeddingMatrix` validates that vector rows match vocabulary size.** The constructor raises
`ValidationError` when `vectors.shape[0] != len(vocabulary)`, and `load` re-checks both the row
count against the stored vocabulary and the width against the recorded dimension. Rebuilding a
vocabulary with a different `min_count` and reusing old vectors is the classic silent
corruption: every row now means a different token, with no exception and no NaN to warn you.

**Normalisation guards zero rows.** `_normalize_rows` divides by `np.maximum(norms, 1e-12)`.
The pad row is all zeros by construction, and dividing it by its own norm would produce NaN
across that row, which then propagates into every score in a `most_similar` call.

**`SimilarityIndex` is exact brute force, and there is a ceiling.** Rows are stored already
normalised, so a query is one matmul and cosine reduces to a dot product; the result is the
true nearest-neighbour set with no build step, no tuning and no recall to measure. That costs
O(n·d) per query, which is the right choice up to roughly 10^5–10^6 items. Past that an
approximate index (HNSW, IVF-PQ) is needed, and this framework does not provide one —
wrapping a poor ANN implementation would be worse than being honest about the ceiling.

**`SifEncoder` needs a batch to remove the first principal component.** The direction being
projected out is estimated from a set of sentences by SVD in `_fit_component`, and a single
sentence in isolation is its own principal component — removing it would zero the vector
outright. `encode` on an unfitted encoder therefore returns the SIF-weighted average and skips
the removal, and `is_fitted` reports which behaviour you are getting. The intended order is
`encode_batch` over the corpus first, then `encode` for queries.

## Usage

The static path, end to end:

```python
from multilingual_embedding.config.base import EmbeddingConfig
from multilingual_embedding.embedding.index import SimilarityIndex
from multilingual_embedding.embedding.sentence import MeanPoolingEncoder
from multilingual_embedding.embedding.word2vec import Word2Vec

sentences = [
    "the cat sat on the mat",
    "the dog sat on the mat",
    "a cat and a dog play",
    "the river is wide and deep",
    "the river runs deep",
] * 40

model = Word2Vec(EmbeddingConfig(dimension=16, window=2, min_count=2, epochs=5, seed=7))

matrix = model.train(sentences)

print(matrix)
print("similarity(cat, dog):", round(matrix.similarity("cat", "dog"), 4))

encoder = MeanPoolingEncoder(matrix)

index = SimilarityIndex.from_texts(
    ["the cat sat on the mat", "the river is wide and deep"],
    encoder,
)

for hit in index.search("a dog on the mat", top_k=2):
    print(hit.index, round(hit.score, 4), hit.label)
```

Actual output:

```
EmbeddingMatrix(size=18, dimension=16)
similarity(cat, dog): 0.5769
0 0.9437 the cat sat on the mat
1 0.8151 the river is wide and deep
```

Note that `sentences` is a list, not a generator. `Word2Vec.train` traverses the stream once
per epoch plus once more to build the vocabulary, so a one-shot generator is not sufficient;
the pipeline layer passes a re-iterable `SentenceStream`.

The contextual path, which requires the `neural` extra:

```python
from multilingual_embedding.embedding.neural import (
    ContrastiveConfig,
    ContrastiveTrainer,
    EncoderConfig,
    NeuralTextEncoder,
    TextPair,
    TransformerEncoderModel,
)
from multilingual_embedding.tokenizer.tokenizer import WordTokenizer

pairs = [
    TextPair("how do I reset the device", "hold the power button for ten seconds"),
    TextPair("what is the refund window", "returns are accepted within thirty days"),
    TextPair("where do I change my password", "open account settings and choose security"),
    TextPair("how is shipping charged", "delivery is free above the order threshold"),
]

tokenizer = WordTokenizer().train([p.anchor for p in pairs] + [p.positive for p in pairs])

model = TransformerEncoderModel(
    EncoderConfig(vocabulary_size=tokenizer.vocabulary_size, dimension=32, layers=2, heads=2)
)

encoder = NeuralTextEncoder(model, tokenizer, device="cpu")

trainer = ContrastiveTrainer(
    encoder,
    ContrastiveConfig(epochs=30, batch_size=4, learning_rate=1e-3),
)

report = trainer.train(pairs)

print("steps:", report.steps)
print("initial_loss:", round(report.initial_loss, 4))
print("final_loss:", round(report.final_loss, 4))
print("improved:", report.improved)
print(encoder)
```

Actual output:

```
steps: 30
initial_loss: 2.2373
final_loss: 0.0
improved: True
NeuralTextEncoder(dimension=32, parameters=34784, device=cpu)
```

Read that final loss as a mechanical check, not a quality claim. `ln(4) ≈ 1.386` is the loss
of random guessing among four candidates, the opening loss is above it because the untrained
model's outputs are near-random, and driving four memorisable pairs to zero demonstrates only
that gradients flow and the objective is wired the right way round. Separating held-out pairs
at a realistic batch size is a different question, and `ROADMAP.md` records the exit criterion
used for it.

`report.improved` compares the final epoch's mean loss against the first, which is the minimum
evidence that training did anything. The resulting `encoder` satisfies `TextEncoder`, so it
goes straight into `SemanticSearchPipeline` with no adapter, and `encoder.save(directory)`
writes weights and architecture together — a state dictionary alone cannot be loaded without
knowing the shape that produced it.

The adapted path, which requires the `neural` extra and downloads a checkpoint on first use:

```python
from multilingual_embedding.embedding.neural.adapter import load_adapter, save_adapter
from multilingual_embedding.embedding.neural.lora import LoRAConfig, apply_lora
from multilingual_embedding.embedding.neural.pretrained import PretrainedTextEncoder

encoder = PretrainedTextEncoder.load("intfloat/multilingual-e5-small", pooling="mean")

lora = LoRAConfig(rank=16, targets=("query", "value"))

apply_lora(encoder.model, lora)

# ... train with ContrastiveTrainer over mined pairs ...

save_adapter(
    encoder,
    "models/indic-v1",
    lora=lora,
    query_prefix="query: ",
    passage_prefix="passage: ",
)

encoder, metadata = load_adapter("models/indic-v1", local_files_only=True)
```

In practice this is driven by `scripts/adapt_pretrained.py`, which also holds the baseline
comparison and the experiment-design checks — see [`scripts/README.md`](../../../scripts/README.md).
Serving the result is one call:

```python
from multilingual_embedding.pipelines.search import SemanticSearchPipeline

pipeline = SemanticSearchPipeline.from_adapter("models/indic-v1")
```

which carries the recorded prefixes into indexing and querying automatically. Writing
`SemanticSearchPipeline(load_adapter("models/indic-v1")[0])` instead loads the right weights
and then uses them wrongly, silently — which is exactly why the factory exists.

## Dependencies

May import from `common`, `core`, `utils`, `config`, `corpus`, `vocabulary` and `tokenizer`.
In practice it uses `config.base` (`EmbeddingConfig`, and `ComputeConfig` in the neural
trainer), `core.exceptions`, `core.logging`, `core.registry`, `utils.filesystem`, `utils.io`,
`utils.serialization`, `utils.validation` and `vocabulary`. It does not import `tokenizer`:
tokenisation enters as a plain `Callable[[str], list[str]]` for the static path, and as the
narrow `Tokenizes` Protocol — anything with `.encode(text).ids` — for the neural one. Either
way a whitespace split, a SentencePiece model or a test double can be passed without an
adapter, which is what lets the encoder be tested without training a subword model first.

`evaluation` and `pipelines` import this package. Neither may be imported from here.
`tqdm` is imported lazily inside `_progress` and its absence degrades to no progress bar
rather than failing.

**torch is an optional dependency, and the boundary is deliberate.** `neural/` is the only
part of the framework that imports it, and `multilingual_embedding.embedding` does **not**
import `neural` — nothing is pulled in transitively. The corpus, tokenizer, vocabulary and
evaluation layers therefore stay installable without a training stack, which matters for
callers that only need text preparation. Install it with `uv sync --extra neural`, and import
explicitly:

```python
from multilingual_embedding.embedding.neural import NeuralTextEncoder
```

`neural/__init__.py` catches the missing import and re-raises it with that instruction rather
than letting a bare `ModuleNotFoundError: torch` surface. The extra is platform-pinned:
torch stopped shipping x86_64 macOS wheels after 2.2, so an Intel Mac caps at `torch<2.3`
and, because that build predates NumPy 2 and its interop fails outright, at `numpy<2` on that
platform alone.

## Tests

| File | Tests | Covers |
|---|---|---|
| `tests/embedding/test_neural.py` | 38 | Architecture, encoder, contrastive training, padding, precision |
| `tests/embedding/test_lora_gradcache.py` | 24 | Zero-init identity, adapter-only state dicts, gradient equivalence, chunk invariance |
| `tests/embedding/test_matrix.py` | 20 | Similarity, analogy, normalisation, versioned persistence |
| `tests/embedding/test_sentence.py` | 20 | Mean pooling, SIF, the registry |
| `tests/embedding/test_index.py` | 19 | Exact search, labels, persistence |
| `tests/embedding/test_word2vec.py` | 17 | Sampling tables, subsampling, the update rules |
| `tests/embedding/test_encoder_contract.py` | 15 | `TextEncoder`: shape, order, zero-not-NaN |
| `tests/embedding/test_pretrained.py` | 15 | Pooling strategies, the wrapper's forward shape, error paths |
| `tests/embedding/test_adapter.py` | 9 | Save/load round trip, byte-identical vectors, metadata survival |

`pytest tests/embedding -q` reports **177 passed** on a full install. `test_neural.py`,
`test_lora_gradcache.py`, `test_pretrained.py` and `test_adapter.py` call
`pytest.importorskip("torch")`, so a core-only checkout collects fewer and still runs green.
`test_pretrained.py` and `test_adapter.py` build a small BERT locally rather than
downloading one — nothing here reaches the network. The layering rule described above is
enforced separately by `tests/test_architecture.py` (17 tests).

## What is not here

There is no character n-gram model, no approximate nearest-neighbour index, no hard-negative
mining, no Matryoshka truncation and no checkpoint resumption. There is also no loader for
external weights into `TransformerEncoderModel` — the pre-norm arrangement above makes that a
real piece of work rather than a `load_state_dict` call, and `pretrained.py` sidesteps it by
using the upstream library instead, which is why external checkpoints *are* supported while
that loader is not. `ROADMAP.md` covers the rest.
