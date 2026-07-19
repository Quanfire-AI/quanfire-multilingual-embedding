# embedding

> Learns word vectors from tokenised text, composes them into sentence vectors, and searches over the result.

## Purpose

This layer turns a stream of tokens into geometry. `Word2Vec` fits skip-gram word
vectors with negative sampling in pure numpy, `EmbeddingMatrix` binds those vectors
to the vocabulary that indexes them, the sentence encoders compose word vectors into
sentence vectors, and `SimilarityIndex` answers nearest-neighbour queries over a set
of those. It is its own layer because everything below it deals in text and ids while
everything here deals in float arrays, and because the `EmbeddingModel` contract lets
the pipeline layer swap the model without knowing which one it holds.

There is no PyTorch, no transformer encoder, no fastText and no contrastive learning
in this package. `base.py` names fastText and transformers as models the ABC is shaped
to accommodate; neither is implemented. Training is numpy only.

## Modules

| Module | Responsibility |
|---|---|
| `base.py` | `EmbeddingModel` ABC: `train`, `matrix`, `save`, `load`. Deliberately thin — anything one model needs and the others do not lives on that model. |
| `word2vec.py` | `Word2Vec`, the skip-gram negative-sampling trainer, its noise and subsampling tables, and the linear learning-rate decay. |
| `matrix.py` | `EmbeddingMatrix`: vectors paired with their vocabulary, plus `similarity`, `most_similar`, `analogy`, `normalized`, and versioned save/load. |
| `sentence.py` | `SentenceEncoder` ABC, `MeanPoolingEncoder`, `SifEncoder`, and the `SENTENCE_ENCODERS` registry (`"mean"`, `"sif"`). |
| `index.py` | `SimilarityIndex` and `SearchResult`: exact brute-force cosine search over labelled vectors, with persistence. |

## Key design decisions

**Negative sampling from the unigram distribution raised to 0.75.** Raw unigram
frequency would make almost every negative a function word, which teaches the model
very little; a uniform distribution would make negatives too easy. The `_NOISE_EXPONENT
= 0.75` dampening in `word2vec.py` lifts rare tokens without flattening the
distribution, and is the value from the original paper.

**The noise table is a cumulative distribution sampled with `np.searchsorted`, not
`np.random.choice(p=...)`.** `_prepare_tables` computes `np.cumsum(weights / total)`
once; `_sample_negatives` then draws in O(log V) from a plain uniform. `np.random.choice`
with a `p=` argument rebuilds its internal structure on every call and is orders of
magnitude too slow at the rate this loop needs — one draw per negative per context pair
per epoch. Special-token weights are zeroed before the cumsum so pad and unk are never
drawn as impostors.

**Subsampling and a dynamic window.** `_subsample_probabilities` applies the classic
word2vec keep formula against `config.subsample_threshold` (default `1e-3`), so very
frequent tokens are randomly discarded before pairs are formed. Independently,
`_train_sentence` redraws the window per centre token with `self._rng.integers(1, window
+ 1)`. A context word at distance 3 is therefore only sampled when the draw is at least
3, which weights nearer context more heavily without any explicit distance term in the
gradient. The tradeoff is that the effective window is smaller than the configured one;
that is intended.

**`W_in` uniform in ±0.5/dimension, `W_out` zeroed, and `W_out` discarded after
training.** The small input scale keeps early dot products near zero where the sigmoid
gradient is largest; a zeroed output matrix means the first updates are driven by labels
rather than by initialisation noise. `train` sets `self._output_weights = None` before
building the matrix, because `W_out` modelled "is this a real context word", not word
meaning. The consequence is that a loaded model can be used for lookup but not resumed
for further training, and `load` says so.

**The centre-word gradient is accumulated once.** In `_update_pair`, `centre_gradient =
gradient @ output_weights[targets]` sums the contribution of the positive and all
negatives, and `input_weights[centre] += centre_gradient` is applied a single time.
Updating `W_in` inside a loop over samples would let later samples see a moved centre
vector, which silently changes the objective rather than raising anything.

**`np.add.at` for the `W_out` update — a correctness fix, not a micro-optimisation.**
Negatives are drawn independently, so a sampled negative can collide with the context
token or with another negative, and `targets` then contains a repeated index. Fancy-index
`+=` evaluates the right-hand side once and keeps only the last write for a duplicated
index, dropping the other updates without any error. `np.add.at(output_weights, targets,
...)` accumulates every one. It is slower than buffered fancy indexing; that is the price
of the update being right.

**Out-of-vocabulary tokens are dropped during training rather than folded into `<unk>`.**
`_select_ids` skips any token whose id falls below `vocabulary.special_tokens.count`.
Routing every rare word through one shared row would train a single vector that is the
centroid of unrelated meanings, and that vector then appears as a plausible neighbour to
everything it absorbed. The same reasoning drives `SentenceEncoder._known_ids`, which
drops unknown tokens rather than mapping them to the unknown row.

**`EmbeddingMatrix` validates that vector rows match vocabulary size.** The constructor
raises `ValidationError` when `vectors.shape[0] != len(vocabulary)`, and `load` re-checks
both the row count against the stored vocabulary and the width against the recorded
dimension. Rebuilding a vocabulary with a different `min_count` and reusing old vectors
is the classic silent corruption: every row now means a different token, with no exception
and no NaN to warn you.

**Normalisation guards zero rows.** `_normalize_rows` divides by `np.maximum(norms,
1e-12)`. The pad row is all zeros by construction, and dividing it by its own norm would
produce NaN across that row, which then propagates into every score in a `most_similar`
call.

**`SimilarityIndex` is exact brute force, and there is a ceiling.** Rows are stored
already normalised, so a query is one matmul and cosine reduces to a dot product; the
result is the true nearest-neighbour set with no build step, no tuning and no recall to
measure. That costs O(n·d) per query, which is the right choice up to roughly 10^5–10^6
items. Past that an approximate index (HNSW, IVF-PQ) is needed, and this framework does
not provide one — wrapping a poor ANN implementation would be worse than being honest
about the ceiling.

**`SifEncoder` needs a batch to remove the first principal component.** The direction
being projected out is estimated from a set of sentences by SVD in `_fit_component`, and
a single sentence in isolation is its own principal component — removing it would zero
the vector outright. `encode` on an unfitted encoder therefore returns the SIF-weighted
average and skips the removal, and `is_fitted` reports which behaviour you are getting.
The intended order is `encode_batch` over the corpus first, then `encode` for queries.

## Usage

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

Note that `sentences` is a list, not a generator. `Word2Vec.train` traverses the stream
once per epoch plus once more to build the vocabulary, so a one-shot generator is not
sufficient; the pipeline layer passes a re-iterable `SentenceStream`.

## Dependencies

May import from `common`, `core`, `utils`, `config`, `corpus`, `vocabulary` and
`tokenizer`. In practice it uses `config.base.EmbeddingConfig`, `core.exceptions`,
`core.logging`, `core.registry`, `utils.filesystem`, `utils.io`, `utils.serialization`
and `vocabulary`. It does not import `tokenizer`: tokenisation enters as a plain
`Callable[[str], list[str]]` parameter, which is what lets a whitespace split, a
SentencePiece model or a test double all be passed without an adapter.

`evaluation` and `pipelines` import this package. Neither may be imported from here.
`tqdm` is imported lazily inside `_progress` and its absence degrades to no progress bar
rather than failing.

## Tests

| File | Tests |
|---|---|
| `tests/embedding/test_word2vec.py` | 17 |
| `tests/embedding/test_matrix.py` | 20 |
| `tests/embedding/test_sentence.py` | 20 |
| `tests/embedding/test_index.py` | 19 |

`.venv/bin/python -m pytest tests/embedding -q` reports **76 passed**. The layering rule
described above is enforced separately by `tests/test_architecture.py` (14 tests).
