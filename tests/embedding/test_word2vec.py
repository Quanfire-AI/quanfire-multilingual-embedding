import random

import numpy as np
import pytest

from multilingual_embedding.config.base import EmbeddingConfig
from multilingual_embedding.core.exceptions import NotFittedError
from multilingual_embedding.corpus.iterator import SentenceStream
from multilingual_embedding.embedding.matrix import EmbeddingMatrix
from multilingual_embedding.embedding.word2vec import Word2Vec

WATER = ["river", "lake", "ocean", "stream", "water", "wave"]

MACHINE = ["laptop", "server", "keyboard", "monitor", "router", "cpu"]


def two_topic_sentences(count=120, length=8, seed=0):
    """Sentences drawn from one of two disjoint word sets, never mixed."""

    rng = random.Random(seed)

    sentences = []

    for _ in range(count):
        topic = WATER if rng.random() < 0.5 else MACHINE

        sentences.append(" ".join(rng.choice(topic) for _ in range(length)))

    return sentences


def make_stream(sentences):
    return SentenceStream(lambda: iter(sentences))


def fast_config(**overrides):
    settings = {
        "dimension": 24,
        "window": 3,
        "min_count": 1,
        "negative_samples": 5,
        "epochs": 5,
        "learning_rate": 0.05,
        "subsample_threshold": 0.0,
        "seed": 13,
    }

    settings.update(overrides)

    return EmbeddingConfig(**settings)


def mean_similarities(matrix):
    within = []

    cross = []

    for group in (WATER, MACHINE):
        for index, first in enumerate(group):
            for second in group[index + 1 :]:
                within.append(matrix.similarity(first, second))

    for first in WATER:
        for second in MACHINE:
            cross.append(matrix.similarity(first, second))

    return float(np.mean(within)), float(np.mean(cross))


def test_matrix_before_training_raises():
    model = Word2Vec(fast_config())

    with pytest.raises(NotFittedError):
        _ = model.matrix


def test_most_similar_before_training_raises():
    model = Word2Vec(fast_config())

    with pytest.raises(NotFittedError):
        model.most_similar("river")


def test_vocabulary_before_training_raises():
    model = Word2Vec(fast_config())

    with pytest.raises(NotFittedError):
        _ = model.vocabulary


def test_train_returns_matched_matrix():
    model = Word2Vec(fast_config())

    matrix = model.train(make_stream(two_topic_sentences()))

    assert isinstance(matrix, EmbeddingMatrix)

    assert matrix.dimension == 24

    assert len(matrix) == len(matrix.vocabulary)

    assert model.is_trained

    assert not np.isnan(matrix.vectors).any()


def test_word2vec_actually_learns_topic_structure():
    model = Word2Vec(fast_config())

    matrix = model.train(make_stream(two_topic_sentences()))

    within, cross = mean_similarities(matrix)

    assert within > cross

    # A real margin, not a rounding artefact: the two topics never
    # co-occur, so their vectors should be clearly separated.
    assert within - cross > 0.3


def test_nearest_neighbours_stay_within_topic():
    model = Word2Vec(fast_config())

    matrix = model.train(make_stream(two_topic_sentences()))

    neighbours = [token for token, _ in matrix.most_similar("river", top_k=3)]

    assert all(token in WATER for token in neighbours)


def test_same_seed_gives_identical_vectors():
    sentences = two_topic_sentences()

    first = Word2Vec(fast_config(seed=7)).train(make_stream(sentences))

    second = Word2Vec(fast_config(seed=7)).train(make_stream(sentences))

    np.testing.assert_array_equal(first.vectors, second.vectors)


def test_different_seed_gives_different_vectors():
    sentences = two_topic_sentences()

    first = Word2Vec(fast_config(seed=7)).train(make_stream(sentences))

    second = Word2Vec(fast_config(seed=8)).train(make_stream(sentences))

    assert not np.array_equal(first.vectors, second.vectors)


def test_training_without_negative_samples_still_learns():
    model = Word2Vec(fast_config(negative_samples=0, epochs=3))

    matrix = model.train(make_stream(two_topic_sentences()))

    assert not np.isnan(matrix.vectors).any()

    within, cross = mean_similarities(matrix)

    # Without negatives every vector is pushed together, so the ranking
    # is all that survives; the point is that the code path runs.
    assert within > cross


def test_subsampling_path_runs():
    model = Word2Vec(fast_config(subsample_threshold=0.01))

    matrix = model.train(make_stream(two_topic_sentences()))

    assert not np.isnan(matrix.vectors).any()

    assert len(matrix) == len(matrix.vocabulary)


def test_min_count_prunes_rare_tokens():
    sentences = [*two_topic_sentences(count=40), "hapax hapax_two"]

    model = Word2Vec(fast_config(min_count=3, epochs=2))

    matrix = model.train(make_stream(sentences))

    assert "hapax" not in matrix.vocabulary

    assert "river" in matrix.vocabulary


def test_prebuilt_vocabulary_is_used():
    from multilingual_embedding.vocabulary.vocabulary import Vocabulary

    vocabulary = Vocabulary.from_counter(dict.fromkeys(WATER, 20))

    model = Word2Vec(fast_config(epochs=2), vocabulary=vocabulary)

    matrix = model.train(make_stream(two_topic_sentences()))

    assert len(matrix.vocabulary) == len(vocabulary)

    assert "laptop" not in matrix.vocabulary


def test_custom_tokenizer_is_honoured():
    sentences = ["river|lake|ocean"] * 30

    model = Word2Vec(fast_config(epochs=2))

    matrix = model.train(make_stream(sentences), tokenize=lambda text: text.split("|"))

    assert "river" in matrix.vocabulary

    assert "river|lake|ocean" not in matrix.vocabulary


def test_frozen_vocabulary_after_training():
    model = Word2Vec(fast_config(epochs=2))

    matrix = model.train(make_stream(two_topic_sentences(count=40)))

    assert matrix.vocabulary.is_frozen


def test_save_load_round_trip(tmp_path):
    model = Word2Vec(fast_config(epochs=2))

    model.train(make_stream(two_topic_sentences(count=40)))

    model.save(tmp_path / "model")

    restored = Word2Vec.load(tmp_path / "model")

    np.testing.assert_array_equal(restored.matrix.vectors, model.matrix.vectors)

    assert restored.matrix.vocabulary == model.matrix.vocabulary

    assert restored.config.dimension == model.config.dimension

    assert restored.config.seed == model.config.seed


def test_save_before_training_raises(tmp_path):
    model = Word2Vec(fast_config())

    with pytest.raises(NotFittedError):
        model.save(tmp_path / "model")


def test_repr_reports_training_state():
    model = Word2Vec(fast_config())

    assert "trained=False" in repr(model)
