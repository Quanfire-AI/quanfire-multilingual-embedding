import numpy as np
import pytest

from multilingual_embedding.core.exceptions import ValidationError
from multilingual_embedding.embedding.matrix import EmbeddingMatrix
from multilingual_embedding.embedding.sentence import (
    SENTENCE_ENCODERS,
    MeanPoolingEncoder,
    SifEncoder,
)
from multilingual_embedding.vocabulary.vocabulary import Vocabulary

DIMENSION = 16

SENTENCES = [
    "the river is wide",
    "the lake is deep",
    "the server is fast",
    "the laptop is new",
]


def make_matrix(seed=0):
    vocabulary = Vocabulary.from_counter(
        {
            "the": 500,
            "is": 400,
            "river": 30,
            "lake": 25,
            "wide": 20,
            "deep": 18,
            "server": 15,
            "laptop": 12,
            "fast": 10,
            "new": 8,
        }
    )

    rng = np.random.default_rng(seed)

    vectors = rng.normal(size=(len(vocabulary), DIMENSION)).astype(np.float32)

    vectors[0] = 0.0

    return EmbeddingMatrix(vocabulary, vectors)


# ----------------------------------------------------------------------
# Mean pooling
# ----------------------------------------------------------------------


def test_mean_pooling_shape():
    encoder = MeanPoolingEncoder(make_matrix())

    assert encoder.encode(SENTENCES[0]).shape == (DIMENSION,)


def test_mean_pooling_is_the_average_of_known_tokens():
    matrix = make_matrix()

    encoder = MeanPoolingEncoder(matrix)

    expected = np.mean(
        [matrix.vector_for(token) for token in ("the", "river", "is", "wide")],
        axis=0,
    )

    np.testing.assert_allclose(encoder.encode("the river is wide"), expected, rtol=1e-5)


def test_mean_pooling_skips_out_of_vocabulary_tokens():
    matrix = make_matrix()

    encoder = MeanPoolingEncoder(matrix)

    np.testing.assert_allclose(
        encoder.encode("river zzz"),
        encoder.encode("river"),
        rtol=1e-5,
    )


def test_mean_pooling_empty_input_is_zero_not_nan():
    encoder = MeanPoolingEncoder(make_matrix())

    vector = encoder.encode("")

    assert not np.isnan(vector).any()

    np.testing.assert_array_equal(vector, np.zeros(DIMENSION, dtype=np.float32))


def test_mean_pooling_oov_only_input_is_zero_not_nan():
    encoder = MeanPoolingEncoder(make_matrix())

    vector = encoder.encode("zzz qqq")

    assert not np.isnan(vector).any()

    assert float(np.linalg.norm(vector)) == 0.0


def test_mean_pooling_batch_shape():
    encoder = MeanPoolingEncoder(make_matrix())

    batch = encoder.encode_batch(SENTENCES)

    assert batch.shape == (len(SENTENCES), DIMENSION)

    assert not np.isnan(batch).any()


def test_mean_pooling_empty_batch():
    encoder = MeanPoolingEncoder(make_matrix())

    assert encoder.encode_batch([]).shape == (0, DIMENSION)


# ----------------------------------------------------------------------
# SIF
# ----------------------------------------------------------------------


def test_sif_shapes():
    encoder = SifEncoder(make_matrix())

    assert encoder.encode(SENTENCES[0]).shape == (DIMENSION,)

    assert encoder.encode_batch(SENTENCES).shape == (len(SENTENCES), DIMENSION)


def test_sif_rejects_non_positive_alpha():
    with pytest.raises(ValidationError):
        SifEncoder(make_matrix(), alpha=0.0)


def test_sif_downweights_frequent_tokens():
    matrix = make_matrix()

    encoder = SifEncoder(matrix)

    # "the" is 500/1038 of the corpus and "new" is 8/1038, so SIF must
    # weight the rare token far more heavily.
    frequent = encoder.encode("the")

    rare = encoder.encode("new")

    assert np.dot(rare, matrix.vector_for("new")) > np.dot(frequent, matrix.vector_for("the"))


def test_sif_empty_input_is_zero_not_nan():
    encoder = SifEncoder(make_matrix())

    vector = encoder.encode("")

    assert not np.isnan(vector).any()

    np.testing.assert_array_equal(vector, np.zeros(DIMENSION, dtype=np.float32))


def test_sif_oov_only_input_is_zero_not_nan():
    encoder = SifEncoder(make_matrix())

    encoder.encode_batch(SENTENCES)

    vector = encoder.encode("zzz qqq")

    assert not np.isnan(vector).any()

    assert float(np.linalg.norm(vector)) == 0.0


def test_sif_single_sentence_skips_component_removal():
    encoder = SifEncoder(make_matrix())

    assert not encoder.is_fitted

    encoder.encode(SENTENCES[0])

    assert not encoder.is_fitted


def test_sif_batch_fits_the_component():
    encoder = SifEncoder(make_matrix())

    encoder.encode_batch(SENTENCES)

    assert encoder.is_fitted


def test_sif_component_removal_changes_the_output():
    unfitted = SifEncoder(make_matrix())

    before = unfitted.encode(SENTENCES[0])

    fitted = SifEncoder(make_matrix())

    fitted.encode_batch(SENTENCES)

    after = fitted.encode(SENTENCES[0])

    assert not np.allclose(before, after)


def test_sif_removed_component_is_orthogonal_to_outputs():
    encoder = SifEncoder(make_matrix())

    encoded = encoder.encode_batch(SENTENCES)

    component = encoder._component

    assert component is not None

    np.testing.assert_allclose(encoded @ component, np.zeros(len(SENTENCES)), atol=1e-5)


def test_sif_empty_batch():
    encoder = SifEncoder(make_matrix())

    assert encoder.encode_batch([]).shape == (0, DIMENSION)


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------


def test_encoders_are_registered():
    assert "mean" in SENTENCE_ENCODERS

    assert "sif" in SENTENCE_ENCODERS


def test_registry_creates_working_encoders():
    matrix = make_matrix()

    for key, expected in (("mean", MeanPoolingEncoder), ("sif", SifEncoder)):
        encoder = SENTENCE_ENCODERS.create(key, matrix)

        assert isinstance(encoder, expected)

        assert encoder.encode(SENTENCES[0]).shape == (DIMENSION,)


def test_custom_tokenizer_is_honoured():
    matrix = make_matrix()

    encoder = MeanPoolingEncoder(matrix, tokenize=lambda text: text.split("|"))

    np.testing.assert_allclose(
        encoder.encode("river|lake"), encoder.encode("lake|river"), rtol=1e-5
    )

    assert float(np.linalg.norm(encoder.encode("river lake"))) == 0.0
