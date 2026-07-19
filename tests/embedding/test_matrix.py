import numpy as np
import pytest

from multilingual_embedding.core.exceptions import ValidationError
from multilingual_embedding.embedding.matrix import EmbeddingMatrix
from multilingual_embedding.vocabulary.vocabulary import Vocabulary

DIMENSION = 8


def make_vocabulary():
    return Vocabulary.from_counter({"king": 9, "queen": 8, "man": 7, "woman": 6, "apple": 5})


def make_matrix(seed=0):
    vocabulary = make_vocabulary()

    rng = np.random.default_rng(seed)

    vectors = rng.normal(size=(len(vocabulary), DIMENSION)).astype(np.float32)

    # Pad rows are zero in a trained model; keep that true here so the
    # zero-row paths are exercised by every test using this fixture.
    vectors[0] = 0.0

    return EmbeddingMatrix(vocabulary, vectors)


def test_size_mismatch_raises():
    vocabulary = make_vocabulary()

    vectors = np.zeros((len(vocabulary) + 3, DIMENSION), dtype=np.float32)

    with pytest.raises(ValidationError) as error:
        EmbeddingMatrix(vocabulary, vectors)

    assert error.value.context["rows"] == len(vocabulary) + 3

    assert error.value.context["vocabulary_size"] == len(vocabulary)


def test_rejects_one_dimensional_array():
    vocabulary = make_vocabulary()

    with pytest.raises(ValidationError):
        EmbeddingMatrix(vocabulary, np.zeros(len(vocabulary), dtype=np.float32))


def test_dimension_and_length():
    matrix = make_matrix()

    assert matrix.dimension == DIMENSION

    assert len(matrix) == len(make_vocabulary())


def test_dtype_is_float32():
    matrix = make_matrix()

    assert matrix.vectors.dtype == np.float32


def test_vector_for_unknown_token_returns_unk_row():
    matrix = make_matrix()

    np.testing.assert_array_equal(matrix.vector_for("nonexistent"), matrix.vector_for_id(1))


def test_vector_for_id_out_of_range_raises():
    matrix = make_matrix()

    with pytest.raises(ValidationError):
        matrix.vector_for_id(len(matrix))


def test_normalized_handles_zero_row_without_nan():
    matrix = make_matrix()

    normalized = matrix.normalized()

    assert not np.isnan(normalized.vectors).any()

    # The zero pad row must stay zero rather than becoming NaN or unit.
    np.testing.assert_array_equal(normalized.vectors[0], np.zeros(DIMENSION, dtype=np.float32))

    norms = np.linalg.norm(normalized.vectors[1:], axis=1)

    np.testing.assert_allclose(norms, np.ones_like(norms), rtol=1e-5)


def test_similarity_of_token_with_itself_is_one():
    matrix = make_matrix()

    assert matrix.similarity("king", "king") == pytest.approx(1.0, rel=1e-5)


def test_similarity_with_zero_vector_is_zero():
    vocabulary = make_vocabulary()

    vectors = np.zeros((len(vocabulary), DIMENSION), dtype=np.float32)

    vectors[4] = 1.0

    matrix = EmbeddingMatrix(vocabulary, vectors)

    assert matrix.similarity("king", vocabulary.token_of(0)) == 0.0


def test_most_similar_excludes_query_and_special_tokens():
    matrix = make_matrix()

    results = matrix.most_similar("king", top_k=10)

    tokens = [token for token, _ in results]

    assert "king" not in tokens

    for special in matrix.vocabulary.special_tokens.as_tuple():
        assert special not in tokens


def test_most_similar_can_include_special_tokens():
    matrix = make_matrix()

    unrestricted = matrix.most_similar("king", top_k=10, exclude_special=False)

    assert len(unrestricted) > len(matrix.most_similar("king", top_k=10))


def test_most_similar_respects_top_k_and_ordering():
    matrix = make_matrix()

    results = matrix.most_similar("king", top_k=2)

    assert len(results) == 2

    assert results[0][1] >= results[1][1]


def test_most_similar_accepts_a_raw_vector():
    matrix = make_matrix()

    results = matrix.most_similar(matrix.vector_for("queen"), top_k=1)

    # A raw vector query has no token to exclude, so the query's own
    # token is the expected top hit.
    assert results[0][0] == "queen"


def test_most_similar_rejects_bad_dimension():
    matrix = make_matrix()

    with pytest.raises(ValidationError):
        matrix.most_similar(np.ones(DIMENSION + 1, dtype=np.float32))


def test_most_similar_rejects_non_positive_top_k():
    matrix = make_matrix()

    with pytest.raises(ValidationError):
        matrix.most_similar("king", top_k=0)


def test_analogy_excludes_its_inputs():
    matrix = make_matrix()

    results = matrix.analogy(["king", "woman"], ["man"], top_k=5)

    tokens = [token for token, _ in results]

    for term in ("king", "woman", "man"):
        assert term not in tokens


def test_analogy_requires_input():
    matrix = make_matrix()

    with pytest.raises(ValidationError):
        matrix.analogy([], [])


def test_save_load_round_trip(tmp_path):
    matrix = make_matrix()

    matrix.save(tmp_path / "embedding")

    restored = EmbeddingMatrix.load(tmp_path / "embedding")

    np.testing.assert_array_equal(restored.vectors, matrix.vectors)

    assert restored.vocabulary == matrix.vocabulary

    assert restored.dimension == matrix.dimension


def test_load_rejects_bad_format_version(tmp_path):
    from multilingual_embedding.utils.io import read_json, write_json

    target = tmp_path / "embedding"

    make_matrix().save(target)

    metadata = read_json(target / "metadata.json")

    metadata["format_version"] = 99

    write_json(target / "metadata.json", metadata)

    with pytest.raises(ValidationError):
        EmbeddingMatrix.load(target)


def test_contains_and_repr():
    matrix = make_matrix()

    assert "king" in matrix

    assert "nonexistent" not in matrix

    assert "EmbeddingMatrix" in repr(matrix)
