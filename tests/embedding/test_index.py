import numpy as np
import pytest

from multilingual_embedding.core.exceptions import ValidationError
from multilingual_embedding.embedding.index import SearchResult, SimilarityIndex
from multilingual_embedding.embedding.matrix import EmbeddingMatrix
from multilingual_embedding.embedding.sentence import MeanPoolingEncoder
from multilingual_embedding.vocabulary.vocabulary import Vocabulary

DIMENSION = 8

LABELS = ["alpha", "beta", "gamma", "delta", "epsilon"]


def make_items(seed=0):
    rng = np.random.default_rng(seed)

    vectors = rng.normal(size=(len(LABELS), DIMENSION)).astype(np.float32)

    return list(zip(LABELS, vectors, strict=True))


def make_index(seed=0):
    return SimilarityIndex.build(make_items(seed))


def make_encoder():
    vocabulary = Vocabulary.from_counter({"river": 10, "lake": 8, "server": 6, "laptop": 4})

    rng = np.random.default_rng(3)

    vectors = rng.normal(size=(len(vocabulary), DIMENSION)).astype(np.float32)

    return MeanPoolingEncoder(EmbeddingMatrix(vocabulary, vectors))


def test_build_infers_dimension_and_size():
    index = make_index()

    assert index.dimension == DIMENSION

    assert len(index) == len(LABELS)

    assert index.labels == LABELS


def test_build_rejects_empty_without_dimension():
    with pytest.raises(ValidationError):
        SimilarityIndex.build([])


def test_build_accepts_empty_with_dimension():
    index = SimilarityIndex.build([], dimension=DIMENSION)

    assert len(index) == 0

    assert index.search(np.ones(DIMENSION, dtype=np.float32)) == []


def test_exact_query_returns_its_own_item():
    items = make_items()

    index = SimilarityIndex.build(items)

    for position, (label, vector) in enumerate(items):
        top = index.search(vector, top_k=1)[0]

        assert top.label == label

        assert top.index == position

        assert top.score == pytest.approx(1.0, rel=1e-5)


def test_top_k_is_respected():
    index = make_index()

    for k in (1, 3, len(LABELS)):
        assert len(index.search(np.ones(DIMENSION, dtype=np.float32), top_k=k)) == k


def test_top_k_larger_than_index_is_clamped():
    index = make_index()

    results = index.search(np.ones(DIMENSION, dtype=np.float32), top_k=100)

    assert len(results) == len(LABELS)


def test_results_are_descending_by_score():
    index = make_index()

    scores = [hit.score for hit in index.search(np.ones(DIMENSION, dtype=np.float32), top_k=5)]

    assert scores == sorted(scores, reverse=True)


def test_search_result_is_frozen():
    result = SearchResult(label="alpha", score=1.0, index=0)

    with pytest.raises(AttributeError):
        result.score = 0.5  # type: ignore[misc]


def test_non_positive_top_k_raises():
    index = make_index()

    with pytest.raises(ValidationError):
        index.search(np.ones(DIMENSION, dtype=np.float32), top_k=0)


def test_wrong_dimension_query_raises():
    index = make_index()

    with pytest.raises(ValidationError):
        index.search(np.ones(DIMENSION + 2, dtype=np.float32))


def test_adding_wrong_dimension_raises():
    index = make_index()

    with pytest.raises(ValidationError):
        index.add([("bad", np.ones(DIMENSION + 1, dtype=np.float32))])


def test_add_extends_the_index():
    index = make_index()

    extra = np.ones(DIMENSION, dtype=np.float32)

    assert index.add([("zeta", extra)]) == len(LABELS) + 1

    assert index.search(extra, top_k=1)[0].label == "zeta"


def test_text_query_requires_an_encoder():
    index = make_index()

    with pytest.raises(ValidationError):
        index.search("river")


def test_text_query_with_encoder():
    encoder = make_encoder()

    texts = ["river lake", "server laptop", "river"]

    index = SimilarityIndex.from_texts(texts, encoder)

    assert len(index) == len(texts)

    assert index.search("server laptop", top_k=1)[0].label == "server laptop"


def test_zero_query_vector_does_not_produce_nan():
    index = make_index()

    results = index.search(np.zeros(DIMENSION, dtype=np.float32), top_k=3)

    assert all(not np.isnan(hit.score) for hit in results)


def test_stored_vectors_are_normalized():
    index = make_index()

    norms = np.linalg.norm(index.vectors, axis=1)

    np.testing.assert_allclose(norms, np.ones_like(norms), rtol=1e-5)


def test_save_load_round_trip(tmp_path):
    index = make_index()

    index.save(tmp_path / "index")

    restored = SimilarityIndex.load(tmp_path / "index")

    np.testing.assert_array_equal(restored.vectors, index.vectors)

    assert restored.labels == index.labels

    assert restored.dimension == index.dimension

    query = make_items()[2][1]

    assert restored.search(query, top_k=1)[0].label == "gamma"


def test_load_rejects_bad_format_version(tmp_path):
    from multilingual_embedding.utils.io import read_json, write_json

    target = tmp_path / "index"

    make_index().save(target)

    metadata = read_json(target / "index.json")

    metadata["format_version"] = 99

    write_json(target / "index.json", metadata)

    with pytest.raises(ValidationError):
        SimilarityIndex.load(target)


def test_repr():
    assert "SimilarityIndex" in repr(make_index())
