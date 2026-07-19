from multilingual_embedding.corpus.metadata import BaseMetadata


def test_default_metadata() -> None:
    metadata = BaseMetadata()

    assert metadata.id is None
    assert metadata.language is None
    assert metadata.attributes == {}
    assert metadata.created_at is not None
    assert metadata.updated_at is not None
