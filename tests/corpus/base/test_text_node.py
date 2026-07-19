from multilingual_embedding.common.span import Span
from multilingual_embedding.corpus.base import TextNode
from multilingual_embedding.corpus.metadata import BaseMetadata


class DummyNode(TextNode):
    pass


def test_length():
    node = DummyNode(
        text="Hello",
        span=Span(0, 5),
        metadata=BaseMetadata(),
    )

    assert len(node) == 5


def test_character_count():
    node = DummyNode(
        text="Hello",
        span=Span(0, 5),
        metadata=BaseMetadata(),
    )

    assert node.character_count == 5


def test_empty():
    node = DummyNode(
        text="",
        span=Span(0, 0),
        metadata=BaseMetadata(),
    )

    assert node.is_empty


def test_strip():
    node = DummyNode(
        text=" hello ",
        span=Span(0, 7),
        metadata=BaseMetadata(),
    )

    assert node.strip() == "hello"


def test_contains():
    node = DummyNode(
        text="Hello world",
        span=Span(0, 11),
        metadata=BaseMetadata(),
    )

    assert node.contains("world")
