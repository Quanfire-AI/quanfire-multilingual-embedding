from multilingual_embedding.common.span import Span


def test_length():
    span = Span(2, 8)

    assert span.length == 6


def test_slice():
    text = "Hello world"

    span = Span(0, 5)

    assert span.slice(text) == "Hello"


def test_contains():
    span = Span(5, 10)

    assert span.contains(5)

    assert span.contains(9)

    assert not span.contains(10)


def test_overlap():
    a = Span(0, 5)

    b = Span(3, 7)

    assert a.overlaps(b)


def test_touch():
    a = Span(0, 5)

    b = Span(5, 10)

    assert a.touches(b)


def test_merge():
    a = Span(0, 5)

    b = Span(5, 10)

    c = a.merge(b)

    assert c.start == 0

    assert c.end == 10


def test_shift():
    span = Span(2, 5)

    shifted = span.shift(10)

    assert shifted.start == 12

    assert shifted.end == 15
