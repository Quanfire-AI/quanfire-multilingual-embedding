from multilingual_embedding.common.version import __version__


def test_version_is_string() -> None:
    assert isinstance(__version__, str)


def test_version_not_empty() -> None:
    assert __version__ != ""
