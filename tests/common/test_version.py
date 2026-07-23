from importlib.metadata import version as installed_version

from multilingual_embedding.common.version import __version__


def test_version_is_string() -> None:
    assert isinstance(__version__, str)


def test_version_not_empty() -> None:
    assert __version__ != ""


def test_the_installed_distribution_reports_the_same_version() -> None:
    """The packaged version and the importable one must not drift apart.

    ``pyproject.toml`` declares ``dynamic = ["version"]`` and points
    hatchling at :mod:`multilingual_embedding.common.version`, so there is
    one literal. This asserts that wiring still holds, because reverting it
    to a hardcoded ``version =`` in ``pyproject.toml`` would break nothing
    visible: the package would still build, still install and still import.
    It would simply publish one number while every evaluation report
    stamped another, and the provenance recorded beside a published metric
    would be wrong with nothing raising.

    It matters more now that sibling repositories pin this one by version.
    """

    assert installed_version("quanfire-multilingual-embedding") == __version__
