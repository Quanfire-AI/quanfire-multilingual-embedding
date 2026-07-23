import tomllib
from pathlib import Path

from multilingual_embedding.common.version import __version__

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_version_is_string() -> None:
    assert isinstance(__version__, str)


def test_version_not_empty() -> None:
    assert __version__ != ""


def test_the_packaged_version_is_read_from_this_module() -> None:
    """There must be exactly one version literal in the repository.

    ``pyproject.toml`` declares ``dynamic = ["version"]`` and points
    hatchling at this module. Reverting that to a hardcoded ``version =``
    would break nothing visible — the package would still build, install
    and import — while allowing a wheel published as one number to
    generate evaluation reports stamped with another, so the provenance
    recorded beside a published metric would be wrong with nothing
    raising. It matters more now that sibling repositories pin by version.

    The build wiring is asserted rather than the installed metadata.
    Comparing against ``importlib.metadata`` would be closer to the thing
    that matters, but an editable install writes its version at install
    time, so it would fail after every bump until the next ``uv sync`` —
    and a test that routinely fails for an uninteresting reason gets
    ignored when it fails for an interesting one.
    """

    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))

    assert "version" in pyproject["project"]["dynamic"]

    assert "version" not in pyproject["project"], (
        "pyproject.toml carries a static version again — it is now a second "
        "source of truth that nothing keeps in step with this module"
    )

    assert pyproject["tool"]["hatch"]["version"]["path"] == (
        "src/multilingual_embedding/common/version.py"
    )
