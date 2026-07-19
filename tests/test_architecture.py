"""
Architectural invariants.

The framework claims a layered design with an acyclic import graph. That
claim is easy to state and easy to break — one convenience import from a
lower layer into a higher one is all it takes, and nothing else in the
suite would notice.

These tests parse the source and enforce the rule directly.
"""

from __future__ import annotations

import ast
import collections
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parent.parent / "src" / "multilingual_embedding"

# Ordered from lowest to highest. A package may import from packages
# strictly to its left, and from nothing else inside the framework.
LAYERS: tuple[str, ...] = (
    "common",
    "core",
    "utils",
    "config",
    "corpus",
    "vocabulary",
    "tokenizer",
    "embedding",
    "evaluation",
    "pipelines",
)

LAYER_RANK: dict[str, int] = {name: index for index, name in enumerate(LAYERS)}


def _internal_imports() -> dict[str, set[str]]:
    """Map each layer to the set of framework layers it imports."""

    edges: dict[str, set[str]] = collections.defaultdict(set)

    for path in SOURCE_ROOT.rglob("*.py"):
        relative = path.relative_to(SOURCE_ROOT)

        # Files directly under the package root (cli.py, __init__.py) are
        # composition roots and are allowed to import anything.
        if len(relative.parts) == 1:
            continue

        package = relative.parts[0]

        if package not in LAYER_RANK:
            continue

        tree = ast.parse(path.read_text(encoding="utf-8"))

        for node in ast.walk(tree):
            for module in _imported_modules(node):
                if not module.startswith("multilingual_embedding."):
                    continue

                target = module.split(".")[1]

                if target in LAYER_RANK and target != package:
                    edges[package].add(target)

    return dict(edges)


def _imported_modules(node: ast.AST) -> list[str]:
    """Return absolute module names imported by one AST node."""

    if isinstance(node, ast.ImportFrom):
        # Relative imports stay inside their own package by construction.
        return [node.module] if node.module and node.level == 0 else []

    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]

    return []


@pytest.fixture(scope="module")
def imports() -> dict[str, set[str]]:
    return _internal_imports()


def test_every_layer_directory_exists() -> None:
    """The declared layer list must match what is on disk."""

    for layer in LAYERS:
        assert (SOURCE_ROOT / layer).is_dir(), layer


@pytest.mark.parametrize("layer", LAYERS)
def test_layer_imports_only_from_below(layer: str, imports: dict[str, set[str]]) -> None:
    """
    No layer may import from itself-or-above.

    An upward import would introduce a cycle; a sideways one would blur
    the boundary between two peers.
    """

    offenders = sorted(
        target for target in imports.get(layer, set()) if LAYER_RANK[target] >= LAYER_RANK[layer]
    )

    assert not offenders, f"{layer} imports from {offenders}, which is not below it"


def test_foundation_layers_have_no_internal_dependencies(
    imports: dict[str, set[str]],
) -> None:
    """``common`` and ``core`` must stand alone."""

    assert imports.get("common", set()) == set()

    assert imports.get("core", set()) == set()


def test_import_graph_is_acyclic(imports: dict[str, set[str]]) -> None:
    """Depth-first search over the layer graph must find no back edge."""

    visiting: set[str] = set()

    visited: set[str] = set()

    def walk(layer: str, trail: list[str]) -> None:
        if layer in visiting:
            raise AssertionError(f"Import cycle: {' -> '.join([*trail, layer])}")

        if layer in visited:
            return

        visiting.add(layer)

        for target in sorted(imports.get(layer, set())):
            walk(target, [*trail, layer])

        visiting.discard(layer)

        visited.add(layer)

    for layer in LAYERS:
        walk(layer, [])


def test_package_ships_py_typed_marker() -> None:
    """
    The library is fully typed, so it must declare it.

    Without this marker, mypy in a downstream project silently ignores
    every annotation the framework provides.
    """

    assert (SOURCE_ROOT / "py.typed").is_file()
