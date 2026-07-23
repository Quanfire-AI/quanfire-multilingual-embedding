"""
The contract this package offers to the repositories that pin it.

Sibling QuanFire repositories install this one as a pinned git dependency
rather than copying from it. That makes two things load-bearing that were
previously only conventions, and neither of them fails loudly on its own:

*Which names are public.* A rename or a move inside a package that no
longer has a single owner breaks a consumer at import time, in their CI,
for a change that looked internal here.

*Which layers a base install can reach.* Torch lives behind the ``neural``
extra so that a consumer wanting text preparation keeps control of its own
training stack. That guarantee is a property of the import graph, not of
anything declared, and it is one careless import in an ``__init__.py``
away from silently ceasing to hold — for the person who adds the import,
everything still works, because they have torch.

The import checks run in a subprocess so they test the graph rather than
this environment, which has every extra installed.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# Documented in CHANGELOG.md as the public surface for consumers. Removing
# or renaming anything here is a breaking change and needs a minor bump,
# which before 1.0 is what this project uses for breaking changes.
PUBLIC_MODULES = (
    "multilingual_embedding.corpus",
    "multilingual_embedding.vocabulary",
    "multilingual_embedding.tokenizer",
    "multilingual_embedding.evaluation",
    "multilingual_embedding.core.exceptions",
)

# Named individually because a consumer pins against these specifically:
# the tokenizer fairness instrument, and the exception base it must catch.
PUBLIC_NAMES = (
    # Named here because it exists to close a hole a consumer found: the
    # trainer class is public and the only type its constructor accepts is
    # not, so this is the supported way to build one. Removing it would
    # push consumers back onto importing `config`.
    ("multilingual_embedding.tokenizer", "trainer_for"),
    ("multilingual_embedding.tokenizer", "SentencePieceTrainerAdapter"),
    ("multilingual_embedding.corpus", "reader_for"),
    ("multilingual_embedding.corpus", "corpus_from"),
    ("multilingual_embedding.corpus", "documents_from"),
    ("multilingual_embedding.corpus", "sentences_from"),
    ("multilingual_embedding.evaluation", "TokenizerEvaluator"),
    ("multilingual_embedding.evaluation", "TokenizerMetrics"),
    ("multilingual_embedding.evaluation", "language_fairness"),
    ("multilingual_embedding.evaluation", "evaluate_tokenizer"),
    ("multilingual_embedding.core.exceptions", "MultilingualEmbeddingError"),
)


def _import_in_a_clean_interpreter(statement: str) -> str:
    """Run one statement in a fresh interpreter and return its stdout."""

    completed = subprocess.run(
        [sys.executable, "-c", statement],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr

    return completed.stdout.strip()


@pytest.mark.parametrize("module", PUBLIC_MODULES)
def test_the_public_modules_import(module: str) -> None:
    _import_in_a_clean_interpreter(f"import {module}")


@pytest.mark.parametrize(("module", "name"), PUBLIC_NAMES)
def test_the_public_names_are_reachable_from_their_package(module: str, name: str) -> None:
    """Reachable from the package, not from the file that defines them.

    Consumers are promised ``from multilingual_embedding.evaluation import
    TokenizerEvaluator``, never ``...evaluation.tokenizer_eval``. Keeping
    the promise at the package boundary is what leaves the module layout
    free to change without breaking anyone.
    """

    assert _import_in_a_clean_interpreter(f"from {module} import {name}; print({name}.__name__)")


@pytest.mark.parametrize("module", PUBLIC_MODULES)
def test_importing_a_public_module_does_not_pull_in_torch(module: str) -> None:
    """The base install must stay free of the training stack.

    Asserted against ``sys.modules`` in a clean interpreter rather than by
    uninstalling torch, so the guarantee is tested here — where every extra
    is installed — and not only on a machine that happens to lack it.

    The path most likely to break this is ``embedding/__init__.py``, which
    ``evaluation`` imports through: today it reaches base, encoder, index,
    matrix, sentence and word2vec, none of which touch torch. One new
    convenience re-export of anything under ``embedding/neural/`` would end
    that, and nothing else would fail.
    """

    loaded = _import_in_a_clean_interpreter(
        f"import sys; import {module}; print('torch' in sys.modules)"
    )

    assert loaded == "False", (
        f"importing {module} pulled in torch, so the corpus, vocabulary, "
        "tokenizer and evaluation layers are no longer usable on a base "
        "install — check what was added to an __init__.py"
    )
