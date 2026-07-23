"""
Which data this repository is allowed to carry in its history.

Customer text may not be used to train anything here, and may not enter
this repository at all. `.gitignore` states most of that already — `data/*`
ignores everything and re-admits `data/sample/` and `data/README.md` — but
a `.gitignore` rule is a convention held up by whoever writes the next
commit, and the failure mode is permanent. Text committed once stays in
the history after the file is deleted, so this is not a mistake anybody
gets to correct later.

So the policy is asserted rather than described. Every JSON Lines file
git tracks, anywhere in the repository, must declare on every record where
its text came from, and that answer must be ``synthetic-`` something. A
sample added in a hurry with no ``source`` field fails, which is the point:
the rule fails closed, because the file most likely to be real is the one
whose provenance nobody thought to write down.

What this cannot see, and what therefore still needs judgement:

*Text pasted somewhere that is not a corpus file* — into a Python fixture,
a docstring, a markdown example. Nothing here parses prose looking for a
client name.

*The weights.* A model adapted on customer text carries that text, and an
adapter file is opaque to every check below. `models/indic-v1` was trained
on Wikipedia, and that fact lives in a note in `adapter.json` rather than
in anything enforced.

The enumerator is git rather than a filesystem walk, deliberately. The
question is what the *history* carries, so a real corpus sitting ignored
under `data/corpora/` is correctly invisible here — it is on the machine,
not in the repository.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

# Every record in a tracked corpus file must name its origin, and the name
# must begin with this. Not a boolean flag: "synthetic-sample" and
# "synthetic-domain-sample" say which fixture a record belongs to, and
# "synthetic-broken" says why one is deliberately damaged.
PROVENANCE_PREFIX = "synthetic-"

# Tracked under `data/`, and nothing else may be. The sample corpora are
# generated from templates; the directory's README explains why so little
# is here.
PERMITTED_UNDER_DATA = frozenset(
    {
        "data/README.md",
        "data/sample/corpus.jsonl",
        "data/sample/domain-corpus.jsonl",
    }
)

# Extensions a corpus, a dump or a mined pair file would arrive with. None
# of them belongs in the history — every one is large, and the ones that
# are not large are the ones worth worrying about.
CORPUS_ARCHIVE_SUFFIXES = (
    ".gz",
    ".bz2",
    ".zip",
    ".csv",
    ".tsv",
    ".parquet",
    ".xml",
)


def _tracked_paths(pathspec: str = ".") -> list[str]:
    """
    Repository-relative paths git tracks, or a skip if git cannot say.

    Skips rather than passes when the answer is unavailable — a suite run
    from an installed wheel has no checkout to ask about, and a guard that
    reports green in that situation is worse than one that is absent.
    """

    try:
        completed = subprocess.run(
            ["git", "ls-files", "-z", "--", pathspec],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:  # pragma: no cover - git is absent
        pytest.skip("git is not installed, so tracked files cannot be listed")

    if completed.returncode != 0:  # pragma: no cover - not a checkout
        pytest.skip("not a git checkout, so tracked files cannot be listed")

    return [name for name in completed.stdout.split("\0") if name]


def _tracked_corpus_files() -> list[str]:
    """Tracked JSON Lines files, wherever in the repository they live."""

    return [name for name in _tracked_paths() if name.endswith(".jsonl")]


def _records(name: str) -> list[tuple[int, Any]]:
    """Parse one JSON Lines file into (line number, record) pairs."""

    text = (REPOSITORY_ROOT / name).read_text(encoding="utf-8")

    return [
        (number, json.loads(line))
        for number, line in enumerate(text.splitlines(), start=1)
        if line.strip()
    ]


class TestTrackedCorpusData:
    """Every corpus file in the history declares itself synthetic."""

    def test_the_enumeration_finds_the_files_it_is_meant_to_guard(self) -> None:
        """
        The guard is looking at something.

        A pathspec typo, a rename, or a `git ls-files` that returns nothing
        would make every other test in this file pass over an empty list.
        That is the way an assertion of this shape fails: silently, in the
        direction of green.
        """

        tracked = _tracked_corpus_files()

        assert "data/sample/corpus.jsonl" in tracked
        assert "data/sample/domain-corpus.jsonl" in tracked
        assert len(tracked) >= 3

    def test_every_record_declares_a_source(self) -> None:
        """A record with no `source` is a record whose origin is unknown."""

        for name in _tracked_corpus_files():
            for number, record in _records(name):
                assert isinstance(record, dict), f"{name}:{number} is not an object"
                assert "source" in record, f"{name}:{number} declares no source"

    def test_every_declared_source_is_synthetic(self) -> None:
        """
        And the declaration says the text was generated, not collected.

        This is the assertion the file exists for. Everything else is
        scaffolding that stops it from passing vacuously.
        """

        for name in _tracked_corpus_files():
            for number, record in _records(name):
                source = record["source"]
                assert isinstance(source, str), f"{name}:{number} source is not a string"
                assert source.startswith(PROVENANCE_PREFIX), (
                    f"{name}:{number} declares source {source!r}, which is not "
                    f"{PROVENANCE_PREFIX}something — real text must not be committed"
                )


class TestWhatDataTracks:
    """`data/` carries the sample corpora and its own documentation."""

    def test_nothing_unexpected_is_tracked(self) -> None:
        """
        Adding a file here is a decision, not a side effect of `git add -A`.

        The `.gitignore` rule ignores `data/*` and re-admits `data/sample/`,
        so a new file dropped into `data/sample/` is committable without
        anyone editing an ignore rule and thinking about it.
        """

        tracked = set(_tracked_paths("data"))

        assert tracked, "expected data/ to track the sample corpora"
        assert tracked <= PERMITTED_UNDER_DATA, (
            f"unexpected tracked files under data/: {sorted(tracked - PERMITTED_UNDER_DATA)}"
        )

    def test_no_corpus_archive_is_tracked_anywhere(self) -> None:
        """
        No dump, no extracted corpus, no mined pair file, anywhere.

        Size is the usual argument and it is the weaker one. The reason
        that matters is that a compressed corpus is unreadable in review:
        nobody looking at the diff can tell what text went in.
        """

        offenders = [name for name in _tracked_paths() if name.endswith(CORPUS_ARCHIVE_SUFFIXES)]

        assert not offenders, f"corpus archives must not be tracked: {offenders}"
