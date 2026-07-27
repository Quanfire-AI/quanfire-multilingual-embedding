"""
Reading a Wikipedia langlinks SQL dump.

The parser is the risk here. A dump is a stream of MySQL tuples whose
last column is an arbitrary article title, so the escaping and the
punctuation inside a title are exactly what a naive split gets wrong —
and a mis-parse does not raise, it silently drops or corrupts an
alignment and shrinks the evaluation set with nothing to point at. These
tests plant the punctuation and escapes that break the naive approach.
"""

from __future__ import annotations

import gzip
from pathlib import Path

from multilingual_embedding.corpus.langlinks import (
    build_target_title_map,
    iter_langlinks,
    normalize_title,
)

# A dump line as mysqldump writes it: one INSERT, several tuples, a
# trailing semicolon. ll_from is a bare int; ll_lang and ll_title are
# single-quoted.
LINE = (
    "INSERT INTO `langlinks` VALUES "
    "(12,'ta','தமிழ் தலைப்பு'),"
    "(12,'ml','മലയാളം'),"
    "(34,'ta','New_Delhi'),"
    "(56,'ta','Massachusetts_(band)'),"
    "(78,'ta','It\\'s_a_title'),"
    "(90,'fr','Paris');\n"
)


def write(path: Path, *lines: str) -> Path:
    opener = gzip.open if path.suffix == ".gz" else open

    with opener(path, "wt", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line)

    return path


class TestIterLanglinks:
    def test_it_reads_every_tuple(self, tmp_path: Path) -> None:
        rows = list(iter_langlinks(write(tmp_path / "ll.sql", LINE)))

        assert len(rows) == 6

    def test_the_page_id_comes_back_an_int(self, tmp_path: Path) -> None:
        rows = list(iter_langlinks(write(tmp_path / "ll.sql", LINE)))

        assert rows[0][0] == 12

        assert isinstance(rows[0][0], int)

    def test_a_title_with_parentheses_is_not_split(self, tmp_path: Path) -> None:
        """The parenthesis is title text, not tuple structure."""

        rows = list(iter_langlinks(write(tmp_path / "ll.sql", LINE)))

        titles = [title for _, lang, title in rows if lang == "ta"]

        assert "Massachusetts_(band)" in titles

    def test_an_escaped_apostrophe_is_unescaped(self, tmp_path: Path) -> None:
        rows = list(iter_langlinks(write(tmp_path / "ll.sql", LINE)))

        titles = [title for _, lang, title in rows if lang == "ta"]

        assert "It's_a_title" in titles

    def test_non_ascii_titles_survive(self, tmp_path: Path) -> None:
        rows = list(iter_langlinks(write(tmp_path / "ll.sql", LINE)))

        assert (12, "ta", "தமிழ் தலைப்பு") in rows

    def test_gzip_is_read_by_extension(self, tmp_path: Path) -> None:
        rows = list(iter_langlinks(write(tmp_path / "ll.sql.gz", LINE)))

        assert len(rows) == 6

    def test_lines_that_are_not_inserts_are_ignored(self, tmp_path: Path) -> None:
        """A real dump opens with comments, DDL and lock statements."""

        path = write(
            tmp_path / "ll.sql",
            "-- MySQL dump\n",
            "DROP TABLE IF EXISTS `langlinks`;\n",
            "LOCK TABLES `langlinks` WRITE;\n",
            LINE,
            "UNLOCK TABLES;\n",
        )

        assert len(list(iter_langlinks(path))) == 6

    def test_multiple_insert_lines_accumulate(self, tmp_path: Path) -> None:
        path = write(
            tmp_path / "ll.sql",
            "INSERT INTO `langlinks` VALUES (1,'ta','A');\n",
            "INSERT INTO `langlinks` VALUES (2,'ta','B');\n",
        )

        assert len(list(iter_langlinks(path))) == 2


class TestBuildTargetTitleMap:
    def test_it_keeps_only_the_target_language(self, tmp_path: Path) -> None:
        mapping = build_target_title_map(write(tmp_path / "ll.sql", LINE), target_language="ta")

        # 12, 34, 56, 78 link to Tamil; the Malayalam and French rows drop.
        assert set(mapping) == {"12", "34", "56", "78"}

    def test_the_page_id_key_is_a_string(self, tmp_path: Path) -> None:
        """The corpus writes id as a string, so the join key must match."""

        mapping = build_target_title_map(write(tmp_path / "ll.sql", LINE), target_language="ta")

        assert mapping["34"] == "New_Delhi"

    def test_the_title_is_left_un_normalised(self, tmp_path: Path) -> None:
        """Normalisation happens at lookup, on both sides at once."""

        mapping = build_target_title_map(write(tmp_path / "ll.sql", LINE), target_language="ta")

        assert mapping["34"] == "New_Delhi"


class TestNormalizeTitle:
    def test_underscores_fold_to_spaces(self) -> None:
        assert normalize_title("New_Delhi") == normalize_title("New Delhi")

    def test_it_is_case_folded(self) -> None:
        assert normalize_title("New Delhi") == normalize_title("new delhi")

    def test_surrounding_and_repeated_whitespace_collapses(self) -> None:
        assert normalize_title("  New   Delhi  ") == normalize_title("New Delhi")

    def test_a_non_latin_title_is_unchanged_but_stripped(self) -> None:
        assert normalize_title(" தமிழ் ") == "தமிழ்"
