from __future__ import annotations

from pathlib import Path

import pytest

from multilingual_embedding.config.base import CorpusConfig
from multilingual_embedding.corpus.corpus import Corpus
from multilingual_embedding.corpus.exceptions import CorpusFormatError
from multilingual_embedding.corpus.loader import build_reader, load_corpus, stream_sentences
from multilingual_embedding.corpus.reader import (
    JsonlReader,
    LineReader,
    TextFileReader,
    reader_for,
)
from multilingual_embedding.corpus.writer import (
    JsonlCorpusWriter,
    PlainTextCorpusWriter,
    write_sentences,
)


class TestReaders:
    def test_text_reader_one_document_per_file(self, text_corpus_directory: Path) -> None:
        documents = list(TextFileReader(text_corpus_directory).iter_documents())

        assert len(documents) == 2

    def test_text_reader_visits_files_in_sorted_order(self, text_corpus_directory: Path) -> None:
        sources = [
            Path(document.metadata.base.source or "").name
            for document in TextFileReader(text_corpus_directory).iter_documents()
        ]

        assert sources == sorted(sources)

    def test_line_reader_one_document_per_line(self, tmp_path: Path) -> None:
        path = tmp_path / "lines.txt"

        path.write_text("First line.\n\nSecond line.\nThird line.\n", encoding="utf-8")

        documents = list(LineReader(path).iter_documents())

        assert len(documents) == 3

        assert all(document.sentence_count == 1 for document in documents)

    def test_line_reader_does_not_resegment(self, tmp_path: Path) -> None:
        """A line holding two sentences must stay one unit."""

        path = tmp_path / "lines.txt"

        path.write_text("One. Two.\n", encoding="utf-8")

        documents = list(LineReader(path).iter_documents())

        assert documents[0].sentence_count == 1

    def test_jsonl_reader_reads_fields(self, jsonl_corpus_file: Path) -> None:
        documents = list(JsonlReader(jsonl_corpus_file).iter_documents())

        assert len(documents) == 3

        assert documents[0].identifier == "d1"

        assert documents[1].metadata.base.language == "hi"

    def test_jsonl_reader_carries_extra_fields(self, jsonl_corpus_file: Path) -> None:
        documents = list(JsonlReader(jsonl_corpus_file).iter_documents())

        assert documents[0].metadata.base.attributes["topic"] == "weather"

    def test_jsonl_missing_text_field_is_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"

        path.write_text('{"id": "x"}\n', encoding="utf-8")

        with pytest.raises(CorpusFormatError) as error:
            list(JsonlReader(path).iter_documents())

        assert error.value.context["line"] == 1

    def test_jsonl_non_object_record_is_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"

        path.write_text('["not", "an", "object"]\n', encoding="utf-8")

        with pytest.raises(CorpusFormatError):
            list(JsonlReader(path).iter_documents())

    def test_reader_for_selects_by_extension(
        self, jsonl_corpus_file: Path, text_corpus_directory: Path
    ) -> None:
        assert isinstance(reader_for(jsonl_corpus_file), JsonlReader)

        assert isinstance(reader_for(text_corpus_directory / "english.txt"), TextFileReader)

    def test_reader_for_explicit_format(self, tmp_path: Path) -> None:
        path = tmp_path / "data.txt"

        path.write_text("One.\n", encoding="utf-8")

        assert isinstance(reader_for(path, format="lines"), LineReader)

    def test_reader_handles_gzip(self, tmp_path: Path) -> None:
        import gzip

        path = tmp_path / "data.txt.gz"

        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write("Hello there. Goodbye.")

        documents = list(TextFileReader(path).iter_documents())

        assert documents[0].sentence_count == 2

    def test_iter_sentences_streams(self, text_corpus_directory: Path) -> None:
        texts = list(TextFileReader(text_corpus_directory).iter_sentence_texts())

        assert len(texts) == 6

        assert all(isinstance(text, str) for text in texts)


class TestWriters:
    def test_jsonl_writer_round_trip(self, small_corpus: Corpus, tmp_path: Path) -> None:
        path = tmp_path / "out.jsonl"

        assert JsonlCorpusWriter(path).write(small_corpus) == 2

        reloaded = list(JsonlReader(path).iter_documents())

        assert len(reloaded) == 2

        assert [document.text for document in reloaded] == [
            document.text for document in small_corpus
        ]

    def test_plain_text_writer_one_sentence_per_line(
        self, small_corpus: Corpus, tmp_path: Path
    ) -> None:
        path = tmp_path / "out.txt"

        count = PlainTextCorpusWriter(path).write(small_corpus)

        lines = path.read_text(encoding="utf-8").splitlines()

        assert count == len(lines) == 6

    def test_write_sentences_collapses_newlines(self, tmp_path: Path) -> None:
        """Embedded newlines would break the one-sentence-per-line contract."""

        path = tmp_path / "out.txt"

        write_sentences(path, ["one\ntwo", "three"])

        assert path.read_text(encoding="utf-8").splitlines() == ["one two", "three"]

    def test_write_sentences_skips_blank(self, tmp_path: Path) -> None:
        path = tmp_path / "out.txt"

        assert write_sentences(path, ["ok", "   ", ""]) == 1


class TestLoader:
    def test_load_corpus_from_directory(self, text_corpus_directory: Path) -> None:
        config = CorpusConfig(source=text_corpus_directory, patterns=["*.txt"])

        corpus = load_corpus(config, name="demo")

        assert corpus.document_count == 2

    def test_min_length_filter_is_applied(self, tmp_path: Path) -> None:
        path = tmp_path / "data.txt"

        path.write_text("Ok sentence here. A. Another good sentence.", encoding="utf-8")

        config = CorpusConfig(source=path, min_sentence_characters=5)

        corpus = load_corpus(config)

        assert all(len(text) >= 5 for text in corpus.sentence_texts())

    def test_deduplication_drops_repeats(self, tmp_path: Path) -> None:
        for name in ["a.txt", "b.txt"]:
            (tmp_path / name).write_text("Identical content here.", encoding="utf-8")

        config = CorpusConfig(source=tmp_path, patterns=["*.txt"])

        assert load_corpus(config, deduplicate=True).document_count == 1

        assert load_corpus(config, deduplicate=False).document_count == 2

    def test_lowercase_applies_at_every_level(self, tmp_path: Path) -> None:
        path = tmp_path / "data.txt"

        path.write_text("HELLO WORLD.", encoding="utf-8")

        config = CorpusConfig(source=path, lowercase=True)

        corpus = load_corpus(config)

        assert corpus[0].text == "hello world."

        assert all(text.islower() for text in corpus.sentence_texts())

    def test_stream_sentences_is_reiterable(self, text_corpus_directory: Path) -> None:
        """Multi-epoch training depends on the stream restarting."""

        config = CorpusConfig(source=text_corpus_directory, patterns=["*.txt"])

        stream = stream_sentences(config)

        assert list(stream) == list(stream)

        assert len(list(stream)) == 6

    def test_stream_limit(self, text_corpus_directory: Path) -> None:
        config = CorpusConfig(source=text_corpus_directory, patterns=["*.txt"])

        assert len(list(stream_sentences(config, limit=2))) == 2


class TestConfiguredTextField:
    """
    ``CorpusConfig.text_field`` must reach the JSON Lines reader.

    A corpus keyed on anything but ``"text"`` is common enough that a
    setting which validates, persists and then does nothing would send
    users chasing an error message naming a key they never configured.
    """

    @staticmethod
    def _write_content_keyed_corpus(tmp_path: Path) -> Path:
        path = tmp_path / "corpus.jsonl"

        path.write_text(
            '{"id": "d1", "content": "First sentence here. Second one too."}\n'
            '{"id": "d2", "content": "A third sentence."}\n',
            encoding="utf-8",
        )

        return path

    def test_build_reader_forwards_the_configured_field(self, tmp_path: Path) -> None:
        path = self._write_content_keyed_corpus(tmp_path)

        reader = build_reader(CorpusConfig(source=path, text_field="content"))

        assert isinstance(reader, JsonlReader)

        assert reader.text_field == "content"

    def test_corpus_keyed_on_another_field_loads(self, tmp_path: Path) -> None:
        path = self._write_content_keyed_corpus(tmp_path)

        corpus = load_corpus(CorpusConfig(source=path, text_field="content"))

        assert corpus.document_count == 2

        assert corpus.sentence_count == 3

    def test_explicit_jsonl_format_also_forwards_it(self, tmp_path: Path) -> None:
        """The setting must survive the non-``auto`` branch too."""

        path = self._write_content_keyed_corpus(tmp_path)

        config = CorpusConfig(source=path, format="jsonl", text_field="content")

        assert load_corpus(config).document_count == 2

    def test_default_field_still_reports_the_missing_key(self, tmp_path: Path) -> None:
        """Leaving the setting alone must not change the existing error."""

        path = self._write_content_keyed_corpus(tmp_path)

        with pytest.raises(CorpusFormatError):
            load_corpus(CorpusConfig(source=path))

    def test_text_sources_are_unaffected(self, text_corpus_directory: Path) -> None:
        """A plain text reader has no fields and must not be handed one."""

        config = CorpusConfig(
            source=text_corpus_directory,
            patterns=["*.txt"],
            text_field="content",
        )

        assert isinstance(build_reader(config), TextFileReader)

        assert load_corpus(config).document_count == 2


class TestFormatNamesAgree:
    """
    Three places name the corpus formats: the reader registry, the
    configuration validator and the command-line ``--format`` choices.
    They are written out separately and drifted once already — ``lines``
    was registered and usable from Python, yet rejected by both the
    config and the CLI, so a sentence-per-line corpus could not be read
    through either entry point.
    """

    def registry_names(self) -> set[str]:
        from multilingual_embedding.corpus.reader import READERS

        return set(READERS.keys())

    def test_config_accepts_every_registered_reader(self) -> None:
        from multilingual_embedding.config.base import CorpusConfig

        for name in self.registry_names():
            CorpusConfig(source="corpus.txt", format=name)

    def test_command_line_offers_every_registered_reader(self) -> None:
        """
        Checked per subcommand, not unioned across them. A union passes
        while one subcommand alone is missing a format, which is exactly
        the shape the drift took.
        """

        from multilingual_embedding.cli import build_parser

        parser = build_parser()

        expected = self.registry_names()

        checked = 0

        for action in parser._subparsers._group_actions:  # type: ignore[union-attr]
            for name, sub in action.choices.items():  # type: ignore[attr-defined]
                for candidate in sub._actions:
                    if candidate.dest != "format" or not candidate.choices:
                        continue

                    checked += 1

                    missing = expected - set(candidate.choices)

                    assert not missing, f"{name} --format omits {sorted(missing)}"

        assert checked, "no subcommand exposed --format; the test found nothing to check"


class TestTheCliDocstringMatchesTheCli:
    """
    The module docstring of ``cli.py`` is the first thing a reader sees,
    and it drifted: it said "Four subcommands" for as long as there were
    five, having never been revisited when ``validate`` was added.

    This is the same failure as ``TestFormatNamesAgree`` above — a second
    place restating a fact that only one place owns — so it gets the same
    treatment rather than a promise to remember.
    """

    def test_every_subcommand_appears_in_the_docstring(self) -> None:
        import multilingual_embedding.cli as cli

        parser = cli.build_parser()

        registered = {
            name
            for action in parser._subparsers._group_actions  # type: ignore[union-attr]
            for name in action.choices  # type: ignore[attr-defined]
        }

        docstring = cli.__doc__ or ""

        missing = {name for name in registered if f"qfme {name}" not in docstring}

        assert not missing, (
            f"registered subcommands absent from the cli.py docstring: {sorted(missing)}"
        )

    def test_the_docstring_does_not_advertise_a_subcommand_that_is_gone(self) -> None:
        import re

        import multilingual_embedding.cli as cli

        parser = cli.build_parser()

        registered = {
            name
            for action in parser._subparsers._group_actions  # type: ignore[union-attr]
            for name in action.choices  # type: ignore[attr-defined]
        }

        # `[\w-]` rather than `\w`: subcommand names may be hyphenated,
        # and `\w` captured "mine" from "mine-pairs", which then looked
        # like an advertised command that does not exist.
        advertised = set(re.findall(r"qfme ([\w-]+)", cli.__doc__ or ""))

        assert advertised <= registered, (
            f"the docstring advertises subcommands that do not exist: "
            f"{sorted(advertised - registered)}"
        )
