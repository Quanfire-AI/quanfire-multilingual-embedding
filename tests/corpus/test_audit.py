"""
Corpus auditing.

The audit exists for extraction pipelines, so the tests mirror the ways
an extraction actually fails: markup that survived cleaning, a language
column nobody populated, a wrong encoding guess, the same article twice.

None of those raise on their own. Each produces a corpus that loads and
trains and yields a worse model, which is why they need naming.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from multilingual_embedding.corpus.audit import Severity, audit_corpus
from multilingual_embedding.corpus.document import Document


def documents(*texts: str, language: str | None = "en") -> list[Document]:
    return [
        Document.from_text(text, identifier=f"d{index}", language=language)
        for index, text in enumerate(texts)
    ]


def codes(audit) -> set[str]:  # type: ignore[no-untyped-def]
    return {finding.code for finding in audit.findings}


class TestCleanCorpus:
    def test_a_good_corpus_has_no_findings(self) -> None:
        audit = audit_corpus(
            documents(
                "The first document is long enough to be real. It has two sentences.",
                "The second document is also long enough to count. It also has two.",
            )
        )

        assert audit.usable

        assert audit.findings == []

    def test_counts_are_reported(self) -> None:
        audit = audit_corpus(
            documents("A sentence that is long enough to pass the length check. And two.")
        )

        assert audit.documents == 1

        assert audit.sentences == 2

        assert audit.languages == {"en": 1}

    def test_empty_source_is_an_error(self) -> None:
        """
        Nothing at all almost always means a wrong path or glob, and it
        should stop a pipeline rather than train on nothing.
        """

        audit = audit_corpus([])

        assert not audit.usable

        assert "empty_corpus" in codes(audit)


class TestExtractionFailures:
    def test_surviving_wiki_markup_is_an_error(self) -> None:
        """
        The characteristic failure of a Wikipedia extraction.

        Markup left in body text trains the tokenizer on syntax rather
        than on language, and it is invisible in aggregate statistics.
        """

        audit = audit_corpus(
            documents("'''Mumbai''' is a city in [[Maharashtra]] and it is very large indeed.")
        )

        assert not audit.usable

        assert "markup" in codes(audit)

    def test_html_markup_is_caught_too(self) -> None:
        audit = audit_corpus(
            documents("A paragraph of text.<ref>a citation</ref> More text follows here.")
        )

        assert "markup" in codes(audit)

    def test_replacement_characters_are_an_error(self) -> None:
        """A wrong encoding guess, which no amount of training recovers."""

        audit = audit_corpus(documents("Encoding went wrong: caf� na�ve r�sum�."))

        assert not audit.usable

        assert "encoding_damage" in codes(audit)

    def test_empty_documents_are_an_error(self) -> None:
        audit = audit_corpus(documents("   ", "Real text that is quite long enough here."))

        assert "empty" in codes(audit)


class TestQualityWarnings:
    def test_missing_language_is_a_warning(self) -> None:
        audit = audit_corpus(
            documents("Some text long enough to avoid the length warning here.", language=None)
        )

        assert "no_language" in codes(audit)

        assert audit.usable, "a missing language should not block training"

    def test_duplicates_are_reported(self) -> None:
        text = "The very same document text appearing twice in one corpus."

        audit = audit_corpus(documents(text, text))

        assert "duplicate" in codes(audit)

    def test_whitespace_variants_count_as_duplicates(self) -> None:
        audit = audit_corpus(
            documents(
                "The same document text appearing twice here.",
                "The  same   document text  appearing twice here.",
            )
        )

        assert "duplicate" in codes(audit)

    def test_short_fragments_are_reported(self) -> None:
        audit = audit_corpus(documents("Too short."))

        assert "too_short" in codes(audit)

    def test_scriptless_text_is_reported(self) -> None:
        audit = audit_corpus(documents("1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17"))

        assert "unknown_script" in codes(audit)


class TestFindingShape:
    def test_findings_are_ordered_by_severity(self) -> None:
        audit = audit_corpus(
            documents(
                "[[markup]] left in a document that is long enough to be real text.",
                "Short.",
            )
        )

        severities = [finding.severity for finding in audit.findings]

        assert severities == sorted(
            severities, key=lambda s: {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}[s]
        )

    def test_findings_carry_counts_shares_and_examples(self) -> None:
        audit = audit_corpus(documents("Short.", "Also short."))

        finding = next(f for f in audit.findings if f.code == "too_short")

        assert finding.count == 2

        assert finding.share == pytest.approx(1.0)

        assert finding.examples

        assert finding.remedy

    def test_examples_are_capped(self) -> None:
        audit = audit_corpus(*[documents(*(["Short."] * 10))], example_limit=2)

        finding = next(f for f in audit.findings if f.code == "too_short")

        assert len(finding.examples) == 2

    def test_audit_serialises(self) -> None:
        audit = audit_corpus(documents("[[markup]] here in a long enough document body."))

        json.dumps(audit.to_dict())

    def test_errors_property_filters(self) -> None:
        audit = audit_corpus(documents("[[markup]] here in a long enough document body."))

        assert audit.errors

        assert all(f.severity is Severity.ERROR for f in audit.errors)


class TestCommandLine:
    def test_clean_corpus_exits_zero(self, tmp_path: Path) -> None:
        from multilingual_embedding.cli import main

        path = tmp_path / "clean.jsonl"

        path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "id": f"d{i}",
                        "language": "en",
                        "text": f"Document {i} is long enough to be real. It has two sentences.",
                    }
                )
                for i in range(3)
            ),
            encoding="utf-8",
        )

        assert main(["validate", "--source", str(path)]) == 0

    def test_broken_corpus_exits_nonzero(self, tmp_path: Path) -> None:
        """A data pipeline should be able to gate on this."""

        from multilingual_embedding.cli import main

        path = tmp_path / "broken.jsonl"

        path.write_text(
            json.dumps({"id": "w1", "text": "'''Bold''' with [[links]] left in the text here."}),
            encoding="utf-8",
        )

        assert main(["validate", "--source", str(path)]) == 1

    def test_strict_promotes_warnings(self, tmp_path: Path) -> None:
        from multilingual_embedding.cli import main

        path = tmp_path / "warn.jsonl"

        path.write_text(
            "\n".join(
                json.dumps(
                    {"id": f"d{i}", "text": "A document with no declared language at all here."}
                )
                for i in range(2)
            ),
            encoding="utf-8",
        )

        assert main(["validate", "--source", str(path)]) == 0

        assert main(["validate", "--source", str(path), "--strict"]) == 1

    def test_json_output(self, tmp_path: Path) -> None:
        from multilingual_embedding.cli import main

        source = tmp_path / "corpus.jsonl"

        source.write_text(
            json.dumps(
                {"id": "d0", "language": "en", "text": "Long enough document text. Two sentences."}
            ),
            encoding="utf-8",
        )

        output = tmp_path / "audit.json"

        assert main(["validate", "--source", str(source), "--output", str(output)]) == 0

        payload = json.loads(output.read_text(encoding="utf-8"))

        assert payload["documents"] == 1

        assert payload["usable"] is True


class TestSeverityDependsOnPrevalence:
    """
    A verdict on a corpus is not the same as a finding about a document.

    Hindi Wikipedia carries mojibake in 2 of 118,571 articles — its own
    source is damaged, and always was. The audit called the whole corpus
    unusable and would have blocked training on the other 118,569.

    What ERROR is meant to mean is "the extraction did not finish", and
    that looks entirely different: markup or encoding damage in most
    documents, not two of them.
    """

    def _corpus(self, damaged: int, clean: int) -> list[Document]:
        broken = [
            Document.from_text(f"Damaged text {i}: caf� na�ve r�sum�.", identifier=f"b{i}")
            for i in range(damaged)
        ]

        fine = [
            Document.from_text(
                f"A perfectly ordinary document number {i}. It has two sentences.",
                identifier=f"g{i}",
            )
            for i in range(clean)
        ]

        return broken + fine

    def test_rare_damage_is_a_warning_and_the_corpus_stays_usable(self) -> None:
        audit = audit_corpus(self._corpus(damaged=2, clean=2000))

        finding = next(f for f in audit.findings if f.code == "encoding_damage")

        assert finding.severity is Severity.WARNING

        assert audit.usable, "a handful of bad source documents must not block training"

    def test_widespread_damage_is_an_error(self) -> None:
        """The extraction really is broken here, and should block."""

        audit = audit_corpus(self._corpus(damaged=400, clean=600))

        finding = next(f for f in audit.findings if f.code == "encoding_damage")

        assert finding.severity is Severity.ERROR

        assert not audit.usable

    def test_a_downgraded_finding_says_what_share_it_affects(self) -> None:
        """
        Otherwise the reader cannot tell a warning about two documents
        from a warning about two thousand.
        """

        audit = audit_corpus(self._corpus(damaged=2, clean=2000))

        finding = next(f for f in audit.findings if f.code == "encoding_damage")

        assert "%" in finding.remedy

        assert "source" in finding.remedy

    def test_the_finding_is_still_reported_either_way(self) -> None:
        """Downgrading severity must not mean hiding the problem."""

        audit = audit_corpus(self._corpus(damaged=1, clean=5000))

        codes = {f.code for f in audit.findings}

        assert "encoding_damage" in codes
