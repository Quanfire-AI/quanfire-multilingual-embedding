"""
Evaluation reports.

A report bundles corpus statistics, tokenizer metrics and embedding
metrics with the configuration that produced them. Keeping the
configuration inside the report is the point: a metric without the
settings that produced it cannot be compared against anything.

Reports are written as JSON for machines and Markdown for humans.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from multilingual_embedding.common.version import __version__
from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.utils.filesystem import ensure_directory
from multilingual_embedding.utils.io import write_json, write_text

__all__ = ["EvaluationReport"]

_logger = get_logger(__name__)


@dataclass(slots=True)
class EvaluationReport:
    """
    A complete evaluation record.

    Attributes
    ----------
    name:
        Experiment name.

    framework_version:
        Version of the framework that produced the report.

    created_at:
        UTC timestamp.

    config:
        Resolved configuration, as primitives.

    corpus, tokenizer, embedding:
        Per stage metric dictionaries.

    tokenizer_by_language:
        Per language tokenizer metrics, used for the fairness section.

    notes:
        Free-form remarks, such as which stages were skipped.
    """

    name: str = "default"

    framework_version: str = __version__

    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    config: dict[str, Any] = field(default_factory=dict)

    corpus: dict[str, Any] = field(default_factory=dict)

    tokenizer: dict[str, Any] = field(default_factory=dict)

    embedding: dict[str, Any] = field(default_factory=dict)

    tokenizer_by_language: dict[str, Any] = field(default_factory=dict)

    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Reduce to primitives."""

        return {
            "name": self.name,
            "framework_version": self.framework_version,
            "created_at": self.created_at,
            "config": self.config,
            "corpus": self.corpus,
            "tokenizer": self.tokenizer,
            "tokenizer_by_language": self.tokenizer_by_language,
            "embedding": self.embedding,
            "notes": self.notes,
        }

    def save(self, directory: str | Path) -> Path:
        """
        Write ``report.json`` and ``report.md`` into a directory.

        Returns the directory.
        """

        target = ensure_directory(directory)

        write_json(target / "report.json", self.to_dict(), indent=2)

        write_text(target / "report.md", self.to_markdown())

        _logger.info("Wrote evaluation report", extra={"directory": str(target)})

        return target

    def to_markdown(self) -> str:
        """Render a human readable report."""

        lines: list[str] = [
            f"# Evaluation report: {self.name}",
            "",
            f"- Framework version: `{self.framework_version}`",
            f"- Created: {self.created_at}",
            "",
        ]

        if self.corpus:
            lines += _section("Corpus", self.corpus)

        if self.tokenizer:
            lines += _section("Tokenizer", self.tokenizer)

        if self.tokenizer_by_language:
            lines += self._language_table()

        if self.embedding:
            lines += _section("Embeddings", self.embedding)

        if self.notes:
            lines += ["## Notes", ""]

            lines += [f"- {note}" for note in self.notes]

            lines += [""]

        return "\n".join(lines)

    def _language_table(self) -> list[str]:
        """Render the per language tokenizer efficiency table."""

        lines = [
            "## Tokenizer efficiency by language",
            "",
            "Characters per token measures compression. A wide spread means the",
            "vocabulary serves some languages better than others.",
            "",
            "| Language | Sentences | Tokens | Chars/token | Fertility | Unknown rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]

        for language, metrics in sorted(self.tokenizer_by_language.items()):
            if not isinstance(metrics, dict):
                continue

            lines.append(
                f"| {language} "
                f"| {metrics.get('sentence_count', 0)} "
                f"| {metrics.get('token_count', 0)} "
                f"| {metrics.get('characters_per_token', 0)} "
                f"| {metrics.get('fertility', 0)} "
                f"| {metrics.get('unknown_rate', 0)} |"
            )

        lines.append("")

        return lines


def _section(title: str, values: dict[str, Any]) -> list[str]:
    """Render a flat key/value section, skipping nested structures."""

    lines = [f"## {title}", "", "| Metric | Value |", "| --- | ---: |"]

    for key, value in values.items():
        if isinstance(value, (dict, list)):
            continue

        lines.append(f"| {key} | {value} |")

    lines.append("")

    return lines
