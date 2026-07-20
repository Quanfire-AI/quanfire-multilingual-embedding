"""
Corpus auditing.

Statistics describe a corpus; an audit judges it. The difference matters
when text arrives from an extraction pipeline, because the failure modes
are quiet: markup that survived cleaning, a language column that was
never populated, an encoding guessed wrongly, the same article ingested
twice under different ids.

None of those raise. They produce a corpus that loads, trains, and
yields a worse model than it should, for reasons that are invisible by
the time anyone looks at the metrics.

Findings carry a severity so a caller can decide what blocks a run.
``ERROR`` means the corpus is unusable as it stands; ``WARNING`` means it
will train but something was probably lost upstream; ``INFO`` is context
worth reading before trusting a result.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.utils.hashing import hash_text

from .document import Document
from .script import Script, detect_script

__all__ = ["CorpusAudit", "Finding", "Severity", "audit_corpus"]

_logger = get_logger(__name__)

# Below this, a "document" is usually a fragment left by a failed
# extraction rather than a real one.
_MINIMUM_CHARACTERS = 40

# Wiki and HTML markup that survives a careless extraction. Their presence
# in body text is close to proof that cleaning did not finish.
_MARKUP_MARKERS: tuple[str, ...] = (
    "[[",
    "]]",
    "{{",
    "}}",
    "<ref",
    "</ref>",
    "<div",
    "&nbsp;",
    "==",
)

_REPLACEMENT_CHARACTER = "�"

# The share above which an ERROR-class problem is treated as a broken
# extraction rather than a damaged source. Below it, the same problem is
# reported as a warning naming the affected share.
#
# One percent is a judgement, not a measurement. It sits far above the
# 0.0017% of Hindi Wikipedia that carries mojibake in its own source, and
# far below what a genuinely unfinished extraction produces — markup
# survives in most documents or almost none, rarely in between.
_SYSTEMATIC_SHARE = 0.01


class Severity(StrEnum):
    """How much a finding should worry the reader."""

    ERROR = "error"

    WARNING = "warning"

    INFO = "info"


@dataclass(slots=True)
class Finding:
    """
    One observation about a corpus.

    Attributes
    ----------
    severity:
        Whether this blocks use, degrades it, or merely informs.

    code:
        Stable machine-readable identifier.

    message:
        What was observed, in one sentence.

    count:
        Number of documents affected.

    share:
        Affected share of the corpus, in [0, 1].

    examples:
        A few offending document identifiers, for going and looking.

    remedy:
        What to do about it.
    """

    severity: Severity

    code: str

    message: str

    count: int = 0

    share: float = 0.0

    examples: list[str] = field(default_factory=list)

    remedy: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Reduce to primitives for reporting."""

        return {
            "severity": str(self.severity),
            "code": self.code,
            "message": self.message,
            "count": self.count,
            "share": round(self.share, 5),
            "examples": self.examples,
            "remedy": self.remedy,
        }


@dataclass(slots=True)
class CorpusAudit:
    """
    The verdict on a corpus.

    Attributes
    ----------
    documents, sentences:
        Volume seen.

    languages, scripts:
        Document counts per declared language and detected script.

    findings:
        Everything observed, most severe first.
    """

    documents: int = 0

    sentences: int = 0

    languages: dict[str, int] = field(default_factory=dict)

    scripts: dict[str, int] = field(default_factory=dict)

    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        """Findings that make the corpus unusable as it stands."""

        return [f for f in self.findings if f.severity is Severity.ERROR]

    @property
    def usable(self) -> bool:
        """True when nothing blocks a training run."""

        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        """Reduce to primitives for reporting."""

        return {
            "documents": self.documents,
            "sentences": self.sentences,
            "usable": self.usable,
            "languages": dict(sorted(self.languages.items())),
            "scripts": dict(sorted(self.scripts.items())),
            "findings": [finding.to_dict() for finding in self.findings],
        }


def audit_corpus(
    documents: Iterable[Document],
    *,
    example_limit: int = 3,
) -> CorpusAudit:
    """
    Inspect a corpus and report what is wrong with it.

    Consumes the iterable once and holds only counters and content
    hashes, so it runs over a corpus far larger than memory.

    Parameters
    ----------
    documents:
        Any iterable of documents — a corpus or a streaming reader.

    example_limit:
        Offending identifiers retained per finding.

    Example
    -------
    ::

        audit = audit_corpus(stream_documents(config))

        if not audit.usable:
            for finding in audit.errors:
                print(finding.message)
    """

    audit = CorpusAudit()

    languages: Counter[str] = Counter()

    scripts: Counter[str] = Counter()

    problems: dict[str, list[str]] = {
        "empty": [],
        "too_short": [],
        "no_language": [],
        "unknown_script": [],
        "markup": [],
        "encoding_damage": [],
        "duplicate": [],
        "single_sentence": [],
    }

    seen: set[str] = set()

    for document in documents:
        audit.documents += 1

        audit.sentences += document.sentence_count

        identifier = document.identifier or f"#{audit.documents}"

        text = document.text

        if not text.strip():
            problems["empty"].append(identifier)

            continue

        language = document.metadata.base.language

        if language:
            languages[language] += 1
        else:
            problems["no_language"].append(identifier)

        profile = detect_script(text)

        scripts[profile.dominant.value] += 1

        if profile.dominant is Script.UNKNOWN:
            problems["unknown_script"].append(identifier)

        if len(text.strip()) < _MINIMUM_CHARACTERS:
            problems["too_short"].append(identifier)

        if any(marker in text for marker in _MARKUP_MARKERS):
            problems["markup"].append(identifier)

        if text.count(_REPLACEMENT_CHARACTER) / max(1, len(text)) > 0.001:
            problems["encoding_damage"].append(identifier)

        digest = hash_text(" ".join(text.split()))

        if digest in seen:
            problems["duplicate"].append(identifier)
        else:
            seen.add(digest)

        if document.sentence_count <= 1:
            problems["single_sentence"].append(identifier)

    audit.languages = dict(languages)

    audit.scripts = dict(scripts)

    audit.findings = _build_findings(problems, audit.documents, example_limit)

    if audit.documents == 0:
        audit.findings.insert(
            0,
            Finding(
                severity=Severity.ERROR,
                code="empty_corpus",
                message="The source yielded no documents.",
                remedy="Check the path, the format and the glob patterns.",
            ),
        )

    _logger.info(
        "Audited corpus",
        extra={
            "documents": audit.documents,
            "usable": audit.usable,
            "findings": len(audit.findings),
        },
    )

    return audit


_SPECIFICATIONS: tuple[tuple[str, Severity, str, str], ...] = (
    (
        "empty",
        Severity.ERROR,
        "documents have no text at all",
        "Drop them upstream, or check that the text field name is right.",
    ),
    (
        "markup",
        Severity.ERROR,
        "documents still contain wiki or HTML markup",
        "Finish the extraction. Markup trains the tokenizer on syntax rather than language.",
    ),
    (
        "encoding_damage",
        Severity.ERROR,
        "documents contain replacement characters",
        "The source was decoded with the wrong encoding. Re-extract as UTF-8.",
    ),
    (
        "unknown_script",
        Severity.WARNING,
        "documents are in no recognised script",
        "Usually numeric tables or symbol dumps. Filter them out.",
    ),
    (
        "duplicate",
        Severity.WARNING,
        "documents duplicate earlier ones",
        "Deduplicate. Repeats inflate the apparent frequency of whatever they contain.",
    ),
    (
        "no_language",
        Severity.WARNING,
        "documents declare no language",
        "Populate the language field. Inference cannot recover it for shared scripts.",
    ),
    (
        "too_short",
        Severity.WARNING,
        f"documents are shorter than {_MINIMUM_CHARACTERS} characters",
        "Usually extraction fragments rather than real documents.",
    ),
    (
        "single_sentence",
        Severity.INFO,
        "documents hold a single sentence",
        "Expected for sentence-per-line corpora; suspicious for article text.",
    ),
)


def _build_findings(
    problems: dict[str, list[str]],
    total: int,
    example_limit: int,
) -> list[Finding]:
    """
    Turn raw problem lists into ordered findings.

    Severity depends on how widespread a problem is, not only on what it
    is. The distinction that matters is between a *broken extraction* and
    a *damaged source*: markup in half the documents means the pipeline
    did not finish and the corpus should not be trained on, while two
    articles in a hundred thousand carrying mojibake means Wikipedia
    contains two bad articles and always did.

    This was wrong until real data showed it. Hindi Wikipedia has
    replacement characters in 2 of 118,571 documents — 0.0017% — and the
    audit called the whole corpus unusable, which would have blocked
    training on the other 118,569 for no reason.
    """

    findings: list[Finding] = []

    for code, severity, message, remedy in _SPECIFICATIONS:
        affected = problems[code]

        if not affected:
            continue

        share = len(affected) / total if total else 0.0

        # An ERROR-class problem below the threshold is a real finding
        # about a few documents rather than a verdict on the corpus, so
        # it is reported as a warning with the remedy adjusted to say so.
        graded = severity

        adjusted = remedy

        if severity is Severity.ERROR and share < _SYSTEMATIC_SHARE:
            graded = Severity.WARNING

            adjusted = (
                f"{remedy} Affects {share * 100:.3g}% of documents, so this "
                f"reads as damage in the source rather than a broken "
                f"extraction — dropping those documents is usually enough."
            )

        findings.append(
            Finding(
                severity=graded,
                code=code,
                message=f"{len(affected)} {message}.",
                count=len(affected),
                share=share,
                examples=affected[:example_limit],
                remedy=adjusted,
            )
        )

    order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}

    findings.sort(key=lambda finding: (order[finding.severity], -finding.count))

    return findings
