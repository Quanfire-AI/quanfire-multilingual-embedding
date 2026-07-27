"""
Manufacturing *aligned* cross-lingual pairs.

The ordinary pair miner (:mod:`multilingual_embedding.corpus.pairs`)
builds pairs whose two sides are in the same language — a title and its
lead, both Hindi. That trains and measures *in-language* retrieval. It
cannot answer the question the evaluation report keeps flagging as the
open gap: does a query in one language retrieve the right passage in
*another*?

Answering it needs pairs whose anchor and positive are in different
languages but about the same subject — a Hindi title against the Tamil
lead of the same article. This module builds those, given a way to know
which documents correspond. That correspondence comes from Wikipedia's
interlanguage links, read by
:mod:`multilingual_embedding.corpus.langlinks`; this module is only the
pairing logic, kept separate so it can be tested on hand-built
alignments with no dump in sight.

Two properties are deliberate.

**Both directions are emitted.** From one aligned article this yields a
Hindi-query/Tamil-answer pair *and* a Tamil-query/Hindi-answer pair, so a
mixed set exercises retrieval in both directions rather than privileging
whichever language happened to be the source corpus.

**Both languages are recorded.** A :class:`MinedPair` carries a single
``language`` field, which the retrieval evaluator groups by. That field
is set to the *positive's* language — the side actually scored — while
the record also carries ``anchor_language`` and ``positive_language`` so
the direction of each pair survives to analysis. ``MinedPair.from_record``
ignores the two extra keys, so an aligned file still loads as ordinary
pairs wherever the richer view is not needed.

Overlap is measured, not filtered. Across two scripts it is ~0 by
construction — different alphabets share no word units — which is the
whole point: an aligned pair cannot be solved by string matching, so a
gain on it is evidence of meaning. For a same-script pairing (Hindi and
Marathi, both Devanagari) overlap can be real, so it is still computed
and a leaky mean is still warned about.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from multilingual_embedding.core.exceptions import ValidationError
from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.utils.hashing import hash_text
from multilingual_embedding.utils.validation import require_positive

from .document import Document
from .langlinks import normalize_title, resolve_target_title
from .pairs import _LEAKY_OVERLAP, _paragraphs, token_overlap

__all__ = [
    "AlignedDocument",
    "AlignedPair",
    "AlignedPairConfig",
    "AlignedPairKind",
    "AlignedStatistics",
    "iter_aligned_documents",
    "iter_aligned_pairs",
    "to_aligned_document",
]

_logger = get_logger(__name__)


class AlignedPairKind:
    """
    Where an aligned pair came from.

    Distinct string values from :class:`~multilingual_embedding.corpus.pairs.PairKind`
    so an aligned file and a monolingual file can be concatenated and a
    ``--kinds`` filter still selects cleanly.
    """

    TITLE_LEAD = "aligned_title_lead"

    LEAD_LEAD = "aligned_lead"


@dataclass(slots=True, frozen=True)
class AlignedDocument:
    """
    The little a document needs to contribute to an aligned pair.

    Only the title, language and lead paragraph — not the paragraph tree.
    Building the target-language index out of these rather than whole
    :class:`Document` objects is what keeps a second corpus from having
    to fit in memory beside the first.
    """

    identifier: str | None

    title: str

    language: str | None

    lead: str


@dataclass(slots=True)
class AlignedPairConfig:
    """
    What counts as a usable aligned pair.

    The anchor floor is lower than the monolingual miner's, because a
    legitimate cross-lingual query is often a bare title — ``भारत`` is
    four characters and a perfectly good query — and rejecting it would
    throw away the shortest, most retrieval-like anchors.
    """

    minimum_anchor_characters: int = 3

    minimum_positive_characters: int = 100

    maximum_positive_characters: int = 2000

    kinds: tuple[str, ...] = (
        AlignedPairKind.TITLE_LEAD,
        AlignedPairKind.LEAD_LEAD,
    )

    def __post_init__(self) -> None:
        require_positive(self.minimum_anchor_characters, name="minimum_anchor_characters")

        require_positive(self.minimum_positive_characters, name="minimum_positive_characters")

        require_positive(self.maximum_positive_characters, name="maximum_positive_characters")

        if self.minimum_positive_characters > self.maximum_positive_characters:
            raise ValidationError(
                "minimum_positive_characters must not exceed maximum_positive_characters",
                minimum=self.minimum_positive_characters,
                maximum=self.maximum_positive_characters,
            )

        unknown = set(self.kinds) - {AlignedPairKind.TITLE_LEAD, AlignedPairKind.LEAD_LEAD}

        if unknown:
            raise ValidationError("Unknown aligned pair kind", unknown=sorted(unknown))


@dataclass(slots=True, frozen=True)
class AlignedPair:
    """
    One cross-lingual anchor/positive, with the language of each side.

    ``language`` is the positive's language — the passage actually
    scored — so the retrieval evaluator's per-language breakdown reads as
    "how well is language X retrieved". The two per-side fields preserve
    which way round the pair runs.
    """

    anchor: str

    positive: str

    kind: str

    document: str

    anchor_language: str | None

    positive_language: str | None

    overlap: float

    def to_record(self) -> dict[str, Any]:
        """Reduce to a JSON Lines record compatible with ``MinedPair``."""

        return {
            "anchor": self.anchor,
            "positive": self.positive,
            "kind": self.kind,
            "document": self.document,
            "language": self.positive_language,
            "anchor_language": self.anchor_language,
            "positive_language": self.positive_language,
            "overlap": round(self.overlap, 4),
        }


@dataclass(slots=True)
class AlignedStatistics:
    """
    What alignment and pairing produced, and what they could not.

    The join numbers matter as much as the yield: a low ``aligned`` share
    against ``source_documents`` means the langlinks dump and the target
    corpus disagree about titles far more than expected, which is a
    reason to distrust the run rather than a detail to bury.
    """

    source_documents: int = 0

    aligned: int = 0

    missing_langlink: int = 0

    missing_target: int = 0

    produced: int = 0

    by_kind: dict[str, int] = field(default_factory=dict)

    mean_overlap_by_kind: dict[str, float] = field(default_factory=dict)

    rejected_short_anchor: int = 0

    rejected_short_positive: int = 0

    rejected_duplicate: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Reduce to primitives for reporting."""

        return {
            "source_documents": self.source_documents,
            "aligned": self.aligned,
            "unaligned": {
                "missing_langlink": self.missing_langlink,
                "missing_target": self.missing_target,
            },
            "produced": self.produced,
            "by_kind": dict(sorted(self.by_kind.items())),
            "mean_overlap_by_kind": {
                kind: round(value, 4) for kind, value in sorted(self.mean_overlap_by_kind.items())
            },
            "rejected": {
                "short_anchor": self.rejected_short_anchor,
                "short_positive": self.rejected_short_positive,
                "duplicate": self.rejected_duplicate,
            },
        }


def to_aligned_document(
    document: Document,
    *,
    language: str | None = None,
) -> AlignedDocument | None:
    """
    Reduce a document to its title, language and lead paragraph.

    Returns ``None`` when the document has no title or no lead — it
    cannot contribute a title/lead or lead/lead pair, so it is dropped
    rather than turned into an empty anchor.

    ``language`` is a fallback used only when the document does not
    already declare one; several Indian languages share the Devanagari
    script, so a document's language is not always inferable and the
    caller passes the corpus's known language here.
    """

    title = (document.metadata.title or "").strip()

    paragraphs = _paragraphs(document.text)

    if not title or not paragraphs:
        return None

    return AlignedDocument(
        identifier=document.identifier,
        title=title,
        language=document.metadata.base.language or language,
        lead=paragraphs[0],
    )


def iter_aligned_documents(
    source_documents: Iterable[Document],
    target_index: Mapping[str, AlignedDocument],
    title_map: Mapping[str, str],
    *,
    source_language: str | None = None,
    statistics: AlignedStatistics | None = None,
) -> Iterator[tuple[AlignedDocument, AlignedDocument]]:
    """
    Join source documents to their aligned target counterparts.

    For each source document, its interlanguage link gives a target
    title (:func:`resolve_target_title`), and the target index gives the
    document at that title. A source with no link, or a link to a title
    the target corpus does not contain, is counted against its reason and
    skipped — the alignment is simply absent, and inventing one would
    poison the evaluation set it feeds.

    ``target_index`` is keyed by :func:`normalize_title` of the target
    document's title, matching how the title from the langlinks dump is
    folded at lookup.
    """

    counts = statistics if statistics is not None else AlignedStatistics()

    for document in source_documents:
        source = to_aligned_document(document, language=source_language)

        if source is None:
            # A source with no title/lead cannot be aligned meaningfully;
            # it is not a join failure, so it is not counted as one.
            continue

        counts.source_documents += 1

        target_title = resolve_target_title(source.identifier, title_map)

        if target_title is None:
            counts.missing_langlink += 1

            continue

        target = target_index.get(normalize_title(target_title))

        if target is None:
            counts.missing_target += 1

            continue

        counts.aligned += 1

        yield source, target


def _emit(
    anchor_side: AlignedDocument,
    positive_side: AlignedDocument,
    *,
    use_title_anchor: bool,
    kind: str,
    config: AlignedPairConfig,
    seen: set[str],
    counts: AlignedStatistics,
    overlap_totals: dict[str, float],
) -> AlignedPair | None:
    """
    Build one directed pair, or ``None`` if it fails a quality rule.

    ``use_title_anchor`` selects the anchor side's title (a title/lead
    pair) versus its lead (a lead/lead pair). The positive is always a
    lead, truncated like the monolingual miner so an over-long section
    does not dominate the batch.
    """

    raw_anchor = anchor_side.title if use_title_anchor else anchor_side.lead

    anchor = " ".join(raw_anchor.split())

    positive = " ".join(positive_side.lead.split())

    if len(anchor) < config.minimum_anchor_characters:
        counts.rejected_short_anchor += 1

        return None

    if len(positive) < config.minimum_positive_characters:
        counts.rejected_short_positive += 1

        return None

    positive = positive[: config.maximum_positive_characters]

    digest = hash_text(f"{anchor}\x00{positive}")

    if digest in seen:
        counts.rejected_duplicate += 1

        return None

    seen.add(digest)

    overlap = token_overlap(anchor, positive)

    counts.produced += 1

    counts.by_kind[kind] = counts.by_kind.get(kind, 0) + 1

    overlap_totals[kind] = overlap_totals.get(kind, 0.0) + overlap

    counts.mean_overlap_by_kind = {
        name: overlap_totals[name] / counts.by_kind[name] for name in counts.by_kind
    }

    document = f"{anchor_side.identifier or ''}|{positive_side.identifier or ''}"

    return AlignedPair(
        anchor=anchor,
        positive=positive,
        kind=kind,
        document=document,
        anchor_language=anchor_side.language,
        positive_language=positive_side.language,
        overlap=overlap,
    )


def iter_aligned_pairs(
    document_pairs: Iterable[tuple[AlignedDocument, AlignedDocument]],
    config: AlignedPairConfig | None = None,
    statistics: AlignedStatistics | None = None,
) -> Iterator[AlignedPair]:
    """
    Yield cross-lingual pairs from aligned document pairs.

    Each aligned ``(a, b)`` produces up to four pairs — a title/lead and
    a lead/lead in each direction — subject to the configured kinds and
    the length and duplicate rules. Deduplication shares one set across
    the whole run, so the same anchor/positive text emitted from two
    aligned articles is written once.

    Pass an :class:`AlignedStatistics` to have it filled in; it is only
    complete once the iterator is exhausted, at which point a leaky mean
    overlap is warned about.
    """

    settings = config if config is not None else AlignedPairConfig()

    counts = statistics if statistics is not None else AlignedStatistics()

    seen: set[str] = set()

    overlap_totals: dict[str, float] = {}

    for source, target in document_pairs:
        # (a, b) and (b, a) so both languages appear as queries. The
        # ordered list is what makes the emitted set symmetric.
        for anchor_side, positive_side in ((source, target), (target, source)):
            if AlignedPairKind.TITLE_LEAD in settings.kinds:
                pair = _emit(
                    anchor_side,
                    positive_side,
                    use_title_anchor=True,
                    kind=AlignedPairKind.TITLE_LEAD,
                    config=settings,
                    seen=seen,
                    counts=counts,
                    overlap_totals=overlap_totals,
                )

                if pair is not None:
                    yield pair

            if AlignedPairKind.LEAD_LEAD in settings.kinds:
                pair = _emit(
                    anchor_side,
                    positive_side,
                    use_title_anchor=False,
                    kind=AlignedPairKind.LEAD_LEAD,
                    config=settings,
                    seen=seen,
                    counts=counts,
                    overlap_totals=overlap_totals,
                )

                if pair is not None:
                    yield pair

    _report(counts)


def _report(statistics: AlignedStatistics) -> None:
    """Log the yield and the join rate, and warn about any leaky kind."""

    _logger.info(
        "Mined aligned cross-lingual pairs",
        extra={
            "source_documents": statistics.source_documents,
            "aligned": statistics.aligned,
            "pairs": statistics.produced,
            "by_kind": statistics.by_kind,
        },
    )

    for kind, mean in statistics.mean_overlap_by_kind.items():
        if mean > _LEAKY_OVERLAP:
            _logger.warning(
                "Aligned pair kind overlaps heavily; the two languages likely "
                "share a script, so a gain on it is weaker evidence of meaning",
                extra={"kind": kind, "mean_overlap": round(mean, 4)},
            )
