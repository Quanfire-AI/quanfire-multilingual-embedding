"""
Tests for parallel-corpus ingestion.

The behaviour that matters most here is the one that *refuses*: two files
of unequal length are a misaligned corpus, and pairing sentences with the
wrong translation is the failure this whole module exists to make loud.
"""

from __future__ import annotations

import pytest

from multilingual_embedding.core.exceptions import ValidationError
from multilingual_embedding.corpus.parallel import (
    ParallelConfig,
    ParallelStatistics,
    iter_parallel_pairs,
)


def _ingest(anchors, positives, **kwargs):
    statistics = ParallelStatistics()
    pairs = list(
        iter_parallel_pairs(
            anchors,
            positives,
            anchor_language=kwargs.pop("anchor_language", "en"),
            positive_language=kwargs.pop("positive_language", "hi"),
            statistics=statistics,
            **kwargs,
        )
    )
    return pairs, statistics


def test_line_aligned_files_become_one_pair_each() -> None:
    anchors = ["the sky is blue today", "water boils at one hundred"]
    positives = ["आज आसमान नीला है", "पानी सौ पर उबलता है"]

    pairs, stats = _ingest(anchors, positives)

    assert [p.anchor for p in pairs] == anchors
    assert [p.positive for p in pairs] == positives
    assert stats.read == 2
    assert stats.produced == 2


def test_the_positive_language_is_the_record_language() -> None:
    pairs, _ = _ingest(["a source sentence here"], ["एक स्रोत वाक्य यहाँ"])

    # `language` is the positive's language, matching AlignedPair, so the
    # evaluator's per-language breakdown reads as "how well is X retrieved".
    assert pairs[0].language == "hi"
    assert pairs[0].to_record()["language"] == "hi"


def test_each_line_gets_a_unique_document_id() -> None:
    anchors = ["first source sentence", "second source sentence"]
    positives = ["पहला वाक्य यहाँ है", "दूसरा वाक्य यहाँ है"]

    pairs, _ = _ingest(anchors, positives, document_prefix="samanantar-hi")

    documents = [p.document for p in pairs]
    assert documents == ["samanantar-hi:1", "samanantar-hi:2"]
    assert len(set(documents)) == len(documents)


def test_a_shorter_anchor_file_raises_rather_than_truncating() -> None:
    anchors = ["only one anchor line here"]
    positives = ["पहली पंक्ति यहाँ", "दूसरी पंक्ति जिसका कोई जोड़ नहीं"]

    with pytest.raises(ValidationError) as excinfo:
        list(
            iter_parallel_pairs(
                anchors,
                positives,
                anchor_language="en",
                positive_language="hi",
            )
        )

    assert excinfo.value.context["anchor_exhausted"] is True


def test_a_shorter_positive_file_raises_rather_than_truncating() -> None:
    anchors = ["first anchor sentence here", "second anchor sentence here"]
    positives = ["केवल एक पंक्ति यहाँ"]

    with pytest.raises(ValidationError) as excinfo:
        list(
            iter_parallel_pairs(
                anchors,
                positives,
                anchor_language="en",
                positive_language="hi",
            )
        )

    assert excinfo.value.context["positive_exhausted"] is True


def test_short_sides_are_rejected_and_counted() -> None:
    anchors = ["ok", "a good long anchor sentence"]
    positives = ["एक अच्छा लंबा वाक्य यहाँ", "ठीक"]

    pairs, stats = _ingest(anchors, positives)

    assert stats.read == 2
    assert stats.produced == 0
    assert stats.rejected_short_anchor == 1
    assert stats.rejected_short_positive == 1


def test_a_positive_over_the_ceiling_is_rejected() -> None:
    anchors = ["a perfectly ordinary anchor"]
    positives = ["य" * 3000]

    pairs, stats = _ingest(anchors, positives)

    assert stats.produced == 0
    assert stats.rejected_long_positive == 1


def test_identical_sides_are_dropped_by_default() -> None:
    anchors = ["https://example.org/page", "a real anchor sentence here"]
    positives = ["https://example.org/page", "एक असली वाक्य यहाँ है"]

    pairs, stats = _ingest(anchors, positives)

    assert stats.rejected_identical == 1
    assert stats.produced == 1
    assert pairs[0].anchor == "a real anchor sentence here"


def test_identical_sides_are_kept_when_asked() -> None:
    anchors = ["https://example.org/page"]
    positives = ["https://example.org/page"]

    pairs, stats = _ingest(
        anchors, positives, config=ParallelConfig(drop_identical=False)
    )

    assert stats.rejected_identical == 0
    assert stats.produced == 1


def test_duplicate_pairs_are_dropped_by_default() -> None:
    anchors = ["a repeated anchor sentence", "a repeated anchor sentence"]
    positives = ["एक दोहराया गया वाक्य", "एक दोहराया गया वाक्य"]

    pairs, stats = _ingest(anchors, positives)

    assert stats.rejected_duplicate == 1
    assert stats.produced == 1


def test_duplicates_survive_when_dedup_is_off() -> None:
    anchors = ["a repeated anchor sentence", "a repeated anchor sentence"]
    positives = ["एक दोहराया गया वाक्य", "एक दोहराया गया वाक्य"]

    pairs, stats = _ingest(
        anchors, positives, config=ParallelConfig(deduplicate=False)
    )

    assert stats.rejected_duplicate == 0
    assert stats.produced == 2


def test_statistics_to_dict_carries_the_rejection_breakdown() -> None:
    anchors = ["x", "a good anchor sentence here", "a good anchor sentence here"]
    positives = ["एक अच्छा वाक्य यहाँ है", "एक अच्छा वाक्य यहाँ है", "एक अच्छा वाक्य यहाँ है"]

    _, stats = _ingest(anchors, positives)
    summary = stats.to_dict()

    assert summary["read"] == 3
    assert summary["produced"] == 1
    assert summary["rejected"]["short_anchor"] == 1
    assert summary["rejected"]["duplicate"] == 1
    assert 0.0 <= summary["mean_overlap"] <= 1.0


def test_a_bad_character_threshold_is_rejected_at_construction() -> None:
    with pytest.raises(ValidationError):
        ParallelConfig(minimum_positive_characters=3000, maximum_positive_characters=2000)
