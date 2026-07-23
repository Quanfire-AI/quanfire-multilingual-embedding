"""
The hard-negative path end to end.

Documents in, negatives mined against a real contextual encoder, pairs
prefixed, model trained. Every piece has unit tests; what those cannot
see is the chain, and this chain crosses three layers with two silent
failure modes in it.

The first is the encoder contract. :func:`mine_negatives` takes a
``TextEncoder`` and never mentions torch, which is what keeps it out of
``embedding/neural/``. Whether a torch-backed encoder actually satisfies
that contract — numpy out, right shape, zero rather than NaN — is not
something the miner's own tests can establish, because they use a
planted encoder written to satisfy it.

The second is the prefix. Negatives are stored unprefixed and prefixed
again at training time. Get that wrong in either direction and the run
still completes: mining against unprefixed text finds the negatives of a
model nobody deploys, and storing prefixed text produces
``passage: passage: ...`` when the pair set reaches training. Neither
raises, and both would show a falling loss.
"""

from __future__ import annotations

import zlib

import pytest

torch = pytest.importorskip("torch", reason="the neural extra is not installed")

from multilingual_embedding.corpus.document import Document  # noqa: E402
from multilingual_embedding.corpus.pairs import mine_pairs  # noqa: E402
from multilingual_embedding.embedding.negatives import (  # noqa: E402
    NegativeConfig,
    mine_negatives,
)
from multilingual_embedding.embedding.neural import (  # noqa: E402
    ContrastiveConfig,
    ContrastiveTrainer,
    EncoderConfig,
    NeuralTextEncoder,
    TextPair,
    TransformerEncoderModel,
)
from multilingual_embedding.pipelines.adaptation import prefixed  # noqa: E402

pytestmark = pytest.mark.slow

VOCABULARY = 64


class WordTokenizer:
    """
    A hashing tokenizer, so this file needs no trained subword model.

    ``crc32`` rather than ``hash``: Python randomises string hashing per
    process, which would give every word a different identifier on every
    run and make the seeded model below reproducible in name only. The
    counts asserted in this file are properties of one fixed corpus
    under one fixed encoder, so the tokenizer has to be fixed too.
    """

    class _Encoding:
        def __init__(self, ids: list[int]) -> None:
            self.ids = ids

    def encode(self, text: str) -> WordTokenizer._Encoding:
        return self._Encoding(
            [zlib.crc32(word.encode("utf-8")) % VOCABULARY for word in text.split()] or [0]
        )


def build_encoder() -> NeuralTextEncoder:
    """A small encoder on CPU, seeded so a failure here is reproducible."""

    torch.manual_seed(0)

    model = TransformerEncoderModel(
        EncoderConfig(
            vocabulary_size=VOCABULARY,
            dimension=32,
            layers=2,
            heads=4,
            max_length=32,
            dropout=0.0,
        )
    )

    return NeuralTextEncoder(model, WordTokenizer(), device="cpu")


def corpus() -> list[Document]:
    """
    Twelve articles on two subjects, so passages exist that are related
    without being answers to each other — which is what a hard negative
    is, and what a corpus of unrelated documents cannot supply.
    """

    documents = []

    for index in range(12):
        subject = "weather" if index % 2 == 0 else "banking"

        lead = (
            f"This opening paragraph concerns {subject} in region {index}, and it "
            f"is written at sufficient length to survive the minimum positive "
            f"length filter that rejects fragments rather than passages."
        )

        body = (
            f"A further block of text about {subject} in region {index}, also long "
            f"enough to pass the same length filter applied to every candidate."
        )

        documents.append(
            Document.from_text(
                f"{lead}\n\n{body}",
                identifier=str(index),
                language="en",
                title=f"Region {index} {subject}",
            )
        )

    return documents


@pytest.fixture(scope="module")
def pairs():  # type: ignore[no-untyped-def]
    mined, _ = mine_pairs(corpus())

    assert len(mined) > 10, "the fixture corpus must yield a real pair set"

    return mined


class TestARealEncoderDrivesTheMiner:
    def test_the_chain_produces_negatives(self, pairs) -> None:  # type: ignore[no-untyped-def]
        """
        An untrained transformer places texts near-arbitrarily, which is
        fine here: the assertion is that the plumbing carries vectors
        from a torch model into a ranking, not that the ranking is good.
        """

        mined, statistics = mine_negatives(
            pairs,
            build_encoder(),
            NegativeConfig(count=2, pool=8, minimum_similarity=-1.0, maximum_similarity=0.999),
        )

        assert statistics.accepted > 0

        assert any(item.negatives for item in mined)

    def test_the_guards_are_reported_rather_than_assumed(self, pairs) -> None:  # type: ignore[no-untyped-def]
        """
        Two pairs come from each article in this corpus, so the
        provenance guard has something real to reject.

        The count is asserted alongside the property, because an
        untrained encoder ranks near-arbitrarily and a run in which the
        guard simply never had to fire would satisfy the property on its
        own. Asking for more negatives than the pool can supply forces
        the whole frontier to be scanned, so both siblings are reached.
        """

        mined, statistics = mine_negatives(
            pairs,
            build_encoder(),
            NegativeConfig(count=16, pool=16, minimum_similarity=-1.0, maximum_similarity=0.999),
        )

        assert statistics.rejected_as_positive > 0

        assert statistics.rejected_same_document > 0

        by_document = {item.positive: item.document for item in mined}

        assert all(
            by_document.get(negative) != item.document
            for item in mined
            for negative in item.negatives
        )

    def test_nothing_mined_is_its_own_pairs_positive(self, pairs) -> None:  # type: ignore[no-untyped-def]
        mined, _ = mine_negatives(
            pairs,
            build_encoder(),
            NegativeConfig(count=2, pool=8, minimum_similarity=-1.0, maximum_similarity=0.999),
        )

        assert all(item.positive not in item.negatives for item in mined)


class TestTheNegativesReachTraining:
    def test_a_mined_pair_set_trains(self, pairs) -> None:  # type: ignore[no-untyped-def]
        """
        The whole point of the chain. If the extra candidate columns
        were shaped wrong, the loss would raise here rather than fall.
        """

        mined, _ = mine_negatives(
            pairs,
            build_encoder(),
            NegativeConfig(count=2, pool=8, minimum_similarity=-1.0, maximum_similarity=0.999),
        )

        report = ContrastiveTrainer(
            build_encoder(),
            ContrastiveConfig(epochs=3, batch_size=8, learning_rate=3e-3, seed=1),
        ).train([TextPair(p.anchor, p.positive, p.negatives) for p in mined])

        assert report.steps > 0

        assert report.improved, f"loss did not fall: {report.losses}"

    def test_the_prefixes_are_applied_once(self, pairs) -> None:  # type: ignore[no-untyped-def]
        """
        Mining strips them and ``prefixed`` puts them back, so a
        negative that reaches training carries exactly one marker. Two
        would be a string no checkpoint was trained on, and nothing
        anywhere would raise about it.
        """

        mined, _ = mine_negatives(
            pairs,
            build_encoder(),
            NegativeConfig(
                count=2,
                pool=8,
                minimum_similarity=-1.0,
                maximum_similarity=0.999,
                query_prefix="query: ",
                passage_prefix="passage: ",
            ),
        )

        assert all(
            not negative.startswith("passage: ") for item in mined for negative in item.negatives
        )

        ready = prefixed(mined, "query: ", "passage: ")

        negatives = [negative for item in ready for negative in item.negatives]

        assert negatives, "the fixture must produce at least one negative to check"

        assert all(negative.count("passage: ") == 1 for negative in negatives)
