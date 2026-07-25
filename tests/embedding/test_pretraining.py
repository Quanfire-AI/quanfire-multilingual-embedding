"""
Masked-language pretraining, and the proof that it teaches an embedding.

The decisive tests are in :class:`TestItActuallyLearns`. The rest check
the masking recipe and the module shapes — necessary, but a model that
learned nothing would pass them. Stage one of the from-scratch path only
matters if a random embedding table comes out of it able to predict a
hidden token from its neighbours *better than guessing the frequent one*,
and that is the one claim a shape test cannot make.

The tokenizer here is a fixed lookup table, not the hashing one the other
neural tests use. Two reasons: the masked-accuracy baseline must be exact,
which rules out hash collisions folding two words onto one id, and the
corpus is a tiny closed vocabulary anyway, so a table is the honest model
of it. Everything runs on CPU in seconds; the same code path pretrains a
real model, differing only in configuration.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

torch = pytest.importorskip("torch", reason="requires the neural extra")

from multilingual_embedding.core.exceptions import ValidationError  # noqa: E402
from multilingual_embedding.embedding.neural import (  # noqa: E402
    ContrastiveConfig,
    ContrastiveTrainer,
    EncoderConfig,
    MaskedLanguageConfig,
    MaskedLanguageModel,
    MaskedLanguageModelTrainer,
    NeuralTextEncoder,
    TextPair,
    TransformerEncoderModel,
    mask_tokens,
)
from multilingual_embedding.embedding.neural.pretraining import IGNORE_INDEX  # noqa: E402

# Ids 0..3 are the framework's PAD/UNK/BOS/EOS; 4 stands in for [MASK],
# which the framework vocabulary does not reserve. Content words start
# above both, so the table never collides with a reserved id.
MASK_ID = 4

CONTENT_OFFSET = 5

DIMENSION = 32


class TableTokenizer:
    """
    A fixed word-to-id table, so the vocabulary is exactly known.

    Unlike the hashing tokenizer elsewhere, every distinct word gets its
    own id with no chance of collision, which is what lets a test compute
    the unigram-frequency baseline exactly and compare the model against
    it. Unknown words are dropped rather than mapped to ``UNK``, because
    these tests only ever feed words the table was built from.
    """

    def __init__(self, words: list[str]) -> None:
        vocabulary = sorted(set(words))

        self._table = {word: CONTENT_OFFSET + index for index, word in enumerate(vocabulary)}

        self.vocabulary_size = CONTENT_OFFSET + len(self._table)

    def encode(self, text: str) -> TableTokenizer._Encoding:
        return TableTokenizer._Encoding(
            [self._table[word] for word in text.split() if word in self._table]
        )

    def ids_for(self, text: str) -> list[int]:
        return self.encode(text).ids

    class _Encoding:
        def __init__(self, ids: list[int]) -> None:
            self.ids = ids


# Two fixed templates over disjoint words. Every position is fully
# determined by which template a sequence belongs to, so a hidden token is
# perfectly predictable from context — but only to a model that has
# learned the structure. Guessing the most frequent token scores 0.1, one
# word in ten.
TEMPLATES = (
    "alpha bravo charlie delta echo",
    "one two three four five",
)


def build_corpus(repeats: int = 40) -> list[str]:
    """The two templates, each repeated, shuffled into one corpus."""

    return [template for template in TEMPLATES for _ in range(repeats)]


def corpus_vocabulary() -> list[str]:
    return [word for template in TEMPLATES for word in template.split()]


def build_encoder(**overrides: object) -> NeuralTextEncoder:
    """A small encoder on CPU over the corpus vocabulary."""

    tokenizer = TableTokenizer(corpus_vocabulary())

    settings: dict[str, object] = {
        "vocabulary_size": tokenizer.vocabulary_size,
        "dimension": DIMENSION,
        "layers": 2,
        "heads": 4,
        "max_length": 16,
        "dropout": 0.0,
    }

    settings.update(overrides)

    torch.manual_seed(0)

    model = TransformerEncoderModel(EncoderConfig(**settings))  # type: ignore[arg-type]

    return NeuralTextEncoder(model, tokenizer, device="cpu")


def unigram_baseline() -> float:
    """
    The accuracy of always predicting the single most frequent token.

    This is the bar masked-language modelling has to clear to have taught
    anything: a model that clears it is using context, not just betting on
    the common word. Computed over the real tokens of the corpus, which is
    the set the trainer measures accuracy over.
    """

    tokenizer = TableTokenizer(corpus_vocabulary())

    counts: Counter[int] = Counter()

    for document in build_corpus():
        counts.update(tokenizer.ids_for(document))

    return max(counts.values()) / sum(counts.values())


class TestHiddenStates:
    """
    The per-token states the pooled vector and the MLM head both read.

    ``forward`` was refactored to pool ``hidden_states``; if that refactor
    changed the pooled result, every downstream encoder test would drift,
    so this pins the equivalence directly.
    """

    def test_hidden_states_are_one_vector_per_token(self) -> None:
        model = TransformerEncoderModel(
            EncoderConfig(vocabulary_size=32, dimension=DIMENSION, layers=2)
        ).eval()

        ids = torch.randint(1, 32, (3, 7))

        mask = torch.ones((3, 7), dtype=torch.long)

        assert model.hidden_states(ids, mask).shape == (3, 7, DIMENSION)

    def test_forward_is_hidden_states_pooled(self) -> None:
        model = TransformerEncoderModel(
            EncoderConfig(vocabulary_size=32, dimension=DIMENSION, layers=2, dropout=0.0)
        ).eval()

        ids = torch.tensor([[5, 9, 13, 0, 0]])

        mask = torch.tensor([[1, 1, 1, 0, 0]])

        with torch.no_grad():
            hidden = model.hidden_states(ids, mask)

            weights = mask.unsqueeze(-1).to(hidden.dtype)

            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1)

            torch.testing.assert_close(model(ids, mask), pooled, rtol=1e-5, atol=1e-6)


class TestMaskTokens:
    """
    The corruption recipe, tested as the pure function it is.

    Every way this can be wrong trains a model quietly: masking padding
    teaches it to predict nothing, masking a protected token corrupts a
    structural marker, and supervising an unselected position leaks the
    answer.
    """

    @staticmethod
    def _generator() -> torch.Generator:
        return torch.Generator().manual_seed(0)

    def test_only_selected_positions_are_supervised(self) -> None:
        ids = torch.randint(CONTENT_OFFSET, 20, (4, 10))

        mask = torch.ones_like(ids)

        corrupted, labels = mask_tokens(
            ids,
            mask,
            mask_token_id=MASK_ID,
            vocabulary_size=20,
            protected_ids=(),
            probability=0.5,
            generator=self._generator(),
        )

        supervised = labels != IGNORE_INDEX

        # Where a position is supervised, its label is the original token.
        assert torch.equal(labels[supervised], ids[supervised])

        # Any change to the ids happens only at a supervised position — the
        # unchanged 10% band means the converse need not hold.
        assert torch.all(supervised | (corrupted == ids))

    def test_padding_is_never_selected(self) -> None:
        ids = torch.randint(CONTENT_OFFSET, 20, (4, 10))

        mask = torch.ones_like(ids)

        mask[:, 6:] = 0  # a padded tail

        _, labels = mask_tokens(
            ids,
            mask,
            mask_token_id=MASK_ID,
            vocabulary_size=20,
            protected_ids=(),
            probability=0.9,
            generator=self._generator(),
        )

        assert torch.all(labels[:, 6:] == IGNORE_INDEX)

    def test_protected_ids_are_never_selected(self) -> None:
        # Row full of a protected token; nothing in it may be masked.
        ids = torch.full((3, 8), 7)

        mask = torch.ones_like(ids)

        corrupted, labels = mask_tokens(
            ids,
            mask,
            mask_token_id=MASK_ID,
            vocabulary_size=20,
            protected_ids=(7,),
            probability=0.9,
            generator=self._generator(),
        )

        assert torch.all(labels == IGNORE_INDEX)

        assert torch.equal(corrupted, ids)

    def test_random_replacements_stay_in_vocabulary(self) -> None:
        ids = torch.randint(CONTENT_OFFSET, 20, (8, 12))

        mask = torch.ones_like(ids)

        corrupted, _ = mask_tokens(
            ids,
            mask,
            mask_token_id=MASK_ID,
            vocabulary_size=20,
            protected_ids=(),
            probability=0.9,
            generator=self._generator(),
        )

        assert int(corrupted.min()) >= 0

        assert int(corrupted.max()) < 20

    def test_it_is_reproducible(self) -> None:
        ids = torch.randint(CONTENT_OFFSET, 20, (4, 10))

        mask = torch.ones_like(ids)

        def run() -> tuple[torch.Tensor, torch.Tensor]:
            return mask_tokens(
                ids,
                mask,
                mask_token_id=MASK_ID,
                vocabulary_size=20,
                protected_ids=(),
                probability=0.5,
                generator=self._generator(),
            )

        first_ids, first_labels = run()

        second_ids, second_labels = run()

        assert torch.equal(first_ids, second_ids)

        assert torch.equal(first_labels, second_labels)

    def test_the_eighty_ten_ten_split(self) -> None:
        """
        Of the selected tokens, roughly 80% become the mask, 10% a random
        token, 10% are left intact — the split that stops the model only
        ever seeing the mask id at positions it must predict.
        """

        ids = torch.full((200, 40), CONTENT_OFFSET + 1)

        mask = torch.ones_like(ids)

        corrupted, labels = mask_tokens(
            ids,
            mask,
            mask_token_id=MASK_ID,
            vocabulary_size=64,
            protected_ids=(),
            probability=0.5,
            generator=self._generator(),
        )

        selected = labels != IGNORE_INDEX

        total = int(selected.sum())

        became_mask = int((selected & (corrupted == MASK_ID)).sum())

        stayed = int((selected & (corrupted == ids)).sum())

        assert became_mask / total == pytest.approx(0.8, abs=0.05)

        assert stayed / total == pytest.approx(0.1, abs=0.05)

    @pytest.mark.parametrize("probability", [0.0, 1.0, -0.1, 1.5])
    def test_an_out_of_range_probability_is_rejected(self, probability: float) -> None:
        with pytest.raises(ValidationError, match="probability"):
            mask_tokens(
                torch.ones((2, 3), dtype=torch.long),
                torch.ones((2, 3), dtype=torch.long),
                mask_token_id=MASK_ID,
                vocabulary_size=20,
                protected_ids=(),
                probability=probability,
                generator=self._generator(),
            )


class TestTheModel:
    def test_forward_returns_a_distribution_per_token(self) -> None:
        model = TransformerEncoderModel(
            EncoderConfig(vocabulary_size=30, dimension=DIMENSION, layers=2)
        )

        mlm = MaskedLanguageModel(model)

        ids = torch.randint(1, 30, (2, 6))

        mask = torch.ones((2, 6), dtype=torch.long)

        assert mlm(ids, mask).shape == (2, 6, 30)

    def test_tying_shares_the_embedding_matrix(self) -> None:
        """
        The decoder and the embedding table are one tensor when tied, so a
        gradient to either shapes both — the point of tying.
        """

        model = TransformerEncoderModel(
            EncoderConfig(vocabulary_size=30, dimension=DIMENSION, layers=1)
        )

        tied = MaskedLanguageModel(model, tie_weights=True)

        assert tied.head.decoder.weight is model.token_embedding.weight

        untied = MaskedLanguageModel(model, tie_weights=False)

        assert untied.head.decoder.weight is not model.token_embedding.weight


class TestItActuallyLearns:
    """
    The tests that matter: the embedding must learn to use context.
    """

    def test_masked_accuracy_beats_the_unigram_baseline(self) -> None:
        """
        The whole justification for the stage. A model betting on the
        frequent token scores the baseline; one that has learned the
        template structure scores far higher, and the gap is the evidence
        the embedding became meaningful.
        """

        encoder = build_encoder()

        report = MaskedLanguageModelTrainer(
            encoder,
            mask_token_id=MASK_ID,
            config=MaskedLanguageConfig(
                epochs=10, batch_size=16, learning_rate=3e-3, seed=1
            ),
        ).train(build_corpus())

        assert report.improved, f"loss did not fall: {report.losses}"

        assert report.final_accuracy > report.initial_accuracy

        baseline = unigram_baseline()

        assert report.final_accuracy > baseline + 0.2, (
            f"final={report.final_accuracy:.3f} baseline={baseline:.3f}"
        )

    def test_pretraining_is_reproducible(self) -> None:
        corpus = build_corpus()

        def run() -> list[float]:
            encoder = build_encoder()

            return (
                MaskedLanguageModelTrainer(
                    encoder,
                    mask_token_id=MASK_ID,
                    config=MaskedLanguageConfig(epochs=3, batch_size=16, seed=7),
                )
                .train(corpus)
                .losses
            )

        assert run() == run()

    def test_the_pretrained_encoder_feeds_the_contrastive_stage(self) -> None:
        """
        The two stages share one encoder. Pretraining modifies it in
        place, so the same object hands straight to the contrastive
        trainer — this is the from-scratch recipe end to end, in
        miniature.
        """

        encoder = build_encoder()

        MaskedLanguageModelTrainer(
            encoder,
            mask_token_id=MASK_ID,
            config=MaskedLanguageConfig(epochs=4, batch_size=16, learning_rate=3e-3, seed=1),
        ).train(build_corpus())

        pairs = [
            TextPair("alpha bravo", "charlie delta echo"),
            TextPair("one two", "three four five"),
        ] * 8

        report = ContrastiveTrainer(
            encoder,
            ContrastiveConfig(epochs=4, batch_size=8, learning_rate=3e-3, seed=1),
        ).train(pairs)

        assert report.steps > 0


class TestConfigAndReport:
    def test_no_documents_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at least one document"):
            MaskedLanguageModelTrainer(build_encoder(), mask_token_id=MASK_ID).train([])

    @pytest.mark.parametrize("probability", [0.0, 1.0, 1.2])
    def test_an_out_of_range_mask_probability_is_rejected(self, probability: float) -> None:
        with pytest.raises(ValidationError, match="mask_probability"):
            MaskedLanguageConfig(mask_probability=probability)

    def test_an_invalid_precision_is_caught_when_the_config_is_built(self) -> None:
        with pytest.raises(ValidationError):
            MaskedLanguageConfig(precision="fp16")

    def test_a_single_epoch_run_is_unmeasurable(self) -> None:
        report = MaskedLanguageModelTrainer(
            build_encoder(),
            mask_token_id=MASK_ID,
            config=MaskedLanguageConfig(epochs=1, batch_size=16, seed=1),
        ).train(build_corpus())

        assert not report.measurable

        assert not report.improved

    def test_the_report_serialises(self) -> None:
        report = MaskedLanguageModelTrainer(
            build_encoder(),
            mask_token_id=MASK_ID,
            config=MaskedLanguageConfig(epochs=2, batch_size=16, seed=1),
        ).train(build_corpus())

        payload = json.loads(json.dumps(report.to_dict()))

        assert payload["documents"] == len(build_corpus())

        assert len(payload["accuracies"]) == 2
