"""
The contextual encoder, and the proof that it learns.

The decisive tests are in :class:`TestItActuallyLearns`. Everything else
here checks shapes and contracts, which a model that has learned nothing
would pass just as easily. A transformer that emits well-formed vectors
containing no information is the default failure mode of a hand-written
training loop, and only a retrieval test catches it.

The model used throughout is deliberately tiny — two layers, 32
dimensions — so the whole module runs in seconds on CPU. That is the
point: the same code path trains a real model, differing only in
configuration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="requires the neural extra")

from multilingual_embedding.core.exceptions import ValidationError  # noqa: E402
from multilingual_embedding.embedding.encoder import TextEncoder  # noqa: E402
from multilingual_embedding.embedding.neural import (  # noqa: E402
    ContrastiveConfig,
    ContrastiveTrainer,
    EncoderConfig,
    NeuralTextEncoder,
    TextPair,
    TransformerEncoderModel,
)
from multilingual_embedding.pipelines.search import SemanticSearchPipeline  # noqa: E402

VOCABULARY = 64

DIMENSION = 32


class WordTokenizer:
    """
    A hashing tokenizer, so these tests need no trained subword model.

    Deterministic, and that is what matters: the same word always maps to
    the same id, which is all the encoder requires.
    """

    def __init__(self, vocabulary_size: int = VOCABULARY) -> None:
        self.vocabulary_size = vocabulary_size

    def encode(self, text: str) -> WordTokenizer._Encoding:
        ids = [
            # 0 is padding, so ids start at 1.
            1 + (abs(hash(word)) % (self.vocabulary_size - 1))
            for word in text.split()
        ]

        return WordTokenizer._Encoding(ids)

    class _Encoding:
        def __init__(self, ids: list[int]) -> None:
            self.ids = ids


def build_encoder(**overrides: object) -> NeuralTextEncoder:
    """A small encoder on CPU, for speed and determinism."""

    settings: dict[str, object] = {
        "vocabulary_size": VOCABULARY,
        "dimension": DIMENSION,
        "layers": 2,
        "heads": 4,
        "max_length": 32,
        "dropout": 0.0,
    }

    settings.update(overrides)

    torch.manual_seed(0)

    model = TransformerEncoderModel(EncoderConfig(**settings))  # type: ignore[arg-type]

    return NeuralTextEncoder(model, WordTokenizer(), device="cpu")


class TestArchitecture:
    def test_rejects_dimension_not_divisible_by_heads(self) -> None:
        with pytest.raises(ValidationError, match="divide evenly"):
            EncoderConfig(vocabulary_size=10, dimension=10, heads=4)

    def test_feedforward_defaults_to_four_times_width(self) -> None:
        assert EncoderConfig(vocabulary_size=10, dimension=64).feedforward_dimension == 256

    def test_forward_returns_one_vector_per_sequence(self) -> None:
        model = TransformerEncoderModel(
            EncoderConfig(vocabulary_size=VOCABULARY, dimension=DIMENSION, layers=1)
        )

        ids = torch.randint(1, VOCABULARY, (3, 7))

        mask = torch.ones((3, 7), dtype=torch.long)

        assert model(ids, mask).shape == (3, DIMENSION)

    def test_sequence_longer_than_position_table_is_rejected(self) -> None:
        model = TransformerEncoderModel(
            EncoderConfig(vocabulary_size=VOCABULARY, dimension=DIMENSION, max_length=8)
        )

        with pytest.raises(ValidationError, match="position table"):
            model(torch.ones((1, 9), dtype=torch.long), torch.ones((1, 9), dtype=torch.long))

    def test_padding_does_not_change_a_sequence_encoding(self) -> None:
        """
        The property that makes batching safe.

        If padding leaked into attention or pooling, a sentence would
        encode differently depending on what happened to be batched
        alongside it — a bug that survives every shape test and quietly
        degrades retrieval.
        """

        model = TransformerEncoderModel(
            EncoderConfig(vocabulary_size=VOCABULARY, dimension=DIMENSION, layers=2, dropout=0.0)
        ).eval()

        ids = torch.tensor([[5, 9, 13]])

        mask = torch.ones((1, 3), dtype=torch.long)

        padded_ids = torch.tensor([[5, 9, 13, 0, 0]])

        padded_mask = torch.tensor([[1, 1, 1, 0, 0]])

        with torch.no_grad():
            plain = model(ids, mask)

            padded = model(padded_ids, padded_mask)

        torch.testing.assert_close(plain, padded, rtol=1e-4, atol=1e-5)

    def test_parameter_count_is_reported(self) -> None:
        model = TransformerEncoderModel(
            EncoderConfig(vocabulary_size=VOCABULARY, dimension=DIMENSION, layers=2)
        )

        assert model.parameter_count() > 0


class TestEncoderContract:
    def test_satisfies_the_framework_contract(self) -> None:
        assert isinstance(build_encoder(), TextEncoder)

    def test_encode_returns_a_vector_of_declared_width(self) -> None:
        assert build_encoder().encode("hello world").shape == (DIMENSION,)

    def test_encode_batch_shape_and_order(self) -> None:
        encoder = build_encoder()

        texts = ["alpha beta", "gamma delta", "epsilon"]

        batch = encoder.encode_batch(texts)

        assert batch.shape == (3, DIMENSION)

        for row, text in enumerate(texts):
            np.testing.assert_allclose(batch[row], encoder.encode(text), rtol=1e-5, atol=1e-6)

    def test_empty_batch_keeps_its_shape(self) -> None:
        assert build_encoder().encode_batch([]).shape == (0, DIMENSION)

    def test_empty_text_gives_zeros_not_nan(self) -> None:
        """The contract's promise for unencodable input."""

        vector = build_encoder().encode("")

        assert not np.isnan(vector).any()

        assert not np.any(vector)

    def test_output_is_l2_normalised(self) -> None:
        vector = build_encoder().encode("some words here")

        assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-5)

    def test_overlong_input_is_truncated_not_rejected(self) -> None:
        """
        A caller encoding a long document should get a vector for its
        opening rather than an exception.
        """

        encoder = build_encoder(max_length=8)

        assert encoder.encode(" ".join(["word"] * 200)).shape == (DIMENSION,)

    def test_batching_does_not_change_results(self) -> None:
        texts = [f"sentence number {index}" for index in range(7)]

        large = build_encoder()

        small = build_encoder()

        small._batch_size = 2

        np.testing.assert_allclose(
            large.encode_batch(texts), small.encode_batch(texts), rtol=1e-5, atol=1e-6
        )


class TestItActuallyLearns:
    """
    The tests that matter.

    A model producing correctly-shaped vectors that encode nothing passes
    every other test in this module. These are the ones that would fail.
    """

    @staticmethod
    def _topic_pairs() -> list[TextPair]:
        """
        Pairs drawn from two disjoint vocabularies.

        No word appears in both topics, so a model that has learned the
        structure must place each topic's texts together and apart from
        the other's.
        """

        weather = ["rain", "storm", "cloud", "wind", "thunder", "drizzle"]

        finance = ["bank", "loan", "credit", "market", "invoice", "ledger"]

        pairs: list[TextPair] = []

        for index in range(24):
            first, second = weather[index % 6], weather[(index + 3) % 6]

            pairs.append(TextPair(f"{first} {second}", f"{second} {first} today"))

            third, fourth = finance[index % 6], finance[(index + 3) % 6]

            pairs.append(TextPair(f"{third} {fourth}", f"{fourth} {third} today"))

        return pairs

    def test_loss_decreases(self) -> None:
        encoder = build_encoder()

        report = ContrastiveTrainer(
            encoder,
            ContrastiveConfig(epochs=6, batch_size=8, learning_rate=3e-3, seed=1),
        ).train(self._topic_pairs())

        assert report.improved, f"loss did not fall: {report.losses}"

        assert report.steps > 0

    def test_training_separates_the_two_topics(self) -> None:
        """
        The retrieval property, which is what the model is for.

        Within-topic similarity must exceed cross-topic similarity after
        training. This is the assertion a model that learned nothing
        cannot pass.
        """

        encoder = build_encoder()

        ContrastiveTrainer(
            encoder,
            ContrastiveConfig(epochs=8, batch_size=8, learning_rate=3e-3, seed=1),
        ).train(self._topic_pairs())

        weather = encoder.encode_batch(["rain storm", "cloud wind", "thunder drizzle"])

        finance = encoder.encode_batch(["bank loan", "credit market", "invoice ledger"])

        within = float(np.mean(weather @ weather.T)) + float(np.mean(finance @ finance.T))

        across = float(np.mean(weather @ finance.T)) * 2

        assert within > across, f"within={within / 2:.3f} across={across / 2:.3f}"

    def test_training_is_reproducible(self) -> None:
        pairs = self._topic_pairs()

        def run() -> list[float]:
            encoder = build_encoder()

            return (
                ContrastiveTrainer(encoder, ContrastiveConfig(epochs=2, batch_size=8, seed=7))
                .train(pairs)
                .losses
            )

        assert run() == run()

    def test_too_few_pairs_is_rejected(self) -> None:
        """
        A batch of one has no negatives, so its loss is identically zero.

        Training would appear to succeed while teaching nothing, which is
        worse than failing.
        """

        with pytest.raises(ValidationError, match="at least two pairs"):
            ContrastiveTrainer(build_encoder()).train([TextPair("a", "b")])

    def test_report_serialises(self) -> None:
        import json

        report = ContrastiveTrainer(
            build_encoder(), ContrastiveConfig(epochs=1, batch_size=8)
        ).train(self._topic_pairs())

        json.dumps(report.to_dict())


class TestPersistence:
    def test_round_trip_preserves_vectors(self, tmp_path: Path) -> None:
        encoder = build_encoder()

        text = "round trip check"

        before = encoder.encode(text)

        encoder.save(tmp_path / "encoder")

        restored = NeuralTextEncoder.load(tmp_path / "encoder", WordTokenizer(), device="cpu")

        np.testing.assert_allclose(restored.encode(text), before, rtol=1e-5, atol=1e-6)

    def test_round_trip_after_training(self, tmp_path: Path) -> None:
        """Trained weights, not just initial ones, must survive."""

        encoder = build_encoder()

        ContrastiveTrainer(encoder, ContrastiveConfig(epochs=2, batch_size=8, seed=3)).train(
            TestItActuallyLearns._topic_pairs()
        )

        before = encoder.encode("rain storm")

        encoder.save(tmp_path / "trained")

        restored = NeuralTextEncoder.load(tmp_path / "trained", WordTokenizer(), device="cpu")

        np.testing.assert_allclose(restored.encode("rain storm"), before, rtol=1e-5, atol=1e-6)

    def test_version_mismatch_is_rejected(self, tmp_path: Path) -> None:
        from multilingual_embedding.utils.io import read_json, write_json

        encoder = build_encoder()

        encoder.save(tmp_path / "encoder")

        payload = read_json(tmp_path / "encoder" / "encoder.json")

        payload["format_version"] = 999

        write_json(tmp_path / "encoder" / "encoder.json", payload)

        with pytest.raises(ValidationError, match="format version"):
            NeuralTextEncoder.load(tmp_path / "encoder", WordTokenizer(), device="cpu")


class TestServesThroughTheExistingPipeline:
    """
    Phase 0's exit criterion, now met by a real model.

    The search pipeline was decoupled from the embedding matrix so that a
    contextual encoder could be served without rewriting it. This is that
    claim tested against an actual transformer rather than a stub.
    """

    def test_pipeline_accepts_the_neural_encoder(self) -> None:
        pipeline = SemanticSearchPipeline(build_encoder())

        assert pipeline.matrix is None

        assert pipeline.encoder.dimension == DIMENSION

    def test_search_returns_ranked_results(self) -> None:
        pipeline = SemanticSearchPipeline(build_encoder())

        corpus = ["rain storm today", "bank loan today", "cloud wind today"]

        assert pipeline.index(corpus) == 3

        hits = pipeline.search("rain storm today", top_k=3)

        assert [hit.rank for hit in hits] == [1, 2, 3]

        assert all(hits[index].score >= hits[index + 1].score for index in range(len(hits) - 1))

    def test_exact_query_ranks_itself_first(self) -> None:
        pipeline = SemanticSearchPipeline(build_encoder())

        pipeline.index(["rain storm today", "bank loan today", "cloud wind today"])

        assert pipeline.search("bank loan today", top_k=1)[0].text == "bank loan today"


class TestPrecision:
    """
    Mixed precision, and the join between a machine profile and a run.

    Everything here executes on CPU. bf16 *autocast* is exercised
    genuinely — the ops really do run in bfloat16 — but CUDA kernel
    selection and the speed and memory claims that motivate bf16 are not
    reachable from this machine and remain unverified until a run happens
    on the GPU box.
    """

    def _pairs(self) -> list[TextPair]:
        words = ["rain", "storm", "cloud", "wind", "thunder", "drizzle"]

        return [
            TextPair(f"{words[i % 6]} {words[(i + 2) % 6]}", f"{words[(i + 2) % 6]} today")
            for i in range(16)
        ]

    def test_autocast_selects_bfloat16_on_cpu(self) -> None:
        """The context is real, not a no-op that quietly does nothing."""

        from multilingual_embedding.embedding.neural.encoder import autocast_for

        with autocast_for(torch.device("cpu"), "bf16"):
            product = torch.randn(4, 4) @ torch.randn(4, 4)

        assert product.dtype is torch.bfloat16

    def test_fp32_context_is_inert(self) -> None:
        from multilingual_embedding.embedding.neural.encoder import autocast_for

        with autocast_for(torch.device("cpu"), "fp32"):
            product = torch.randn(4, 4) @ torch.randn(4, 4)

        assert product.dtype is torch.float32

    def test_autocast_state_is_restored_after_the_block(self) -> None:
        """
        One context object is reused for every step of a run rather than
        rebuilt, so it has to leave the global autocast state as it found
        it. A leak here would silently put inference in bf16 too.
        """

        from multilingual_embedding.embedding.neural.encoder import autocast_for

        context = autocast_for(torch.device("cpu"), "bf16")

        for _ in range(3):
            with context:
                assert torch.is_autocast_cpu_enabled()

            assert not torch.is_autocast_cpu_enabled()

    def test_bf16_training_still_learns(self) -> None:
        """
        The claim that matters. bf16 trades mantissa bits, and the
        question is whether the loss still falls — if it did not, the GPU
        profile would train a worse model than the laptop one and the
        whole split would be unsound.
        """

        encoder = build_encoder()

        # Load-bearing, and easy to lose. On Metal this trainer falls back
        # to fp32 by design, so an encoder that resolved to MPS would make
        # this a second fp32 test wearing a bf16 name.
        assert encoder.device.type == "cpu", "this test only means anything off Metal"

        report = ContrastiveTrainer(
            encoder,
            ContrastiveConfig(epochs=6, batch_size=8, learning_rate=3e-3, seed=1, precision="bf16"),
        ).train(self._pairs())

        assert report.improved, f"bf16 loss did not fall: {report.losses}"

    def test_bf16_is_requested_but_ignored_on_metal(self) -> None:
        """
        Ignoring it is deliberate. Metal's autocast support is uneven, and
        falling back loudly beats training in a precision nobody chose.
        """

        from multilingual_embedding.embedding.neural.encoder import autocast_for

        context = autocast_for(torch.device("mps"), "bf16")

        with context:
            product = torch.randn(4, 4) @ torch.randn(4, 4)

        assert product.dtype is torch.float32

    def test_for_compute_joins_machine_to_experiment(self) -> None:
        """
        The machine supplies batch size and precision; the experiment
        supplies everything that determines the result.
        """

        from multilingual_embedding.config.base import ComputeConfig

        config = ContrastiveConfig.for_compute(
            ComputeConfig(batch_size=256, precision="bf16"),
            epochs=9,
            temperature=0.02,
        )

        assert config.batch_size == 256

        assert config.precision == "bf16"

        assert config.epochs == 9

        assert config.temperature == 0.02

    def test_an_invalid_precision_is_caught_when_the_config_is_built(self) -> None:
        with pytest.raises(ValidationError):
            ContrastiveConfig(precision="fp16")


class TestGradientCachingInTheTrainer:
    """
    The trainer's use of gradient caching, which is what makes the GPU
    profile's batch size reachable.

    Worth its own tests because the wiring is easy to get subtly wrong in
    ways that still train: zeroing gradients at the wrong point, or
    normalising on the wrong side of the cache.
    """

    def _pairs(self) -> list[TextPair]:
        words = ["rain", "storm", "cloud", "wind", "thunder", "drizzle"]

        return [
            TextPair(f"{words[i % 6]} {words[(i + 2) % 6]}", f"{words[(i + 2) % 6]} today")
            for i in range(16)
        ]

    def test_caching_trains_the_same_model_as_not_caching(self) -> None:
        """
        The claim gradient caching makes: same result, less memory.

        Dropout is off here deliberately. Chunked encoding draws
        different dropout masks than unchunked — eight rows in one call
        is not eight calls of one row — so with dropout on, the two runs
        could not agree however correct the implementation. That property
        is covered separately, against a reference that shares the
        chunking.
        """

        pairs = self._pairs()

        def train(chunk: int) -> list[float]:
            encoder = build_encoder(dropout=0.0)

            report = ContrastiveTrainer(
                encoder,
                ContrastiveConfig(
                    epochs=3,
                    batch_size=8,
                    learning_rate=3e-3,
                    seed=1,
                    gradient_checkpoint_chunk=chunk,
                ),
            ).train(pairs)

            return report.losses

        plain = train(0)

        cached = train(4)

        for epoch, (expected, found) in enumerate(zip(plain, cached, strict=True)):
            assert found == pytest.approx(expected, abs=1e-4), (
                f"epoch {epoch}: caching changed the loss, {expected} vs {found}"
            )

    def test_caching_still_learns(self) -> None:
        encoder = build_encoder(dropout=0.0)

        report = ContrastiveTrainer(
            encoder,
            ContrastiveConfig(
                epochs=6,
                batch_size=8,
                learning_rate=3e-3,
                seed=1,
                gradient_checkpoint_chunk=2,
            ),
        ).train(self._pairs())

        assert report.improved, f"cached loss did not fall: {report.losses}"

    def test_a_chunk_larger_than_the_batch_is_harmless(self) -> None:
        """Degenerates to a single chunk rather than failing."""

        encoder = build_encoder(dropout=0.0)

        report = ContrastiveTrainer(
            encoder,
            ContrastiveConfig(
                epochs=2,
                batch_size=8,
                learning_rate=3e-3,
                seed=1,
                gradient_checkpoint_chunk=1024,
            ),
        ).train(self._pairs())

        assert report.steps > 0


class TestLossIsNotEvidenceOnItsOwn:
    """
    `improved` compares the first epoch's loss with the last, so for a
    single epoch it compares a value with itself.

    A real adaptation run reported `loss: [1.17487, 1.17487]` and
    `improved: False` while its retrieval score rose 20%. Silently
    returning False there invites the opposite conclusion from the true
    one.
    """

    def _pairs(self) -> list[TextPair]:
        words = ["rain", "storm", "cloud", "wind", "thunder", "drizzle"]

        return [
            TextPair(f"{words[i % 6]} {words[(i + 2) % 6]}", f"{words[(i + 2) % 6]} today")
            for i in range(16)
        ]

    def test_a_single_epoch_run_is_marked_unmeasurable(self) -> None:
        report = ContrastiveTrainer(
            build_encoder(dropout=0.0),
            ContrastiveConfig(epochs=1, batch_size=8, learning_rate=3e-3, seed=1),
        ).train(self._pairs())

        assert not report.measurable

        assert report.initial_loss == report.final_loss

        assert not report.improved, "improved cannot be true when nothing was compared"

    def test_a_multi_epoch_run_is_measurable(self) -> None:
        report = ContrastiveTrainer(
            build_encoder(dropout=0.0),
            ContrastiveConfig(epochs=4, batch_size=8, learning_rate=3e-3, seed=1),
        ).train(self._pairs())

        assert report.measurable

        assert report.improved

    def test_the_distinction_is_serialised(self) -> None:
        """A report read later must carry it too."""

        report = ContrastiveTrainer(
            build_encoder(dropout=0.0),
            ContrastiveConfig(epochs=1, batch_size=8, learning_rate=3e-3, seed=1),
        ).train(self._pairs())

        assert report.to_dict()["measurable"] is False
