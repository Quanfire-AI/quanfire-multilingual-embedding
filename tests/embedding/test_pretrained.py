"""
Adapting a published encoder.

The whole point of this module is that a pretrained model becomes
indistinguishable from a native one to everything downstream, so the
tests are largely about *absence of special cases*: the same trainer, the
same LoRA, the same retrieval evaluation, no branch anywhere on which
kind of model is in hand.

Every model here is built from a config rather than downloaded. A test
suite that needs the network is a test suite that fails on a train, and
these assertions are about wiring rather than about any particular
checkpoint's quality.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="requires the neural extra")

pytest.importorskip("transformers", reason="requires the pretrained extra")

from transformers import BertConfig, BertModel, BertTokenizerFast  # noqa: E402

from multilingual_embedding.embedding.encoder import TextEncoder  # noqa: E402
from multilingual_embedding.embedding.neural.pretrained import (  # noqa: E402
    PretrainedEncoderError,
    PretrainedTextEncoder,
)

VOCABULARY = 1000


def tokenizer() -> BertTokenizerFast:
    """A real tokenizer over a throwaway vocabulary, built not fetched."""

    directory = Path(tempfile.mkdtemp())

    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    tokens += [f"tok{index}" for index in range(VOCABULARY - len(tokens))]

    (directory / "vocab.txt").write_text("\n".join(tokens), encoding="utf-8")

    return BertTokenizerFast(vocab_file=str(directory / "vocab.txt"))


def model() -> BertModel:
    torch.manual_seed(0)

    return BertModel(
        BertConfig(
            vocab_size=VOCABULARY,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            max_position_embeddings=64,
        )
    )


def encoder(**options: object) -> PretrainedTextEncoder:
    return PretrainedTextEncoder(model(), tokenizer(), device="cpu", **options)  # type: ignore[arg-type]


class TestItSatisfiesTheFrameworkContract:
    def test_it_is_a_text_encoder(self) -> None:
        """
        The assertion the whole design rests on. If this fails, every
        downstream component needs to know what it is holding.
        """

        assert isinstance(encoder(), TextEncoder)

    def test_dimension_comes_from_the_model_config(self) -> None:
        assert encoder().dimension == 64

    def test_encode_returns_one_vector(self) -> None:
        assert encoder().encode("tok1 tok2").shape == (64,)

    def test_encode_batch_preserves_order(self) -> None:
        subject = encoder()

        texts = ["tok1 tok2", "tok3 tok4", "tok5"]

        batch = subject.encode_batch(texts)

        assert batch.shape == (3, 64)

        for row, text in enumerate(texts):
            np.testing.assert_allclose(batch[row], subject.encode(text), rtol=1e-4, atol=1e-5)

    def test_output_is_l2_normalised(self) -> None:
        vector = encoder().encode("tok1 tok2 tok3")

        assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-5)

    def test_an_empty_batch_keeps_its_shape(self) -> None:
        assert encoder().encode_batch([]).shape == (0, 64)


class TestPooling:
    def test_padding_does_not_change_a_vector(self) -> None:
        """
        The property that makes batching safe. If padding leaked into the
        mean, a text would encode differently depending on what happened
        to be batched beside it — invisible to every shape test, and it
        degrades retrieval.
        """

        subject = encoder(pooling="mean")

        alone = subject.encode("tok1 tok2")

        with_longer_neighbour = subject.encode_batch(
            ["tok1 tok2", "tok3 tok4 tok5 tok6 tok7 tok8 tok9"]
        )[0]

        np.testing.assert_allclose(alone, with_longer_neighbour, rtol=1e-4, atol=1e-5)

    def test_mean_and_cls_are_different(self) -> None:
        """
        Otherwise the setting is decorative, and a checkpoint served with
        the wrong one would be indistinguishable from one served right.
        """

        text = "tok1 tok2 tok3 tok4"

        assert not np.allclose(
            encoder(pooling="mean").encode(text),
            encoder(pooling="cls").encode(text),
            atol=1e-4,
        )

    def test_an_unknown_pooling_strategy_is_refused(self) -> None:
        with pytest.raises(PretrainedEncoderError, match="pooling"):
            encoder(pooling="attention")


class TestNoSpecialCasesDownstream:
    """
    A pretrained model must work with what already exists, unchanged.
    """

    def _pairs(self) -> list[object]:
        from dataclasses import dataclass

        @dataclass
        class Pair:
            anchor: str

            positive: str

            language: str = "hi"

            kind: str = "adjacent"

            overlap: float = 0.2

        return [Pair(f"tok{i} tok{i + 1}", f"tok{i} tok{i + 1} tok{i + 2}") for i in range(1, 60)]

    def test_the_retrieval_evaluation_accepts_it(self) -> None:
        from multilingual_embedding.evaluation.retrieval import evaluate_retrieval

        report = evaluate_retrieval(encoder(), self._pairs())

        assert report.overall.queries > 0

    def test_lora_applies_to_the_upstream_module_names(self) -> None:
        """
        Our own attention fuses q, k and v into one `qkv` projection;
        published encoders keep them separate. The targets are
        configurable precisely so a checkpoint we did not design can be
        adapted, and this is what proves it.
        """

        from multilingual_embedding.embedding.neural.lora import (
            LoRAConfig,
            apply_lora,
            parameter_summary,
        )

        subject = encoder()

        apply_lora(subject._model, LoRAConfig(rank=8, alpha=16, targets=("query", "value")))

        summary = parameter_summary(subject._model)

        assert 0 < summary["trainable"] < summary["total"]

    def test_the_existing_trainer_trains_it_and_leaves_the_base_frozen(self) -> None:
        """
        The claim LoRA makes, checked on someone else's architecture:
        adapters move, base weights do not.
        """

        from multilingual_embedding.embedding.neural import (
            ContrastiveConfig,
            ContrastiveTrainer,
            TextPair,
        )
        from multilingual_embedding.embedding.neural.lora import (
            LoRAConfig,
            apply_lora,
            lora_state_dict,
        )

        subject = encoder()

        apply_lora(subject._model, LoRAConfig(rank=8, alpha=16, targets=("query", "value")))

        adapters_before = {
            name: tensor.clone() for name, tensor in lora_state_dict(subject._model).items()
        }

        query = subject._model.model.encoder.layer[0].attention.self.query

        base_before = query.base.weight.clone()

        ContrastiveTrainer(
            subject,
            ContrastiveConfig(epochs=2, batch_size=16, learning_rate=1e-3, seed=1),
        ).train([TextPair(p.anchor, p.positive) for p in self._pairs()])  # type: ignore[attr-defined]

        adapters_after = lora_state_dict(subject._model)

        moved = max(
            float((adapters_after[name] - tensor).abs().max())
            for name, tensor in adapters_before.items()
        )

        assert moved > 0, "the adapters did not train"

        assert float((query.base.weight - base_before).abs().max()) == 0.0, (
            "LoRA changed the frozen base weights"
        )

    def test_gradient_caching_works_against_it(self) -> None:
        """Chunked encoding must not care whose architecture it is."""

        from multilingual_embedding.embedding.neural import (
            ContrastiveConfig,
            ContrastiveTrainer,
            TextPair,
        )

        report = ContrastiveTrainer(
            encoder(),
            ContrastiveConfig(
                epochs=1,
                batch_size=16,
                learning_rate=1e-4,
                seed=1,
                gradient_checkpoint_chunk=4,
            ),
        ).train([TextPair(p.anchor, p.positive) for p in self._pairs()])  # type: ignore[attr-defined]

        assert report.steps > 0


class TestLoading:
    def test_a_missing_checkpoint_is_reported_clearly(self) -> None:
        """
        With local_files_only, so this never reaches the network however
        it fails.
        """

        with pytest.raises(PretrainedEncoderError, match="Could not load"):
            PretrainedTextEncoder.load(
                "definitely-not-a-real-model-name-9f3a", local_files_only=True
            )

    def test_a_model_without_a_readable_width_is_refused(self) -> None:
        """Better than guessing a dimension and being wrong downstream."""

        from torch import nn

        class Opaque(nn.Module):
            config = object()

        with pytest.raises(PretrainedEncoderError, match="width"):
            PretrainedTextEncoder(Opaque(), tokenizer(), device="cpu")


class TestRevisionPinning:
    """
    A pinned revision is the other half of reproducibility. ``local_files_only``
    stops a run reaching the network; a revision stops a bare name resolving to
    whatever the cache happens to hold. It has to reach the model *and* the
    tokenizer — a checkpoint is the pair, and pinning one while the other floats
    resolves to a different tokenizer build on another machine and encodes the
    same text differently, silently.
    """

    def test_revision_reaches_both_the_model_and_the_tokenizer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import transformers

        captured: dict[str, object] = {}

        built_model = model()

        built_tokenizer = tokenizer()

        def fake_model(name: str, **kwargs: object) -> object:
            captured["model_revision"] = kwargs.get("revision")

            return built_model

        def fake_tokenizer(name: str, **kwargs: object) -> object:
            captured["tokenizer_revision"] = kwargs.get("revision")

            return built_tokenizer

        monkeypatch.setattr(transformers.AutoModel, "from_pretrained", fake_model)

        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake_tokenizer)

        PretrainedTextEncoder.load("any/name", device="cpu", revision="abc123")

        assert captured["model_revision"] == "abc123"

        assert captured["tokenizer_revision"] == "abc123"

    def test_an_unpinned_load_passes_no_revision(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default is unchanged behaviour: name resolves as it always did."""

        import transformers

        captured: dict[str, object] = {}

        built_model = model()

        built_tokenizer = tokenizer()

        def fake_model(name: str, **kwargs: object) -> object:
            captured["model_revision"] = kwargs.get("revision")

            return built_model

        def fake_tokenizer(name: str, **kwargs: object) -> object:
            captured["tokenizer_revision"] = kwargs.get("revision")

            return built_tokenizer

        monkeypatch.setattr(transformers.AutoModel, "from_pretrained", fake_model)

        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake_tokenizer)

        PretrainedTextEncoder.load("any/name", device="cpu")

        assert captured["model_revision"] is None

        assert captured["tokenizer_revision"] is None
