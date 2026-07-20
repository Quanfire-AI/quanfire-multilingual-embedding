"""
Low-rank adaptation and gradient caching.

Two claims here are the ones worth testing, because both are easy to
implement in a way that appears to work and is silently wrong.

**LoRA must be a no-op at initialisation.** If the adapter's up-projection
is not zeroed, the adapted model starts as a corrupted version of its
base and the first optimizer steps go on undoing that damage. The model
still trains, so nothing looks broken — it just starts from a worse place
than it should.

**Gradient caching must be exact.** It is sold as identical to a large
batch, not an approximation of one. If it were merely close, the whole
justification for the extra forward pass would collapse, and the error
would be invisible against the noise of training.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="requires the neural extra")

from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from multilingual_embedding.core.exceptions import ValidationError  # noqa: E402
from multilingual_embedding.embedding.neural import (  # noqa: E402
    ContrastiveConfig,
    ContrastiveTrainer,
    EncoderConfig,
    TransformerEncoderModel,
)
from multilingual_embedding.embedding.neural.gradcache import (  # noqa: E402
    cached_contrastive_backward,
    suggest_chunk_size,
)
from multilingual_embedding.embedding.neural.lora import (  # noqa: E402
    LoRAConfig,
    LoRALinear,
    apply_lora,
    load_lora_state_dict,
    lora_state_dict,
    merge_lora,
    parameter_summary,
)

from .test_neural import VOCABULARY, build_encoder  # noqa: E402

DIMENSION = 32


def build_model(**overrides: object) -> TransformerEncoderModel:
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

    return TransformerEncoderModel(EncoderConfig(**settings))  # type: ignore[arg-type]


class TestLoRAIsANoOpAtInit:
    """
    The property that makes adaptation safe to start from.
    """

    def test_adapted_model_matches_base_exactly(self) -> None:
        base = build_model().eval()

        ids = torch.randint(1, VOCABULARY, (4, 6))

        mask = torch.ones((4, 6), dtype=torch.long)

        with torch.no_grad():
            before = base(ids, mask)

        apply_lora(base, LoRAConfig(rank=8))

        base.eval()

        with torch.no_grad():
            after = base(ids, mask)

        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_up_projection_starts_at_zero(self) -> None:
        """The mechanism behind the property above."""

        model = build_model()

        apply_lora(model, LoRAConfig(rank=4))

        adapters = [m for m in model.modules() if isinstance(m, LoRALinear)]

        assert adapters

        for adapter in adapters:
            assert not torch.any(adapter.lora_up.weight)

            assert torch.any(adapter.lora_down.weight), "down projection should not be zero"


class TestLoRAFreezing:
    def test_only_adapters_are_trainable(self) -> None:
        model = build_model()

        apply_lora(model, LoRAConfig(rank=8))

        for name, parameter in model.named_parameters():
            is_adapter = ".lora_down." in name or ".lora_up." in name

            assert parameter.requires_grad == is_adapter, name

    def test_trainable_share_is_small(self) -> None:
        """
        The number that justifies the technique.

        If this is not a small percentage the freeze did not take, and
        the memory saving LoRA exists for has been lost.
        """

        model = build_model(dimension=128, layers=4)

        apply_lora(model, LoRAConfig(rank=8))

        summary = parameter_summary(model)

        assert summary["trainable_share"] < 0.10, summary

        assert summary["trainable"] > 0

    def test_base_weights_do_not_move_during_training(self) -> None:
        encoder = build_encoder()

        apply_lora(encoder.model, LoRAConfig(rank=8))

        reference = {
            name: parameter.detach().clone()
            for name, parameter in encoder.model.named_parameters()
            if not parameter.requires_grad
        }

        from .test_neural import TestItActuallyLearns

        ContrastiveTrainer(
            encoder, ContrastiveConfig(epochs=2, batch_size=8, learning_rate=1e-2, seed=5)
        ).train(TestItActuallyLearns._topic_pairs())

        for name, original in reference.items():
            current = dict(encoder.model.named_parameters())[name]

            torch.testing.assert_close(current.detach(), original, rtol=0, atol=0)

    def test_unmatched_targets_are_rejected(self) -> None:
        """
        A silent no-match would present as a model that refuses to learn,
        which is a much harder thing to diagnose than an exception.
        """

        with pytest.raises(ValidationError, match="matched no layers"):
            apply_lora(build_model(), LoRAConfig(targets=("nonexistent",)))


class TestLoRALearnsAndMerges:
    def test_adapters_alone_can_learn(self) -> None:
        from .test_neural import TestItActuallyLearns

        encoder = build_encoder()

        apply_lora(encoder.model, LoRAConfig(rank=8))

        report = ContrastiveTrainer(
            encoder, ContrastiveConfig(epochs=6, batch_size=8, learning_rate=1e-2, seed=5)
        ).train(TestItActuallyLearns._topic_pairs())

        assert report.improved, f"LoRA did not learn: {report.losses}"

    def test_merging_preserves_outputs_exactly(self) -> None:
        """
        Merging is algebraic, not approximate: the adapter was only ever
        an additive term on the weight.
        """

        model = build_model()

        apply_lora(model, LoRAConfig(rank=8))

        # Give the adapters non-zero values, or merging a zero adapter
        # would prove nothing.
        for module in model.modules():
            if isinstance(module, LoRALinear):
                nn.init.normal_(module.lora_up.weight, std=0.05)

        model.eval()

        ids = torch.randint(1, VOCABULARY, (3, 5))

        mask = torch.ones((3, 5), dtype=torch.long)

        with torch.no_grad():
            adapted = model(ids, mask)

        merged = merge_lora(model)

        assert merged > 0

        model.eval()

        with torch.no_grad():
            after = model(ids, mask)

        torch.testing.assert_close(adapted, after, rtol=1e-5, atol=1e-6)

    def test_merged_model_has_no_adapters_left(self) -> None:
        model = build_model()

        apply_lora(model, LoRAConfig(rank=4))

        merge_lora(model)

        assert not any(isinstance(m, LoRALinear) for m in model.modules())


class TestAdapterCheckpoints:
    def test_checkpoint_holds_only_adapters(self) -> None:
        """
        The practical reason to prefer LoRA even when memory allows more:
        many domain adaptations of one base, each a few megabytes.
        """

        # rank must be small relative to width for the saving to exist;
        # rank 8 against a 32-wide toy model is 25%, not low rank.
        model = build_model(dimension=128, layers=4)

        apply_lora(model, LoRAConfig(rank=8))

        state = lora_state_dict(model)

        assert state

        assert all(".lora_down." in name or ".lora_up." in name for name in state)

        full = sum(p.numel() for p in model.parameters())

        adapter = sum(t.numel() for t in state.values())

        assert adapter < full * 0.10

    def test_round_trip_restores_behaviour(self) -> None:
        source = build_model()

        apply_lora(source, LoRAConfig(rank=8))

        for module in source.modules():
            if isinstance(module, LoRALinear):
                nn.init.normal_(module.lora_up.weight, std=0.05)

        source.eval()

        ids = torch.randint(1, VOCABULARY, (2, 5))

        mask = torch.ones((2, 5), dtype=torch.long)

        with torch.no_grad():
            expected = source(ids, mask)

        target = build_model()

        apply_lora(target, LoRAConfig(rank=8))

        load_lora_state_dict(target, lora_state_dict(source))

        target.eval()

        with torch.no_grad():
            restored = target(ids, mask)

        torch.testing.assert_close(expected, restored, rtol=1e-5, atol=1e-6)

    def test_loading_into_an_unadapted_model_is_rejected(self) -> None:
        source = build_model()

        apply_lora(source, LoRAConfig(rank=4))

        with pytest.raises(ValidationError, match="no LoRA adapters"):
            load_lora_state_dict(build_model(), lora_state_dict(source))


class TestGradientCachingIsExact:
    """
    The claim that justifies the extra forward pass.

    Gradient caching is advertised as producing gradients identical to a
    single large batch. These tests hold it to that, rather than to being
    approximately right.
    """

    @staticmethod
    def _fixture() -> tuple[TransformerEncoderModel, torch.Tensor, torch.Tensor]:
        model = build_model(layers=1).eval()

        torch.manual_seed(3)

        ids = torch.randint(1, VOCABULARY, (16, 6))

        mask = torch.ones((16, 6), dtype=torch.long)

        return model, ids, mask

    @staticmethod
    def _info_nce(vectors: torch.Tensor) -> torch.Tensor:
        anchors, positives = vectors.chunk(2, dim=0)

        anchors = F.normalize(anchors, dim=-1)

        positives = F.normalize(positives, dim=-1)

        logits = anchors @ positives.T / 0.05

        return F.cross_entropy(logits, torch.arange(anchors.shape[0]))

    def test_gradients_match_a_single_large_batch(self) -> None:
        model, ids, mask = self._fixture()

        # Reference: one unchunked backward pass.
        model.zero_grad(set_to_none=True)

        self._info_nce(model(ids, mask)).backward()

        reference = {
            name: p.grad.detach().clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        }

        # Same computation, chunked through the cache.
        model.zero_grad(set_to_none=True)

        cached_contrastive_backward(
            model,
            range(16),
            lambda chunk: model(ids[list(chunk)], mask[list(chunk)]),
            self._info_nce,
            chunk_size=4,
        )

        assert reference

        for name, expected in reference.items():
            actual = dict(model.named_parameters())[name].grad

            assert actual is not None, name

            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)

    def test_chunk_size_does_not_change_the_result(self) -> None:
        model, ids, mask = self._fixture()

        def gradients_at(chunk_size: int) -> dict[str, torch.Tensor]:
            model.zero_grad(set_to_none=True)

            cached_contrastive_backward(
                model,
                range(16),
                lambda chunk: model(ids[list(chunk)], mask[list(chunk)]),
                self._info_nce,
                chunk_size=chunk_size,
            )

            return {
                name: p.grad.detach().clone()
                for name, p in model.named_parameters()
                if p.grad is not None
            }

        small = gradients_at(2)

        large = gradients_at(8)

        for name, value in small.items():
            torch.testing.assert_close(large[name], value, rtol=1e-4, atol=1e-6)

    def test_returns_the_loss(self) -> None:
        model, ids, mask = self._fixture()

        model.zero_grad(set_to_none=True)

        loss = cached_contrastive_backward(
            model,
            range(16),
            lambda chunk: model(ids[list(chunk)], mask[list(chunk)]),
            self._info_nce,
            chunk_size=4,
        )

        assert loss.ndim == 0

        assert float(loss) > 0.0

    def test_empty_batch_is_rejected(self) -> None:
        model, ids, mask = self._fixture()

        with pytest.raises(ValidationError, match="non-empty batch"):
            cached_contrastive_backward(
                model, [], lambda chunk: model(ids, mask), self._info_nce, chunk_size=4
            )

    def test_non_positive_chunk_size_is_rejected(self) -> None:
        model, ids, mask = self._fixture()

        with pytest.raises(ValidationError):
            cached_contrastive_backward(
                model, range(4), lambda chunk: model(ids, mask), self._info_nce, chunk_size=0
            )


class TestChunkSizeSuggestion:
    def test_caps_at_the_batch_size(self) -> None:
        assert suggest_chunk_size(32, available_gb=100.0, bytes_per_example=1.0) == 32

    def test_shrinks_when_memory_is_tight(self) -> None:
        suggested = suggest_chunk_size(1024, available_gb=1.0, bytes_per_example=50e6)

        assert 1 <= suggested < 1024

    def test_never_returns_zero(self) -> None:
        assert suggest_chunk_size(1024, available_gb=0.001, bytes_per_example=1e12) == 1


class TestTogether:
    """LoRA and gradient caching are meant to be used at the same time."""

    def test_cached_backward_reaches_only_adapter_parameters(self) -> None:
        model = build_model(layers=1)

        apply_lora(model, LoRAConfig(rank=4))

        torch.manual_seed(3)

        ids = torch.randint(1, VOCABULARY, (8, 5))

        mask = torch.ones((8, 5), dtype=torch.long)

        model.zero_grad(set_to_none=True)

        cached_contrastive_backward(
            model,
            range(8),
            lambda chunk: model(ids[list(chunk)], mask[list(chunk)]),
            TestGradientCachingIsExact._info_nce,
            chunk_size=2,
        )

        with_gradients = [n for n, p in model.named_parameters() if p.grad is not None]

        assert with_gradients

        assert all(".lora_down." in n or ".lora_up." in n for n in with_gradients), with_gradients
