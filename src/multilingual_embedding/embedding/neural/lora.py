"""
Low-rank adaptation.

Fine-tuning every weight of a 568M-parameter model needs roughly 8.5 GB
for the weights, their gradients and the optimizer's two moment
estimates. That leaves too little of a 16 GB card for activations, and
activations are where contrastive training actually needs room.

LoRA sidesteps this. A weight update learned by fine-tuning turns out to
be well approximated by a low-rank matrix, so instead of learning the
full ``ΔW`` we learn ``B @ A`` where ``A`` is ``rank × in`` and ``B`` is
``out × rank``. With rank 16 against a 768-wide layer that is 25k
parameters instead of 590k. The base weights are frozen, so they need no
gradients and no optimizer state at all.

The scaling factor is ``alpha / rank``. It exists so that changing the
rank does not change the effective learning rate of the adapter: doubling
the rank halves the scale, and the update keeps roughly the same
magnitude.

**The initialisation is load-bearing.** ``A`` is random and ``B`` is
zero, so ``B @ A`` is exactly zero when training starts and the adapted
model is numerically identical to the base. Were both random, the model
would begin from a corrupted version of its pretrained weights and the
first optimizer steps would be spent undoing the damage. There is a test
for this property.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from multilingual_embedding.core.exceptions import ValidationError
from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.utils.validation import require_positive

__all__ = [
    "LoRAConfig",
    "LoRALinear",
    "apply_lora",
    "load_lora_state_dict",
    "lora_state_dict",
    "merge_lora",
    "parameter_summary",
]

_logger = get_logger(__name__)

# Attention projections are the conventional target. The feed-forward
# layers can be added, at roughly triple the adapter cost for a usually
# modest gain.
DEFAULT_TARGETS: tuple[str, ...] = ("qkv", "projection")


@dataclass(slots=True)
class LoRAConfig:
    """
    Low-rank adaptation settings.

    Attributes
    ----------
    rank:
        Inner dimension of the adapter. 8 to 32 covers most cases; higher
        buys capacity at linear cost in adapter size.

    alpha:
        Scaling numerator. The effective scale is ``alpha / rank``.
        Setting it equal to the rank gives a scale of 1.

    dropout:
        Applied to the adapter input, not the base path.

    targets:
        Substrings matched against module names. A linear layer whose
        name contains any of them is adapted.
    """

    rank: int = 16

    alpha: int = 16

    dropout: float = 0.0

    targets: tuple[str, ...] = DEFAULT_TARGETS

    def __post_init__(self) -> None:
        require_positive(self.rank, name="rank")

        require_positive(self.alpha, name="alpha")

        if not self.targets:
            raise ValidationError("LoRA needs at least one target module name")

    @property
    def scaling(self) -> float:
        """The factor applied to the adapter's contribution."""

        return self.alpha / self.rank


class LoRALinear(nn.Module):
    """
    A frozen linear layer with a trainable low-rank adapter beside it.

    The forward pass is ``base(x) + scaling * B(A(dropout(x)))``. The base
    layer's parameters are frozen in place rather than copied, so wrapping
    costs no additional memory for the weights themselves.
    """

    def __init__(self, base: nn.Linear, config: LoRAConfig) -> None:
        super().__init__()

        self.base = base

        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

        self.rank = config.rank

        self.scaling = config.scaling

        self.lora_down = nn.Linear(base.in_features, config.rank, bias=False)

        self.lora_up = nn.Linear(config.rank, base.out_features, bias=False)

        self.dropout = nn.Dropout(config.dropout) if config.dropout else nn.Identity()

        # A is drawn from the Kaiming-uniform scheme a linear layer would
        # normally use; B is zeroed so the product starts at exactly zero
        # and the wrapped layer is initially indistinguishable from the
        # base it wraps.
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))

        nn.init.zeros_(self.lora_up.weight)

    @property
    def in_features(self) -> int:
        """Input width of the wrapped layer."""

        return int(self.base.in_features)

    @property
    def out_features(self) -> int:
        """Output width of the wrapped layer."""

        return int(self.base.out_features)

    def forward(self, hidden: Tensor) -> Tensor:
        base_output: Tensor = self.base(hidden)

        adapter: Tensor = self.lora_up(self.lora_down(self.dropout(hidden)))

        combined: Tensor = base_output + self.scaling * adapter

        return combined

    def merged_weight(self) -> Tensor:
        """
        The base weight with the adapter folded in.

        Used to collapse the adapter for inference, after which the layer
        costs exactly what an unadapted one would.
        """

        delta = self.lora_up.weight @ self.lora_down.weight

        return self.base.weight + self.scaling * delta

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, scaling={self.scaling:.3f}"
        )


def apply_lora(model: nn.Module, config: LoRAConfig | None = None) -> int:
    """
    Freeze a model and attach adapters to its targeted linear layers.

    Every parameter is frozen first, then adapters are added — so anything
    not explicitly adapted stays frozen, rather than relying on the
    caller to have frozen it.

    Parameters
    ----------
    model:
        Modified in place.

    config:
        Adapter settings. Defaults apply rank 16 to attention projections.

    Returns
    -------
    Number of layers adapted.

    Raises
    ------
    ValidationError
        If no layer matched, which almost always means the target names
        are wrong for this architecture and would otherwise present as a
        model that silently refuses to learn.
    """

    settings = config if config is not None else LoRAConfig()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    adapted = _replace_targets(model, settings, prefix="")

    if adapted == 0:
        raise ValidationError(
            "LoRA matched no layers; check the target names against this model",
            targets=list(settings.targets),
            available=sorted(
                {
                    name.rsplit(".", 1)[-1]
                    for name, module in model.named_modules()
                    if isinstance(module, nn.Linear)
                }
            ),
        )

    summary = parameter_summary(model)

    _logger.info(
        "Applied LoRA",
        extra={
            "layers": adapted,
            "rank": settings.rank,
            "trainable": summary["trainable"],
            "total": summary["total"],
            "trainable_share": summary["trainable_share"],
        },
    )

    return adapted


def _replace_targets(module: nn.Module, config: LoRAConfig, prefix: str) -> int:
    """Recursively swap matching linear children for adapted ones."""

    replaced = 0

    for name, child in list(module.named_children()):
        qualified = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Linear) and any(target in qualified for target in config.targets):
            setattr(module, name, LoRALinear(child, config))

            replaced += 1
        else:
            replaced += _replace_targets(child, config, qualified)

    return replaced


def parameter_summary(model: nn.Module) -> dict[str, float]:
    """
    Count trainable against total parameters.

    The share is the number worth watching: if it is not a small
    percentage, the freeze did not take and the memory saving that
    justifies LoRA has been lost.
    """

    total = sum(p.numel() for p in model.parameters())

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "trainable_share": round(trainable / total, 6) if total else 0.0,
    }


def lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """
    Extract only the adapter weights.

    An adapter checkpoint is a few megabytes against several hundred for
    a full model, so many domain adaptations of one base can be stored
    and swapped cheaply. That is the practical reason to prefer LoRA even
    where memory is not the constraint.
    """

    # Selected by walking for LoRALinear instances rather than by matching
    # names. A base module may legitimately contain layers called `up` or
    # `down` -- the feed-forward block does -- and a name filter would pull
    # those base weights into what is supposed to be an adapter-only file.
    adapter_keys: set[str] = set()

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            adapter_keys.add(f"{name}.lora_down.weight")

            adapter_keys.add(f"{name}.lora_up.weight")

    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.state_dict().items()
        if name in adapter_keys
    }


def load_lora_state_dict(model: nn.Module, state: dict[str, Tensor]) -> None:
    """
    Load adapter weights into an already-adapted model.

    Raises
    ------
    ValidationError
        If the model has no adapters, since loading would otherwise
        silently do nothing and leave the base model in place.
    """

    if not any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValidationError("Model has no LoRA adapters to load into")

    missing, unexpected = model.load_state_dict(state, strict=False)

    if unexpected:
        raise ValidationError(
            "Adapter checkpoint contains unknown keys",
            unexpected=sorted(unexpected)[:5],
        )

    # Missing keys are expected and correct: the checkpoint deliberately
    # holds adapters only, so every base weight is absent from it.
    _logger.debug("Loaded adapters", extra={"tensors": len(state), "base_keys": len(missing)})


def merge_lora(model: nn.Module) -> int:
    """
    Fold every adapter into its base weight and remove it.

    After merging the model is an ordinary network with no adapter
    overhead at inference. The operation is exact rather than an
    approximation — the adapter was only ever an additive term.

    Returns
    -------
    Number of layers merged.
    """

    merged = 0

    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                with torch.no_grad():
                    child.base.weight.copy_(child.merged_weight())

                child.base.weight.requires_grad_(True)

                if child.base.bias is not None:
                    child.base.bias.requires_grad_(True)

                setattr(module, name, child.base)

                merged += 1

    _logger.info("Merged LoRA adapters", extra={"layers": merged})

    return merged
