"""
Compute profiles.

The premise is that a laptop and a training box run the *same* code and
the same experiment, differing only in a ``compute`` section. These tests
hold that premise up: a GPU profile must be readable on a machine with no
GPU, the two halves of a configuration must not leak into each other, and
the profiles committed to ``configs/`` must actually be valid — a typo
there would otherwise surface on the training box, which is the one place
nobody wants to be debugging YAML.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from multilingual_embedding.config.base import ComputeConfig, ExperimentConfig
from multilingual_embedding.config.loader import load_config
from multilingual_embedding.core.exceptions import ConfigurationError, ValidationError

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


class TestComputeConfig:
    def test_defaults_are_laptop_shaped(self) -> None:
        """
        The default must be the conservative one. A default of bf16 or a
        large batch would fail on the machine least able to say why.
        """

        compute = ComputeConfig()

        assert compute.device == "auto"

        assert compute.precision == "fp32"

        assert compute.gradient_checkpoint_chunk == 0

    def test_cuda_is_accepted_on_a_machine_without_cuda(self) -> None:
        """
        The property the whole approach rests on.

        A GPU profile is written, diffed, validated in CI and committed
        from machines that have no GPU. Validating the device against
        what is present would make that impossible and push the failure
        to the training box.
        """

        assert ComputeConfig(device="cuda").device == "cuda"

        assert ComputeConfig(device="cuda:1").device == "cuda:1"

    def test_unknown_device_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError):
            ComputeConfig(device="tpu")

    def test_unknown_precision_is_rejected(self) -> None:
        """
        fp16 in particular. It is a plausible thing to write and it is
        wrong here — it needs loss scaling, which this trainer does not
        do, so accepting it would train quietly and badly.
        """

        with pytest.raises(ConfigurationError):
            ComputeConfig(precision="fp16")

    def test_nested_dict_is_coerced_and_validated(self) -> None:
        """
        A plain dict is the natural thing to write in a notebook, and it
        must not bypass validation the way a stored dict would.

        The two failure types are the framework's existing split rather
        than an inconsistency here: the precondition helpers raise
        ValidationError, an explicit check raises ConfigurationError, and
        ``load_config`` unifies both at its boundary.
        """

        experiment = ExperimentConfig(compute={"batch_size": 64})

        assert isinstance(experiment.compute, ComputeConfig)

        assert experiment.compute.batch_size == 64

        with pytest.raises(ValidationError):
            ExperimentConfig(compute={"batch_size": 0})

        with pytest.raises(ConfigurationError):
            ExperimentConfig(compute={"precision": "fp8"})

    def test_every_setting_is_actually_read_by_something(self) -> None:
        """
        No dead knobs.

        This section was written with a ``workers`` field that nothing
        consumed — training is single-process and there is no data
        loader — which reads as a tuning knob and silently does nothing.
        The same defect had already been fixed once in EmbeddingConfig
        (see ``test_removed_fields_are_not_silently_accepted``), so this
        pins the rule rather than trusting it to be remembered.

        The check is deliberately crude: each field name must appear
        somewhere in the source outside the config package. That catches
        a field wired to nothing, which is the failure that actually
        happened, without pretending to prove reachability.
        """

        from dataclasses import fields

        import multilingual_embedding

        root = Path(multilingual_embedding.__file__).parent

        sources = [
            path.read_text(encoding="utf-8")
            for path in root.rglob("*.py")
            if "config" not in path.parts
        ]

        for field in fields(ComputeConfig):
            assert any(field.name in text for text in sources), (
                f"ComputeConfig.{field.name} is read by nothing outside the "
                f"config package; either wire it up or remove it"
            )

    def test_survives_a_round_trip(self) -> None:
        """The resolved config is persisted beside the artefacts."""

        original = ExperimentConfig(compute={"precision": "bf16", "batch_size": 128})

        restored = ExperimentConfig.from_dict(original.to_dict())

        assert restored.compute == original.compute


class TestProfileOverlay:
    def test_profile_overrides_only_what_it_names(self, tmp_path: Path) -> None:
        """
        The point of the split. A profile changes the machine and must
        leave the experiment alone, or results stop being comparable
        across the two boxes.
        """

        base = tmp_path / "experiment.yaml"

        base.write_text(
            "name: indic\nseed: 7\nembedding:\n  dimension: 256\ncompute:\n  batch_size: 16\n",
            encoding="utf-8",
        )

        profile = tmp_path / "gpu.yaml"

        profile.write_text("compute:\n  batch_size: 256\n  precision: bf16\n", encoding="utf-8")

        config = load_config(base, profile=profile, use_environment=False)

        assert config.compute.batch_size == 256

        assert config.compute.precision == "bf16"

        # Untouched by the profile.
        assert config.name == "indic"

        assert config.seed == 7

        assert config.embedding.dimension == 256

    def test_profile_merge_is_deep(self, tmp_path: Path) -> None:
        """
        A profile naming one compute key must not blank the others.
        Shallow replacement of the section would silently reset
        batch_size to its default, which is a quiet way to lose a
        result.
        """

        base = tmp_path / "base.yaml"

        base.write_text(
            "compute:\n  batch_size: 128\n  gradient_checkpoint_chunk: 8\n",
            encoding="utf-8",
        )

        profile = tmp_path / "p.yaml"

        profile.write_text("compute:\n  precision: bf16\n", encoding="utf-8")

        config = load_config(base, profile=profile, use_environment=False)

        assert config.compute.batch_size == 128

        assert config.compute.gradient_checkpoint_chunk == 8

        assert config.compute.precision == "bf16"

    def test_explicit_overrides_still_beat_the_profile(self, tmp_path: Path) -> None:
        """Halving the batch after an out-of-memory failure, without editing a file."""

        profile = tmp_path / "gpu.yaml"

        profile.write_text("compute:\n  batch_size: 256\n", encoding="utf-8")

        config = load_config(
            profile=profile,
            overrides={"compute": {"batch_size": 128}},
            use_environment=False,
        )

        assert config.compute.batch_size == 128

    def test_a_broken_profile_names_itself(self, tmp_path: Path) -> None:
        """
        The error must point at the profile rather than the experiment,
        since they are separate files and the wrong one wastes real time.
        """

        profile = tmp_path / "bad.yaml"

        profile.write_text("compute:\n  precision: fp8\n", encoding="utf-8")

        with pytest.raises(ConfigurationError) as caught:
            load_config(profile=profile, use_environment=False)

        assert caught.value.context.get("config_stage") == "profile"


class TestShippedProfiles:
    """
    The committed profiles are loaded here rather than assumed correct.
    They are only exercised on a real run otherwise, and one of them only
    ever runs on the machine furthest from a debugger.
    """

    @pytest.mark.parametrize("name", ["cpu.yaml", "gpu.yaml"])
    def test_profile_loads(self, name: str) -> None:
        config = load_config(profile=CONFIGS / name, use_environment=False)

        assert config.compute.batch_size > 0

    def test_the_gpu_profile_actually_differs(self) -> None:
        """
        A profile pair that resolved to the same thing would be pure
        ceremony. The GPU profile should raise the batch — that is the
        contrastive quality knob — and enable gradient caching to afford
        it.
        """

        cpu = load_config(profile=CONFIGS / "cpu.yaml", use_environment=False).compute

        gpu = load_config(profile=CONFIGS / "gpu.yaml", use_environment=False).compute

        assert gpu.batch_size > cpu.batch_size

        assert gpu.precision == "bf16"

        assert gpu.gradient_checkpoint_chunk > 0
