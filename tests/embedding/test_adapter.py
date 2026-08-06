"""
Persisting an adapted encoder.

An adaptation run that measures an improvement and then exits has
produced a number, not a model. These tests are about the artefact being
worth trusting: that a reloaded adapter is the same model, that it
carries how it must be used, and that a mismatch fails loudly rather than
serving plausible vectors that encode the wrong thing.

The identity test is the one that matters. A saved model scoring
differently on reload is worse than no saved model, because it is
trusted.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="requires the neural extra")

pytest.importorskip("transformers", reason="requires the pretrained extra")

from transformers import BertConfig, BertModel, BertTokenizerFast  # noqa: E402

from multilingual_embedding.embedding.neural.adapter import (  # noqa: E402
    load_adapter,
    save_adapter,
)
from multilingual_embedding.embedding.neural.lora import LoRAConfig, apply_lora  # noqa: E402
from multilingual_embedding.embedding.neural.pretrained import (  # noqa: E402
    PretrainedEncoderError,
    PretrainedTextEncoder,
)

PROBE = ["tok5 tok6 tok7", "tok50 tok51", "tok200"]


def base_checkpoint() -> Path:
    """A real HF checkpoint on disk, built rather than downloaded."""

    directory = Path(tempfile.mkdtemp())

    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    tokens += [f"tok{index}" for index in range(995)]

    (directory / "vocab.txt").write_text("\n".join(tokens), encoding="utf-8")

    BertTokenizerFast(vocab_file=str(directory / "vocab.txt")).save_pretrained(directory / "base")

    torch.manual_seed(0)

    BertModel(
        BertConfig(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            max_position_embeddings=64,
        )
    ).save_pretrained(directory / "base")

    return directory


def adapted(directory: Path, rank: int = 8) -> tuple[PretrainedTextEncoder, LoRAConfig]:
    """An encoder with a trained, non-trivial adapter."""

    encoder = PretrainedTextEncoder.load(str(directory / "base"), device="cpu", max_length=32)

    config = LoRAConfig(rank=rank, alpha=2 * rank, targets=("query", "value"))

    apply_lora(encoder._model, config)

    # Give the adapter real values; a zeroed one would round-trip
    # trivially and prove nothing.
    for module in encoder._model.modules():
        if hasattr(module, "lora_up"):
            torch.nn.init.normal_(module.lora_up.weight, std=0.02)

    return encoder, config


class TestTheReloadedModelIsTheSameModel:
    def test_vectors_are_identical_after_a_round_trip(self) -> None:
        """
        The assertion the artefact exists for. Anything less than
        identical means the saved model is not the one that was scored.
        """

        directory = base_checkpoint()

        encoder, config = adapted(directory)

        before = encoder.encode_batch(PROBE)

        save_adapter(encoder, directory / "saved", lora=config, data_provenance="public")

        reloaded, _ = load_adapter(directory / "saved", device="cpu")

        np.testing.assert_array_equal(reloaded.encode_batch(PROBE), before)

    def test_an_unadapted_reload_would_differ(self) -> None:
        """
        The control. If the base model alone produced the same vectors,
        the test above would pass whether or not the adapter was loaded.
        """

        directory = base_checkpoint()

        encoder, _ = adapted(directory)

        adapted_vectors = encoder.encode_batch(PROBE)

        plain = PretrainedTextEncoder.load(
            str(directory / "base"), device="cpu", max_length=32
        ).encode_batch(PROBE)

        assert not np.allclose(adapted_vectors, plain, atol=1e-6), (
            "the adapter changes nothing, so the round-trip test proves nothing"
        )


class TestTheArtefactCarriesHowToUseIt:
    def test_prefixes_survive(self) -> None:
        """
        An E5 model served without its prefixes returns worse vectors and
        says nothing, so the artefact has to remember them.
        """

        directory = base_checkpoint()

        encoder, config = adapted(directory)

        save_adapter(
            encoder,
            directory / "saved",
            lora=config,
            data_provenance="public",
            query_prefix="query: ",
            passage_prefix="passage: ",
        )

        _, metadata = load_adapter(directory / "saved", device="cpu")

        assert metadata.query_prefix == "query: "

        assert metadata.passage_prefix == "passage: "

    def test_normalization_survives(self) -> None:
        """
        The manifest has recorded this since format 1, and the loader was
        not reading it.

        An encoder saved with normalization off and reloaded with it on
        returns unit vectors for text that should have produced vectors
        of varying magnitude. Cosine scores are unaffected, so a search
        pipeline looks fine; a dot-product index built over those vectors
        ranks differently, and nothing anywhere raises.
        """

        directory = base_checkpoint()

        encoder = PretrainedTextEncoder.load(
            str(directory / "base"), device="cpu", max_length=32, normalize=False
        )

        config = LoRAConfig(rank=8, alpha=16, targets=("query", "value"))

        apply_lora(encoder._model, config)

        save_adapter(encoder, directory / "saved", lora=config, data_provenance="public")

        reloaded, _ = load_adapter(directory / "saved", device="cpu")

        assert reloaded._normalize is False

        # The property, not just the flag: unnormalized vectors have no
        # reason to land on the unit sphere.
        assert not np.allclose(np.linalg.norm(reloaded.encode_batch(PROBE), axis=1), 1.0)

    def test_provenance_survives(self) -> None:
        directory = base_checkpoint()

        encoder, config = adapted(directory)

        save_adapter(
            encoder,
            directory / "saved",
            lora=config,
            data_provenance="public",
            notes={"trained_on": "hi-pairs", "recall_at_1_after": 0.65},
        )

        _, metadata = load_adapter(directory / "saved", device="cpu")

        assert metadata.get("notes")["trained_on"] == "hi-pairs"

    def test_only_the_adapter_is_stored(self) -> None:
        """
        The base is named, not copied. Storing it would make a 3 MB
        artefact a 400 MB one for no gain.
        """

        directory = base_checkpoint()

        encoder, config = adapted(directory)

        saved = save_adapter(encoder, directory / "saved", lora=config, data_provenance="public")

        written = sum(path.stat().st_size for path in saved.iterdir())

        base_size = sum(
            path.stat().st_size for path in (directory / "base").iterdir() if path.is_file()
        )

        assert written < base_size / 4, f"adapter {written} is not small against base {base_size}"


class TestItFailsLoudly:
    def test_saving_without_an_adapter_is_refused(self) -> None:
        """
        Otherwise an empty file is written and believed.
        """

        directory = base_checkpoint()

        plain = PretrainedTextEncoder.load(str(directory / "base"), device="cpu")

        with pytest.raises(PretrainedEncoderError, match="no LoRA adapter"):
            save_adapter(
                plain, directory / "saved", lora=LoRAConfig(rank=8), data_provenance="public"
            )

    def test_loading_a_directory_that_is_not_an_adapter_is_refused(self) -> None:
        directory = base_checkpoint()

        with pytest.raises(PretrainedEncoderError, match="does not look like"):
            load_adapter(directory / "base", device="cpu")

    def test_mismatched_lora_settings_are_refused(self) -> None:
        """
        Weights that do not fit the reconstructed shapes must raise
        rather than load partially and serve a half-adapted model.
        """

        import json

        directory = base_checkpoint()

        encoder, config = adapted(directory, rank=8)

        saved = save_adapter(encoder, directory / "saved", lora=config, data_provenance="public")

        metadata = json.loads((saved / "adapter.json").read_text(encoding="utf-8"))

        metadata["lora"]["rank"] = 32

        (saved / "adapter.json").write_text(json.dumps(metadata), encoding="utf-8")

        with pytest.raises(PretrainedEncoderError, match="do not fit"):
            load_adapter(saved, device="cpu")

    def test_an_incompatible_format_version_is_refused(self) -> None:
        import json

        directory = base_checkpoint()

        encoder, config = adapted(directory)

        saved = save_adapter(encoder, directory / "saved", lora=config, data_provenance="public")

        metadata = json.loads((saved / "adapter.json").read_text(encoding="utf-8"))

        metadata["format_version"] = 99

        (saved / "adapter.json").write_text(json.dumps(metadata), encoding="utf-8")

        with pytest.raises(PretrainedEncoderError, match="incompatible format"):
            load_adapter(saved, device="cpu")


class TestTheBaseRevisionIsPinned:
    """
    An adapter names its base rather than storing it, so the base can change
    under the name after the adapter was measured. Recording the revision it was
    built against closes that gap: reloaded with it, the encoder resolves to the
    weights that produced the numbers — not whatever the cache now holds.
    """

    def test_the_recorded_revision_survives_a_round_trip(self) -> None:
        import json

        directory = base_checkpoint()

        encoder, config = adapted(directory)

        saved = save_adapter(
            encoder,
            directory / "saved",
            lora=config,
            data_provenance="public",
            checkpoint_revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
        )

        manifest = json.loads((saved / "adapter.json").read_text(encoding="utf-8"))

        assert manifest["checkpoint_revision"] == "614241f622f53c4eeff9890bdc4f31cfecc418b3"

        _, metadata = load_adapter(saved, device="cpu")

        assert metadata.checkpoint_revision == "614241f622f53c4eeff9890bdc4f31cfecc418b3"

    def test_an_unpinned_adapter_omits_the_key_and_reads_none(self) -> None:
        """
        A null key would read as a revision considered and left blank; its
        absence reads as an adapter that names its base without pinning it,
        which is the true state of an unpinned save.
        """

        import json

        directory = base_checkpoint()

        encoder, config = adapted(directory)

        saved = save_adapter(encoder, directory / "saved", lora=config, data_provenance="public")

        manifest = json.loads((saved / "adapter.json").read_text(encoding="utf-8"))

        assert "checkpoint_revision" not in manifest

        _, metadata = load_adapter(saved, device="cpu")

        assert metadata.checkpoint_revision is None

    def test_the_pinned_revision_reaches_the_base_load(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Writing the revision is pointless if the reload does not use it. This
        asserts the value the manifest records is the value handed to the base
        load — the link between the two that makes the pin real.
        """

        directory = base_checkpoint()

        encoder, config = adapted(directory)

        saved = save_adapter(
            encoder,
            directory / "saved",
            lora=config,
            data_provenance="public",
            checkpoint_revision="deadbeefcafe",
        )

        captured: dict[str, object] = {}

        real_load = PretrainedTextEncoder.load.__func__  # type: ignore[attr-defined]

        def spy(cls: type, name: str, **kwargs: object) -> object:
            captured["revision"] = kwargs.get("revision")

            # The base here is a local directory, which has no revisions; drop
            # it before delegating so the reload succeeds while the assertion
            # still sees what load_adapter computed.
            kwargs["revision"] = None

            return real_load(cls, name, **kwargs)

        monkeypatch.setattr(PretrainedTextEncoder, "load", classmethod(spy))

        load_adapter(saved, device="cpu")

        assert captured["revision"] == "deadbeefcafe"

    def test_an_explicit_revision_overrides_the_manifest(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        The manifest is the default, not a lock. Loading against a different
        base build than the one saved is done by passing a revision, and it
        must win over the recorded one.
        """

        directory = base_checkpoint()

        encoder, config = adapted(directory)

        saved = save_adapter(
            encoder,
            directory / "saved",
            lora=config,
            data_provenance="public",
            checkpoint_revision="the-recorded-one",
        )

        captured: dict[str, object] = {}

        real_load = PretrainedTextEncoder.load.__func__  # type: ignore[attr-defined]

        def spy(cls: type, name: str, **kwargs: object) -> object:
            captured["revision"] = kwargs.get("revision")

            kwargs["revision"] = None

            return real_load(cls, name, **kwargs)

        monkeypatch.setattr(PretrainedTextEncoder, "load", classmethod(spy))

        load_adapter(saved, device="cpu", revision="an-override")

        assert captured["revision"] == "an-override"


class TestProvenanceIsRecordedAndRequired:
    """
    Where the training data came from is a legal fact about the model, so
    it is written into the artefact, required before one can be saved, and
    an old adapter that predates the field reads as undeclared rather than
    silently claiming a value it never recorded.
    """

    def test_the_declared_value_survives_a_round_trip(self) -> None:
        directory = base_checkpoint()

        encoder, config = adapted(directory)

        save_adapter(encoder, directory / "saved", lora=config, data_provenance="synthetic")

        _, metadata = load_adapter(directory / "saved", device="cpu")

        assert metadata.data_provenance == "synthetic"

    def test_the_manifest_declares_format_two(self) -> None:
        import json

        directory = base_checkpoint()

        encoder, config = adapted(directory)

        saved = save_adapter(encoder, directory / "saved", lora=config, data_provenance="public")

        manifest = json.loads((saved / "adapter.json").read_text(encoding="utf-8"))

        # The field is top-level, not buried in notes, because it gates
        # load compatibility: format 2 guarantees its presence.
        assert manifest["format_version"] == 2

        assert manifest["data_provenance"] == "public"

    def test_an_unrecognised_provenance_is_refused(self) -> None:
        """
        A value that is not one of the declared kinds reads as a
        declaration while meaning nothing, so it is refused rather than
        written through.
        """

        directory = base_checkpoint()

        encoder, config = adapted(directory)

        with pytest.raises(ValueError, match="data_provenance"):
            save_adapter(encoder, directory / "saved", lora=config, data_provenance="customer")

    def test_a_format_one_adapter_reads_as_undeclared(self) -> None:
        """
        An adapter written before the field existed is still loadable, and
        its provenance is legibly absent rather than defaulted to a value
        that would claim more than the file records.
        """

        import json

        directory = base_checkpoint()

        encoder, config = adapted(directory)

        saved = save_adapter(encoder, directory / "saved", lora=config, data_provenance="public")

        manifest = json.loads((saved / "adapter.json").read_text(encoding="utf-8"))

        # Roll the manifest back to what format 1 wrote: no provenance key,
        # older version number.
        manifest["format_version"] = 1

        del manifest["data_provenance"]

        (saved / "adapter.json").write_text(json.dumps(manifest), encoding="utf-8")

        _, metadata = load_adapter(saved, device="cpu")

        assert metadata.data_provenance == "undeclared"
