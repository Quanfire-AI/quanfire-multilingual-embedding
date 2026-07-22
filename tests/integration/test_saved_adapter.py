"""
Verifying an adapter that was trained somewhere else.

Training happens on the GPU box; serving and inspection happen here. What
crosses between them is a directory of a few megabytes, copied by hand.
That copy is the weakest link in the whole pipeline: it is the one step
with no checksum, no config and no test — and a truncated ``adapter.pt``
or a manifest from a different run reloads into an encoder that produces
vectors of the right shape, the right norm and free of NaN, while
encoding something other than what was measured.

So these tests assert the artefact contract that
:func:`multilingual_embedding.embedding.neural.adapter.save_adapter`
writes, not the numbers of any particular run. A future ``models/`` entry
trained on different data with a different rank passes them unchanged;
a corrupted one does not.

They are skipped, not failed, when the artefact is absent — a checkout
without a trained model is a normal state of this repository. They are
also skipped when the base checkpoint is not in the local Hugging Face
cache, because :file:`tests/README.md` promises nothing reaches the
network and honouring that is worth more than a test that only runs
online. Warm the cache once and they execute from then on.

The split matters: **loading the base is a skip gate, loading the adapter
is an assertion.** A missing base means this machine is not set up; a
base that loads while the adapter does not means the copy is broken,
which is exactly what this file exists to catch.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="requires the neural extra")

pytest.importorskip("transformers", reason="requires the pretrained extra")

from multilingual_embedding.embedding.neural.adapter import (  # noqa: E402
    load_adapter,
)
from multilingual_embedding.embedding.neural.pretrained import (  # noqa: E402
    PretrainedEncoderError,
    PretrainedTextEncoder,
)
from multilingual_embedding.pipelines.search import (  # noqa: E402
    SemanticSearchPipeline,
)
from multilingual_embedding.utils.io import read_json  # noqa: E402

pytestmark = pytest.mark.slow


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

_ADAPTER_DIRECTORY = _REPOSITORY_ROOT / "models" / "indic-v1"

# Short, deliberately multi-script. Any adapter this repository produces
# is multilingual, and a comparison run only on ASCII would miss a
# tokenizer that lost its Devanagari or Tamil vocabulary in transit.
_TEXTS = (
    "भारत का संविधान मौलिक अधिकारों की गारंटी देता है",
    "இந்திய அரசியலமைப்பு அடிப்படை உரிமைகளை உறுதி செய்கிறது",
    "The constitution guarantees a set of fundamental rights",
)


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    """The adapter's own description of itself, or a skip."""

    if not _ADAPTER_DIRECTORY.is_dir():
        pytest.skip(
            f"no adapter at {_ADAPTER_DIRECTORY.relative_to(_REPOSITORY_ROOT)} "
            "— copy one there to run these"
        )

    metadata_path = _ADAPTER_DIRECTORY / "adapter.json"

    if not metadata_path.is_file():
        pytest.fail(f"{_ADAPTER_DIRECTORY} exists but has no adapter.json — the copy is incomplete")

    payload = read_json(metadata_path)

    assert isinstance(payload, dict)

    return payload


@pytest.fixture(scope="module")
def base_encoder(manifest: dict[str, Any]) -> PretrainedTextEncoder:
    """
    The unadapted checkpoint, for before-and-after comparison.

    A skip gate rather than an assertion: the checkpoint is an external
    dependency, and its absence from the cache says nothing about
    whether the adapter copied correctly.
    """

    try:
        return PretrainedTextEncoder.load(
            str(manifest["checkpoint"]),
            pooling=str(manifest.get("pooling", "mean")),
            device="cpu",
            max_length=int(manifest.get("max_length", 512)),
            local_files_only=True,
        )
    except PretrainedEncoderError as error:  # pragma: no cover - environment
        pytest.skip(
            f"base checkpoint {manifest['checkpoint']!r} is not in the local "
            f"Hugging Face cache ({error})"
        )


@pytest.fixture(scope="module")
def adapted_encoder(
    manifest: dict[str, Any],
    base_encoder: PretrainedTextEncoder,
) -> PretrainedTextEncoder:
    """The reloaded adapter. Failing to build this is a real failure."""

    encoder, _ = load_adapter(_ADAPTER_DIRECTORY, device="cpu", local_files_only=True)

    return encoder


class TestTheArtefactIsComplete:
    """Checks that need only the copied directory, not the base model."""

    def test_both_files_are_present(self, manifest: dict[str, Any]) -> None:
        assert (_ADAPTER_DIRECTORY / "adapter.json").is_file()

        assert (_ADAPTER_DIRECTORY / "adapter.pt").is_file()

    def test_format_version_is_one(self, manifest: dict[str, Any]) -> None:
        # A newer writer would bump this, and load_adapter refuses a
        # version it does not know. Asserting it here names the reason
        # rather than leaving a confusing load error downstream.
        assert manifest["format_version"] == 1

    def test_the_base_checkpoint_is_named(self, manifest: dict[str, Any]) -> None:
        # Only the low-rank update is stored, so an adapter that does not
        # say what it adapts is unusable, however intact its weights are.
        assert isinstance(manifest["checkpoint"], str)

        assert manifest["checkpoint"].strip()

    def test_the_lora_shape_is_recorded(self, manifest: dict[str, Any]) -> None:
        lora = manifest["lora"]

        assert lora["rank"] >= 1

        assert lora["alpha"] > 0

        assert lora["targets"], "an adapter with no target modules adapts nothing"

    def test_prefixes_are_recorded_as_a_pair(self, manifest: dict[str, Any]) -> None:
        query = manifest["query_prefix"]

        passage = manifest["passage_prefix"]

        # ("", "") is a valid answer — it means the model is symmetric.
        # One side set and the other empty is not: it would apply an
        # asymmetry the model was never trained with.
        assert bool(query) == bool(passage)

    def test_an_e5_checkpoint_carries_its_prefixes(self, manifest: dict[str, Any]) -> None:
        # The failure this guards against is silent by construction: an
        # E5 model served without `query: ` returns well-formed vectors
        # that encode the wrong thing.
        if "e5" not in manifest["checkpoint"].lower():
            pytest.skip("not an E5 checkpoint, so no prefixes are expected")

        assert manifest["query_prefix"] == "query: "

        assert manifest["passage_prefix"] == "passage: "

    def test_the_weights_match_the_manifest_parameter_count(self, manifest: dict[str, Any]) -> None:
        # The corruption check. A truncated or half-written copy is the
        # likeliest way this artefact breaks in transit, and it is the
        # one failure mode that produces no error anywhere else.
        state = torch.load(
            _ADAPTER_DIRECTORY / "adapter.pt",
            map_location="cpu",
            weights_only=True,
        )

        counted = sum(tensor.numel() for tensor in state.values())

        assert counted == manifest["adapter_parameters"]

    def test_the_adapter_is_not_a_no_op(self, manifest: dict[str, Any]) -> None:
        # The up-projection is the one initialised to zero, so that an
        # untrained adapter reloads *identical* to its base. That is the
        # right default and it is also the quietest possible failure:
        # every shape, norm and score stays plausible while the model is
        # simply the checkpoint again. Checking the down-projections
        # instead would pass on a no-op adapter, because they are
        # random-initialised and never zero.
        state = torch.load(
            _ADAPTER_DIRECTORY / "adapter.pt",
            map_location="cpu",
            weights_only=True,
        )

        assert state, "no tensors in adapter.pt"

        up = [tensor for name, tensor in state.items() if name.endswith("lora_up.weight")]

        assert up, "no up-projection weights — this is not an adapter this repository wrote"

        assert any(bool(tensor.abs().sum().item()) for tensor in up), (
            "every up-projection is zero — the adapter is a no-op, "
            "which means training never ran or never saved"
        )


class TestItReloadsIntoAWorkingEncoder:
    """Checks that need the base checkpoint present as well."""

    def test_the_dimension_survives_the_round_trip(
        self,
        manifest: dict[str, Any],
        adapted_encoder: PretrainedTextEncoder,
    ) -> None:
        assert adapted_encoder.dimension == manifest["dimension"]

    def test_it_encodes_every_script_finitely(self, adapted_encoder: PretrainedTextEncoder) -> None:
        vectors = adapted_encoder.encode_batch(list(_TEXTS))

        assert vectors.shape == (len(_TEXTS), adapted_encoder.dimension)

        assert np.isfinite(vectors).all()

        # A row of zeros would mean the text tokenised to nothing, which
        # for these strings means the tokenizer, not the model, is wrong.
        assert (np.linalg.norm(vectors, axis=1) > 0).all()

    def test_it_differs_from_the_base_model(
        self,
        base_encoder: PretrainedTextEncoder,
        adapted_encoder: PretrainedTextEncoder,
    ) -> None:
        # The end-to-end version of the zero-init check above: if the
        # adapter did not load, or loaded as zeros, these two encoders
        # are the same model and every recorded gain is unexplained.
        before = base_encoder.encode_batch(list(_TEXTS))

        after = adapted_encoder.encode_batch(list(_TEXTS))

        assert not np.allclose(before, after, atol=1e-4)

    def test_two_loads_agree_exactly(
        self,
        adapted_encoder: PretrainedTextEncoder,
    ) -> None:
        # Reproducibility of serving, not of training. A saved model that
        # scores differently on reload is worse than no saved model,
        # because it is trusted.
        again, _ = load_adapter(_ADAPTER_DIRECTORY, device="cpu", local_files_only=True)

        first = adapted_encoder.encode_batch(list(_TEXTS))

        second = again.encode_batch(list(_TEXTS))

        assert np.array_equal(first, second)


class TestItServesThroughTheSearchPipeline:
    """The path a serving layer would actually take."""

    @pytest.fixture(scope="class")
    def pipeline(
        self,
        manifest: dict[str, Any],
        base_encoder: PretrainedTextEncoder,
    ) -> SemanticSearchPipeline:
        return SemanticSearchPipeline.from_adapter(
            _ADAPTER_DIRECTORY, device="cpu", local_files_only=True
        )

    def test_the_recorded_prefixes_reach_the_pipeline(
        self,
        manifest: dict[str, Any],
        pipeline: SemanticSearchPipeline,
    ) -> None:
        # from_adapter exists for exactly this. Loading the encoder alone
        # and wrapping it by hand would drop them, silently.
        assert pipeline.prefixes == (
            manifest["query_prefix"],
            manifest["passage_prefix"],
        )

    def test_it_indexes_and_ranks(self, pipeline: SemanticSearchPipeline) -> None:
        indexed = pipeline.index(list(_TEXTS))

        assert indexed == len(_TEXTS)

        hits = pipeline.search(_TEXTS[0], top_k=len(_TEXTS))

        assert hits

        # Structure, not quality: ranks are 1..k and scores do not rise.
        assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))

        assert all(earlier.score >= later.score for earlier, later in pairwise(hits))

    def test_an_exact_query_finds_its_own_passage(self, pipeline: SemanticSearchPipeline) -> None:
        # A sanity check on wiring, not a retrieval claim. A model that
        # cannot find a passage identical to the query has been loaded
        # or prefixed wrongly; one that can has proved nothing about
        # quality, which is what the evaluation reports are for.
        pipeline.index(list(_TEXTS))

        for text in _TEXTS:
            assert pipeline.search(text, top_k=1)[0].text == text
