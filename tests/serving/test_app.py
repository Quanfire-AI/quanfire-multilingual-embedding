"""
The endpoint, against a real adapter.

The model here is built rather than downloaded — a two-layer BERT with a
non-zero LoRA adapter written by ``save_adapter`` — so these tests run
offline and exercise the same loading path a deployment uses.

The assertion this file exists for is the one about prefixes. A server
that stores ``query: ``, reports it in ``prefix_applied`` and never
prepends it would pass every shallow check: 200, right dimension, unit
norm, no NaN. So the test below compares the *vectors* returned for the
same text on each side and requires them to differ. If they did not, the
prefix would be decoration and an E5 model would be served as a symmetric
one, with nothing to show for it but a worse result list.

These tests need torch, transformers and the serve extra, and skip
without them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="requires the neural extra")

pytest.importorskip("transformers", reason="requires the pretrained extra")

pytest.importorskip("fastapi", reason="requires the serve extra")

pytest.importorskip("httpx", reason="the test client needs httpx")

from fastapi.testclient import TestClient  # noqa: E402
from transformers import BertConfig, BertModel, BertTokenizerFast  # noqa: E402

from multilingual_embedding.common.version import __version__  # noqa: E402
from multilingual_embedding.embedding.neural.adapter import save_adapter  # noqa: E402
from multilingual_embedding.embedding.neural.lora import LoRAConfig, apply_lora  # noqa: E402
from multilingual_embedding.embedding.neural.pretrained import (  # noqa: E402
    PretrainedTextEncoder,
)
from multilingual_embedding.serving.app import ServingConfig, create_app  # noqa: E402


def _saved_adapter(root: Path, *, query_prefix: str, passage_prefix: str) -> Path:
    """An adapter directory, written the way an adaptation run writes it."""

    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    tokens += [f"tok{index}" for index in range(995)]

    tokens += ["query", "passage", ":"]

    (root / "vocab.txt").write_text("\n".join(tokens), encoding="utf-8")

    base = root / "base"

    BertTokenizerFast(vocab_file=str(root / "vocab.txt")).save_pretrained(base)

    torch.manual_seed(0)

    BertModel(
        BertConfig(
            vocab_size=len(tokens),
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            max_position_embeddings=64,
        )
    ).save_pretrained(base)

    encoder = PretrainedTextEncoder.load(str(base), device="cpu", max_length=32)

    lora = LoRAConfig(rank=8, alpha=16, targets=("query", "value"))

    apply_lora(encoder._model, lora)

    # A zeroed adapter would make base and adapted identical, so several
    # assertions below would hold whether or not it had been loaded.
    for module in encoder._model.modules():
        if hasattr(module, "lora_up"):
            torch.nn.init.normal_(module.lora_up.weight, std=0.02)

    return save_adapter(
        encoder,
        root / "asymmetric" if query_prefix else root / "symmetric",
        lora=lora,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
    )


@pytest.fixture(scope="module")
def asymmetric_adapter(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("asymmetric")

    return _saved_adapter(root, query_prefix="query: ", passage_prefix="passage: ")


@pytest.fixture(scope="module")
def symmetric_adapter(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("symmetric")

    return _saved_adapter(root, query_prefix="", passage_prefix="")


@pytest.fixture(scope="module")
def client(asymmetric_adapter: Path) -> TestClient:
    return TestClient(create_app(ServingConfig(adapter_directory=asymmetric_adapter)))


class TestThePrefixIsAppliedAndNotMerelyReported:
    def test_the_two_sides_return_different_vectors(self, client: TestClient) -> None:
        """
        The decisive test.

        Same text, same model, different declared side. If the prefixes
        reached the encoder the vectors differ; if the server only echoed
        them in ``prefix_applied`` they would be identical and every
        other test here would still pass.
        """

        def embed(side: str) -> np.ndarray:
            response = client.post(
                "/v1/embeddings", json={"input": "tok5 tok6 tok7", "input_type": side}
            )

            assert response.status_code == 200, response.text

            return np.asarray(response.json()["data"][0]["embedding"], dtype=np.float64)

        assert not np.allclose(embed("query"), embed("passage"), atol=1e-6)

    def test_the_reported_prefix_matches_the_side_asked_for(self, client: TestClient) -> None:
        for side, expected in (("query", "query: "), ("passage", "passage: ")):
            body = client.post("/v1/embeddings", json={"input": "tok5", "input_type": side}).json()

            assert body["prefix_applied"] == expected


class TestTheServerRefusesToGuessASide:
    def test_an_asymmetric_model_rejects_a_request_with_no_side(self, client: TestClient) -> None:
        response = client.post("/v1/embeddings", json={"input": "tok5"})

        assert response.status_code == 400

    def test_the_refusal_names_both_valid_values(self, client: TestClient) -> None:
        """
        A 400 that does not say what to send makes the caller guess,
        which is the thing this endpoint refuses to do on their behalf.
        """

        message = client.post("/v1/embeddings", json={"input": "tok5"}).json()["error"]["message"]

        assert "query" in message

        assert "passage" in message

        assert (
            client.post("/v1/embeddings", json={"input": "tok5"}).json()["error"]["param"]
            == "input_type"
        )

    def test_a_configured_default_resolves_it(self, asymmetric_adapter: Path) -> None:
        """The operator override, for a deployment that is single-sided."""

        configured = TestClient(
            create_app(
                ServingConfig(
                    adapter_directory=asymmetric_adapter,
                    default_input_type="passage",
                )
            )
        )

        response = configured.post("/v1/embeddings", json={"input": "tok5"})

        assert response.status_code == 200

        assert response.json()["prefix_applied"] == "passage: "

    def test_an_explicit_side_still_beats_the_default(self, asymmetric_adapter: Path) -> None:
        configured = TestClient(
            create_app(
                ServingConfig(
                    adapter_directory=asymmetric_adapter,
                    default_input_type="passage",
                )
            )
        )

        body = configured.post(
            "/v1/embeddings", json={"input": "tok5", "input_type": "query"}
        ).json()

        assert body["prefix_applied"] == "query: "


class TestASymmetricModelHasNoSidesToChoose:
    def test_it_answers_without_an_input_type(self, symmetric_adapter: Path) -> None:
        client = TestClient(create_app(ServingConfig(adapter_directory=symmetric_adapter)))

        response = client.post("/v1/embeddings", json={"input": "tok5"})

        assert response.status_code == 200

    def test_prefix_applied_is_null_rather_than_empty(self, symmetric_adapter: Path) -> None:
        """
        Null and "" would both be falsy and neither would raise, but they
        say different things: "this model has no sides" versus "an
        asymmetric model was served without a prefix", which is a bug.
        """

        client = TestClient(create_app(ServingConfig(adapter_directory=symmetric_adapter)))

        body = client.post("/v1/embeddings", json={"input": "tok5"}).json()

        assert body["prefix_applied"] is None

    def test_an_input_type_is_accepted_and_ignored(self, symmetric_adapter: Path) -> None:
        client = TestClient(create_app(ServingConfig(adapter_directory=symmetric_adapter)))

        body = client.post("/v1/embeddings", json={"input": "tok5", "input_type": "query"}).json()

        assert body["prefix_applied"] is None


class TestTheResponseShape:
    def test_a_batch_comes_back_in_order(self, client: TestClient) -> None:
        """
        Indices are the only thing tying a vector to its input, so a
        reordering here would silently mislabel every result.
        """

        texts = ["tok5", "tok50 tok51", "tok200 tok201 tok202"]

        body = client.post("/v1/embeddings", json={"input": texts, "input_type": "passage"}).json()

        assert [item["index"] for item in body["data"]] == [0, 1, 2]

        singles = [
            client.post("/v1/embeddings", json={"input": text, "input_type": "passage"}).json()[
                "data"
            ][0]["embedding"]
            for text in texts
        ]

        for batched, alone in zip(body["data"], singles, strict=True):
            assert np.allclose(batched["embedding"], alone, atol=1e-5)

    def test_every_vector_has_the_model_dimension(self, client: TestClient) -> None:
        body = client.post(
            "/v1/embeddings", json={"input": ["tok5", "tok6"], "input_type": "query"}
        ).json()

        dimension = client.get("/v1/models").json()["data"][0]["dimension"]

        assert all(len(item["embedding"]) == dimension for item in body["data"])

    def test_usage_is_reported(self, client: TestClient) -> None:
        """Clients written against the standard schema read this field."""

        usage = client.post(
            "/v1/embeddings", json={"input": "tok5 tok6 tok7", "input_type": "query"}
        ).json()["usage"]

        assert usage["prompt_tokens"] > 0

        assert usage["total_tokens"] == usage["prompt_tokens"]


class TestTheModelCard:
    def test_it_reports_the_real_context_limit(self, client: TestClient) -> None:
        """
        Read from the artefact, not guessed off the encoder.

        A getattr fallback published ``max_length: 0`` for a model whose
        real limit is 32, and a client sizing its chunks from that field
        would have had no way to know.
        """

        card = client.get("/v1/models").json()["data"][0]

        assert card["max_length"] == 32

    def test_it_reports_the_prefixes_so_a_client_can_reproduce_a_vector(
        self, client: TestClient
    ) -> None:
        card = client.get("/v1/models").json()["data"][0]

        assert card["query_prefix"] == "query: "

        assert card["passage_prefix"] == "passage: "

    def test_a_symmetric_card_reports_no_prefixes(self, symmetric_adapter: Path) -> None:
        client = TestClient(create_app(ServingConfig(adapter_directory=symmetric_adapter)))

        card = client.get("/v1/models").json()["data"][0]

        assert card["query_prefix"] is None

        assert card["passage_prefix"] is None

    def test_it_stamps_the_framework_version(self, client: TestClient) -> None:
        assert client.get("/v1/models").json()["data"][0]["framework_version"] == __version__

    def test_the_model_id_defaults_to_the_directory_name(
        self, client: TestClient, asymmetric_adapter: Path
    ) -> None:
        """
        Ties the served name to a versioned artefact rather than to
        whatever the operator typed.
        """

        assert client.get("/v1/models").json()["data"][0]["id"] == asymmetric_adapter.name

    def test_an_explicit_model_id_overrides_it(self, asymmetric_adapter: Path) -> None:
        client = TestClient(
            create_app(ServingConfig(adapter_directory=asymmetric_adapter, model_id="indic-v9"))
        )

        assert client.get("/v1/models").json()["data"][0]["id"] == "indic-v9"


class TestHealth:
    def test_it_names_which_deployment_answered(self, client: TestClient) -> None:
        """
        Enough identity to tell two deployments apart. A bare ``{"status":
        "ok"}`` cannot distinguish the model you meant to roll out from
        the one still running.
        """

        body = client.get("/health").json()

        assert body["status"] == "ok"

        assert body["framework_version"] == __version__

        assert body["model"] == client.get("/v1/models").json()["data"][0]["id"]


class TestErrors:
    def test_asking_for_a_model_this_server_does_not_serve(self, client: TestClient) -> None:
        response = client.post(
            "/v1/embeddings",
            json={"input": "tok5", "input_type": "query", "model": "something-else"},
        )

        assert response.status_code == 404

        assert "something-else" in response.json()["error"]["message"]

    def test_naming_the_served_model_explicitly_is_accepted(
        self, client: TestClient, asymmetric_adapter: Path
    ) -> None:
        response = client.post(
            "/v1/embeddings",
            json={"input": "tok5", "input_type": "query", "model": asymmetric_adapter.name},
        )

        assert response.status_code == 200

    def test_an_empty_input_is_refused(self, client: TestClient) -> None:
        assert client.post("/v1/embeddings", json={"input": "  "}).status_code == 422

    def test_errors_use_the_standard_envelope(self, client: TestClient) -> None:
        """So a client's existing failure handling parses them too."""

        body = client.post("/v1/embeddings", json={"input": "tok5"}).json()

        assert set(body["error"]) == {"message", "type", "param"}

        assert body["error"]["type"] == "invalid_request_error"


class TestStartup:
    def test_a_missing_adapter_fails_at_startup_not_on_first_request(self, tmp_path: Path) -> None:
        """
        A deployment notices a process that will not start. It does not
        notice one that starts and then fails whatever the first caller
        happens to ask for.
        """

        with pytest.raises(Exception):  # noqa: B017 - the layer below chooses the type
            create_app(ServingConfig(adapter_directory=tmp_path / "nothing-here"))
