"""
Loading a saved adapter into a working search path.

An adaptation run that measures a 39% improvement and writes 3.4 MB to
disk has produced an artefact nobody can query. This is the step that
closes that gap, and the thing it must get right is not the weights —
:mod:`tests.embedding.test_adapter` already proves those round-trip
byte-identically — but the *prefixes*.

``save_adapter`` records them precisely so the model cannot be served
without them, and that recording is wasted if the pipeline built from the
adapter drops it on the floor. So the assertion here is that the prefixes
travel from the artefact into the encoder calls, not merely into an
attribute.

These tests need torch and transformers, and skip without them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="requires the neural extra")

pytest.importorskip("transformers", reason="requires the pretrained extra")

from transformers import BertConfig, BertModel, BertTokenizerFast  # noqa: E402

from multilingual_embedding.embedding.neural.adapter import save_adapter  # noqa: E402
from multilingual_embedding.embedding.neural.lora import LoRAConfig, apply_lora  # noqa: E402
from multilingual_embedding.embedding.neural.pretrained import (  # noqa: E402
    PretrainedTextEncoder,
)
from multilingual_embedding.pipelines.search import SemanticSearchPipeline  # noqa: E402

CORPUS = ["tok5 tok6 tok7", "tok50 tok51 tok52", "tok200 tok201"]


def base_checkpoint(root: Path) -> Path:
    """A real HF checkpoint on disk, built rather than downloaded."""

    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    tokens += [f"tok{index}" for index in range(995)]

    (root / "vocab.txt").write_text("\n".join(tokens), encoding="utf-8")

    BertTokenizerFast(vocab_file=str(root / "vocab.txt")).save_pretrained(root / "base")

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
    ).save_pretrained(root / "base")

    return root / "base"


def saved_adapter(root: Path, *, query_prefix: str, passage_prefix: str) -> Path:
    """An adapter directory, written the way an adaptation run writes it."""

    base = base_checkpoint(root)

    encoder = PretrainedTextEncoder.load(str(base), device="cpu", max_length=32)

    config = LoRAConfig(rank=8, alpha=16, targets=("query", "value"))

    apply_lora(encoder._model, config)

    # A zeroed adapter would make the base and adapted models identical,
    # so every assertion below would hold whether or not it was loaded.
    for module in encoder._model.modules():
        if hasattr(module, "lora_up"):
            torch.nn.init.normal_(module.lora_up.weight, std=0.02)

    return save_adapter(
        encoder,
        root / "saved",
        lora=config,
        data_provenance="public",
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
    )


class TestTheFactoryCarriesThePrefixes:
    def test_prefixes_come_from_the_artefact(self, tmp_path: Path) -> None:
        directory = saved_adapter(tmp_path, query_prefix="query: ", passage_prefix="passage: ")

        pipeline = SemanticSearchPipeline.from_adapter(directory, device="cpu")

        assert pipeline.prefixes == ("query: ", "passage: ")

    def test_the_query_prefix_is_actually_applied(self, tmp_path: Path) -> None:
        """
        The decisive test, and the reason this factory exists.

        Storing the prefix on the pipeline and then not using it fails
        every way except visibly, so it is not enough to read it back off
        an attribute. Two pipelines are built over the same encoder and
        the same corpus, differing only in the query prefix. Their scores
        must differ — and if they do, the prefix reached the encoder.
        """

        directory = saved_adapter(tmp_path, query_prefix="query: ", passage_prefix="passage: ")

        loaded = SemanticSearchPipeline.from_adapter(directory, device="cpu")

        # The control: if the prefix did not change this model's vectors,
        # the comparison below would hold whether or not it was applied.
        assert not np.allclose(
            loaded.encoder.encode("query: tok5 tok6"),
            loaded.encoder.encode("tok5 tok6"),
            atol=1e-6,
        ), "the prefix changes nothing in this model, so this test proves nothing"

        unprefixed_query = SemanticSearchPipeline(loaded.encoder, passage_prefix="passage: ")

        loaded.index(CORPUS)

        unprefixed_query.index(CORPUS)

        assert loaded.search("tok5 tok6")[0].score != pytest.approx(
            unprefixed_query.search("tok5 tok6")[0].score
        )

    def test_the_passage_prefix_reaches_the_stored_vectors(self, tmp_path: Path) -> None:
        """
        The same argument for the other side.

        Both pipelines here prefix the query identically, so any
        difference in score is attributable to the passage side alone. If
        there were none, the recorded prefix would be decorative and an
        E5 model would be served as a symmetric one, with nothing to show
        for it but a lower score.
        """

        directory = saved_adapter(tmp_path, query_prefix="query: ", passage_prefix="passage: ")

        loaded = SemanticSearchPipeline.from_adapter(directory, device="cpu")

        unprefixed_passages = SemanticSearchPipeline(loaded.encoder, query_prefix="query: ")

        loaded.index(CORPUS)

        unprefixed_passages.index(CORPUS)

        assert loaded.search("tok5 tok6")[0].score != pytest.approx(
            unprefixed_passages.search("tok5 tok6")[0].score
        )

    def test_a_symmetric_adapter_gets_no_prefixes(self, tmp_path: Path) -> None:
        """An adapter saved without prefixes must not acquire any."""

        directory = saved_adapter(tmp_path, query_prefix="", passage_prefix="")

        assert SemanticSearchPipeline.from_adapter(directory, device="cpu").prefixes == ("", "")


class TestTheFactoryProducesAWorkingPipeline:
    def test_it_indexes_and_searches(self, tmp_path: Path) -> None:
        directory = saved_adapter(tmp_path, query_prefix="query: ", passage_prefix="passage: ")

        pipeline = SemanticSearchPipeline.from_adapter(directory, device="cpu")

        assert pipeline.index(CORPUS) == len(CORPUS)

        hits = pipeline.search("tok5 tok6", top_k=2)

        assert len(hits) == 2

        # Results are the corpus as given, not as the model had to see it.
        assert all(hit.text in CORPUS for hit in hits)

    def test_there_is_no_matrix_and_no_separable_tokenizer(self, tmp_path: Path) -> None:
        """
        A contextual model has no per-token table, so ``similar_tokens``
        has nothing to be similar in. It returns nothing rather than
        raising, since a pipeline that serves search should not fail on
        an inspection helper that does not apply to it.
        """

        directory = saved_adapter(tmp_path, query_prefix="query: ", passage_prefix="passage: ")

        pipeline = SemanticSearchPipeline.from_adapter(directory, device="cpu")

        assert pipeline.matrix is None

        assert pipeline.tokenizer is None

        assert pipeline.similar_tokens("tok5") == []
