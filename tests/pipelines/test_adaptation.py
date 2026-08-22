"""
Adapting a published checkpoint, and the guards around what it claims.

Two different things are tested here, and only one of them is about
machine learning.

The first is the arithmetic of the run: score, adapt, score again, and
report the difference. That part needs torch and a checkpoint, and the
end-to-end test builds a two-layer BERT on disk rather than downloading
one, so the whole path runs on a laptop with no network.

The second is the part that decides whether the number means anything.
An adaptation run is only interpretable if exactly one thing changed
between training and evaluation, and ``--adaptation task`` on a run whose
kinds are identical produces a report labelled as a transfer result that
measured nothing of the sort. The label is what gets quoted six months
later, long after the command line is gone, so it must not be able to be
wrong. Those guards are pure functions and are tested without torch.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import unicodedata
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import pytest

from multilingual_embedding.config.base import (
    ADAPTATIONS,
    AdaptationConfig,
    ComputeConfig,
    ExperimentConfig,
)
from multilingual_embedding.core.exceptions import ConfigurationError
from multilingual_embedding.corpus.pairs import MinedPair
from multilingual_embedding.pipelines.adaptation import (
    AdaptationPipeline,
    AdaptationResult,
    check_adaptation,
    only,
    prefixed,
    values,
    varying,
    without_held_out,
)

needs_neural = pytest.mark.skipif(
    find_spec("torch") is None or find_spec("transformers") is None,
    reason="requires the neural and pretrained extras",
)


def pair(
    anchor: str = "a query",
    positive: str = "an answer",
    *,
    kind: str = "adjacent",
    language: str = "hi",
) -> MinedPair:
    return MinedPair(
        anchor=anchor,
        positive=positive,
        kind=kind,
        document="doc-1",
        language=language,
        overlap=0.1,
    )


class TestPrefixes:
    """
    The asymmetry an E5-family checkpoint was trained with.

    Serving one of these models without its prefixes returns vectors of
    the right shape, the right norm and free of NaN, which encode the
    wrong thing. Nothing raises; the score is merely lower. So the
    prefixes are applied to the text and recorded with the adapter.
    """

    def test_each_side_gets_its_own_marker(self) -> None:
        [result] = prefixed([pair("क्या", "उत्तर")], "query: ", "passage: ")

        assert result.anchor == "query: क्या"

        assert result.positive == "passage: उत्तर"

    def test_provenance_survives(self) -> None:
        """The breakdowns are computed after this, off these fields."""

        [result] = prefixed([pair(kind="title_lead", language="ta")], "q: ", "p: ")

        assert (result.kind, result.language, result.overlap) == ("title_lead", "ta", 0.1)

    def test_no_prefixes_is_not_a_rewrite(self) -> None:
        original = [pair()]

        assert prefixed(original, "", "") == original

    def test_mined_negatives_take_the_passage_marker(self) -> None:
        """
        A negative is a passage — the model is being taught not to
        retrieve it — so marking it as a query would put a whole class of
        candidate columns in the wrong half of the space. The batch would
        then find them easy for the wrong reason, and the loss would fall
        to prove it.
        """

        one = MinedPair(
            anchor="क्या",
            positive="उत्तर",
            kind="adjacent",
            document="doc-1",
            language="hi",
            overlap=0.1,
            negatives=("गलत उत्तर",),
        )

        [result] = prefixed([one], "query: ", "passage: ")

        assert result.negatives == ("passage: गलत उत्तर",)

    def test_a_pair_without_negatives_gains_none(self) -> None:
        [result] = prefixed([pair()], "query: ", "passage: ")

        assert result.negatives == ()


class TestFacetFilter:
    def test_naming_nothing_keeps_everything(self) -> None:
        pairs = [pair(kind="adjacent"), pair(kind="title_lead")]

        assert only(pairs, (), "kind") == pairs

    def test_it_keeps_only_what_was_named(self) -> None:
        pairs = [pair(kind="adjacent"), pair(kind="title_lead")]

        assert [p.kind for p in only(pairs, ("title_lead",), "kind")] == ["title_lead"]

    def test_an_empty_name_is_not_a_filter(self) -> None:
        """
        ``--train-kinds ""`` is how a config-set filter is cleared, and
        an empty string must not become a kind nothing matches.
        """

        pairs = [pair(kind="adjacent")]

        assert only(pairs, ("",), "kind") == pairs

    def test_selecting_nothing_raises_rather_than_training_on_zero(self) -> None:
        """
        Almost always a typo, and an empty training set is the worst
        available outcome: the run completes, reports a delta of zero,
        and looks like a result about the pairs.
        """

        with pytest.raises(ConfigurationError) as caught:
            only([pair(kind="adjacent")], ("heading_section",), "kind")

        assert "heading_section" in str(caught.value)

    def test_it_says_what_the_file_actually_holds(self) -> None:
        """Naming the available values is what turns the error into a fix."""

        with pytest.raises(ConfigurationError) as caught:
            only([pair(kind="adjacent"), pair(kind="title_lead")], ("adjacnet",), "kind")

        assert caught.value.context["available"] == ["adjacent", "title_lead"]

    def test_values_reports_what_is_present(self) -> None:
        assert values([pair(language="hi"), pair(language="ta")], "language") == ["hi", "ta"]

    def test_values_omits_the_unlabelled(self) -> None:
        """A hand-built pair set need not carry a language, and an empty
        string is not one — reporting it would make the run look as
        though it trained on a language called ``""``."""

        assert values([pair(language="hi"), pair(language="")], "language") == ["hi"]


class TestHeldOutNormalisation:
    """
    The text rule compares normalised forms, not bytes.

    Legal source formats are where exact matching breaks: EU Formex carries
    non-breaking spaces and decomposed combining forms, so the *same sentence*
    can appear under two document ids in two byte sequences. Neither rule sees
    it then — the ids differ, and the strings are not equal — and it trains.
    """

    @staticmethod
    def _pair(anchor_text: str, positive: str, document: str) -> MinedPair:
        return MinedPair(
            anchor=anchor_text,
            positive=positive,
            kind="adjacent",
            document=document,
            language="en",
            overlap=0.1,
        )

    def test_a_whitespace_variant_under_another_document_is_excluded(self) -> None:
        held_text = "This Regulation shall enter into force on the twentieth day"
        variant = held_text.replace(" shall ", "\u00a0shall\u00a0")

        assert variant != held_text, "the fixture no longer varies the bytes"

        evaluation = [self._pair("when does it apply", held_text, "celex:A:1")]
        pool = [self._pair("a different question", variant, "celex:B:9")]

        kept, facts = without_held_out(pool, evaluation)

        assert kept == [], "the normalised form of a held-out text trained"

        # The document rule cannot have done this — the ids are disjoint — and
        # neither can exact text. Attributing the drop is what makes the test
        # about normalisation rather than about exclusion in general.
        assert facts["pool_dropped_by_normalization"] == 1

        assert facts["pool_dropped_by_document"] == 0

        assert facts["pool_dropped_by_text"] == 0

    def test_a_decomposed_form_is_excluded(self) -> None:
        """NFD vs NFC: identical to a reader, unequal to ``in``."""

        held_text = "L'acquis de l'Union européenne s'applique à cette décision"
        decomposed = unicodedata.normalize("NFD", held_text)

        assert decomposed != held_text, "the fixture no longer varies the bytes"

        evaluation = [self._pair("a query", held_text, "celex:A:1")]
        pool = [self._pair("another query", decomposed, "celex:B:9")]

        kept, facts = without_held_out(pool, evaluation)

        assert kept == []

        assert facts["pool_dropped_by_normalization"] == 1

    def test_a_genuinely_different_text_survives(self) -> None:
        """
        Without this the class above is satisfied by a rule that drops
        everything, which would fail safe on integrity and destroy the run.
        """

        evaluation = [self._pair("a query", "the first provision", "celex:A:1")]
        pool = [self._pair("another query", "an unrelated provision", "celex:B:9")]

        kept, facts = without_held_out(pool, evaluation)

        assert kept == pool

        assert facts["pool_dropped_as_held_out"] == 0

    def test_a_pool_pair_reusing_a_held_out_anchor_is_excluded(self) -> None:
        """
        The anchor side of the text rule, which nothing above reaches.

        Every other text-rule test here matches a held-out *positive* against
        a pool *positive* — the one case the pre-fix rule already handled, so
        they all pass against it. Replaying the exact rule that produced the
        withdrawn numbers (``held_texts`` built from positives only, and only
        the pool pair's positive compared) leaves this whole file green. What
        actually stops that rule in the fixtures above is the *document* rule;
        the text rule is never the thing under test.

        Here a multi-language alignment emits the same English anchor against
        several translations. Hold out EN -> ES and the corpus still offers
        EN -> FR: same held-out query text, a positive that was never held,
        and its own document id, so the document rule cannot reach it either.
        Training on it hands the model the exact anchor it is about to be
        scored on.
        """

        held_anchor = "when does this regulation start to apply"

        evaluation = [self._pair(held_anchor, "entra en vigor a los veinte dias", "celex:A:es")]

        pool = [self._pair(held_anchor, "entre en vigueur le vingtieme jour", "celex:A:fr")]

        kept, facts = without_held_out(pool, evaluation)

        assert kept == [], "a pool pair carrying the held-out anchor trained"

        # Attributed, so a future change to the document rule cannot make this
        # pass without the text rule doing the work.
        assert facts["pool_dropped_by_text"] == 1

        assert facts["pool_dropped_by_document"] == 0

    def test_the_reverse_direction_is_excluded_when_the_documents_differ(self) -> None:
        """
        The eulaw leak on a corpus that ids each language separately.

        ``test_the_reverse_of_a_held_out_pair_is_not_trained_on`` covers the
        reverse direction, but its two directions share one document, and it
        asserts as much — so the document rule closes it and the text rule is
        never exercised. A corpus that files each language manifestation under
        its own id does not give the document rule that handle, and then the
        reverse pair's positive is the held-out *anchor*, which the pre-fix
        rule never looked at.
        """

        english = "the Commission shall adopt implementing acts"
        spanish = "la Comision adoptara actos de ejecucion"

        evaluation = [self._pair(english, spanish, "celex:A:en")]

        pool = [self._pair(spanish, english, "celex:A:es")]

        kept, facts = without_held_out(pool, evaluation)

        assert kept == [], "the reverse of a held-out pair trained"

        assert facts["pool_dropped_by_text"] == 1

        assert facts["pool_dropped_by_document"] == 0


class TestVarying:
    def test_the_same_values_are_held_fixed(self) -> None:
        assert varying(["adjacent"], ["adjacent"]) is False

    def test_disjoint_values_vary(self) -> None:
        assert varying(["adjacent"], ["title_lead"]) is True

    def test_partial_overlap_is_not_varying(self) -> None:
        """
        Neither held fixed nor varied. A run built on it answers no clean
        question — a gain could come from the kinds it shares — so it is
        reported as not varying and the declaration check rejects it.
        """

        assert varying(["adjacent", "title_lead"], ["title_lead"]) is False

    def test_an_unlabelled_side_cannot_vary(self) -> None:
        """Absence of a label is not evidence of a difference."""

        assert varying([], ["title_lead"]) is False


class TestDeclaration:
    """
    The two-sided check between the label and the data.

    A facet that should vary and does not is one failure. A facet that
    varies when it should have been held fixed is the other, and it is
    the more dangerous: the result cannot be attributed to either change,
    and the report gives no sign of it.
    """

    def test_a_matching_declaration_passes(self) -> None:
        check_adaptation("task", {"kind"})

    def test_in_distribution_requires_nothing_to_vary(self) -> None:
        check_adaptation("in-distribution", set())

    def test_a_facet_that_should_vary_and_does_not_is_refused(self) -> None:
        with pytest.raises(ConfigurationError) as caught:
            check_adaptation("task", set())

        assert "held fixed" in str(caught.value)

    def test_a_facet_that_varies_and_should_not_is_refused(self) -> None:
        with pytest.raises(ConfigurationError) as caught:
            check_adaptation("in-distribution", {"language"})

        assert "cannot be attributed" in str(caught.value)

    def test_the_right_facet_varying_is_not_enough(self) -> None:
        """
        Declaring ``task`` while the language also changed. The kind
        requirement is satisfied, so a one-sided check would let this
        through and the report would attribute a language effect to the
        task shape.
        """

        with pytest.raises(ConfigurationError) as caught:
            check_adaptation("task", {"kind", "language"})

        assert "language varies too" in str(caught.value)

    def test_a_combined_mode_accepts_both(self) -> None:
        check_adaptation("task+language", {"kind", "language"})

    def test_the_error_carries_what_would_fix_it(self) -> None:
        with pytest.raises(ConfigurationError) as caught:
            check_adaptation("domain", set())

        context = caught.value.context

        assert context["required"] == ["corpus"]

        assert context["varying"] == []

        assert sorted(ADAPTATIONS) == context["modes"]

    @pytest.mark.parametrize("mode", sorted(ADAPTATIONS))
    def test_every_mode_accepts_exactly_its_own_facets(self, mode: str) -> None:
        """
        The declaration table and the check are one thing, so a mode
        added to :data:`ADAPTATIONS` cannot arrive without a check that
        implements it.
        """

        check_adaptation(mode, set(ADAPTATIONS[mode]))


def report(recall: float, mrr: float = 0.5) -> Any:
    """A retrieval report with only the fields the verdict reads."""

    from multilingual_embedding.evaluation.retrieval import RetrievalReport, RetrievalScores

    return RetrievalReport(
        overall=RetrievalScores(queries=100, recall_at_1=recall, recall_at_5=recall, mrr=mrr)
    )


def result(before: float, after: float, *, moved: float = 0.01) -> AdaptationResult:
    return AdaptationResult(
        checkpoint="base",
        adaptation="in-distribution",
        measures="…",
        before=report(before),
        after=report(after),
        trained_on=Path("pairs.jsonl"),
        scored_against=Path("pairs.jsonl"),
        train_examples=100,
        eval_examples=50,
        sampled_from=150,
        varying_facets=[],
        moved=moved,
    )


class TestVerdict:
    def test_a_higher_score_helped(self) -> None:
        assert result(0.40, 0.55).helped is True

    def test_an_unchanged_score_did_not_help(self) -> None:
        """
        Not a coin flip. ``helped`` gates the CLI's exit code, and a
        pipeline that ships whatever it trained unless the model got
        strictly worse would ship no-op adapters.
        """

        assert result(0.40, 0.40).helped is False

    def test_a_lower_score_did_not_help(self) -> None:
        assert result(0.55, 0.40).helped is False

    def test_the_delta_is_absolute_and_the_relative_is_a_percentage(self) -> None:
        outcome = result(0.40, 0.50)

        assert outcome.delta == pytest.approx(0.10)

        assert outcome.relative == pytest.approx(25.0)

    def test_a_zero_baseline_does_not_divide(self) -> None:
        """A checkpoint that retrieves nothing is a plausible starting
        point for an unrelated domain, and must not crash the report."""

        assert result(0.0, 0.10).relative == 0.0

    def test_a_model_that_did_not_move_is_distinguishable(self) -> None:
        """
        The distinction the probe exists for. An unchanged score with
        unchanged weights is a training problem — a larger learning
        rate; an unchanged score with changed weights is a result about
        the pairs. The remedies are opposite, and without ``moved`` the
        two are the same line of output.
        """

        assert result(0.40, 0.40, moved=0.0).trained is False

        assert result(0.40, 0.40, moved=0.01).trained is True

    def test_the_summary_says_which_of_the_two_happened(self) -> None:
        stalled = result(0.40, 0.40, moved=0.0).summary()

        assert "the MODEL barely changed" in stalled

        assert "training problem" in stalled

        moved = result(0.40, 0.40, moved=0.01).summary()

        assert "The model changed but the score did not improve" in moved

    def test_to_dict_is_json_serialisable(self) -> None:
        """It is written to disk as the record of the run."""

        payload = json.loads(json.dumps(result(0.40, 0.50).to_dict()))

        assert payload["delta"] == 0.1

        assert payload["adapter_directory"] is None


# ----------------------------------------------------------------------
# End to end, on a checkpoint built rather than downloaded
# ----------------------------------------------------------------------


def base_checkpoint(root: Path) -> Path:
    """A real HF checkpoint on disk, small enough to train on a laptop."""

    import torch
    from transformers import BertConfig, BertModel, BertTokenizerFast

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


def pair_file(
    path: Path,
    count: int = 64,
    *,
    kinds: tuple[str, ...] = ("adjacent",),
    languages: tuple[str, ...] = ("hi",),
) -> Path:
    """
    A pair file over the toy vocabulary.

    Each anchor and positive share one token and differ in the rest, so
    the task is solvable but not trivially, and the pool is large enough
    for recall@1 to be worth reading.
    """

    opener = gzip.open if path.suffix == ".gz" else open

    with opener(path, "wt", encoding="utf-8") as handle:  # type: ignore[operator]
        for index in range(count):
            record = MinedPair(
                anchor=f"tok{index} tok{index + 100}",
                positive=f"tok{index} tok{index + 300} tok{index + 400}",
                kind=kinds[index % len(kinds)],
                document=f"doc-{index}",
                language=languages[index % len(languages)],
                overlap=0.33,
            ).to_record()

            handle.write(json.dumps(record) + "\n")

    return path


def bidirectional_pair_file(path: Path, units: int = 96) -> Path:
    """
    A cross-lingual pair file shaped like the ones we actually mine.

    Every aligned unit is emitted **twice** — once each way — sharing one
    document id, which is what ``aligned``, ``eulaw``, ``pib`` and ``trade``
    all do. That shape is the whole point: the reverse of a held-out pair has
    the held-out *anchor* as its positive, so a filter that inspects only
    positives cannot see it.
    """

    with open(path, "w", encoding="utf-8") as handle:
        for index in range(units):
            # Every eighth unit repeats the previous unit's text under a fresh
            # document id, the way an entry-into-force article recurs verbatim
            # across regulations. Document identity cannot see that, so the
            # split has to fall back on the text for these.
            source = index - 1 if index % 8 == 7 else index

            left = f"tok{source} tok{source + 100}"
            right = f"tok{source} tok{source + 300} tok{source + 400}"

            forward = MinedPair(
                anchor=left,
                positive=right,
                kind="adjacent",
                document=f"doc-{index}",
                language="en",
                positive_language="hi",
                overlap=0.33,
            )

            handle.write(json.dumps(forward.to_record(), ensure_ascii=False) + "\n")

            handle.write(
                json.dumps(
                    dataclasses.replace(
                        forward,
                        anchor=right,
                        positive=left,
                        language="hi",
                        positive_language="en",
                    ).to_record(),
                    ensure_ascii=False,
                )
                + "\n"
            )

    return path


def sibling_pair_file(path: Path, documents: int = 96, units: int = 2) -> Path:
    """
    A corpus where each document contributes several *different* units.

    Every pair here has text that appears nowhere else, so no text rule can
    tell that two of them come from the same regulation. Only the document
    identity can. That makes this the one shape that isolates what the
    document rule is actually for.
    """

    with open(path, "w", encoding="utf-8") as handle:
        for document in range(documents):
            for unit in range(units):
                tag = document * 10 + unit

                handle.write(
                    json.dumps(
                        MinedPair(
                            anchor=f"a{tag} a{tag + 1000}",
                            positive=f"b{tag} b{tag + 2000} b{tag + 3000}",
                            kind="adjacent",
                            document=f"doc-{document}",
                            language="en",
                            overlap=0.2,
                        ).to_record(),
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    return path


BOILERPLATE = "shall enter into force on the twentieth day following publication"


def repeated_text_pair_file(path: Path, documents: int = 160) -> Path:
    """
    A corpus where one sentence recurs verbatim under many *different* documents.

    This is the shape the document rule cannot see. An entry-into-force
    article is the same sentence in a hundred regulations, so a pair carrying
    it under `celex:B` gives away a held-out pair carrying it under
    `celex:A` — and both documents are properly identified, so a text check
    that only runs when `document` is empty never looks.

    Each document contributes exactly two pairs, one repeated and one unique,
    which bounds what the document rule alone could possibly drop at two per
    held-out document and leaves something to train on after the text rule.
    """

    with open(path, "w", encoding="utf-8") as handle:
        for document in range(documents):
            pairs = (
                (f"a{document} a{document + 10000}", BOILERPLATE),
                (f"c{document} c{document + 20000}", f"e{document} e{document + 30000}"),
            )

            for anchor, positive in pairs:
                handle.write(
                    json.dumps(
                        MinedPair(
                            anchor=anchor,
                            positive=positive,
                            kind="adjacent",
                            document=f"celex-{document}",
                            language="en",
                            overlap=0.2,
                        ).to_record(),
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    return path


def experiment(root: Path, **settings: Any) -> ExperimentConfig:
    """A runnable config over the built checkpoint and a fresh pair file."""

    fields: dict[str, Any] = {
        "checkpoint": str(base_checkpoint(root)),
        "pairs": pair_file(root / "pairs.jsonl"),
        "train_pairs": 32,
        "eval_pairs": 16,
        "sample_pairs": 64,
        "epochs": 1,
        # Large for real work, and deliberately so: the point of the
        # end-to-end test is that the weights move far enough for the
        # probe to see it, on two layers and thirty-two pairs.
        "learning_rate": 5e-3,
        "rank": 4,
        "max_length": 16,
        "seed": 0,
    }

    fields.update(settings)

    return ExperimentConfig(
        name="test-adaptation",
        seed=0,
        adaptation=AdaptationConfig(**fields),
        compute=ComputeConfig(device="cpu", precision="fp32", batch_size=8),
    )


class TestPipelineGuards:
    """What is refused, and how early."""

    def test_a_missing_checkpoint_is_named(self) -> None:
        config = ExperimentConfig(adaptation=AdaptationConfig(pairs=Path("pairs.jsonl")))

        with pytest.raises(ConfigurationError, match="checkpoint"):
            AdaptationPipeline(config, echo=lambda _: None).run()

    def test_a_missing_pair_file_is_named(self) -> None:
        config = ExperimentConfig(adaptation=AdaptationConfig(checkpoint="some/model"))

        with pytest.raises(ConfigurationError, match="pairs"):
            AdaptationPipeline(config, echo=lambda _: None).run()

    @needs_neural
    def test_a_wrong_declaration_is_caught_before_the_model_loads(self, tmp_path: Path) -> None:
        """
        The ordering is the point. ``adaptation.checkpoint`` here names a
        model that does not exist, so anything that reached the loader
        would fail with a download error instead. Getting a
        ConfigurationError proves the declaration was checked first —
        which on a GPU box is the difference between seconds and an hour.
        """

        config = experiment(tmp_path, checkpoint="not-a-real/checkpoint", adaptation="task")

        with pytest.raises(ConfigurationError, match="held fixed"):
            AdaptationPipeline(config, echo=lambda _: None).run()


@needs_neural
class TestEndToEnd:
    def test_it_scores_adapts_and_scores_again(self, tmp_path: Path) -> None:
        outcome = AdaptationPipeline(experiment(tmp_path), echo=lambda _: None).run()

        assert outcome.before.overall.queries == outcome.after.overall.queries

        assert outcome.train_examples > 0

        assert outcome.eval_examples > 0

        assert 0.0 < outcome.trainable_share < 100.0

    def test_the_weights_actually_move(self, tmp_path: Path) -> None:
        """
        LoRA initialises its up-projection to zero, so an adapted model
        that was never trained is byte-identical to the published one and
        every score below would be unchanged for a reason that has
        nothing to do with the pairs.
        """

        assert AdaptationPipeline(experiment(tmp_path), echo=lambda _: None).run().trained

    def test_the_held_out_set_is_not_trained_on(self, tmp_path: Path) -> None:
        """
        The evaluation and training samples are drawn from the same file
        by default, so without an explicit exclusion the model would be
        scored on pairs it had memorised and the delta would measure
        recall of the training set.
        """

        config = experiment(tmp_path)

        pipeline = AdaptationPipeline(config, echo=lambda _: None)

        train, held, _ = pipeline._select()

        assert {p.positive for p in train} & {p.positive for p in held} == set()

    def test_the_reverse_of_a_held_out_pair_is_not_trained_on(self, tmp_path: Path) -> None:
        """
        The leak that inflated eulaw-multi-e1.

        A cross-lingual corpus emits both directions of every alignment, so
        holding out (A -> B) while training on (B -> A) feeds a symmetric
        bi-encoder the same two texts with the roles swapped. Excluding by
        positive text alone cannot catch it, because the reverse pair's
        positive is the held-out *anchor*. Measured on the real run: 23.4% of
        held-out pairs had their exact reverse in training and only 7.4% were
        unseen on both sides, which is what makes this a measurement bug
        rather than a modelling preference.
        """

        config = experiment(
            tmp_path,
            pairs=bidirectional_pair_file(tmp_path / "xling.jsonl"),
            sample_pairs=192,
            train_pairs=64,
            eval_pairs=32,
        )

        train, held, _ = AdaptationPipeline(config, echo=lambda _: None)._select()

        assert train, "nothing left to train on — the exclusion is too broad"

        trained = {(pair.anchor, pair.positive) for pair in train}

        reversed_held = {(pair.positive, pair.anchor) for pair in held}

        assert trained & reversed_held == set()

        # Both directions share one document, so the document rule is what
        # closes this. Asserting it directly stops a future text-level
        # workaround from passing the check above by accident.
        assert {p.document for p in train} & {p.document for p in held} == set()

    def test_the_report_carries_the_split_hygiene_counts(self, tmp_path: Path) -> None:
        """
        The audit trail has to survive into the artefact.

        The counts existed inside ``_select`` for a while before they reached
        the report, so the comment promising a report "can be audited for
        split hygiene after the fact" was describing something no reader of a
        report could actually do. gov-indic-e4c is what surfaced it: its JSON
        answered ``None`` to every hygiene question.
        """

        config = experiment(
            tmp_path,
            pairs=sibling_pair_file(tmp_path / "siblings.jsonl"),
            sample_pairs=192,
            train_pairs=64,
            eval_pairs=32,
        )

        payload = AdaptationPipeline(config, echo=lambda _: None).run().to_dict()

        assert payload["held_out_documents"] > 0

        assert payload["held_out_without_document"] == 0

        # More was dropped than the held-out pairs themselves, which is the
        # document rule doing work rather than the text rule alone.
        assert payload["pool_dropped_as_held_out"] > payload["eval_examples"]

    def test_a_sibling_unit_of_a_held_out_document_is_not_trained_on(self, tmp_path: Path) -> None:
        """
        What the document rule is for, isolated from the text rule.

        Two provisions of the same regulation share no wording, so excluding
        by text cannot see that they belong together — and a model that
        trained on Article 3 has been shown the register, subject matter and
        drafting idiom of the very document it is about to be scored on.

        This shape is the one that fails if the document rule is removed. The
        bidirectional test above does not: there, both directions share their
        text as well, so the text rule alone keeps it passing and the document
        assertion succeeds for the wrong reason.
        """

        config = experiment(
            tmp_path,
            pairs=sibling_pair_file(tmp_path / "siblings.jsonl"),
            sample_pairs=192,
            train_pairs=64,
            eval_pairs=32,
        )

        train, held, facts = AdaptationPipeline(config, echo=lambda _: None)._select()

        assert train, "nothing left to train on — the exclusion is too broad"

        assert {pair.document for pair in train} & {pair.document for pair in held} == set()

        # Siblings share no text with the held-out set, so anything dropped
        # beyond the held-out pairs themselves was dropped by document. Without
        # this the assertion above could pass on a corpus where the text rule
        # happened to catch everything anyway.
        held_texts = {pair.anchor for pair in held} | {pair.positive for pair in held}

        assert {pair.anchor for pair in train} & held_texts == set()

        assert facts["pool_dropped_as_held_out"] > len(held)

        assert facts["held_out_without_document"] == 0

    def test_repeated_text_under_another_document_is_not_trained_on(self, tmp_path: Path) -> None:
        """
        What the *text* rule is for, on a corpus that documents itself.

        The pair-level and sibling tests above both pass with the text check
        restricted to undocumented pairs — that was the code before 6fe7e6b,
        and nothing in this file said it was wrong. Here a held-out sentence
        recurs verbatim under a document that was never held out, so the
        document rule cannot reach it and only a text check that runs for
        *every* pair closes it. On the real eulaw corpus this shape left 250
        exact reverses standing after the document rule had done its work.
        """

        config = experiment(
            tmp_path,
            pairs=repeated_text_pair_file(tmp_path / "boilerplate.jsonl"),
            sample_pairs=240,
            train_pairs=32,
            eval_pairs=32,
        )

        train, held, facts = AdaptationPipeline(config, echo=lambda _: None)._select()

        assert train, "nothing left to train on — the exclusion is too broad"

        # State the precondition rather than trusting the sample: if no
        # held-out pair carries the repeated sentence there is nothing here
        # for the text rule to catch and the test would pass vacuously.
        assert any(pair.positive == BOILERPLATE for pair in held)

        held_documents = {pair.document for pair in held}

        assert {pair.document for pair in train} & held_documents == set()

        held_sides = {pair.anchor for pair in held} | {pair.positive for pair in held}

        assert {pair.anchor for pair in train} & held_sides == set()

        assert {pair.positive for pair in train} & held_sides == set()

        # Each document contributes exactly two pairs, so the document rule
        # alone cannot account for more than two drops per held-out document.
        # Anything beyond that was the text rule reaching across documents.
        assert facts["pool_dropped_as_held_out"] > 2 * len(held_documents)

        assert facts["held_out_without_document"] == 0

    def test_a_pair_without_a_document_is_excluded_by_either_side(self, tmp_path: Path) -> None:
        """
        The fallback for a corpus that does not identify its documents.

        There is no document to group on, so the only defence left is text —
        and it has to look at *both* sides. Checking the positive alone is
        what let the reverse direction through in the first place.

        The opt-in is deliberate and load-bearing: this split is the weak
        one, and a corpus only gets it by asking. See the companion test
        that the default refuses.
        """

        path = tmp_path / "anonymous.jsonl"

        with open(path, "w", encoding="utf-8") as handle:
            for index in range(96):
                left = f"tok{index} tok{index + 100}"
                right = f"tok{index} tok{index + 300} tok{index + 400}"

                for anchor, positive in ((left, right), (right, left)):
                    handle.write(
                        json.dumps(
                            MinedPair(
                                anchor=anchor,
                                positive=positive,
                                kind="adjacent",
                                document="",
                                language="en",
                                overlap=0.33,
                            ).to_record(),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

        config = experiment(
            tmp_path,
            pairs=path,
            sample_pairs=192,
            train_pairs=64,
            eval_pairs=32,
            allow_undocumented_fallback=True,
        )

        train, held, facts = AdaptationPipeline(config, echo=lambda _: None)._select()

        held_sides = {pair.anchor for pair in held} | {pair.positive for pair in held}

        assert {pair.anchor for pair in train} & held_sides == set()

        assert {pair.positive for pair in train} & held_sides == set()

        # The run must say it fell back, or a report claims a document-level
        # split it never had.
        assert facts["held_out_without_document"] == len(held)

    def test_an_undocumented_corpus_is_refused_unless_it_asks(self, tmp_path: Path) -> None:
        """
        The same corpus, without the opt-in, must stop the run.

        A warning was the wrong instrument here. It scrolls past in a
        ninety-second run and is gone by the next session, while the split it
        permits is the text-only one this function exists to replace — so the
        failure mode is a report that claims a document-level split it never
        had, with the evidence already off the screen.
        """

        path = tmp_path / "anonymous.jsonl"

        with open(path, "w", encoding="utf-8") as handle:
            for index in range(96):
                left = f"tok{index} tok{index + 100}"
                right = f"tok{index} tok{index + 300} tok{index + 400}"

                for anchor_text, positive in ((left, right), (right, left)):
                    handle.write(
                        json.dumps(
                            MinedPair(
                                anchor=anchor_text,
                                positive=positive,
                                kind="adjacent",
                                document="",
                                language="en",
                                overlap=0.33,
                            ).to_record(),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

        config = experiment(tmp_path, pairs=path, sample_pairs=192, train_pairs=64, eval_pairs=32)

        with pytest.raises(ConfigurationError, match="carry no document id"):
            AdaptationPipeline(config, echo=lambda _: None)._select()

    def test_a_fixed_evaluation_file_is_recorded(self, tmp_path: Path) -> None:
        """
        The corpus axis. Two runs that train on different data are only
        comparable if they are judged by the same thing, and the report
        has to say which file that was.
        """

        held_out = pair_file(tmp_path / "other.jsonl", 32)

        config = experiment(tmp_path, eval_pairs_file=held_out, adaptation="domain")

        outcome = AdaptationPipeline(config, echo=lambda _: None).run()

        assert outcome.scored_against == held_out

        assert outcome.varying_facets == ["corpus"]

    def test_a_kind_filter_reaches_the_report(self, tmp_path: Path) -> None:
        """A run labelled ``task`` must show the two kinds it separated."""

        config = experiment(
            tmp_path,
            pairs=pair_file(tmp_path / "mixed.jsonl", 96, kinds=("adjacent", "title_lead")),
            train_kinds="adjacent",
            eval_kinds="title_lead",
            adaptation="task",
            sample_pairs=96,
        )

        outcome = AdaptationPipeline(config, echo=lambda _: None).run()

        assert outcome.train_kinds == ["adjacent"]

        assert outcome.eval_kinds == ["title_lead"]

        assert outcome.varying_facets == ["kind"]

    def test_the_adapter_is_written_with_its_evidence(self, tmp_path: Path) -> None:
        """
        Without this the run produced a measurement rather than a model.
        The scores go into the artefact so it carries the case for itself
        rather than pointing at a report that may not travel with it.
        """

        destination = tmp_path / "adapter"

        config = experiment(
            tmp_path,
            save_adapter=destination,
            data_provenance="public",
            query_prefix="query: ",
            passage_prefix="passage: ",
        )

        outcome = AdaptationPipeline(config, echo=lambda _: None).run()

        assert outcome.adapter_directory == destination

        manifest = json.loads((destination / "adapter.json").read_text(encoding="utf-8"))

        # The provenance is a legal fact about the model, written at the
        # top level where it gates load compatibility, not buried in notes.
        assert manifest["data_provenance"] == "public"

        # The prefixes are part of the model, not of the command line
        # that produced it. Serving without them returns plausible
        # vectors that encode the wrong thing.
        assert manifest["query_prefix"] == "query: "

        assert manifest["passage_prefix"] == "passage: "

        assert manifest["notes"]["recall_at_1_before"] == pytest.approx(
            outcome.before.overall.recall_at_1, abs=1e-4
        )

    def test_the_report_records_the_run_and_not_the_request(self, tmp_path: Path) -> None:
        """
        ``train_examples`` is what was used, which a filter can cut far
        below what was asked for. A report that echoed the request would
        describe a run that did not happen.
        """

        destination = tmp_path / "report.json"

        config = experiment(tmp_path, report=destination, train_pairs=1_000)

        outcome = AdaptationPipeline(config, echo=lambda _: None).run()

        payload = json.loads(destination.read_text(encoding="utf-8"))

        assert payload["train_examples"] == outcome.train_examples < 1_000

        assert payload["adaptation"] == "in-distribution"

    def test_a_gzipped_pair_file_works(self, tmp_path: Path) -> None:
        """Mining writes compressed; adaptation must read it unaided."""

        config = experiment(tmp_path, pairs=pair_file(tmp_path / "pairs.jsonl.gz"))

        assert AdaptationPipeline(config, echo=lambda _: None).run().eval_examples > 0

    def test_the_same_seed_reproduces_the_run(self, tmp_path: Path) -> None:
        first = AdaptationPipeline(experiment(tmp_path), echo=lambda _: None).run()

        second = AdaptationPipeline(experiment(tmp_path), echo=lambda _: None).run()

        assert first.before.overall.recall_at_1 == second.before.overall.recall_at_1

        assert first.after.overall.recall_at_1 == second.after.overall.recall_at_1


@needs_neural
class TestCommandLine:
    def test_the_exit_code_follows_the_verdict(self, tmp_path: Path) -> None:
        """
        Non-zero when the adapter did not beat the checkpoint it started
        from. A shell pipeline that chains adaptation into a deployment
        step cannot read a verdict off stdout, and shipping a model that
        made retrieval worse is the failure this prevents.
        """

        from multilingual_embedding.cli import EXIT_ERROR, EXIT_SUCCESS, main

        destination = tmp_path / "report.json"

        code = main(
            [
                "adapt",
                "--checkpoint",
                str(base_checkpoint(tmp_path)),
                "--pairs",
                str(pair_file(tmp_path / "pairs.jsonl")),
                "--train-pairs",
                "32",
                "--eval-pairs",
                "16",
                "--sample-pairs",
                "64",
                "--rank",
                "4",
                "--max-length",
                "16",
                "--learning-rate",
                "5e-3",
                "--output",
                str(destination),
                "--set",
                "compute.device=cpu",
                "--set",
                "compute.batch_size=8",
            ]
        )

        payload = json.loads(destination.read_text(encoding="utf-8"))

        assert code == (EXIT_SUCCESS if payload["delta"] > 0 else EXIT_ERROR)

    def test_a_wrong_declaration_exits_without_a_traceback(self, tmp_path: Path) -> None:
        """
        A mode that disagrees with the filters is the user's mistake, and
        the CLI owes them a sentence rather than a stack trace.
        """

        from multilingual_embedding.cli import EXIT_ERROR, main

        code = main(
            [
                "adapt",
                "--checkpoint",
                str(base_checkpoint(tmp_path)),
                "--pairs",
                str(pair_file(tmp_path / "pairs.jsonl")),
                "--adaptation",
                "language",
            ]
        )

        assert code == EXIT_ERROR

    def test_a_profile_supplies_the_machine_half(self, tmp_path: Path) -> None:
        """
        The reason ``qfme adapt`` exists. The experiment file describes
        the science and the profile describes the box, so the same run is
        reproducible on a laptop and on a GPU without editing the part
        that decides what is being measured.
        """

        from multilingual_embedding.config.loader import load_config

        experiment_file = tmp_path / "experiment.yaml"

        experiment_file.write_text(
            "adaptation:\n"
            f"  checkpoint: {base_checkpoint(tmp_path)}\n"
            f"  pairs: {pair_file(tmp_path / 'pairs.jsonl')}\n"
            "  rank: 4\n"
            "compute:\n"
            "  batch_size: 4\n",
            encoding="utf-8",
        )

        profile = tmp_path / "profile.yaml"

        profile.write_text("compute:\n  device: cpu\n  batch_size: 64\n", encoding="utf-8")

        config = load_config(experiment_file, profile=profile)

        assert config.compute.batch_size == 64

        # The science is untouched by the machine it ran on.
        assert config.adaptation.rank == 4
