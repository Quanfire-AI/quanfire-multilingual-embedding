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

import gzip
import json
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
            query_prefix="query: ",
            passage_prefix="passage: ",
        )

        outcome = AdaptationPipeline(config, echo=lambda _: None).run()

        assert outcome.adapter_directory == destination

        manifest = json.loads((destination / "adapter.json").read_text(encoding="utf-8"))

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
