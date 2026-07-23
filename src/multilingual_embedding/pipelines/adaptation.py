"""
Adapting a published encoder to a domain, and measuring whether it helped.

Three steps, and the order is the argument:

1. Score the published checkpoint on held-out pairs. **This is the
   number to beat**, and it is the only honest baseline — beating
   chance, or beating an untrained model, proves nothing about whether
   adaptation was worth doing.
2. Fine-tune it on the mined pairs with LoRA.
3. Score it again on the same held-out pairs.

If step 3 does not beat step 1, the adaptation did not work, and that is
a result worth having rather than a failure to hide. Fine-tuning a
well-pretrained model on a narrow corpus can easily make it worse.

Nothing here trains the base weights. LoRA leaves them frozen, so the
comparison is between one model and itself plus a small adapter, rather
than between two models that differ in unknown ways.

**Which experiment is being run is declared, not inferred.** The same
three steps answer several different questions depending on what is held
fixed between training and evaluation, and the questions are not
interchangeable — see :data:`~multilingual_embedding.config.base.ADAPTATIONS`.
The declaration is checked against what the filters actually do, and a
run whose label and data disagree is refused before it starts. A report
labelled ``task`` that quietly held the kinds identical is worse than no
report, because the label is what gets quoted later.

This module exists so that the experiment is a library object rather
than a script. ``qfme adapt`` resolves a config, a compute profile and
``--set`` overrides into an
:class:`~multilingual_embedding.config.base.AdaptationConfig` and runs
it, which means a GPU run is reproducible from a file that can be
committed and diffed.

Torch is imported inside :meth:`AdaptationPipeline.run` rather than at
module scope, because the neural stack is an optional extra and the
static path must stay installable without it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from multilingual_embedding.config.base import (
    ADAPTATIONS,
    AdaptationConfig,
    ComputeConfig,
    ExperimentConfig,
)
from multilingual_embedding.core.exceptions import ConfigurationError
from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.corpus.pairs import MinedPair, sample_pairs
from multilingual_embedding.evaluation.retrieval import RetrievalReport, evaluate_retrieval

__all__ = [
    "AdaptationPipeline",
    "AdaptationResult",
    "adapt",
    "check_adaptation",
    "only",
    "prefixed",
    "varying",
]

_logger = get_logger(__name__)

# How many anchors to re-encode after training to prove the weights
# actually moved. Sixteen is enough to make a zero maximum difference
# mean something and small enough to be free.
_PROBE = 16


def prefixed(pairs: Sequence[MinedPair], query: str, passage: str) -> list[MinedPair]:
    """
    Apply the asymmetric prefixes a retrieval checkpoint expects.

    E5 and several other families are trained with distinct markers on
    the query and passage sides, and omitting them measurably degrades
    retrieval. The framework's ``TextEncoder`` contract has one
    ``encode_batch`` and no notion of which side a text is, so this is
    applied to the text before it reaches the encoder rather than by
    complicating the contract for one family's convention.

    Mined negatives take the *passage* prefix. They are passages — the
    model is being taught not to retrieve them — and prefixing them as
    queries would put a whole class of candidate columns in the wrong
    half of the space, where they would look easy for the wrong reason
    and the loss would fall to prove it.
    """

    if not query and not passage:
        return list(pairs)

    return [
        MinedPair(
            anchor=query + pair.anchor,
            positive=passage + pair.positive,
            kind=pair.kind,
            document=pair.document,
            language=pair.language,
            overlap=pair.overlap,
            negatives=tuple(passage + negative for negative in pair.negatives),
        )
        for pair in pairs
    ]


def only(pairs: Sequence[MinedPair], wanted: Sequence[str], facet: str) -> list[MinedPair]:
    """
    Keep pairs whose ``facet`` is named, or everything when none are.

    Two facets are filterable, and which one is varied decides what the
    run measures:

    ``kind``
        The shape of the retrieval task. ``adjacent`` retrieves the
        paragraph following a paragraph; ``title_lead`` retrieves a lead
        from a title. Same text, different question asked of it.

    ``language``
        The corpus the text comes from. Same question, different text.

    Vary one and hold the other fixed, and the result attributes itself.
    Vary both and it cannot.

    A filter that selects nothing raises rather than returning an empty
    list, and names what the file actually holds — almost always a typo
    or a mistaken assumption about the pair file, and silently training
    on zero pairs is the worst available outcome.
    """

    names = {name for name in wanted if name}

    if not names:
        return list(pairs)

    kept = [pair for pair in pairs if getattr(pair, facet) in names]

    if not kept:
        available = sorted({getattr(p, facet) for p in pairs if getattr(p, facet)})

        raise ConfigurationError(
            f"no pairs with {facet} in {sorted(names)}",
            facet=facet,
            requested=sorted(names),
            available=available,
        )

    return kept


def values(pairs: Sequence[MinedPair], facet: str) -> list[str]:
    """The distinct values of a facet actually present, for reporting."""

    return sorted({getattr(p, facet) for p in pairs if getattr(p, facet)})


def varying(train: Sequence[str], held: Sequence[str]) -> bool:
    """
    Whether a facet is varied between training and evaluation.

    Varied means disjoint. Partial overlap is neither held fixed nor
    varied, and a run built on it answers no clean question, so it is
    reported as not varying and :func:`check_adaptation` rejects it.
    """

    return bool(train and held and not set(train) & set(held))


def check_adaptation(declared: str, observed: set[str]) -> None:
    """
    Refuse a run whose filters do not implement the experiment it claims.

    The declaration is the point. Naming a run ``adaptation: task`` and
    then leaving the kinds identical produces a report labelled as a
    transfer result that measured nothing of the sort, and the label is
    what gets quoted six months later. So the label is checked against
    what the data actually does, and disagreement stops the run before it
    spends an hour on a GPU.

    The check is two-sided. A facet that should vary and does not is one
    failure; a facet that varies when it should have been held fixed is
    the other, and it is the more dangerous of the two — the result
    cannot be attributed to either change.
    """

    required = set(ADAPTATIONS[declared])

    if observed == required:
        return

    problems = []

    if missing := sorted(required - observed):
        problems.append(
            f"declared adaptation {declared!r}, but {', '.join(missing)} is held fixed "
            f"rather than varied"
        )

    if extra := sorted(observed - required):
        problems.append(
            f"declared adaptation {declared!r}, but {', '.join(extra)} varies too, so the "
            f"result cannot be attributed to one of them"
        )

    raise ConfigurationError(
        "; ".join(problems) + ". Vary a facet by giving the train and eval filters disjoint values "
        "(kind, language) or different files (corpus, via eval_pairs_file)",
        declared=declared,
        varying=sorted(observed),
        required=sorted(required),
        modes=sorted(ADAPTATIONS),
    )


@dataclass(slots=True)
class AdaptationResult:
    """
    What one adaptation run produced.

    Attributes
    ----------
    checkpoint:
        The base model, which was not modified.

    adaptation:
        The declared mode, already checked against the filters.

    before, after:
        Retrieval reports on the *same* held-out pairs, either side of
        training.

    train_examples, eval_examples:
        The counts actually used, not the ones requested. A facet filter
        can cut either far below its configured value, and a report that
        records only the request describes a run that did not happen.

    moved:
        Largest absolute change in a probe vector, before against after.
        Kept so the run can tell "adaptation did not help" apart from
        "adaptation did not happen". Without it an unchanged score is
        ambiguous, and the two have opposite remedies: a bigger learning
        rate, or better pairs.

    trainable_share:
        Percentage of parameters LoRA made trainable.

    adapter_directory:
        Where the adapter was written, or ``None`` when the run produced
        a measurement rather than a model.
    """

    checkpoint: str

    adaptation: str

    measures: str

    before: RetrievalReport

    after: RetrievalReport

    trained_on: Path

    scored_against: Path

    train_examples: int

    eval_examples: int

    sampled_from: int

    varying_facets: list[str]

    train_kinds: list[str] = field(default_factory=list)

    eval_kinds: list[str] = field(default_factory=list)

    train_languages: list[str] = field(default_factory=list)

    eval_languages: list[str] = field(default_factory=list)

    initial_loss: float = 0.0

    final_loss: float = 0.0

    moved: float = 0.0

    trainable_share: float = 0.0

    seconds: float = 0.0

    data_provenance: str = ""

    adapter_directory: Path | None = None

    @property
    def delta(self) -> float:
        """Absolute change in recall@1."""

        return self.after.overall.recall_at_1 - self.before.overall.recall_at_1

    @property
    def relative(self) -> float:
        """Change in recall@1 as a percentage of the baseline."""

        baseline = self.before.overall.recall_at_1

        return (self.delta / baseline * 100.0) if baseline else 0.0

    @property
    def helped(self) -> bool:
        """Whether the adapted model beat the published one."""

        return self.delta > 0

    @property
    def trained(self) -> bool:
        """
        Whether the weights moved at all.

        A run where this is ``False`` has a training problem rather than
        a result, and the distinction decides what to try next.
        """

        return self.moved >= 1e-4

    def summary(self) -> str:
        """The comparison, as the lines worth reading off a terminal."""

        verdict = "BETTER" if self.delta > 0 else ("NO CHANGE" if self.delta == 0 else "WORSE")

        lines = [
            f"{'published checkpoint':24} recall@1 {self.before.overall.recall_at_1:.4f}"
            f"   MRR {self.before.overall.mrr:.4f}",
            f"{'after LoRA adaptation':24} recall@1 {self.after.overall.recall_at_1:.4f}"
            f"   MRR {self.after.overall.mrr:.4f}",
            "",
            f"recall@1 {self.delta:+.4f}  ({self.relative:+.1f}%)   -> {verdict}",
            f"the model itself moved by {self.moved:.6f} (max change in a probe vector)",
        ]

        if self.delta == 0 and not self.trained:
            lines += [
                "",
                "The score did not change because the MODEL barely changed.",
                "That is a training problem rather than a result: try a larger",
                "learning rate, more epochs, or a higher rank before concluding",
                "anything about the pairs.",
            ]
        elif self.delta <= 0:
            lines += [
                "",
                "The model changed but the score did not improve. That is a",
                "result, not a failure — fine-tuning a well-pretrained model on",
                "a narrow corpus can make it worse, and pairs that are largely",
                "solvable by string matching are a likely reason.",
            ]

        lines += ["", "by lexical overlap, before -> after:"]

        for band in sorted(set(self.before.by_overlap) | set(self.after.by_overlap)):
            was, now = self.before.by_overlap.get(band), self.after.by_overlap.get(band)

            if was and now:
                lines.append(
                    f"  {band:14} {was.recall_at_1:.4f} -> {now.recall_at_1:.4f}"
                    f"  ({now.queries:,} queries)"
                )

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Reduce to primitives for persistence."""

        return {
            "checkpoint": self.checkpoint,
            "adaptation": self.adaptation,
            "measures": self.measures,
            "varying_facets": self.varying_facets,
            "trained_on": str(self.trained_on),
            "scored_against": str(self.scored_against),
            "train_kinds": self.train_kinds,
            "eval_kinds": self.eval_kinds,
            "train_languages": self.train_languages,
            "eval_languages": self.eval_languages,
            "train_examples": self.train_examples,
            "eval_examples": self.eval_examples,
            "sampled_from": self.sampled_from,
            "loss": [self.initial_loss, self.final_loss],
            "moved": self.moved,
            "trainable_share": self.trainable_share,
            "data_provenance": self.data_provenance,
            "recall_at_1_before": round(self.before.overall.recall_at_1, 4),
            "recall_at_1_after": round(self.after.overall.recall_at_1, 4),
            "delta": round(self.delta, 4),
            "relative_percent": round(self.relative, 2),
            "seconds": round(self.seconds, 1),
            "adapter_directory": str(self.adapter_directory) if self.adapter_directory else None,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
        }


class AdaptationPipeline:
    """
    Runs the three-step adaptation experiment from a resolved config.

    Parameters
    ----------
    config:
        The experiment. Its ``adaptation`` section carries the science
        and its ``compute`` section carries what the machine dictates,
        which is the split that lets one file describe a run on a laptop
        and on a GPU box.

    echo:
        Where progress goes. Defaults to :func:`print`, because the
        primary caller is a CLI and a training run that says nothing for
        forty minutes is indistinguishable from a hung one. Pass a
        no-op to silence it.

    Example
    -------
    ::

        config = ExperimentConfig(
            name="indic-v1",
            adaptation=AdaptationConfig(
                checkpoint="intfloat/multilingual-e5-small",
                pairs=Path("pairs/hi.jsonl.gz"),
                query_prefix="query: ",
                passage_prefix="passage: ",
            ),
        )

        result = AdaptationPipeline(config).run()

        print(result.summary())
    """

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        echo: Callable[[str], None] = print,
    ) -> None:
        self.config = config

        self.echo = echo

    @property
    def adaptation(self) -> AdaptationConfig:
        """The science half of the run."""

        return self.config.adaptation

    @property
    def compute(self) -> ComputeConfig:
        """The machine half of the run — the section a profile swaps."""

        return self.config.compute

    def run(self) -> AdaptationResult:
        """
        Score, adapt, score again.

        Raises
        ------
        ConfigurationError
            If the checkpoint or pair file is missing, a facet filter
            selects nothing, or the declared adaptation mode disagrees
            with what the filters actually vary. All three are checked
            before the model is loaded, so a mistake costs seconds
            rather than an hour on a GPU.
        """

        settings = self.adaptation

        if not settings.checkpoint:
            raise ConfigurationError(
                "adaptation.checkpoint is required — name a published model or a "
                "local directory to adapt"
            )

        if settings.pairs is None:
            raise ConfigurationError(
                "adaptation.pairs is required — point it at a file written by qfme mine-pairs"
            )

        from multilingual_embedding.embedding.neural import (
            ContrastiveConfig,
            ContrastiveTrainer,
            PretrainedTextEncoder,
            TextPair,
            save_adapter,
        )
        from multilingual_embedding.embedding.neural.lora import (
            LoRAConfig,
            apply_lora,
            parameter_summary,
        )

        started = time.time()

        train, held, facts = self._select()

        # Everything checkable is checked before the checkpoint is
        # downloaded, let alone trained.
        check_adaptation(settings.adaptation, facts["observed"])

        self.echo(f"\n  adaptation {settings.adaptation}: {settings.measures}")

        encoder = PretrainedTextEncoder.load(
            settings.checkpoint,
            pooling=settings.pooling,
            max_length=settings.max_length,
        )

        self.echo(f"device {encoder.device}   dimension {encoder.dimension}")

        self.echo("\n[1] scoring the checkpoint as published — this is the number to beat")

        probe = [pair.anchor for pair in held[:_PROBE]]

        probe_before = encoder.encode_batch(probe)

        before = evaluate_retrieval(encoder, held, limit=None)

        self.echo(before.summary())

        lora = LoRAConfig(
            rank=settings.rank,
            alpha=settings.resolved_alpha,
            dropout=settings.dropout,
            targets=settings.targets,
        )

        apply_lora(encoder._model, lora)

        parameters = parameter_summary(encoder._model)

        share = parameters["trainable"] / parameters["total"] * 100

        self.echo(
            f"\n[2] LoRA rank {settings.rank} on {settings.targets}: "
            f"{parameters['trainable']:,} of {parameters['total']:,} trainable ({share:.2f}%)"
        )

        training = ContrastiveTrainer(
            encoder,
            ContrastiveConfig(
                epochs=settings.epochs,
                batch_size=self.compute.batch_size,
                learning_rate=settings.learning_rate,
                seed=settings.seed or 0,
                precision=self.compute.precision,
                gradient_checkpoint_chunk=self.compute.gradient_checkpoint_chunk,
            ),
        ).train([TextPair(pair.anchor, pair.positive, pair.negatives) for pair in train])

        if training.measurable:
            self.echo(f"    loss {training.initial_loss:.4f} -> {training.final_loss:.4f}")
        else:
            self.echo(
                f"    loss {training.final_loss:.4f} (one epoch, so there is nothing "
                f"to compare it against — the retrieval scores below are the evidence)"
            )

        self.echo("\n[3] scoring the adapted model on the same held-out pairs")

        after = evaluate_retrieval(encoder, held, limit=None)

        self.echo(after.summary())

        moved = float(abs(encoder.encode_batch(probe) - probe_before).max())

        result = AdaptationResult(
            checkpoint=settings.checkpoint,
            adaptation=settings.adaptation,
            measures=settings.measures,
            before=before,
            after=after,
            trained_on=settings.pairs,
            scored_against=facts["evaluation_source"],
            train_examples=len(train),
            eval_examples=len(held),
            sampled_from=settings.sampled,
            varying_facets=sorted(facts["observed"]),
            train_kinds=facts["train_kinds"],
            eval_kinds=facts["eval_kinds"],
            train_languages=facts["train_languages"],
            eval_languages=facts["eval_languages"],
            initial_loss=training.initial_loss,
            final_loss=training.final_loss,
            moved=moved,
            trainable_share=share,
            data_provenance=settings.data_provenance,
            seconds=time.time() - started,
        )

        if settings.save_adapter:
            # Saved after scoring, with the scores in it, so the artefact
            # carries the evidence for itself rather than pointing at a
            # report that may not travel with it.
            save_adapter(
                encoder,
                settings.save_adapter,
                lora=lora,
                data_provenance=settings.data_provenance,
                query_prefix=settings.query_prefix,
                passage_prefix=settings.passage_prefix,
                notes={
                    "experiment": self.config.name,
                    "data_provenance": settings.data_provenance,
                    "trained_on": str(settings.pairs),
                    "scored_against": str(facts["evaluation_source"]),
                    "train_pairs": len(train),
                    "adaptation": settings.adaptation,
                    "train_kinds": facts["train_kinds"],
                    "train_languages": facts["train_languages"],
                    "held_out_languages": facts["eval_languages"],
                    "recall_at_1_before": round(before.overall.recall_at_1, 4),
                    "recall_at_1_after": round(after.overall.recall_at_1, 4),
                    "epochs": settings.epochs,
                    "learning_rate": settings.learning_rate,
                    "batch_size": self.compute.batch_size,
                },
            )

            result.adapter_directory = settings.save_adapter

            self.echo(f"\nadapter written to {settings.save_adapter}")

        if settings.report:
            from multilingual_embedding.utils.io import write_json

            write_json(settings.report, result.to_dict())

            self.echo(f"report written to {settings.report}")

        self.echo(f"\nTOTAL {result.seconds:.0f}s")

        return result

    def _select(self) -> tuple[list[MinedPair], list[MinedPair], dict[str, Any]]:
        """
        Draw the training and held-out sets, and describe what varies.

        The held-out set is drawn from a fixed-size sample of its own, so
        that changing the training volume does not change what the model
        is scored on. It moved once, and two Hindi runs reported
        baselines of 0.4238 and 0.4764 for the same checkpoint — a
        difference in the measuring stick that looked like a difference
        in the model.
        """

        settings = self.adaptation

        assert settings.pairs is not None  # checked by run() before this is reached

        seed = settings.seed or 0

        pool = sample_pairs(settings.pairs, settings.sampled, seed=seed)

        if len(pool) < settings.sampled:
            self.echo(f"\nonly {len(pool)} pairs available; wanted {settings.sampled}")

        evaluation_source = settings.eval_pairs_file or settings.pairs

        evaluation = sample_pairs(evaluation_source, settings.eval_pairs, seed=seed + 10_000)

        held_texts = {pair.positive for pair in evaluation}

        candidates = only(
            only([p for p in pool if p.positive not in held_texts], settings.train_kinds, "kind"),
            settings.train_languages,
            "language",
        )

        train = prefixed(
            candidates[: settings.train_pairs],
            settings.query_prefix,
            settings.passage_prefix,
        )

        held = prefixed(
            only(
                only(evaluation, settings.eval_kinds, "kind"),
                settings.eval_languages,
                "language",
            ),
            settings.query_prefix,
            settings.passage_prefix,
        )

        facts: dict[str, Any] = {
            "evaluation_source": evaluation_source,
            "train_kinds": values(train, "kind"),
            "eval_kinds": values(held, "kind"),
            "train_languages": values(train, "language"),
            "eval_languages": values(held, "language"),
        }

        self.echo(f"\ntrain {len(train):,} pairs   held out {len(held):,} pairs")

        # Laid out as a table because the whole design of the run is which
        # row differs. Reading it off two prose sentences is how a run
        # ends up varying two things at once without anyone noticing.
        self.echo(f"\n  {'facet':10} {'trained on':32} {'scored on':32}")

        observed: set[str] = set()

        axes = (
            ("kind", facts["train_kinds"], facts["eval_kinds"]),
            ("language", facts["train_languages"], facts["eval_languages"]),
            # File names rather than full paths, so a deep pair path
            # cannot push the VARIES column off the terminal — which is
            # the one thing in this table anybody needs to read.
            ("corpus", [settings.pairs.name], [evaluation_source.name]),
        )

        for facet, left, right in axes:
            differs = varying(left, right)

            if differs:
                observed.add(facet)

            marker = "VARIES" if differs else "fixed"

            self.echo(
                f"  {facet:10} {','.join(left) or '-':30.30} "
                f"{','.join(right) or '-':30.30} {marker}"
            )

        facts["observed"] = observed

        # A facet filter that starves the training set turns a comparison
        # of task shapes into a comparison of training-set sizes. Said out
        # loud, because it is invisible in the scores and it silently
        # decided a run: 40,000 requested with `train_kinds: title_lead`
        # drew about 7,000 pairs from a 42,000 sample, while `adjacent`
        # drew 25,000 from the same sample.
        filtered = settings.train_kinds or settings.train_languages

        if filtered and len(train) < settings.train_pairs:
            self.echo(
                f"\n  WARNING: asked for {settings.train_pairs:,} training pairs and the "
                f"filters left {len(train):,}.\n"
                f"  Raise adaptation.sample_pairs (currently {settings.sampled:,}) until this "
                f"stops, or runs naming different facets will differ in data volume as well "
                f"as in shape."
            )

        if settings.eval_pairs_file:
            self.echo(f"scored against {evaluation_source} (held fixed)")

        return train, held, facts


def adapt(config: ExperimentConfig, *, echo: Callable[[str], None] = print) -> AdaptationResult:
    """Run one adaptation experiment. Convenience wrapper over the pipeline."""

    return AdaptationPipeline(config, echo=echo).run()
