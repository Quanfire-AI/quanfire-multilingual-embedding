"""
Typed configuration objects.

Configuration is expressed as dataclasses rather than free-form
dictionaries so that a mistake surfaces at load time, next to the file
that caused it, instead of hours into a training run.

Each config validates itself in ``__post_init__``. Since dataclasses do
not re-run that hook on mutation, treat instances as immutable after
construction and use :meth:`ExperimentConfig.merged` to derive variants.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from multilingual_embedding.common.constants import (
    DEFAULT_CHARACTER_COVERAGE,
    DEFAULT_ENCODING,
    DEFAULT_RANDOM_SEED,
    DEFAULT_VOCAB_SIZE,
)
from multilingual_embedding.common.enums import DataProvenance, TokenizerModel
from multilingual_embedding.core.exceptions import ConfigurationError
from multilingual_embedding.utils.serialization import from_primitive, to_primitive
from multilingual_embedding.utils.validation import (
    require_in_range,
    require_non_negative,
    require_positive,
)

__all__ = [
    "ADAPTATIONS",
    "ADAPTATION_DESCRIPTIONS",
    "AdaptationConfig",
    "ComputeConfig",
    "CorpusConfig",
    "EmbeddingConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "TokenizerConfig",
]


@dataclass(slots=True)
class CorpusConfig:
    """
    Where the text comes from and how it is segmented.

    Attributes
    ----------
    source:
        File or directory to read. Directories are searched using
        ``patterns``.

    format:
        Reader to use: ``"text"``, ``"lines"``, ``"jsonl"`` or
        ``"auto"``. ``"auto"`` selects by file extension, which cannot
        distinguish a sentence-per-line file from prose, so name
        ``"lines"`` explicitly for those.

    patterns:
        Glob patterns applied when ``source`` is a directory.

    language:
        Default ISO 639 code for documents that do not declare one.

    encoding:
        Character encoding of the source files.

    text_field:
        For JSON Lines sources, the record key holding the text.

    min_sentence_characters:
        Sentences shorter than this are dropped. Filters out stray
        fragments left behind by segmentation.

    max_sentence_characters:
        Sentences longer than this are dropped. Guards against
        unsegmented boilerplate such as minified markup.

    lowercase:
        Whether the corpus reader lowercases text on load. Off by
        default: casing is a tokenizer concern, and destroying it here
        would be irreversible.
    """

    source: Path | None = None

    format: str = "auto"

    patterns: list[str] = field(default_factory=lambda: ["*.txt", "*.jsonl"])

    language: str | None = None

    encoding: str = DEFAULT_ENCODING

    text_field: str = "text"

    min_sentence_characters: int = 1

    max_sentence_characters: int = 10_000

    lowercase: bool = False

    def __post_init__(self) -> None:
        # Configs are routinely built from YAML, where paths arrive as
        # strings. Re-wrapping an existing Path is a no-op, so this coerces
        # unconditionally rather than testing the runtime type — which the
        # declared annotation would make look dead to a type checker.
        if self.source is not None:
            self.source = Path(self.source)

        require_positive(self.min_sentence_characters, name="min_sentence_characters")

        require_positive(self.max_sentence_characters, name="max_sentence_characters")

        if self.min_sentence_characters > self.max_sentence_characters:
            raise ConfigurationError(
                "min_sentence_characters must not exceed max_sentence_characters",
                minimum=self.min_sentence_characters,
                maximum=self.max_sentence_characters,
            )

        if self.format not in {"auto", "text", "lines", "jsonl"}:
            raise ConfigurationError(
                "Unsupported corpus format",
                format=self.format,
                supported=["auto", "text", "lines", "jsonl"],
            )


@dataclass(slots=True)
class TokenizerConfig:
    """
    Subword tokenizer training and inference settings.

    Attributes
    ----------
    model_type:
        Subword algorithm. Unigram is the default because it handles
        scripts without whitespace word boundaries more gracefully than
        BPE.

    vocab_size:
        Target vocabulary size, including special tokens.

    character_coverage:
        Fraction of characters in the training corpus the model must
        cover, in ``(0.0, 1.0]``. The 0.9995 default leaves rare
        characters to be encoded as byte fallback; for large-alphabet
        scripts such as Han this matters more than for Latin, where the
        permitted 1.0 — cover every character — is affordable.

    normalizers:
        Ordered normalizer specifications, each a ``{"type": ...}``
        mapping resolved through the normalizer registry. Applied by
        both tokenizers: :class:`WordTokenizer` runs them before
        pre-tokenization, and on the SentencePiece path they are applied
        to the training corpus as it is staged and re-applied by
        :class:`SentencePieceTokenizer` at encode time.

    pretokenizer:
        Pre-tokenizer specification, resolved through its registry.
        Applies to :class:`WordTokenizer` only. SentencePiece consumes a
        raw character stream by design — that is what lets one model
        serve scripts with no whitespace word boundaries — so it has
        nowhere to honour a pre-tokenizer. The trainer logs a warning
        rather than ignoring a non-default value silently.

    max_sentence_length:
        Longest training sentence, in bytes, passed to SentencePiece.

    input_sentence_size:
        Cap on how many sentences SentencePiece feeds into its EM
        training pass. ``0`` (the default, and SentencePiece's own) means
        "use every staged sentence". On a large monolingual corpus the
        uncapped EM step slows to a crawl — a from-scratch Hindi run of
        ~2.2M sentences with a 32k unigram vocabulary stalled here with no
        progress logged. Setting a cap (e.g. ``1_000_000``) subsamples the
        EM training set to that many sentences and completes, at no real
        quality cost: the seed vocabulary is estimated from a large random
        sample, not the whole corpus. Sentences beyond the cap still shape
        character coverage. Paired with ``shuffle_input_sentence`` so the
        subsample is drawn across the corpus, not just its head.

    shuffle_input_sentence:
        Whether SentencePiece shuffles before it subsamples to
        ``input_sentence_size``. Only matters when that cap is below the
        corpus size; then it is what keeps the EM sample representative
        rather than the first N lines. Defaults to True, matching
        SentencePiece.

    model_prefix:
        Base filename for the trained model artefacts.
    """

    model_type: TokenizerModel = TokenizerModel.UNIGRAM

    vocab_size: int = DEFAULT_VOCAB_SIZE

    character_coverage: float = DEFAULT_CHARACTER_COVERAGE

    normalizers: list[dict[str, Any]] = field(
        default_factory=lambda: [{"type": "nfkc"}, {"type": "whitespace"}]
    )

    pretokenizer: dict[str, Any] = field(default_factory=lambda: {"type": "whitespace"})

    max_sentence_length: int = 16_384

    input_sentence_size: int = 0

    shuffle_input_sentence: bool = True

    model_prefix: str = "tokenizer"

    def __post_init__(self) -> None:
        if isinstance(self.model_type, str):
            try:
                self.model_type = TokenizerModel(self.model_type)
            except ValueError as error:
                raise ConfigurationError(
                    "Unsupported tokenizer model",
                    model_type=self.model_type,
                    supported=[member.value for member in TokenizerModel],
                ) from error

        require_positive(self.vocab_size, name="vocab_size")

        # The interval is half-open: 1.0 is a legitimate SentencePiece
        # value meaning "cover every character", while 0.0 would ask the
        # model to cover nothing. `require_in_range` only offers wholly
        # inclusive or wholly exclusive bounds, so the open lower bound
        # is expressed by the additional positivity check.
        require_positive(self.character_coverage, name="character_coverage")

        require_in_range(
            self.character_coverage,
            name="character_coverage",
            minimum=0.0,
            maximum=1.0,
        )

        require_positive(self.max_sentence_length, name="max_sentence_length")

        # Zero is a legitimate value here (SentencePiece's "use every
        # sentence"), so this is non-negative rather than positive.
        require_non_negative(self.input_sentence_size, name="input_sentence_size")


@dataclass(slots=True)
class EmbeddingConfig:
    """
    Word embedding model hyperparameters.

    Attributes
    ----------
    dimension:
        Size of each vector.

    window:
        Maximum context distance on either side of the centre token.

    min_count:
        Tokens appearing fewer times than this are excluded from the
        embedding vocabulary. Rare tokens receive too few updates to
        acquire a meaningful vector and mostly add noise.

    negative_samples:
        Number of negative examples drawn per positive pair.

    epochs:
        Passes over the corpus.

    learning_rate:
        Initial learning rate, decayed linearly to ``min_learning_rate``.

    min_learning_rate:
        Floor for the decayed learning rate.

    subsample_threshold:
        Frequency above which tokens are randomly discarded during
        training. Very frequent tokens carry little information; the
        classic word2vec value of 1e-3 is used.

    seed:
        Random seed for weight initialisation and sampling. ``None``
        means "inherit the experiment seed": when this config is nested
        in an :class:`ExperimentConfig`, that config fills the field in
        from :attr:`ExperimentConfig.seed`. Setting it to an int
        overrides the experiment seed. A standalone config left at
        ``None`` falls back to
        :data:`~multilingual_embedding.common.constants.DEFAULT_RANDOM_SEED`
        via :attr:`resolved_seed`, so training is never accidentally
        unseeded.
    """

    dimension: int = 128

    window: int = 5

    min_count: int = 5

    negative_samples: int = 5

    epochs: int = 5

    learning_rate: float = 0.025

    min_learning_rate: float = 0.0001

    subsample_threshold: float = 1e-3

    seed: int | None = None

    def __post_init__(self) -> None:
        require_positive(self.dimension, name="dimension")

        require_positive(self.window, name="window")

        require_positive(self.min_count, name="min_count")

        require_non_negative(self.negative_samples, name="negative_samples")

        require_positive(self.epochs, name="epochs")

        require_positive(self.learning_rate, name="learning_rate")

        require_positive(self.min_learning_rate, name="min_learning_rate")

        require_non_negative(self.subsample_threshold, name="subsample_threshold")

        # None is the documented "inherit" marker and must survive
        # validation; a supplied seed is held to the same rule as
        # ExperimentConfig.seed so the two cannot disagree about what a
        # legal seed is.
        if self.seed is not None:
            require_non_negative(self.seed, name="seed")

        if self.min_learning_rate > self.learning_rate:
            raise ConfigurationError(
                "min_learning_rate must not exceed learning_rate",
                learning_rate=self.learning_rate,
                min_learning_rate=self.min_learning_rate,
            )

    @property
    def resolved_seed(self) -> int:
        """
        The seed to actually train with.

        Falls back to the framework default when :attr:`seed` is still
        ``None``, which happens only for a config used outside an
        :class:`ExperimentConfig`. Consumers read this rather than
        :attr:`seed` so that no code path can silently train unseeded.
        """

        return self.seed if self.seed is not None else DEFAULT_RANDOM_SEED


@dataclass(slots=True)
class PretrainingConfig:
    """
    From-scratch masked-language pretraining settings.

    Read only by ``qfme pretrain``; the static Word2Vec training pipeline
    ignores it, and a config that omits the section gets defaults rather
    than an error, so one experiment file can carry both paths. This is the
    other, contextual answer to the same question :class:`EmbeddingConfig`
    asks — vectors from a transformer taught by masking, rather than from a
    co-occurrence table.

    The knobs split in two. The first group sizes the encoder that is built
    from scratch; the second governs the masked-language objective that
    teaches it what its tokens mean. Machine-shaped settings — device,
    precision, batch size — live in :class:`ComputeConfig`, not here, so one
    experiment pretrains unchanged on a laptop and a GPU box.

    Attributes
    ----------
    dimension, layers, heads, feedforward_dimension, max_length, dropout:
        The encoder's shape. ``feedforward_dimension`` of ``None`` lets the
        model default it to four times ``dimension``, the usual transformer
        ratio. ``dimension`` must divide evenly by ``heads``.

    epochs, learning_rate, warmup_ratio, weight_decay, max_gradient_norm:
        The optimisation schedule. ``learning_rate`` is the peak reached at
        the end of warmup and decayed after it; it is higher than the
        contrastive default because a from-scratch table has further to
        travel.

    mask_probability:
        Share of eligible tokens hidden per sequence. 0.15 is standard —
        low enough to leave usable context, high enough to supervise
        densely.

    tie_weights:
        Share the output projection with the embedding table.

    seed:
        ``None`` inherits the experiment seed, exactly as
        :attr:`EmbeddingConfig.seed` does; an int overrides it. A standalone
        config left at ``None`` falls back to the framework default via
        :attr:`resolved_seed`, so pretraining is never accidentally
        unseeded.

    data_provenance:
        Where the pretraining corpus came from — ``public``, ``synthetic``
        or ``licensed``. A pretrained encoder is a model trained on that
        text, so ``qfme pretrain`` refuses to save one without this
        declaration, exactly as fine-tuning does. Empty by default so an
        experiment that never pretrains still constructs; the requirement
        is enforced by :class:`~multilingual_embedding.pipelines.pretraining.PretrainingPipeline`
        when a pretrain actually runs. Customer data must never reach it.
    """

    dimension: int = 256

    layers: int = 4

    heads: int = 4

    feedforward_dimension: int | None = None

    max_length: int = 256

    dropout: float = 0.1

    epochs: int = 3

    learning_rate: float = 1e-4

    warmup_ratio: float = 0.1

    weight_decay: float = 0.01

    max_gradient_norm: float = 1.0

    mask_probability: float = 0.15

    tie_weights: bool = True

    seed: int | None = None

    data_provenance: str = ""

    def __post_init__(self) -> None:
        require_positive(self.dimension, name="dimension")

        require_positive(self.layers, name="layers")

        require_positive(self.heads, name="heads")

        if self.feedforward_dimension is not None:
            require_positive(self.feedforward_dimension, name="feedforward_dimension")

        require_positive(self.max_length, name="max_length")

        require_positive(self.epochs, name="epochs")

        require_positive(self.learning_rate, name="learning_rate")

        require_non_negative(self.weight_decay, name="weight_decay")

        require_positive(self.max_gradient_norm, name="max_gradient_norm")

        require_non_negative(self.dropout, name="dropout")

        # The attention heads split the width between them; a remainder
        # would leave a head with a fractional slice, which the encoder
        # cannot build. Caught here so the failure names the config rather
        # than surfacing as a reshape error an hour into a GPU run.
        if self.dimension % self.heads != 0:
            raise ConfigurationError(
                "dimension must be divisible by heads",
                dimension=self.dimension,
                heads=self.heads,
            )

        if self.dropout >= 1.0:
            raise ConfigurationError("dropout must lie in [0, 1)", dropout=self.dropout)

        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ConfigurationError(
                "warmup_ratio must lie in [0, 1)",
                warmup_ratio=self.warmup_ratio,
            )

        if not 0.0 < self.mask_probability < 1.0:
            raise ConfigurationError(
                "mask_probability must lie in (0, 1)",
                mask_probability=self.mask_probability,
            )

        if self.seed is not None:
            require_non_negative(self.seed, name="seed")

        # Validated here if declared; *required* only when a pretrain runs
        # (see PretrainingPipeline), because an ExperimentConfig that never
        # pretrains still builds a default PretrainingConfig and must not
        # be forced to answer a question its run never asks.
        if self.data_provenance and self.data_provenance not in set(DataProvenance):
            raise ConfigurationError(
                f"unknown data_provenance {self.data_provenance!r}",
                supported=sorted(DataProvenance),
            )

    @property
    def resolved_seed(self) -> int:
        """
        The seed to actually pretrain with.

        Falls back to the framework default when :attr:`seed` is still
        ``None``, which happens only for a config used outside an
        :class:`ExperimentConfig`. Consumers read this rather than
        :attr:`seed` so no code path can silently train unseeded.
        """

        return self.seed if self.seed is not None else DEFAULT_RANDOM_SEED


@dataclass(slots=True)
class EvaluationConfig:
    """
    Which metrics to compute and how.

    Attributes
    ----------
    top_k:
        Neighbourhood size for retrieval metrics.

    similarity_dataset:
        Optional path to a word similarity dataset in JSON Lines form,
        each record holding ``word_a``, ``word_b`` and ``score``.

    report_directory:
        Where evaluation reports are written.

    sample_size:
        Number of corpus sentences sampled for tokenizer statistics.
        ``0`` uses the whole corpus.
    """

    top_k: int = 10

    similarity_dataset: Path | None = None

    report_directory: Path = field(default_factory=lambda: Path("reports"))

    sample_size: int = 0

    def __post_init__(self) -> None:
        # See CorpusConfig.__post_init__: YAML supplies these as strings.
        if self.similarity_dataset is not None:
            self.similarity_dataset = Path(self.similarity_dataset)

        self.report_directory = Path(self.report_directory)

        require_positive(self.top_k, name="top_k")

        require_non_negative(self.sample_size, name="sample_size")


@dataclass(slots=True)
class ComputeConfig:
    """
    Settings that describe the machine rather than the experiment.

    These are the values that legitimately differ between a laptop and a
    training box: which device, what numeric precision, how much can be
    held in memory at once. Keeping them in their own section is what
    lets one experiment run unchanged on both — the science lives in the
    other sections and stays fixed, while this section is swapped.

    Nothing here changes what is computed, only how much of it fits and
    how fast it runs. A result should be reproducible across profiles up
    to floating-point tolerance and the batch size, which does change
    the number of in-batch negatives and therefore the outcome. That one
    exception is why ``batch_size`` is recorded alongside the artefacts.

    Attributes
    ----------
    device:
        ``"auto"`` resolves CUDA, then Apple MPS, then CPU. Name one
        explicitly to pin it — most usefully ``"cpu"`` on a GPU machine,
        to tell a real bug apart from a device-specific one.

    precision:
        ``"fp32"`` everywhere, or ``"bf16"`` for mixed precision on
        hardware that supports it. ``bf16`` roughly halves activation
        memory and is materially faster on recent NVIDIA parts. It has
        the same exponent range as fp32, so unlike fp16 it does not need
        loss scaling.

    batch_size:
        Pairs per step. In contrastive training this doubles as the
        number of negatives each query is contrasted against, so it is
        the single knob most worth raising on a larger card.

    gradient_checkpoint_chunk:
        Chunk size for gradient caching. ``0`` disables it and encodes
        the batch in one pass. When set, peak memory follows the chunk
        rather than the batch, which is what allows a batch size the
        card could not otherwise hold. Mathematically exact either way.


    Note what is deliberately absent. There is no ``workers`` setting,
    because training is single-process and nothing would read one. A
    config field that silently does nothing is worse than a missing
    one — it reads as a tuning knob and invites someone to spend an
    afternoon turning it. It belongs here the day a data loader does.
    """

    device: str = "auto"

    precision: str = "fp32"

    batch_size: int = 16

    gradient_checkpoint_chunk: int = 0

    def __post_init__(self) -> None:
        require_positive(self.batch_size, name="batch_size")

        require_non_negative(
            self.gradient_checkpoint_chunk,
            name="gradient_checkpoint_chunk",
        )

        if self.precision not in _PRECISIONS:
            raise ConfigurationError(
                "Unsupported precision",
                precision=self.precision,
                supported=sorted(_PRECISIONS),
            )

        # Devices are validated by shape rather than against what this
        # machine happens to have. A GPU profile must survive being read
        # on the laptop that wrote it — to be diffed, validated in CI, or
        # committed — and rejecting "cuda" here would make that
        # impossible. An unavailable device fails later, at the point of
        # use, where the error can say so precisely.
        root = self.device.split(":", 1)[0]

        if root not in _DEVICES:
            raise ConfigurationError(
                "Unsupported device",
                device=self.device,
                supported=sorted(_DEVICES),
            )


_PRECISIONS = frozenset({"fp32", "bf16"})

_DEVICES = frozenset({"auto", "cpu", "cuda", "mps"})

# What each declared experiment requires to vary. Everything not named
# for a mode must be held fixed, and the check is two-sided: a `task` run
# whose languages also differ is not a task result, it is two changes at
# once wearing one name.
ADAPTATIONS: dict[str, tuple[str, ...]] = {
    "in-distribution": (),
    "task": ("kind",),
    "language": ("language",),
    "domain": ("corpus",),
    "task+language": ("kind", "language"),
    "task+domain": ("kind", "corpus"),
}

ADAPTATION_DESCRIPTIONS: dict[str, str] = {
    "in-distribution": "same task, same corpus — how much adaptation helps where it was trained",
    "task": "same corpus, different task shape — did it learn retrieval or the mining scheme",
    "language": "same task, different language — does the adaptation cross scripts",
    "domain": "same task, different corpus — does it survive contact with your own text",
    "task+language": "task shape and language both change",
    "task+domain": "task shape and corpus both change",
}


@dataclass(slots=True)
class AdaptationConfig:
    """
    Adapting a published checkpoint to a domain, and measuring whether it helped.

    This is the science half of an adaptation run. Everything the
    *machine* dictates — device, precision, batch size, gradient
    caching — lives in :class:`ComputeConfig` and is swapped by
    ``--profile``, so the same experiment file describes the run on a
    laptop and on a GPU box and the two are comparable.

    The one exception is stated in :class:`ComputeConfig`: batch size is
    also the number of in-batch negatives, so it does change the result.
    It is recorded with the artefacts for that reason.

    Attributes
    ----------
    checkpoint:
        Model name or local directory to adapt. The baseline is this
        model as published, which is the only honest one — beating
        chance, or beating an untrained model, says nothing about
        whether adaptation was worth doing.

    pairs:
        Mined pair file to train on, as written by ``qfme mine-pairs``.

    eval_pairs_file:
        Score against this file instead of :attr:`pairs`. Holding it
        fixed is what lets two runs that train on different data be
        compared: without it, a run that changes what it trains on also
        changes what it is judged by, and the two cannot be separated.

    train_pairs, eval_pairs:
        How many pairs to train on and to score against.

    sample_pairs:
        How many pairs to draw from the file *before* filtering by kind
        or language. Defaults to ``train_pairs + eval_pairs``, which is
        right when no filter is set and wrong when one is: a kind holding
        a sixth of the file yields a sixth of the sample, ``train_pairs``
        stops binding, and two runs naming different kinds silently
        differ in data volume as well as in shape.

    train_kinds, eval_kinds, train_languages, eval_languages:
        Facet filters. Empty means everything. Give a pair of them
        disjoint values to vary that facet; leave them equal to hold it
        fixed.

    adaptation:
        What the run claims to measure, one of :data:`ADAPTATIONS`.
        Checked against what the filters actually do, and the run is
        refused if they disagree. The label outlives the command line —
        it is what gets quoted six months later — so it must not be able
        to be wrong.

    epochs, learning_rate:
        Contrastive training schedule.

    rank, alpha, dropout, targets:
        The LoRA update. ``alpha`` defaults to twice the rank when left
        at ``None``, which is the usual convention. Nothing here trains
        the base weights, so the comparison is between one model and
        itself plus a small adapter rather than between two models that
        differ in unknown ways.

    pooling, max_length, query_prefix, passage_prefix:
        Properties of the checkpoint rather than of the experiment. The
        prefixes matter more than they look: an E5-family model served
        without ``"query: "`` and ``"passage: "`` returns vectors that
        are the right shape, the right norm, free of NaN, and encode the
        wrong thing. Nothing raises; the score is simply lower.

    save_adapter:
        Directory to write the trained adapter to. Without it the run
        produces a measurement rather than a model.

    data_provenance:
        Where the training data came from, one of :class:`DataProvenance`
        (``public``, ``synthetic``, ``licensed``). Empty is allowed only
        for a measurement-only run; the moment ``save_adapter`` is set,
        this must be declared, because a shipped model without a recorded
        basis for its training data is the exact thing this field exists
        to prevent.

    report:
        Where to write the full before/after comparison as JSON.

    seed:
        Seeds pair sampling and training. ``None`` inherits
        :attr:`ExperimentConfig.seed`.
    """

    checkpoint: str = ""

    pairs: Path | None = None

    eval_pairs_file: Path | None = None

    train_pairs: int = 20_000

    eval_pairs: int = 2_000

    sample_pairs: int | None = None

    train_kinds: tuple[str, ...] = ()

    eval_kinds: tuple[str, ...] = ()

    train_languages: tuple[str, ...] = ()

    eval_languages: tuple[str, ...] = ()

    adaptation: str = "in-distribution"

    epochs: int = 1

    learning_rate: float = 1e-4

    rank: int = 16

    alpha: int | None = None

    dropout: float = 0.0

    targets: tuple[str, ...] = ("query", "value")

    pooling: str = "mean"

    max_length: int = 256

    query_prefix: str = ""

    passage_prefix: str = ""

    save_adapter: Path | None = None

    data_provenance: str = ""

    report: Path | None = None

    seed: int | None = None

    def __post_init__(self) -> None:
        # See CorpusConfig.__post_init__: YAML supplies paths as strings.
        for name in ("pairs", "eval_pairs_file", "save_adapter", "report"):
            value = getattr(self, name)

            if value is not None:
                setattr(self, name, Path(value))

        # YAML has no tuple, and a comma-separated string is what the CLI
        # hands over. Both are normalised here so that everything
        # downstream sees a tuple and no caller has to guess.
        for name in ("train_kinds", "eval_kinds", "train_languages", "eval_languages", "targets"):
            setattr(self, name, _as_names(getattr(self, name)))

        require_positive(self.train_pairs, name="train_pairs")

        require_positive(self.eval_pairs, name="eval_pairs")

        require_positive(self.epochs, name="epochs")

        require_positive(self.learning_rate, name="learning_rate")

        require_positive(self.rank, name="rank")

        require_positive(self.max_length, name="max_length")

        require_in_range(self.dropout, name="dropout", minimum=0.0, maximum=1.0)

        if self.sample_pairs is not None:
            require_positive(self.sample_pairs, name="sample_pairs")

        if self.seed is not None:
            require_non_negative(self.seed, name="seed")

        if self.alpha is not None:
            require_positive(self.alpha, name="alpha")

        if self.adaptation not in ADAPTATIONS:
            raise ConfigurationError(
                "Unknown adaptation mode",
                adaptation=self.adaptation,
                supported=sorted(ADAPTATIONS),
            )

        if self.data_provenance and self.data_provenance not in set(DataProvenance):
            raise ConfigurationError(
                "Unknown data provenance",
                data_provenance=self.data_provenance,
                supported=sorted(DataProvenance),
            )

        # Fail closed: a run that produces a model must say where its
        # training data came from. A measurement-only run (no save) need
        # not, because it ships nothing. Checked here rather than at save
        # time so the run is refused before it spends the training, not
        # after.
        if self.save_adapter is not None and not self.data_provenance:
            raise ConfigurationError(
                "Saving an adapter requires declaring data_provenance "
                "(where its training data came from)",
                supported=sorted(DataProvenance),
            )

        if self.pooling not in {"mean", "cls"}:
            raise ConfigurationError(
                "Unsupported pooling",
                pooling=self.pooling,
                supported=["mean", "cls"],
            )

        if not self.targets:
            raise ConfigurationError(
                "LoRA must attach to at least one module",
                targets=list(self.targets),
            )

    @property
    def measures(self) -> str:
        """One line saying what this run's declared mode actually answers."""

        return ADAPTATION_DESCRIPTIONS[self.adaptation]

    @property
    def sampled(self) -> int:
        """Pairs to draw before filtering, with the default resolved."""

        return self.sample_pairs or (self.train_pairs + self.eval_pairs)

    @property
    def resolved_alpha(self) -> int:
        """The LoRA numerator, with the twice-the-rank convention applied."""

        return self.alpha if self.alpha is not None else 2 * self.rank


def _as_names(value: object) -> tuple[str, ...]:
    """Normalise a comma-separated string or any sequence into a tuple of names."""

    if isinstance(value, str):
        return tuple(name.strip() for name in value.split(",") if name.strip())

    if isinstance(value, Sequence):
        return tuple(str(name).strip() for name in value if str(name).strip())

    raise ConfigurationError("Expected a comma-separated string or a list", value=repr(value))


@dataclass(slots=True)
class FinetuneConfig:
    """
    Stage two of the from-scratch neural path: contrastive fine-tuning.

    ``qfme pretrain`` produces an encoder that has only ever seen a
    masked-language objective. It is a poor sentence embedder — it was
    never taught that two paraphrases should land near each other — and
    this is the section that fixes that, by fine-tuning the whole encoder
    on mined pairs with an in-batch contrastive loss. Where
    :class:`AdaptationConfig` freezes a published checkpoint and trains a
    LoRA adapter over it, this owns the weights and moves all of them:
    there is no base model to preserve, so there is no adapter.

    Like an adaptation run it scores the encoder on held-out pairs either
    side of training, so the report says whether the fine-tune helped
    rather than merely that it happened. Unlike one it declares no
    ablation mode — this is not an experiment in what transfers, it is the
    step that turns a pretrained artefact into a usable model — so it
    carries none of :class:`AdaptationConfig`'s vary/hold machinery.

    The machine half lives in :class:`ComputeConfig` exactly as it does
    for adaptation, so one file describes the fine-tune on a laptop and on
    a GPU box.

    Attributes
    ----------
    source:
        The ``qfme pretrain`` experiment directory to fine-tune, holding
        ``tokenizer/`` and ``encoder/``. Required.

    pairs:
        Mined pair file to train on, as written by ``qfme mine-pairs``.

    eval_pairs_file:
        Score against this file instead of :attr:`pairs`. Holding it
        fixed is what lets two fine-tunes that train on different data be
        compared on the same measuring stick.

    train_pairs, eval_pairs, sample_pairs:
        How many pairs to train on, to score against, and to draw from
        the file before filtering — the same three knobs, and the same
        defaulting, as :class:`AdaptationConfig`.

    train_kinds, eval_kinds, train_languages, eval_languages:
        Facet filters. Empty means everything. Present so a fine-tune can
        be confined to one language or pair kind; a run that names a facet
        holding nothing is refused rather than trained on zero pairs.

    epochs, learning_rate, temperature, warmup_ratio, weight_decay:
        The contrastive schedule. The learning rate defaults higher than
        adaptation's because full fine-tuning a small from-scratch encoder
        is not the same regime as nudging a well-pretrained checkpoint
        through a low-rank adapter.

    save:
        Whether to write the fine-tuned encoder back out as a servable
        experiment directory. ``True`` — the point of stage two is a
        usable model. Set it ``False`` for a measurement-only run that
        answers "would this fine-tune help?" without shipping anything.

    data_provenance:
        Where the training data came from, one of :class:`DataProvenance`.
        Required the moment :attr:`save` is set: a shipped model must
        carry a recorded basis for its training data, and this fails
        closed so the run is refused before it spends the training rather
        than after.

    report:
        Where to write the full before/after comparison as JSON.

    seed:
        Seeds pair sampling and training. ``None`` inherits
        :attr:`ExperimentConfig.seed`.
    """

    source: Path | None = None

    pairs: Path | None = None

    eval_pairs_file: Path | None = None

    train_pairs: int = 20_000

    eval_pairs: int = 2_000

    sample_pairs: int | None = None

    train_kinds: tuple[str, ...] = ()

    eval_kinds: tuple[str, ...] = ()

    train_languages: tuple[str, ...] = ()

    eval_languages: tuple[str, ...] = ()

    epochs: int = 3

    learning_rate: float = 2e-5

    temperature: float = 0.05

    warmup_ratio: float = 0.1

    weight_decay: float = 0.01

    save: bool = True

    data_provenance: str = ""

    report: Path | None = None

    seed: int | None = None

    def __post_init__(self) -> None:
        # See CorpusConfig.__post_init__: YAML supplies paths as strings.
        for name in ("source", "pairs", "eval_pairs_file", "report"):
            value = getattr(self, name)

            if value is not None:
                setattr(self, name, Path(value))

        for name in ("train_kinds", "eval_kinds", "train_languages", "eval_languages"):
            setattr(self, name, _as_names(getattr(self, name)))

        require_positive(self.train_pairs, name="train_pairs")

        require_positive(self.eval_pairs, name="eval_pairs")

        require_positive(self.epochs, name="epochs")

        require_positive(self.learning_rate, name="learning_rate")

        require_positive(self.temperature, name="temperature")

        # [0, 1), matching ContrastiveConfig: a warmup that never ends is a
        # schedule with no training phase. Refused here so the run is
        # stopped when the config is built, not after the encoder is loaded.
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ConfigurationError(
                "warmup_ratio must lie in [0, 1)",
                warmup_ratio=self.warmup_ratio,
            )

        require_non_negative(self.weight_decay, name="weight_decay")

        if self.sample_pairs is not None:
            require_positive(self.sample_pairs, name="sample_pairs")

        if self.seed is not None:
            require_non_negative(self.seed, name="seed")

        if self.data_provenance and self.data_provenance not in set(DataProvenance):
            raise ConfigurationError(
                "Unknown data provenance",
                data_provenance=self.data_provenance,
                supported=sorted(DataProvenance),
            )

        # Fail closed, in the spirit of AdaptationConfig: a run that writes
        # a model must say where its training data came from. The check is
        # gated on the section being *engaged* — a source or a pair file
        # named — so an ExperimentConfig that carries no finetune section
        # at all keeps a valid default rather than failing to construct,
        # while a real fine-tune that would save without a provenance is
        # still refused before it spends the training.
        engaged = self.source is not None or self.pairs is not None

        if engaged and self.save and not self.data_provenance:
            raise ConfigurationError(
                "Saving a fine-tuned encoder requires declaring data_provenance "
                "(where its training data came from). Pass save=false for a "
                "measurement-only run that ships nothing",
                supported=sorted(DataProvenance),
            )

    @property
    def sampled(self) -> int:
        """Pairs to draw before filtering, with the default resolved."""

        return self.sample_pairs or (self.train_pairs + self.eval_pairs)


@dataclass(slots=True)
class ExperimentConfig:
    """
    Root configuration binding every stage of one experiment.

    Attributes
    ----------
    name:
        Experiment identifier, used for output directory naming.

    seed:
        Global seed. Propagated into :attr:`EmbeddingConfig.seed` when
        that field is left at ``None``; an embedding config that names
        its own seed keeps it.

    output_directory:
        Root for all artefacts produced by this experiment.

    corpus, tokenizer, embedding, evaluation:
        Per stage configuration.

    adaptation:
        Adapting a published checkpoint. Read only by ``qfme adapt``;
        the from-scratch training pipeline ignores it, and a config
        that omits the section gets defaults rather than an error, so
        one experiment file can carry both paths.

    pretraining:
        From-scratch masked-language pretraining. Read only by
        ``qfme pretrain``; the static training pipeline ignores it, and
        an omitted section gets defaults, so one file can carry the
        static and contextual paths side by side.

    compute:
        Machine-shaped settings. The section a profile swaps, and
        the only one expected to differ between a laptop and a
        training box.
    """

    name: str = "default"

    seed: int = DEFAULT_RANDOM_SEED

    output_directory: Path = field(default_factory=lambda: Path("artifacts"))

    corpus: CorpusConfig = field(default_factory=CorpusConfig)

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)

    pretraining: PretrainingConfig = field(default_factory=PretrainingConfig)

    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)

    compute: ComputeConfig = field(default_factory=ComputeConfig)

    def __post_init__(self) -> None:
        # See CorpusConfig.__post_init__: YAML supplies this as a string.
        self.output_directory = Path(self.output_directory)

        # Coercion lives here, not only in `from_primitive`, so that
        # every route to an ExperimentConfig validates its sections.
        # Direct construction with a plain dict is the natural thing to
        # write in a notebook or a test, and it used to store the dict
        # verbatim — producing a config whose `.embedding` was not an
        # EmbeddingConfig at all, holding values validation would have
        # rejected.
        self.corpus = _coerced_section(self.corpus, CorpusConfig, name="corpus")

        self.tokenizer = _coerced_section(self.tokenizer, TokenizerConfig, name="tokenizer")

        self.embedding = _coerced_section(self.embedding, EmbeddingConfig, name="embedding")

        self.evaluation = _coerced_section(self.evaluation, EvaluationConfig, name="evaluation")

        self.adaptation = _coerced_section(self.adaptation, AdaptationConfig, name="adaptation")

        self.pretraining = _coerced_section(self.pretraining, PretrainingConfig, name="pretraining")

        self.finetune = _coerced_section(self.finetune, FinetuneConfig, name="finetune")

        self.compute = _coerced_section(self.compute, ComputeConfig, name="compute")

        require_non_negative(self.seed, name="seed")

        if not self.name.strip():
            raise ConfigurationError("Experiment name must not be empty")

        # Resolve inheritance here rather than at read time so that the
        # config persisted next to the artefacts records the concrete
        # seed the run used. A file saying `seed: null` would not be a
        # record of anything.
        if self.embedding.seed is None:
            self.embedding.seed = self.seed

        if self.adaptation.seed is None:
            self.adaptation.seed = self.seed

        if self.pretraining.seed is None:
            self.pretraining.seed = self.seed

        if self.finetune.seed is None:
            self.finetune.seed = self.seed

    @property
    def experiment_directory(self) -> Path:
        """Directory holding this experiment's artefacts."""

        return self.output_directory / self.name

    @property
    def tokenizer_directory(self) -> Path:
        """Directory holding trained tokenizer artefacts."""

        return self.experiment_directory / "tokenizer"

    @property
    def embedding_directory(self) -> Path:
        """Directory holding trained embedding artefacts."""

        return self.experiment_directory / "embedding"

    @property
    def encoder_directory(self) -> Path:
        """Directory holding the pretrained contextual encoder."""

        return self.experiment_directory / "encoder"

    def to_dict(self) -> dict[str, Any]:
        """Reduce to primitives for persistence alongside the artefacts."""

        result = to_primitive(self)

        assert isinstance(result, dict)

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        """Rebuild from primitives, validating every nested section."""

        return from_primitive(cls, data)

    def merged(self, overrides: dict[str, Any]) -> ExperimentConfig:
        """
        Return a new config with ``overrides`` applied.

        Nested mappings are merged recursively, so a caller can override
        ``embedding.dimension`` without restating the rest of the
        embedding section. The result is revalidated on construction.

        An override of the global ``seed`` also moves an embedding seed
        that was only ever inherited from it — otherwise ``-o seed=11``
        would change the top-level value and leave the embedding stage
        running on the old one, which is the very trap this config is
        meant to avoid.
        """

        base = self.to_dict()

        # A stage seed equal to the experiment seed carries no intent of
        # its own: either it was inherited, or it was set to the same
        # number, in which case moving them together is still what the
        # caller asked for. Reset it to the inherit marker so
        # __post_init__ re-derives it from the incoming seed.
        if "seed" in overrides:
            for section in ("embedding", "adaptation", "pretraining", "finetune"):
                if base[section].get("seed") == base["seed"]:
                    base[section]["seed"] = None

        return ExperimentConfig.from_dict(_deep_merge(base, overrides))


def _coerced_section[SectionT](
    value: object,
    section_type: type[SectionT],
    *,
    name: str,
) -> SectionT:
    """
    Return ``value`` as ``section_type``, validating it on the way.

    Parameters
    ----------
    value:
        Either an instance of ``section_type``, which is returned
        unchanged, or a mapping of field names to values.

    section_type:
        The config dataclass this section must be.

    name:
        Section name, for the error message.

    Returns
    -------
    An instance of ``section_type``.

    Raises
    ------
    ConfigurationError
        If ``value`` is neither an instance nor a mapping.

    SerializationError
        If the mapping names a field the section does not declare.

    ValidationError
        If a value fails the section's own precondition checks.
    """

    if isinstance(value, section_type):
        return value

    if isinstance(value, Mapping):
        # `from_primitive` runs the section's __post_init__, so building
        # through it is what makes the nested validation actually happen.
        return from_primitive(section_type, dict(value))

    raise ConfigurationError(
        "Configuration section has the wrong type",
        section=name,
        expected=section_type.__name__,
        received=type(value).__name__,
    )


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overrides`` into a copy of ``base``."""

    merged = dict(base)

    for key, value in overrides.items():
        existing = merged.get(key)

        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value

    return merged
