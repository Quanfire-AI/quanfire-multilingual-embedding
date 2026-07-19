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

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from multilingual_embedding.common.constants import (
    DEFAULT_CHARACTER_COVERAGE,
    DEFAULT_ENCODING,
    DEFAULT_RANDOM_SEED,
    DEFAULT_VOCAB_SIZE,
)
from multilingual_embedding.common.enums import TokenizerModel
from multilingual_embedding.core.exceptions import ConfigurationError
from multilingual_embedding.utils.serialization import from_primitive, to_primitive
from multilingual_embedding.utils.validation import (
    require_in_range,
    require_non_negative,
    require_positive,
)

__all__ = [
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
        Reader to use: ``"text"``, ``"jsonl"`` or ``"auto"``. ``"auto"``
        selects by file extension.

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

        if self.format not in {"auto", "text", "jsonl"}:
            raise ConfigurationError(
                "Unsupported corpus format",
                format=self.format,
                supported=["auto", "text", "jsonl"],
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
    """

    name: str = "default"

    seed: int = DEFAULT_RANDOM_SEED

    output_directory: Path = field(default_factory=lambda: Path("artifacts"))

    corpus: CorpusConfig = field(default_factory=CorpusConfig)

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

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

        require_non_negative(self.seed, name="seed")

        if not self.name.strip():
            raise ConfigurationError("Experiment name must not be empty")

        # Resolve inheritance here rather than at read time so that the
        # config persisted next to the artefacts records the concrete
        # seed the run used. A file saying `seed: null` would not be a
        # record of anything.
        if self.embedding.seed is None:
            self.embedding.seed = self.seed

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

        embedding = base["embedding"]

        # An embedding seed equal to the experiment seed carries no
        # intent of its own: either it was inherited, or it was set to
        # the same number, in which case moving them together is still
        # what the caller asked for. Reset it to the inherit marker so
        # __post_init__ re-derives it from the incoming seed.
        if "seed" in overrides and embedding.get("seed") == base["seed"]:
            embedding["seed"] = None

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
