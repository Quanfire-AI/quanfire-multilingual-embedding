"""
Persisting an adapted encoder as a few megabytes.

An adaptation run that measures a 30% improvement and then exits has
produced a number, not a model. This is what turns the one into the
other.

Only the adapter is stored. The base checkpoint is named rather than
copied, because it is hundreds of megabytes, it is already on disk in a
cache, and it did not change — LoRA freezes it. What gets written is the
low-rank update, which at rank 32 on a 118M-parameter encoder is about
3.4 MB. One base model plus several of these is several domain models.

**The artefact records how the encoder must be used, not only what it
weighs.** Pooling and the query and passage prefixes belong to the model
as firmly as its weights do: a checkpoint trained mean-pooled and served
CLS-pooled, or an E5 model served without its ``query: `` prefix,
produces plausible vectors that encode the wrong thing and degrades
quietly. Storing them beside the weights is what stops the next person
having to know.

The reload is verified rather than assumed. :func:`load_adapter`
reconstructs the encoder, and the round-trip test asserts it produces
byte-identical vectors — a saved model that scores differently on reload
is worse than no saved model, because it is trusted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from multilingual_embedding.common.enums import DataProvenance
from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.utils.filesystem import atomic_write_path, ensure_directory
from multilingual_embedding.utils.io import read_json, write_json

from .lora import LoRAConfig, apply_lora, load_lora_state_dict, lora_state_dict
from .pretrained import PretrainedEncoderError, PretrainedTextEncoder

__all__ = ["AdapterMetadata", "load_adapter", "save_adapter"]

_logger = get_logger(__name__)

_WEIGHTS_FILENAME = "adapter.pt"

_METADATA_FILENAME = "adapter.json"

# Bumped to 2 when ``data_provenance`` became a required top-level field.
# A format-2 adapter guarantees the field is present; a format-1 one
# predates it, and :func:`load_adapter` still reads those, reporting the
# provenance as undeclared rather than refusing a model that was valid
# when it was written.
_FORMAT_VERSION = 2

_SUPPORTED_LOAD_VERSIONS = frozenset({1, 2})

# What a format-1 adapter's provenance reads as: not one of the declared
# values, and visibly not, so a caller cannot mistake an old model whose
# training data was never recorded for one that declared it.
_UNDECLARED_PROVENANCE = "undeclared"


class AdapterMetadata:
    """
    Everything needed to use an adapter, beyond its weights.

    Held as a plain mapping rather than a dataclass because it is written
    to disk and read back by versions of this code that may not agree on
    fields. Unknown keys survive a round trip.
    """

    __slots__ = ("payload",)

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)

    @property
    def checkpoint(self) -> str:
        """The base model this adapter was trained against."""

        return str(self.payload["checkpoint"])

    @property
    def query_prefix(self) -> str:
        """Marker the query side expects, empty for a symmetric model."""

        return str(self.payload.get("query_prefix", ""))

    @property
    def passage_prefix(self) -> str:
        """Marker the passage side expects."""

        return str(self.payload.get("passage_prefix", ""))

    @property
    def data_provenance(self) -> str:
        """
        Where the training data came from, one of :class:`DataProvenance`.

        Returns ``"undeclared"`` for a format-1 adapter written before the
        field existed, so the absence of a declaration is itself legible
        rather than defaulting to a value that would claim more than the
        file records.
        """

        return str(self.payload.get("data_provenance", _UNDECLARED_PROVENANCE))


def save_adapter(
    encoder: PretrainedTextEncoder,
    directory: str | Path,
    *,
    lora: LoRAConfig,
    data_provenance: str,
    query_prefix: str = "",
    passage_prefix: str = "",
    notes: dict[str, Any] | None = None,
) -> Path:
    """
    Write an adapter and everything needed to use it.

    Parameters
    ----------
    encoder:
        The adapted encoder. Its base weights are not written.

    lora:
        The configuration the adapter was built with. Required, because
        reloading has to rebuild the same shapes before the weights will
        fit, and inferring rank from a tensor is guesswork that fails
        silently on a mismatch.

    data_provenance:
        Where the training data came from, one of :class:`DataProvenance`
        (``public``, ``synthetic``, ``licensed``). Required and with no
        default, because it is a legal fact about the model rather than a
        note about it, and a default would let the one question that must
        be answered be answered by omission. An adapter cannot be written
        without a caller stating this, in the same sense that
        ``AllowAll(acknowledged=True)`` has to be typed: legitimate, and
        the record of a human decision.

    query_prefix, passage_prefix:
        Recorded so the adapter cannot be served without them. An E5
        model used without ``query: `` still returns vectors; they are
        just worse, and nothing says so.

    notes:
        Free-form provenance — what it trained on, what it scored. Kept
        verbatim, so a model file traces back to the run that made it.

    Raises
    ------
    PretrainedEncoderError
        If the encoder carries no adapter, which almost always means
        ``apply_lora`` was never called and the caller is about to save
        an empty file believing otherwise.
    ValueError
        If ``data_provenance`` is not one of the declared values. Refused
        rather than written through, because an unrecognised provenance
        recorded on a model is worse than none: it reads as a declaration
        while meaning nothing.
    """

    if data_provenance not in set(DataProvenance):
        raise ValueError(
            f"data_provenance must be one of {sorted(DataProvenance)}, not {data_provenance!r}"
        )

    target = ensure_directory(directory)

    state = lora_state_dict(encoder._model)

    if not state:
        raise PretrainedEncoderError(
            "This encoder has no LoRA adapter to save; apply_lora was probably never called on it",
            directory=str(target),
        )

    with atomic_write_path(target / _WEIGHTS_FILENAME) as temporary:
        torch.save(state, temporary)

    parameters = sum(tensor.numel() for tensor in state.values())

    write_json(
        target / _METADATA_FILENAME,
        {
            "format_version": _FORMAT_VERSION,
            "checkpoint": encoder.name,
            "dimension": encoder.dimension,
            "pooling": encoder._model.pooling,
            "max_length": encoder._max_length,
            "normalize": encoder._normalize,
            "data_provenance": str(data_provenance),
            "query_prefix": query_prefix,
            "passage_prefix": passage_prefix,
            "lora": {
                "rank": lora.rank,
                "alpha": lora.alpha,
                "dropout": lora.dropout,
                "targets": list(lora.targets),
            },
            "adapter_parameters": parameters,
            "notes": notes or {},
        },
    )

    _logger.info(
        "Saved adapter",
        extra={
            "directory": str(target),
            "adapter_parameters": parameters,
            "megabytes": round(parameters * 4 / 1024**2, 2),
        },
    )

    return target


def load_adapter(
    directory: str | Path,
    *,
    device: str | torch.device | None = None,
    local_files_only: bool = False,
) -> tuple[PretrainedTextEncoder, AdapterMetadata]:
    """
    Rebuild an adapted encoder from disk.

    Returns the encoder and its metadata, because the prefixes are needed
    to use it correctly and returning the encoder alone would invite them
    being forgotten.

    Parameters
    ----------
    local_files_only:
        Refuse to fetch the base checkpoint from the network. Worth
        setting when a result must be reproducible, since an upstream
        repository can change what is behind a name.

    Raises
    ------
    PretrainedEncoderError
        If the directory is not an adapter, was written by an
        incompatible version, or the weights do not fit the
        reconstructed shapes.

    Example
    -------
    ::

        encoder, meta = load_adapter("models/indic-v1")

        vector = encoder.encode(meta.query_prefix + "some question")
    """

    source = Path(directory).expanduser()

    metadata_path = source / _METADATA_FILENAME

    if not metadata_path.exists():
        raise PretrainedEncoderError(
            "No adapter metadata here; this does not look like a saved adapter",
            directory=str(source),
            expected=_METADATA_FILENAME,
        )

    payload = read_json(metadata_path)

    version = payload.get("format_version")

    if version not in _SUPPORTED_LOAD_VERSIONS:
        raise PretrainedEncoderError(
            "Adapter was written by an incompatible format version",
            found=version,
            supported=sorted(_SUPPORTED_LOAD_VERSIONS),
        )

    encoder = PretrainedTextEncoder.load(
        payload["checkpoint"],
        pooling=payload.get("pooling", "mean"),
        device=device,
        max_length=int(payload.get("max_length", 512)),
        # Restored rather than left to default. An encoder saved with
        # normalization off and reloaded with it on returns vectors of a
        # different magnitude for the same text: cosine scores are
        # unchanged, dot-product scores are not, and nothing raises. The
        # manifest has recorded this since format 1; it was simply not
        # being read.
        normalize=bool(payload.get("normalize", True)),
        local_files_only=local_files_only,
    )

    settings = payload.get("lora", {})

    apply_lora(
        encoder._model,
        LoRAConfig(
            rank=int(settings.get("rank", 8)),
            alpha=int(settings.get("alpha", 16)),
            dropout=float(settings.get("dropout", 0.0)),
            targets=tuple(settings.get("targets", ("query", "value"))),
        ),
    )

    state = torch.load(source / _WEIGHTS_FILENAME, map_location=encoder.device)

    try:
        load_lora_state_dict(encoder._model, state)
    except Exception as error:
        raise PretrainedEncoderError(
            "Adapter weights do not fit the reconstructed model; the recorded "
            "LoRA settings and the saved tensors disagree",
            directory=str(source),
            detail=str(error)[:300],
        ) from error

    _logger.info(
        "Loaded adapter",
        extra={
            "directory": str(source),
            "base": payload["checkpoint"],
            "device": str(encoder.device),
        },
    )

    return encoder, AdapterMetadata(payload)
