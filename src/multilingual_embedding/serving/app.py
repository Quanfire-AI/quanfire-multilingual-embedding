"""
The embeddings endpoint.

A web service over the artefact-loading pattern already in place, using
the de facto industry-standard request and response schema so an existing
client migrates by changing a base URL. It serves either kind of model
this project produces — a LoRA adapter over a pretrained checkpoint, or a
from-scratch / fine-tuned encoder written by ``qfme pretrain`` and
``qfme finetune`` — routing on the directory's shape, so the same command
serves both.

**The decision this module exists to get right.** The standard schema has
no field distinguishing a query from a passage. The models this project
serves are asymmetric: an E5-family checkpoint is trained with ``query: ``
on one side and ``passage: `` on the other, and served without them it
returns vectors of the right shape and norm, free of NaN, that encode the
wrong thing. Nothing raises. Retrieval is simply worse, which is
indistinguishable from the model not being very good.

So when the served model records prefixes, this server will not guess. It
answers 400 naming both valid values unless the caller says which side the
text belongs to, or unless whoever started the server chose a default with
``--default-input-type``. That trades one-line client compatibility for
not being quietly wrong, which after the work behind these numbers is the
only defensible trade. An operator serving a single-sided workload — a
query-only search box, a passage-only indexing job — sets the default once
and their clients really do change nothing but the base URL.

Symmetric models are unaffected: with no prefixes recorded there is
nothing to choose, ``input_type`` is accepted and ignored, and
``prefix_applied`` comes back null.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ..common.version import __version__
from ..core.logging import get_logger
from ..pipelines.search import SemanticSearchPipeline
from .schemas import (
    EmbeddingObject,
    EmbeddingsRequest,
    EmbeddingsResponse,
    ErrorBody,
    ErrorResponse,
    ModelCard,
    ModelsResponse,
    Usage,
)

if TYPE_CHECKING:  # pragma: no cover
    from fastapi import FastAPI

__all__ = ["ServingConfig", "create_app"]

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ServingConfig:
    """What to serve, and how to resolve the ambiguity the schema leaves.

    Parameters
    ----------
    adapter_directory:
        The model to serve. Either a LoRA adapter directory written by
        ``save_adapter``, or a from-scratch / fine-tuned experiment
        directory holding ``encoder/`` + ``tokenizer/`` — the kind
        ``qfme pretrain`` and ``qfme finetune`` write. The server routes
        on shape, so the same flag serves both.

    model_id:
        The name clients send and receive. Defaults to the directory's
        own name, which makes ``models/indic-v1`` serve as ``indic-v1``
        and keeps the served name tied to a versioned artefact rather
        than to a branch or a date.

    default_input_type:
        Which side to assume when a request omits ``input_type``. Left
        unset, an asymmetric model refuses the request instead. Set it
        only when every caller of this deployment is on one side.

    local_files_only:
        Refuse to fetch the base checkpoint from the network. Worth
        setting in a container built with the weights baked in, so a
        misconfigured deployment fails at startup rather than silently
        pulling a checkpoint that may have changed upstream.
    """

    adapter_directory: Path

    model_id: str = ""

    default_input_type: Literal["query", "passage"] | None = None

    local_files_only: bool = False

    def resolved_model_id(self) -> str:
        return self.model_id or self.adapter_directory.name


class _ServedModel:
    """A loaded pipeline plus the metadata a client is entitled to see."""

    def __init__(self, config: ServingConfig) -> None:
        self.config = config

        directory = config.adapter_directory

        # Route on shape, exactly as `evaluate` and `mine-negatives` do: a
        # from-scratch or fine-tuned experiment writes `encoder/` +
        # `tokenizer/` and carries no `adapter.json`; a LoRA adapter is the
        # other case. Serving whichever was handed over is what makes
        # `qfme finetune`'s own promise — that `qfme serve` loads the
        # encoder it wrote — finally true, rather than a from_adapter call
        # that fails on the missing adapter.json a moment later.
        if (directory / "encoder").is_dir() and (directory / "tokenizer").is_dir():
            self.pipeline = SemanticSearchPipeline.from_directory(directory)

            # The encoder records its own limit and whether it normalises,
            # in the same encoder.json the loader just read; take the card's
            # numbers from there rather than from an adapter.json this model
            # never wrote. A from-scratch encoder is symmetric, so it has no
            # prefixes and --local-files-only is moot — nothing is fetched.
            payload = json.loads(
                (directory / "encoder" / "encoder.json").read_text(encoding="utf-8")
            )

            self.manifest = {
                "max_length": payload["architecture"]["max_length"],
                "normalize": payload.get("normalize", True),
            }
        else:
            self.pipeline = SemanticSearchPipeline.from_adapter(
                directory,
                local_files_only=config.local_files_only,
            )

            # Read from the artefact rather than the encoder. max_length and
            # normalize are recorded in adapter.json but are not part of the
            # TextEncoder protocol, and a getattr fallback would publish a
            # confident 0 for a model whose real limit is 256 — a model card
            # that states a wrong number is worse than one that omits it.
            self.manifest = json.loads((directory / "adapter.json").read_text(encoding="utf-8"))

        self.query_prefix, self.passage_prefix = self.pipeline.prefixes

        self.model_id = config.resolved_model_id()

    @property
    def is_asymmetric(self) -> bool:
        """Whether the choice of side changes the vector.

        Both prefixes are written as a pair, so either one being present
        means the model was trained with the distinction.
        """

        return bool(self.query_prefix or self.passage_prefix)

    def prefix_for(self, requested: str | None) -> str | None:
        """Resolve which prefix to apply, or raise if it cannot be known.

        Returns None for a symmetric model — not the empty string — so a
        caller reading ``prefix_applied`` can tell "this model has no
        sides" apart from "no prefix was applied to an asymmetric one",
        which would be a bug rather than a property.
        """

        if not self.is_asymmetric:
            return None

        side = requested or self.config.default_input_type

        if side is None:
            raise _InputTypeRequiredError(self.model_id)

        return self.query_prefix if side == "query" else self.passage_prefix

    def card(self) -> ModelCard:
        encoder = self.pipeline.encoder

        return ModelCard(
            id=self.model_id,
            dimension=encoder.dimension,
            # The defaults match load_adapter's exactly, so the card
            # describes the encoder that is actually running rather than
            # a second opinion about it.
            max_length=int(self.manifest.get("max_length", 512)),
            normalized=bool(self.manifest.get("normalize", True)),
            query_prefix=self.query_prefix or None,
            passage_prefix=self.passage_prefix or None,
            framework_version=__version__,
        )


class _InputTypeRequiredError(Exception):
    """Raised when an asymmetric model was asked to guess a side."""

    def __init__(self, model_id: str) -> None:
        super().__init__(model_id)

        self.model_id = model_id


def create_app(config: ServingConfig) -> FastAPI:
    """Build the application, loading the model before accepting traffic.

    The model is loaded here rather than on first request so that a
    broken artefact or a missing checkpoint fails at startup, where a
    deployment notices, instead of on a caller's first query.
    """

    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    served = _ServedModel(config)

    _logger.info(
        "serving %s (dimension %d, %s)",
        served.model_id,
        served.pipeline.encoder.dimension,
        "asymmetric" if served.is_asymmetric else "symmetric",
    )

    app = FastAPI(
        title="Quanfire multilingual embeddings",
        version=__version__,
        summary="An embeddings endpoint over an adapted multilingual encoder.",
    )

    def _error(status: int, message: str, kind: str, param: str | None = None) -> JSONResponse:
        return JSONResponse(
            status_code=status,
            content=ErrorResponse(
                error=ErrorBody(message=message, type=kind, param=param)
            ).model_dump(),
        )

    @app.exception_handler(_InputTypeRequiredError)
    async def _handle_input_type_required(
        _request: Request, error: _InputTypeRequiredError
    ) -> JSONResponse:
        return _error(
            400,
            f"Model {error.model_id!r} is asymmetric: it was trained with distinct "
            "query and passage prefixes, and embedding text on the wrong side "
            "returns a plausible vector that encodes the wrong thing. Send "
            '"input_type": "query" or "passage", or start the server with '
            "--default-input-type if every caller is on one side.",
            "invalid_request_error",
            "input_type",
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Liveness, and enough identity to tell two deployments apart."""

        return {
            "status": "ok",
            "model": served.model_id,
            "framework_version": __version__,
        }

    @app.get("/v1/models", response_model=ModelsResponse)
    async def models() -> ModelsResponse:
        return ModelsResponse(data=[served.card()])

    @app.post("/v1/embeddings", response_model=EmbeddingsResponse)
    async def embeddings(request: EmbeddingsRequest) -> Any:
        if request.model is not None and request.model != served.model_id:
            return _error(
                404,
                f"This server serves {served.model_id!r}, not {request.model!r}. "
                "Omit the field to accept whatever is deployed.",
                "invalid_request_error",
                "model",
            )

        prefix = served.prefix_for(request.input_type)

        texts = request.texts()

        started = time.perf_counter()

        vectors = served.pipeline.encoder.encode_batch(
            [f"{prefix}{text}" for text in texts] if prefix else texts
        )

        _logger.debug(
            "embedded %d input(s) in %.1f ms", len(texts), (time.perf_counter() - started) * 1000
        )

        tokens = _approximate_tokens(served, texts)

        return EmbeddingsResponse(
            data=[
                EmbeddingObject(index=index, embedding=[float(value) for value in vector])
                for index, vector in enumerate(vectors)
            ],
            model=served.model_id,
            usage=Usage(prompt_tokens=tokens, total_tokens=tokens),
            prefix_applied=prefix,
        )

    return app


def _approximate_tokens(served: _ServedModel, texts: list[str]) -> int:
    """Count tokens with the model's own tokenizer where one is reachable.

    A served adapter wraps a Hugging Face tokenizer rather than this
    project's, and reaching into it is not part of any contract, so this
    falls back to a whitespace count. The number is reported for client
    compatibility and is not billed against anything here; a fallback
    that is honest about being approximate beats a field that is absent.
    """

    tokenizer = getattr(served.pipeline.encoder, "_tokenizer", None)

    if tokenizer is not None:
        try:
            return sum(len(tokenizer(text)["input_ids"]) for text in texts)
        except Exception:
            _logger.debug("token counting failed; falling back to a whitespace count")

    return sum(len(text.split()) for text in texts)
