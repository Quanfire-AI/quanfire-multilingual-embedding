"""
The HTTP endpoint.

Serving is one deployment of this project, so everything here sits behind
the ``serve`` extra and nothing below this layer imports it. Training,
evaluation, the CLI's other commands and a library consumer all work
without FastAPI installed.

``create_app`` is imported lazily for the same reason: importing this
package must not require the extra, or the layer becomes mandatory by
accident for anyone who merely touches the namespace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .schemas import (
    EmbeddingObject,
    EmbeddingsRequest,
    EmbeddingsResponse,
    ModelCard,
    ModelsResponse,
    Usage,
)

if TYPE_CHECKING:  # pragma: no cover
    from .app import ServingConfig, create_app

__all__ = [
    "EmbeddingObject",
    "EmbeddingsRequest",
    "EmbeddingsResponse",
    "ModelCard",
    "ModelsResponse",
    "ServingConfig",
    "Usage",
    "create_app",
]


def __getattr__(name: str) -> Any:
    """Defer the FastAPI-dependent names until something asks for them."""

    if name in {"ServingConfig", "create_app"}:
        from . import app

        return getattr(app, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
