"""
Core infrastructure: errors, logging, registries and factories.

This package depends only on the standard library and on
:mod:`multilingual_embedding.common`. Every other package may depend on
it, which keeps the import graph acyclic.
"""

from __future__ import annotations

from .exceptions import (
    ConfigurationError,
    MultilingualEmbeddingError,
    NotFittedError,
    RegistryError,
    ResourceNotFoundError,
    SerializationError,
    ValidationError,
)
from .factory import (
    TYPE_KEY,
    build_all_from_config,
    build_from_config,
)
from .logging import (
    ROOT_LOGGER_NAME,
    JsonFormatter,
    configure_logging,
    get_logger,
)
from .registry import Registry

__all__ = [
    "ROOT_LOGGER_NAME",
    "TYPE_KEY",
    "ConfigurationError",
    "JsonFormatter",
    "MultilingualEmbeddingError",
    "NotFittedError",
    "Registry",
    "RegistryError",
    "ResourceNotFoundError",
    "SerializationError",
    "ValidationError",
    "build_all_from_config",
    "build_from_config",
    "configure_logging",
    "get_logger",
]
