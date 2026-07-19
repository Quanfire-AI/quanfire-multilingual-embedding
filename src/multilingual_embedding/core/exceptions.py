"""
Framework exception hierarchy.

Every error raised deliberately by the framework derives from
:class:`MultilingualEmbeddingError`, so callers can catch framework
failures without also catching unrelated builtins.

Exceptions carry structured context rather than pre-formatted strings.
This keeps messages readable for humans while leaving the individual
values available to logging and tests.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ConfigurationError",
    "MultilingualEmbeddingError",
    "NotFittedError",
    "RegistryError",
    "ResourceNotFoundError",
    "SerializationError",
    "ValidationError",
]


class MultilingualEmbeddingError(Exception):
    """
    Base class for all framework errors.

    Parameters
    ----------
    message:
        Human readable description of the failure.

    context:
        Arbitrary key/value pairs describing the failure site.
        Rendered into the string form and preserved for callers.
    """

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)

        self.message = message

        self.context: dict[str, Any] = context

    def __str__(self) -> str:
        if not self.context:
            return self.message

        details = ", ".join(f"{key}={value!r}" for key, value in sorted(self.context.items()))

        return f"{self.message} ({details})"


class ConfigurationError(MultilingualEmbeddingError):
    """Raised when configuration is missing, malformed, or inconsistent."""


class ValidationError(MultilingualEmbeddingError):
    """Raised when a value fails a precondition check."""


class RegistryError(MultilingualEmbeddingError):
    """Raised for duplicate registrations or unknown registry keys."""


class SerializationError(MultilingualEmbeddingError):
    """Raised when an object cannot be encoded or decoded."""


class ResourceNotFoundError(MultilingualEmbeddingError):
    """Raised when a required file or resource is absent."""


class NotFittedError(MultilingualEmbeddingError):
    """Raised when a trainable component is used before it is trained."""
