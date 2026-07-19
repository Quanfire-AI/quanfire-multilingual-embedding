"""
Loading configuration from files, environment and command line.

Precedence, lowest to highest:

1. dataclass defaults
2. the YAML or JSON file
3. environment variables prefixed ``QFME_``
4. explicit overrides passed by the caller

Later sources win, so a deployment can adjust one value through the
environment without editing a file, and a developer can adjust one value
on the command line without touching the environment.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from multilingual_embedding.core.exceptions import (
    ConfigurationError,
    SerializationError,
    ValidationError,
)
from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.utils.io import read_json, read_yaml, write_yaml
from multilingual_embedding.utils.serialization import to_primitive

from .base import ExperimentConfig

__all__ = [
    "ENV_PREFIX",
    "config_from_env",
    "load_config",
    "parse_override",
    "save_config",
]

ENV_PREFIX = "QFME_"

_logger = get_logger(__name__)


def load_config(
    path: str | Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
    use_environment: bool = True,
) -> ExperimentConfig:
    """
    Build an :class:`ExperimentConfig` from all configured sources.

    Parameters
    ----------
    path:
        YAML or JSON file. When omitted, defaults are used as the base.

    overrides:
        Nested mapping applied last. Highest precedence.

    use_environment:
        Whether to consult ``QFME_`` environment variables.

    Raises
    ------
    ConfigurationError
        If the file does not contain a mapping, if it names a field no
        config section declares, or if any value fails the validation
        rules of its target section. The underlying
        :class:`~multilingual_embedding.core.exceptions.ValidationError`
        or
        :class:`~multilingual_embedding.core.exceptions.SerializationError`
        is preserved as ``__cause__``, and its structured context is
        merged into this error's own.
    """

    data: dict[str, Any] = {}

    if path is not None:
        data = _read_config_file(Path(path))

    with _configuration_errors(path, stage="file"):
        config = ExperimentConfig.from_dict(data)

    if use_environment:
        environment_overrides = config_from_env()

        if environment_overrides:
            _logger.debug(
                "Applying environment configuration overrides",
                extra={"keys": sorted(environment_overrides)},
            )

            with _configuration_errors(path, stage="environment"):
                config = config.merged(environment_overrides)

    if overrides:
        with _configuration_errors(path, stage="overrides"):
            config = config.merged(overrides)

    return config


@contextmanager
def _configuration_errors(path: str | Path | None, *, stage: str) -> Iterator[None]:
    """
    Present construction failures as :class:`ConfigurationError`.

    ``load_config`` is the boundary between a user's config file and the
    framework's internals. Whether a bad value was caught by a
    precondition helper or by the deserialiser is an implementation
    detail of those internals; from outside, both mean "the
    configuration is wrong". Reporting one exception type keeps the
    caller's ``except`` clause honest, and ``stage`` records which of the
    layered sources introduced the problem — a value that is fine in the
    file but broken by an environment variable is otherwise very hard to
    place.

    ``ConfigurationError`` passes through untouched, so an error raised
    by a section's own ``__post_init__`` is not double-wrapped.
    """

    try:
        yield
    except ConfigurationError:
        raise
    except (SerializationError, ValidationError) as error:
        # The original context is carried through flattened so it stays
        # queryable, with the two framing keys applied last so they are
        # always present whatever the wrapped error happened to record.
        context = dict(error.context)

        context["config_path"] = str(path) if path is not None else None

        context["config_stage"] = stage

        raise ConfigurationError(f"Configuration is invalid: {error.message}", **context) from error


def save_config(config: ExperimentConfig, path: str | Path) -> Path:
    """
    Write a config to YAML.

    Every pipeline run persists its resolved configuration next to the
    artefacts it produced. Without that record, a model file cannot be
    traced back to the settings that made it.
    """

    return write_yaml(path, to_primitive(config))


def config_from_env(environ: dict[str, str] | None = None) -> dict[str, Any]:
    """
    Collect overrides from ``QFME_`` prefixed environment variables.

    Double underscores denote nesting and the remainder is lowercased::

        QFME_EMBEDDING__DIMENSION=256  ->  {"embedding": {"dimension": 256}}
        QFME_NAME=trial-a              ->  {"name": "trial-a"}

    Values are parsed as YAML scalars, so ``256`` arrives as an int and
    ``true`` as a bool rather than as strings that would fail validation.
    """

    source = environ if environ is not None else dict(os.environ)

    overrides: dict[str, Any] = {}

    for key, raw_value in source.items():
        if not key.startswith(ENV_PREFIX):
            continue

        remainder = key[len(ENV_PREFIX) :]

        if not remainder:
            continue

        path_parts = [part.lower() for part in remainder.split("__") if part]

        if not path_parts:
            continue

        _assign_nested(overrides, path_parts, _parse_scalar(raw_value))

    return overrides


def parse_override(assignment: str) -> dict[str, Any]:
    """
    Parse a single ``key.path=value`` command line override.

    Example
    -------
    ::

        parse_override("embedding.dimension=256")
        {"embedding": {"dimension": 256}}
    """

    if "=" not in assignment:
        raise ConfigurationError(
            "Override must use key.path=value form",
            assignment=assignment,
        )

    key_path, _, raw_value = assignment.partition("=")

    parts = [part for part in key_path.strip().split(".") if part]

    if not parts:
        raise ConfigurationError(
            "Override is missing a key path",
            assignment=assignment,
        )

    overrides: dict[str, Any] = {}

    _assign_nested(overrides, parts, _parse_scalar(raw_value))

    return overrides


def _read_config_file(path: Path) -> dict[str, Any]:
    """Read a YAML or JSON config file, chosen by suffix."""

    suffix = path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        data = read_yaml(path)
    elif suffix == ".json":
        data = read_json(path)
    else:
        raise ConfigurationError(
            "Unsupported configuration file type",
            path=str(path),
            supported=[".yaml", ".yml", ".json"],
        )

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ConfigurationError(
            "Configuration file must contain a mapping at the top level",
            path=str(path),
            received=type(data).__name__,
        )

    return data


def _assign_nested(target: dict[str, Any], path_parts: list[str], value: Any) -> None:
    """Assign ``value`` into ``target`` following a nested key path."""

    cursor = target

    for part in path_parts[:-1]:
        existing = cursor.get(part)

        if not isinstance(existing, dict):
            existing = {}
            cursor[part] = existing

        cursor = existing

    cursor[path_parts[-1]] = value


def _parse_scalar(raw: str) -> Any:
    """
    Interpret a string as a YAML scalar, falling back to the raw string.

    This is what turns ``"5"`` into ``5`` and ``"[a, b]"`` into a list,
    so that environment and command line overrides carry the same types
    as their file-based counterparts.
    """

    import yaml

    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw

    return raw if parsed is None and raw.strip() != "" else parsed
