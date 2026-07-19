"""
Configuration driven construction of registered components.

A component specification is a mapping with a discriminator key naming
the implementation, plus the keyword arguments to pass to it::

    {"type": "nfkc", "remove_control_characters": true}

:func:`build_from_config` turns that mapping into an instance using the
supplied :class:`~multilingual_embedding.core.registry.Registry`. This is
the bridge between YAML configuration and Python objects.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any

from .exceptions import ConfigurationError
from .registry import Registry

__all__ = [
    "TYPE_KEY",
    "build_all_from_config",
    "build_from_config",
]

TYPE_KEY = "type"


def build_from_config[T](
    registry: Registry[T],
    config: Mapping[str, Any] | str,
    **overrides: Any,
) -> T:
    """
    Construct a component from a specification mapping.

    Parameters
    ----------
    registry:
        Registry to resolve the ``type`` discriminator against.

    config:
        Either a mapping containing a ``type`` key plus constructor
        arguments, or a bare string naming the type when no arguments
        are needed.

    overrides:
        Keyword arguments that take precedence over those in ``config``.
        Used to inject runtime values (a resolved path, a shared
        vocabulary) that do not belong in a static config file.

    Raises
    ------
    ConfigurationError
        If the specification is malformed or names arguments the target
        constructor does not accept.
    """

    if isinstance(config, str):
        spec: dict[str, Any] = {TYPE_KEY: config}
    elif isinstance(config, Mapping):
        spec = dict(config)
    else:
        raise ConfigurationError(
            "Component specification must be a mapping or a string",
            registry=registry.name,
            received=type(config).__name__,
        )

    type_name = spec.pop(TYPE_KEY, None)

    if not type_name:
        raise ConfigurationError(
            f"Component specification is missing the {TYPE_KEY!r} key",
            registry=registry.name,
            keys=sorted(spec),
        )

    if not isinstance(type_name, str):
        raise ConfigurationError(
            f"Component {TYPE_KEY!r} must be a string",
            registry=registry.name,
            received=type(type_name).__name__,
        )

    implementation = registry.get(type_name)

    kwargs = {**spec, **overrides}

    _reject_unknown_arguments(registry, type_name, implementation, kwargs)

    try:
        return implementation(**kwargs)
    except TypeError as error:
        raise ConfigurationError(
            "Could not construct component",
            registry=registry.name,
            type=type_name,
            reason=str(error),
        ) from error


def build_all_from_config[T](
    registry: Registry[T],
    configs: Sequence[Mapping[str, Any] | str] | None,
    **overrides: Any,
) -> list[T]:
    """
    Construct an ordered list of components.

    Used for pipeline stages such as normalizer chains, where order is
    significant. A ``None`` or empty sequence yields an empty list.
    """

    if not configs:
        return []

    if isinstance(configs, (str, Mapping)):
        raise ConfigurationError(
            "Expected a sequence of component specifications",
            registry=registry.name,
            received=type(configs).__name__,
        )

    return [build_from_config(registry, config, **overrides) for config in configs]


def _reject_unknown_arguments(
    registry: Registry[Any],
    type_name: str,
    implementation: type[Any],
    kwargs: Mapping[str, Any],
) -> None:
    """
    Fail fast on constructor arguments the implementation cannot accept.

    Without this a typo in a YAML key surfaces as an opaque ``TypeError``
    from deep inside the constructor. Implementations that accept
    ``**kwargs`` are skipped, since anything is legal for them.
    """

    try:
        signature = inspect.signature(implementation)
    except (TypeError, ValueError):
        return

    accepts_var_keyword = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    if accepts_var_keyword:
        return

    accepted = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }

    unknown = sorted(set(kwargs) - accepted)

    if unknown:
        raise ConfigurationError(
            "Unknown configuration keys for component",
            registry=registry.name,
            type=type_name,
            unknown=unknown,
            accepted=sorted(accepted),
        )
