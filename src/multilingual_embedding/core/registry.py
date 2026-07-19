"""
Generic component registry.

The framework resolves pluggable components by name so that
configuration files can select an implementation without importing it.
Normalizers, pre-tokenizers, tokenizers and embedding models each own a
registry instance.

Example
-------
::

    normalizers: Registry[Normalizer] = Registry("normalizer")

    @normalizers.register("nfkc")
    class NFKCNormalizer(Normalizer):
        ...

    normalizer = normalizers.create("nfkc")
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .exceptions import RegistryError

__all__ = ["Registry"]


class Registry[T]:
    """
    Name to implementation mapping for one component family.

    Keys are case insensitive and stored lowercase, so configuration
    files may use whichever casing reads best.

    Parameters
    ----------
    name:
        Family name, used in error messages (e.g. ``"tokenizer"``).
    """

    __slots__ = ("_entries", "_name")

    def __init__(self, name: str) -> None:
        self._name = name

        self._entries: dict[str, type[T]] = {}

    @property
    def name(self) -> str:
        """Name of the component family."""

        return self._name

    def register(
        self,
        key: str,
        implementation: type[T] | None = None,
        *,
        override: bool = False,
    ) -> Any:
        """
        Register an implementation.

        Usable directly or as a decorator::

            registry.register("bpe", BpeTokenizer)

            @registry.register("bpe")
            class BpeTokenizer: ...

        Parameters
        ----------
        key:
            Lookup name.

        implementation:
            The class to register. Omit to use decorator form.

        override:
            Allow replacing an existing key. Defaults to False so that
            an accidental duplicate fails loudly at import time rather
            than silently shadowing the earlier implementation.
        """

        normalized = self._normalize(key)

        def _register(target: type[T]) -> type[T]:
            if normalized in self._entries and not override:
                raise RegistryError(
                    "Duplicate registry key",
                    registry=self._name,
                    key=normalized,
                    existing=self._entries[normalized].__name__,
                )

            self._entries[normalized] = target

            return target

        if implementation is None:
            return _register

        return _register(implementation)

    def get(self, key: str) -> type[T]:
        """
        Return the implementation registered under ``key``.

        Raises
        ------
        RegistryError
            If the key is unknown. The message lists the available keys,
            which turns a typo in a config file into an obvious fix.
        """

        normalized = self._normalize(key)

        try:
            return self._entries[normalized]
        except KeyError:
            raise RegistryError(
                "Unknown registry key",
                registry=self._name,
                key=normalized,
                available=self.keys(),
            ) from None

    def create(self, key: str, *args: Any, **kwargs: Any) -> T:
        """Instantiate the implementation registered under ``key``."""

        return self.get(key)(*args, **kwargs)

    def keys(self) -> list[str]:
        """Return registered keys in sorted order."""

        return sorted(self._entries)

    def unregister(self, key: str) -> None:
        """Remove a key. Primarily useful for test isolation."""

        self._entries.pop(self._normalize(key), None)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self._normalize(key) in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"Registry(name={self._name!r}, keys={self.keys()!r})"

    @staticmethod
    def _normalize(key: str) -> str:
        if not isinstance(key, str) or not key.strip():
            raise RegistryError("Registry key must be a non-empty string", key=key)

        return key.strip().lower()
