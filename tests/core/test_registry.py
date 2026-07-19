from __future__ import annotations

import pytest

from multilingual_embedding.core.exceptions import RegistryError
from multilingual_embedding.core.registry import Registry


class Base:
    pass


class Alpha(Base):
    pass


class Beta(Base):
    pass


def test_register_and_get() -> None:
    registry: Registry[Base] = Registry("demo")

    registry.register("alpha", Alpha)

    assert registry.get("alpha") is Alpha


def test_register_as_decorator() -> None:
    registry: Registry[Base] = Registry("demo")

    @registry.register("beta")
    class Local(Base):
        pass

    assert registry.get("beta") is Local


def test_keys_are_case_insensitive() -> None:
    registry: Registry[Base] = Registry("demo")

    registry.register("Alpha", Alpha)

    assert registry.get("ALPHA") is Alpha

    assert "alpha" in registry


def test_duplicate_registration_rejected() -> None:
    registry: Registry[Base] = Registry("demo")

    registry.register("alpha", Alpha)

    with pytest.raises(RegistryError):
        registry.register("alpha", Beta)


def test_duplicate_allowed_with_override() -> None:
    registry: Registry[Base] = Registry("demo")

    registry.register("alpha", Alpha)

    registry.register("alpha", Beta, override=True)

    assert registry.get("alpha") is Beta


def test_unknown_key_lists_available() -> None:
    registry: Registry[Base] = Registry("demo")

    registry.register("alpha", Alpha)

    with pytest.raises(RegistryError) as error:
        registry.get("missing")

    assert "alpha" in str(error.value)


def test_empty_key_rejected() -> None:
    registry: Registry[Base] = Registry("demo")

    with pytest.raises(RegistryError):
        registry.register("   ", Alpha)


def test_create_instantiates() -> None:
    registry: Registry[Base] = Registry("demo")

    registry.register("alpha", Alpha)

    assert isinstance(registry.create("alpha"), Alpha)


def test_len_and_iteration() -> None:
    registry: Registry[Base] = Registry("demo")

    registry.register("alpha", Alpha)

    registry.register("beta", Beta)

    assert len(registry) == 2

    assert list(registry) == ["alpha", "beta"]


def test_unregister() -> None:
    registry: Registry[Base] = Registry("demo")

    registry.register("alpha", Alpha)

    registry.unregister("alpha")

    assert "alpha" not in registry
