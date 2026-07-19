from __future__ import annotations

from typing import Any

import pytest

from multilingual_embedding.core.exceptions import ConfigurationError
from multilingual_embedding.core.factory import build_all_from_config, build_from_config
from multilingual_embedding.core.registry import Registry


class Component:
    def __init__(self, size: int = 1, label: str = "default") -> None:
        self.size = size

        self.label = label


class FlexibleComponent:
    def __init__(self, **options: Any) -> None:
        self.options = options


@pytest.fixture
def registry() -> Registry[Any]:
    instance: Registry[Any] = Registry("component")

    instance.register("component", Component)

    instance.register("flexible", FlexibleComponent)

    return instance


def test_build_from_mapping(registry: Registry[Any]) -> None:
    built = build_from_config(registry, {"type": "component", "size": 5})

    assert built.size == 5


def test_build_from_bare_string(registry: Registry[Any]) -> None:
    built = build_from_config(registry, "component")

    assert built.size == 1


def test_overrides_take_precedence(registry: Registry[Any]) -> None:
    built = build_from_config(registry, {"type": "component", "size": 5}, size=9)

    assert built.size == 9


def test_missing_type_rejected(registry: Registry[Any]) -> None:
    with pytest.raises(ConfigurationError):
        build_from_config(registry, {"size": 5})


def test_unknown_key_rejected(registry: Registry[Any]) -> None:
    with pytest.raises(ConfigurationError) as error:
        build_from_config(registry, {"type": "component", "typo": 5})

    assert "typo" in str(error.value)


def test_var_keyword_component_accepts_anything(registry: Registry[Any]) -> None:
    built = build_from_config(registry, {"type": "flexible", "anything": 1})

    assert built.options == {"anything": 1}


def test_non_mapping_rejected(registry: Registry[Any]) -> None:
    with pytest.raises(ConfigurationError):
        build_from_config(registry, 42)  # type: ignore[arg-type]


def test_build_all_preserves_order(registry: Registry[Any]) -> None:
    built = build_all_from_config(
        registry,
        [
            {"type": "component", "label": "first"},
            {"type": "component", "label": "second"},
        ],
    )

    assert [item.label for item in built] == ["first", "second"]


def test_build_all_of_none_is_empty(registry: Registry[Any]) -> None:
    assert build_all_from_config(registry, None) == []


def test_build_all_rejects_single_mapping(registry: Registry[Any]) -> None:
    with pytest.raises(ConfigurationError):
        build_all_from_config(registry, {"type": "component"})  # type: ignore[arg-type]
