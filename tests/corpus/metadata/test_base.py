from datetime import datetime

import pytest

from multilingual_embedding.corpus.metadata import BaseMetadata


def test_default_metadata() -> None:
    metadata = BaseMetadata()

    assert metadata.id is None
    assert metadata.language is None
    assert metadata.attributes == {}


def test_no_wall_clock_field_is_populated_per_document() -> None:
    """
    No field here may call the clock.

    `created_at` and `updated_at` used to, through default factories, on
    every document ever constructed. Nothing read them and nothing
    serialised them, but they took a real pipeline down: on CPython
    3.12.3 under WSL, `datetime.now(UTC)` raised SystemError partway
    through auditing 118,571 Hindi articles.

    Naming either one now raises TypeError. The wider rule is that a
    field nobody reads should not be able to fail a run.
    """

    import dataclasses

    for name in ("created_at", "updated_at"):
        with pytest.raises(TypeError):
            BaseMetadata(**{name: "anything"})

    # And nothing else has quietly acquired a clock-calling default.
    for field_ in dataclasses.fields(BaseMetadata):
        factory = field_.default_factory

        if factory is dataclasses.MISSING:
            continue

        produced = factory()

        assert not isinstance(produced, datetime), (
            f"BaseMetadata.{field_.name} calls the clock on construction"
        )
