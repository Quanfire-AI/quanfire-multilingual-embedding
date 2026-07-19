from __future__ import annotations

import io
import json
import logging

from multilingual_embedding.core.exceptions import MultilingualEmbeddingError
from multilingual_embedding.core.logging import (
    ROOT_LOGGER_NAME,
    configure_logging,
    get_logger,
)


def test_get_logger_namespaces_under_root() -> None:
    assert get_logger("corpus.reader").name == f"{ROOT_LOGGER_NAME}.corpus.reader"


def test_get_logger_does_not_double_prefix() -> None:
    name = f"{ROOT_LOGGER_NAME}.corpus"

    assert get_logger(name).name == name


def test_get_logger_without_name_returns_root() -> None:
    assert get_logger().name == ROOT_LOGGER_NAME


def test_json_format_emits_parseable_records() -> None:
    stream = io.StringIO()

    configure_logging(level=logging.INFO, log_format="json", stream=stream)

    get_logger("demo").info("hello", extra={"documents": 3})

    payload = json.loads(stream.getvalue().strip())

    assert payload["message"] == "hello"

    assert payload["documents"] == 3

    assert payload["level"] == "INFO"


def test_reconfiguring_does_not_duplicate_handlers() -> None:
    stream = io.StringIO()

    configure_logging(log_format="text", stream=stream)

    configure_logging(log_format="text", stream=stream)

    logger = logging.getLogger(ROOT_LOGGER_NAME)

    assert len(logger.handlers) == 1


def test_exception_context_is_rendered() -> None:
    error = MultilingualEmbeddingError("failed", path="a.txt", line=3)

    rendered = str(error)

    assert "failed" in rendered

    assert "line=3" in rendered

    assert error.context["path"] == "a.txt"


def test_exception_without_context_renders_message_only() -> None:
    assert str(MultilingualEmbeddingError("plain")) == "plain"
