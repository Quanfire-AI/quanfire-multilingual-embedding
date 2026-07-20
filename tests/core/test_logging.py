from __future__ import annotations

import io
import json
import logging

import pytest

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


class TestNoLoggingExtraCollidesWithLogRecord:
    """
    `extra=` keys must not shadow a LogRecord attribute.

    `logging` raises KeyError when they do — but only once a handler is
    attached, because an unconfigured logger short-circuits before
    building the record. So the failure waits for a formatter, and then
    appears somewhere unrelated to the code that caused it.

    That is exactly how it surfaced: `extra={"name": ...}` in the
    pretrained encoder passed on its own and took thirteen tests down as
    soon as another test enabled INFO logging.
    """

    RESERVED = frozenset(
        {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
            "taskName",
        }
    )

    def test_no_source_file_passes_a_reserved_key(self) -> None:
        import ast
        import pathlib

        import multilingual_embedding

        root = pathlib.Path(multilingual_embedding.__file__).parent

        offences: list[str] = []

        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))

            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue

                for keyword in node.keywords:
                    if keyword.arg != "extra" or not isinstance(keyword.value, ast.Dict):
                        continue

                    for key in keyword.value.keys:
                        if isinstance(key, ast.Constant) and key.value in self.RESERVED:
                            offences.append(
                                f"{path.relative_to(root)}:{key.lineno} extra key {key.value!r}"
                            )

        assert not offences, (
            "these logging calls will raise KeyError once a handler is attached: "
            + "; ".join(offences)
        )

    def test_a_reserved_key_really_does_raise(self) -> None:
        """
        The control. If logging tolerated this, the test above would be
        guarding against nothing.
        """

        import logging

        logger = logging.getLogger("multilingual_embedding.test.collision")

        logger.addHandler(logging.NullHandler())

        logger.setLevel(logging.INFO)

        with pytest.raises(KeyError):
            logger.info("message", extra={"name": "anything"})
