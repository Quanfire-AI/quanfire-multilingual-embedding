"""
Logging setup for the framework.

The framework never configures logging as an import side effect. Library
code calls :func:`get_logger` and attaches nothing; applications call
:func:`configure_logging` once at startup to decide where records go.

Two formats are supported:

``text``
    Human readable, intended for interactive use.

``json``
    One JSON object per line, intended for log aggregation in production.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Literal

__all__ = [
    "ROOT_LOGGER_NAME",
    "JsonFormatter",
    "configure_logging",
    "get_logger",
]

ROOT_LOGGER_NAME = "multilingual_embedding"

LogFormat = Literal["text", "json"]

_TEXT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s %(message)s"

# Attributes present on every LogRecord. Anything outside this set was
# supplied by the caller via `extra=` and belongs in the JSON payload.
_RESERVED_RECORD_KEYS = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__.keys()
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    """
    Format records as single line JSON objects.

    Any keyword passed through ``extra=`` is merged into the payload,
    which makes structured fields queryable downstream.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key not in _RESERVED_RECORD_KEYS:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(
    level: int | str = logging.INFO,
    log_format: LogFormat = "text",
    stream: Any = None,
) -> logging.Logger:
    """
    Configure the framework logger.

    Safe to call more than once; existing handlers are replaced rather
    than accumulated, which avoids duplicate output when an application
    reconfigures at runtime.

    Parameters
    ----------
    level:
        Minimum level to emit.

    log_format:
        Either ``"text"`` or ``"json"``.

    stream:
        Destination stream. Defaults to ``sys.stderr`` so that stdout
        stays clean for program output.

    Returns
    -------
    The configured framework logger.
    """

    logger = logging.getLogger(ROOT_LOGGER_NAME)

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)

    if log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_TEXT_FORMAT))

    logger.addHandler(handler)

    logger.setLevel(level)

    # Framework records are fully handled here; leaking them to the root
    # logger would duplicate output in applications that configure their own.
    logger.propagate = False

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Return a logger namespaced under the framework root.

    Parameters
    ----------
    name:
        Usually ``__name__``. The framework prefix is added automatically
        and never duplicated.
    """

    if not name or name == ROOT_LOGGER_NAME:
        return logging.getLogger(ROOT_LOGGER_NAME)

    if name.startswith(f"{ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)

    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
