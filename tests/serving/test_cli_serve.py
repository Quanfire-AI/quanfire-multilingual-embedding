"""
The ``qfme serve`` wiring.

Parser-level only: what the flags mean and what the defaults are. The
handler itself blocks in ``uvicorn.run`` and the behaviour it configures
is covered against a real model in :mod:`tests.serving.test_app`.

No optional extra is needed here — building the parser must not import
FastAPI, which is the point of the last test in this file.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from multilingual_embedding.cli import build_parser


@pytest.fixture(scope="module")
def parse():  # type: ignore[no-untyped-def]
    parser = build_parser()

    return lambda *argv: parser.parse_args(["serve", *argv])


class TestTheFlags:
    def test_the_adapter_is_required(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["serve"])

    def test_the_adapter_arrives_as_a_path(self, parse) -> None:  # type: ignore[no-untyped-def]
        assert parse("--adapter", "models/indic-v1").adapter == Path("models/indic-v1")

    def test_it_binds_loopback_by_default(self, parse) -> None:  # type: ignore[no-untyped-def]
        """
        The endpoint has no authentication and no rate limiting, so a
        default of 0.0.0.0 would put an unauthenticated GPU-backed
        service on whatever network the host is on the first time anyone
        runs this. Exposing it has to be a decision someone typed.
        """

        assert parse("--adapter", "m").host == "127.0.0.1"

    def test_the_port_default(self, parse) -> None:  # type: ignore[no-untyped-def]
        assert parse("--adapter", "m").port == 8000

    def test_no_default_side_unless_asked_for(self, parse) -> None:  # type: ignore[no-untyped-def]
        """
        Unset means the server refuses to guess. A default here would
        make every deployment silently single-sided.
        """

        assert parse("--adapter", "m").default_input_type is None

    def test_a_default_side_is_accepted(self, parse) -> None:  # type: ignore[no-untyped-def]
        assert parse("--adapter", "m", "--default-input-type", "query").default_input_type == (
            "query"
        )

    def test_an_invalid_side_is_rejected(self, parse) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(SystemExit):
            parse("--adapter", "m", "--default-input-type", "document")

    def test_local_files_only_is_off_unless_asked_for(self, parse) -> None:  # type: ignore[no-untyped-def]
        assert parse("--adapter", "m").local_files_only is False

        assert parse("--adapter", "m", "--local-files-only").local_files_only is True


def test_building_the_parser_does_not_import_fastapi() -> None:
    """
    Serving is one deployment of this project. Registering its subcommand
    must not make its dependencies mandatory for ``qfme stats``, and an
    import moved to module scope by a later edit would not otherwise show
    up in a checkout that happens to have the extra installed.

    Run in a fresh interpreter so this measures the import graph rather
    than whatever the test session has already loaded.
    """

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from multilingual_embedding.cli import build_parser;"
            " build_parser(); print('fastapi' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr

    assert completed.stdout.strip() == "False"
