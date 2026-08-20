"""The prefix-regime check: what it catches, and what it honestly cannot."""

from __future__ import annotations

import pytest

from multilingual_embedding.corpus.eulaw import EULAW_PROVISION_XLING_KIND
from multilingual_embedding.corpus.prefixes import check_prefix_regime, declared_regimes
from multilingual_embedding.corpus.trade import (
    TRADE_NOTIFICATION_KIND,
    TRADE_SECTION_XLING_KIND,
)


def test_registry_is_sourced_from_the_corpus_modules_not_a_second_copy() -> None:
    """The registry must agree with each module's own table, by construction."""
    from multilingual_embedding.corpus import eulaw, trade

    regimes = declared_regimes()

    assert regimes[TRADE_NOTIFICATION_KIND] == trade.prefix_regime(TRADE_NOTIFICATION_KIND)
    assert regimes[EULAW_PROVISION_XLING_KIND] == eulaw.prefix_regime(EULAW_PROVISION_XLING_KIND)


def test_matching_regime_passes() -> None:
    check = check_prefix_regime([TRADE_NOTIFICATION_KIND], "query: ", "passage: ")
    assert check.ok
    assert check.checked == (TRADE_NOTIFICATION_KIND,)
    assert check.unchecked == ()


def test_empty_prefixes_on_an_asymmetric_kind_is_caught() -> None:
    """The expensive silent failure: notifications trained without their prefixes."""
    check = check_prefix_regime([TRADE_NOTIFICATION_KIND], "", "")
    assert not check.ok
    kind, declared, configured = check.mismatches[0]
    assert kind == TRADE_NOTIFICATION_KIND
    assert declared == ("query: ", "passage: ")
    assert configured == ("", "")


def test_asymmetric_prefixes_on_a_cross_lingual_kind_is_caught() -> None:
    """Exactly the error the EUR-Lex recipe's prose would have produced."""
    check = check_prefix_regime([EULAW_PROVISION_XLING_KIND], "query: ", "passage: ")
    assert not check.ok
    assert check.mismatches[0][1] == ("", "")


def test_mixing_kinds_with_different_regimes_is_caught() -> None:
    """One run applies ONE global prefix pair, so mixed regimes cannot both hold."""
    check = check_prefix_regime(
        [TRADE_NOTIFICATION_KIND, TRADE_SECTION_XLING_KIND], "query: ", "passage: "
    )
    assert not check.ok
    assert [m[0] for m in check.mismatches] == [TRADE_SECTION_XLING_KIND]


def test_undeclared_kind_is_reported_unchecked_never_counted_as_verified() -> None:
    """
    'No mismatches' must never be able to mean 'nothing was inspected'.

    gov-indic's kinds declare no regime; the check must pass without claiming to
    have verified anything.
    """
    check = check_prefix_regime(["pib_title_body"], "", "")
    assert check.ok
    assert check.checked == ()
    assert check.unchecked == ("pib_title_body",)
    assert "NOT checked" in check.describe()


def test_empty_and_falsy_kinds_are_ignored() -> None:
    check = check_prefix_regime(["", None, TRADE_NOTIFICATION_KIND], "query: ", "passage: ")
    assert check.checked == (TRADE_NOTIFICATION_KIND,)


def test_suite_is_not_vacuous() -> None:
    """
    A must-fail sanity mutant.

    If the checker were stubbed to always return ok, every assertion above that
    proves a PASS would still hold. This one would not: it demands that a known
    mismatch actually reports as a mismatch, so the suite cannot go green
    against a checker that has stopped checking.
    """
    good = check_prefix_regime([TRADE_NOTIFICATION_KIND], "query: ", "passage: ")
    bad = check_prefix_regime([TRADE_NOTIFICATION_KIND], "", "")
    assert good.ok is True
    assert bad.ok is False
    assert good.ok != bad.ok, "checker is not discriminating between regimes"
