"""
Check a run's E5 prefixes against the regime its corpus declares.

A corpus module states, per pair kind, which ``(anchor, positive)`` prefixes
that kind trains and serves with. Nothing used to read those tables outside
their own tests: a training run takes ONE global prefix pair from its config
and applies it to every pair, so the declared regime and the actual run agreed
only by discipline.

That is the shape our house rule warns about — a stated guarantee where a check
belongs — and the failure it permits is expensive and quiet. A train/serve
prefix mismatch does not raise, does not look wrong in a loss curve, and shows
up only as retrieval that is worse than it should be, long after the run.

This module closes that gap. It deliberately does NOT invent a second copy of
the regimes: it imports the corpus modules' own tables, so there is one source
of truth and this file cannot drift from them.

**Honest about coverage.** Only kinds that actually declare a regime can be
checked. An undeclared kind is reported as *unchecked*, never folded into the
pass count — "0 mismatches" must never be able to mean "nothing was inspected".
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PrefixCheck", "declared_regimes", "check_prefix_regime"]


def declared_regimes() -> dict[str, tuple[str, str]]:
    """
    Every pair kind that declares a prefix regime, from the owning module.

    Imported lazily and defensively: a corpus module that fails to import must
    narrow this registry, never break a training run that does not use it.
    """
    regimes: dict[str, tuple[str, str]] = {}

    try:
        from . import trade

        for kind in (trade.TRADE_NOTIFICATION_KIND,
                     trade.TRADE_SECTION_KIND,
                     trade.TRADE_SECTION_XLING_KIND):
            regimes[kind] = trade.prefix_regime(kind)
    except Exception:  # pragma: no cover - a missing module just narrows coverage
        pass

    try:
        from . import eulaw

        regimes[eulaw.EULAW_PROVISION_XLING_KIND] = eulaw.prefix_regime(
            eulaw.EULAW_PROVISION_XLING_KIND
        )
    except Exception:  # pragma: no cover
        pass

    return regimes


@dataclass(frozen=True)
class PrefixCheck:
    """What the check could and could not establish."""

    checked: tuple[str, ...] = ()
    unchecked: tuple[str, ...] = ()
    mismatches: tuple[tuple[str, tuple[str, str], tuple[str, str]], ...] = ()
    configured: tuple[str, str] = ("", "")

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def describe(self) -> str:
        parts = [
            f"configured prefixes {self.configured!r}",
            f"{len(self.checked)} kind(s) checked",
        ]
        if self.unchecked:
            parts.append(
                f"{len(self.unchecked)} kind(s) declare no regime and were NOT "
                f"checked: {', '.join(sorted(self.unchecked))}"
            )
        for kind, declared, configured in self.mismatches:
            parts.append(
                f"MISMATCH {kind!r}: corpus declares {declared!r}, run configured {configured!r}"
            )
        return "; ".join(parts)


def check_prefix_regime(
    kinds: object,
    query_prefix: str,
    passage_prefix: str,
) -> PrefixCheck:
    """
    Compare the run's single global prefix pair to each kind's declared regime.

    ``kinds`` is any iterable of the pair kinds a run will train on. Returns a
    :class:`PrefixCheck`; it does not raise, so a caller decides whether a
    mismatch is fatal for its context.
    """
    regimes = declared_regimes()
    configured = (query_prefix, passage_prefix)

    checked: list[str] = []
    unchecked: list[str] = []
    mismatches: list[tuple[str, tuple[str, str], tuple[str, str]]] = []

    for kind in sorted({k for k in kinds if k}):
        declared = regimes.get(kind)
        if declared is None:
            unchecked.append(kind)
            continue
        checked.append(kind)
        if declared != configured:
            mismatches.append((kind, declared, configured))

    return PrefixCheck(
        checked=tuple(checked),
        unchecked=tuple(unchecked),
        mismatches=tuple(mismatches),
        configured=configured,
    )
