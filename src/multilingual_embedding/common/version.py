"""
Framework version information.

This is the single source of truth for the version. ``pyproject.toml``
declares ``dynamic = ["version"]`` and hatchling reads the literal below,
so the distribution on PyPI, the wheel filename, ``qfme --version`` and
the ``framework_version`` stamped into every evaluation report cannot
disagree.

They used to be two independent literals. Nothing asserted they matched,
and the failure was silent in the worst way: a wheel could be published
as one version while every report it generated claimed another, so the
provenance recorded alongside a published number would be wrong with
nothing raising. That matters now that other repositories pin this one.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.3.0"
