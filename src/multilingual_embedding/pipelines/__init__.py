"""
High level workflows composing the layers below.

:class:`TrainingPipeline` runs corpus to evaluated model.
:class:`SemanticSearchPipeline` loads the result and answers queries.

Pipelines own orchestration only — ordering, artefact plumbing and
record keeping. Every stage remains usable on its own.
"""

from __future__ import annotations

from .search import SearchHit, SemanticSearchPipeline
from .training import TrainingPipeline, TrainingResult, train

__all__ = [
    "SearchHit",
    "SemanticSearchPipeline",
    "TrainingPipeline",
    "TrainingResult",
    "train",
]
