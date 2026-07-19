"""
Structural base classes for corpus nodes.
"""

from __future__ import annotations

from .container_node import ContainerNode
from .node import Composite, Spanned
from .text_node import TextNode

__all__ = [
    "Composite",
    "ContainerNode",
    "Spanned",
    "TextNode",
]
