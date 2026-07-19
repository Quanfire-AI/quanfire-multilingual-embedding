"""
Cross cutting helpers: validation, hashing, filesystem, I/O, serialisation.

Modules here depend on :mod:`multilingual_embedding.common` and
:mod:`multilingual_embedding.core` only. They must never import from
domain packages such as ``corpus`` or ``tokenizer``.
"""

from __future__ import annotations

from .filesystem import (
    atomic_write_path,
    ensure_directory,
    human_readable_size,
    iter_files,
    require_directory,
    require_file,
)
from .hashing import (
    DEFAULT_DIGEST_SIZE,
    hash_bytes,
    hash_file,
    hash_iterable,
    hash_object,
    hash_text,
)
from .io import (
    count_lines,
    open_text,
    read_json,
    read_jsonl,
    read_text,
    read_yaml,
    write_json,
    write_jsonl,
    write_text,
    write_yaml,
)
from .serialization import (
    from_primitive,
    is_dataclass_type,
    to_primitive,
)
from .validation import (
    require_in_range,
    require_non_empty_collection,
    require_non_empty_string,
    require_non_negative,
    require_one_of,
    require_positive,
    require_same_length,
)

__all__ = [
    "DEFAULT_DIGEST_SIZE",
    "atomic_write_path",
    "count_lines",
    "ensure_directory",
    "from_primitive",
    "hash_bytes",
    "hash_file",
    "hash_iterable",
    "hash_object",
    "hash_text",
    "human_readable_size",
    "is_dataclass_type",
    "iter_files",
    "open_text",
    "read_json",
    "read_jsonl",
    "read_text",
    "read_yaml",
    "require_directory",
    "require_file",
    "require_in_range",
    "require_non_empty_collection",
    "require_non_empty_string",
    "require_non_negative",
    "require_one_of",
    "require_positive",
    "require_same_length",
    "to_primitive",
    "write_json",
    "write_jsonl",
    "write_text",
    "write_yaml",
]
