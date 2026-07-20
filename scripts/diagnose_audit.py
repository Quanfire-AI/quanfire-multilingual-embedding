"""
Reproduce an audit failure in isolation, with the whole traceback.

Written for a specific report: `audit` raising ``SystemError: attempting
to create PyCMethod with a METH_METHOD flag but no class`` on WSL while
passing on macOS. That error comes from a compiled extension, not from
this project's code, so the useful output is the import chain and the
versions rather than the line that happened to raise.

Reads a few hundred documents and writes nothing.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path


def versions() -> None:
    print("VERSIONS")

    print(f"  python                 {sys.version.split()[0]}")

    for name in ("numpy", "yaml", "sentencepiece", "pandas", "torch", "mwparserfromhell"):
        try:
            module = __import__(name)

            print(f"  {name:22} {getattr(module, '__version__', 'unknown')}")
        except ImportError:
            print(f"  {name:22} not installed")

    runtimes = sorted(m for m in sys.modules if m.startswith("_cython"))

    print(f"  cython runtimes loaded {runtimes or 'none'}")

    if len(runtimes) > 1:
        print("  ^ more than one Cython runtime is loaded. Extensions built")

        print("    against different Cython versions can raise SystemError")

        print("    when one of them defines a method the other cannot accept.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--source", type=Path, required=True, help="An extracted corpus")

    parser.add_argument("--limit", type=int, default=300, help="Documents to read")

    arguments = parser.parse_args()

    print("=" * 60)

    versions()

    print("=" * 60)

    for step, action in (
        ("import reader", lambda: __import__("multilingual_embedding.corpus.reader")),
        ("import audit", lambda: __import__("multilingual_embedding.corpus.audit")),
    ):
        try:
            action()

            print(f"[ok]   {step}")
        except Exception:
            print(f"[FAIL] {step}")

            traceback.print_exc()

            return 1

    from multilingual_embedding.corpus.audit import audit_corpus
    from multilingual_embedding.corpus.reader import reader_for

    print(f"[..]   reading {arguments.limit} documents from {arguments.source}")

    try:
        documents = []

        for index, document in enumerate(reader_for(arguments.source).iter_documents()):
            documents.append(document)

            if index + 1 >= arguments.limit:
                break

        print(f"[ok]   read {len(documents)} documents")
    except Exception:
        print("[FAIL] reading documents")

        traceback.print_exc()

        return 1

    try:
        audit = audit_corpus(documents)

        print(f"[ok]   audited: {audit.documents} documents, usable={audit.usable}")
    except Exception:
        print("[FAIL] audit_corpus -- this is the failure being chased")

        traceback.print_exc()

        return 1

    print()

    print("No failure reproduced at this size. Try a larger --limit.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
