"""
Kisan Call Centre crawl (box-run) — fetch the data.gov.in KCC resource to CSV.

The thin live edge over the tested :mod:`.datagovin` paginator. This runs **on
the GPU box under an explicit go**: it reads the api key from the environment
and opens sockets, so it is never exercised in tests and never run on the
laptop. All the fragile logic — pagination assembly and total reconciliation —
lives in :mod:`.datagovin` and is tested there; this file only supplies the HTTP
transport and the key, then hands off to that tested core.

Usage (on the box, with the key exported — the key is never committed, never
logged, and never printed by this script)::

    export DATA_GOV_IN_API_KEY=...
    python -m multilingual_embedding.corpus.kcc_crawl \
        <resource_id> <out.csv.gz> [max_records] [limit]

The offline path then takes over on the laptop, unchanged::

    read_kcc_rows(out.csv.gz) -> iter_kcc_pairs -> extract_kcc_pairs

Provenance: the KCC resource is published under GODL-India (see :mod:`.kcc` for
the attribution that rides on every emitted pair).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request

from .datagovin import (
    FetchStatistics,
    PageFetcher,
    ResourcePage,
    paginate_resource,
    write_resource_csv,
)
from .kcc import KCC_CSV_FIELDNAMES

# The live JSON endpoint. The public resource page (no key) is used only to
# print an attribution pointer at the end.
_API_URL = "https://api.data.gov.in/resource/{resource_id}"

_PUBLIC_URL = "https://www.data.gov.in/resource/{resource_id}"

# Polite pause between pages, and a hard safety bound analogous to the PIB
# crawl's fetch cap so a misreported total can never spin the walk forever.
_DELAY_SECONDS = 0.5

_DEFAULT_LIMIT = 1000

_DEFAULT_MAX_RECORDS = 500_000


def _live_fetcher(resource_id: str, api_key: str) -> PageFetcher:
    """
    Build a :data:`~.datagovin.PageFetcher` bound to the live API.

    The key is captured in this closure and used only to sign the request; the
    request URL (which carries the key) is never logged — only the offset/limit
    are printed, so the key cannot leak into stdout or a log file.
    """

    endpoint = _API_URL.format(resource_id=resource_id)

    def fetch_page(offset: int, limit: int) -> ResourcePage:
        query = urllib.parse.urlencode(
            {"api-key": api_key, "format": "json", "offset": offset, "limit": limit}
        )

        request = urllib.request.Request(
            f"{endpoint}?{query}", headers={"Accept": "application/json"}
        )

        print(f"[fetch] offset={offset} limit={limit}", flush=True)

        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))

        time.sleep(_DELAY_SECONDS)

        return ResourcePage(
            records=list(payload.get("records") or []),
            total=int(payload.get("total") or 0),
            offset=int(payload.get("offset") or offset),
            limit=int(payload.get("limit") or limit),
        )

    return fetch_page


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(
            "usage: python -m multilingual_embedding.corpus.kcc_crawl "
            "<resource_id> <out.csv[.gz]> [max_records] [limit]",
            flush=True,
        )
        return 2

    resource_id = argv[0]

    output_path = argv[1]

    max_records = int(argv[2]) if len(argv) > 2 else _DEFAULT_MAX_RECORDS

    limit = int(argv[3]) if len(argv) > 3 else _DEFAULT_LIMIT

    api_key = os.environ.get("DATA_GOV_IN_API_KEY", "").strip()

    if not api_key:
        print("[error] DATA_GOV_IN_API_KEY is not set in the environment", flush=True)
        return 3

    stats = FetchStatistics()

    records = paginate_resource(
        _live_fetcher(resource_id, api_key),
        limit=limit,
        max_records=max_records,
        stats=stats,
    )

    written = write_resource_csv(records, output_path, fieldnames=KCC_CSV_FIELDNAMES)

    print(
        f"[done] wrote {written} rows to {output_path} "
        f"(pages={stats.pages}, total_reported={stats.total_reported}, "
        f"shortfall={stats.shortfall}, stop={stats.stopped_reason})",
        flush=True,
    )

    if stats.shortfall > 0:
        print(
            f"[warn] shortfall {stats.shortfall}: assembled fewer rows than the "
            "server reported as total — the pull is incomplete, do not treat it as final",
            flush=True,
        )

    print(
        f"[attribution] source resource: {_PUBLIC_URL.format(resource_id=resource_id)}",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
