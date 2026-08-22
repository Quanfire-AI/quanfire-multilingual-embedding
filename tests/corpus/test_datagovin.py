"""
Paginating a data.gov.in resource into a local CSV.

The fetch is split so the fragile part is the tested part: the live HTTP edge
(:mod:`.kcc_crawl`) runs on the box and is not tested, but the *pagination
assembly* (:mod:`.datagovin`) is pure and pinned here behind a fake in-memory
page source. The three classic pagination bugs each get a test that would fail
if the guard regressed:

- advancing by the requested limit instead of the rows actually returned
  (skips rows when the server caps the page smaller than asked);
- stopping at the first short-but-non-empty page (ends the walk early);
- a truncated pull passing as complete (the shortfall must be loud).

The last test runs the whole offline path — fake fetch → paginate → write CSV →
``read_kcc_rows`` → ``iter_kcc_pairs`` — so the two modules are proven to meet
at the CSV, not just in isolation.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from multilingual_embedding.corpus.datagovin import (
    DataGovInError,
    FetchStatistics,
    ResourcePage,
    paginate_resource,
    write_resource_csv,
)
from multilingual_embedding.corpus.kcc import (
    KCC_CSV_FIELDNAMES,
    iter_kcc_pairs,
    read_kcc_rows,
)


def _records(n: int) -> list[dict[str, str]]:
    """n distinct records, each carrying its own index so order is checkable."""

    return [{"QueryText": f"q{i}", "KccAns": f"answer number {i}"} for i in range(n)]


def _fake_source(records, *, server_cap: int | None = None):
    """
    An in-memory page source mimicking the data.gov.in envelope.

    ``server_cap`` models a server that returns at most that many rows per page
    regardless of the requested limit — the condition that breaks a paginator
    which steps by the requested limit.
    """

    total = len(records)

    def fetch_page(offset: int, limit: int) -> ResourcePage:
        size = limit if server_cap is None else min(limit, server_cap)
        page = records[offset : offset + size]
        return ResourcePage(records=list(page), total=total, offset=offset, limit=limit)

    return fetch_page


def test_paginate_assembles_all_records_once_in_order():
    records = _records(25)
    stats = FetchStatistics()

    out = list(paginate_resource(_fake_source(records), limit=10, stats=stats))

    assert out == records  # every row, exactly once, in order
    assert stats.assembled == 25
    assert stats.total_reported == 25
    assert stats.shortfall == 0
    assert stats.stopped_reason == "complete"


def test_paginate_advances_by_returned_count_not_requested_limit():
    # The server hands back at most 3 rows per page though we ask for 10. A
    # paginator that stepped offset by the requested 10 would fetch rows [0:3]
    # then jump to offset 10 and miss rows 3-9 entirely. Stepping by the 3 that
    # actually arrived keeps every row.
    records = _records(10)
    stats = FetchStatistics()

    out = list(paginate_resource(_fake_source(records, server_cap=3), limit=10, stats=stats))

    assert out == records
    assert stats.assembled == 10


def test_paginate_does_not_stop_on_a_short_nonempty_page():
    # A page shorter than the limit is the server's page size, not the end. With
    # a cap of 4 and 9 rows the pages are 4,4,1,(empty) — the length-1 third page
    # must not be read as "done".
    records = _records(9)
    stats = FetchStatistics()

    out = list(paginate_resource(_fake_source(records, server_cap=4), limit=10, stats=stats))

    assert len(out) == 9
    assert stats.assembled == 9


def test_paginate_stops_on_empty_page():
    stats = FetchStatistics()

    out = list(paginate_resource(_fake_source([]), limit=10, stats=stats))

    assert out == []
    assert stats.stopped_reason == "empty_page"
    assert stats.assembled == 0


def test_paginate_respects_max_records_cap_mid_page():
    records = _records(100)
    stats = FetchStatistics()

    out = list(paginate_resource(_fake_source(records), limit=10, max_records=7, stats=stats))

    assert len(out) == 7
    assert out == records[:7]
    assert stats.assembled == 7
    assert stats.stopped_reason == "max_records"


def test_shortfall_is_loud_when_server_serves_fewer_than_it_reports():
    # The server claims a total of 10 but only ever hands back 6 rows, then an
    # empty page. The walk must end (empty_page) and the ledger must show the
    # gap — a truncated pull cannot pass as complete.
    served = _records(6)

    def fetch_page(offset, limit):
        page = served[offset : offset + limit]
        return ResourcePage(records=list(page), total=10, offset=offset, limit=limit)

    stats = FetchStatistics()
    out = list(paginate_resource(fetch_page, limit=10, stats=stats))

    assert len(out) == 6
    assert stats.total_reported == 10
    assert stats.shortfall == 4
    assert stats.stopped_reason == "empty_page"


def test_paginate_rejects_nonpositive_limit():
    with pytest.raises(DataGovInError):
        list(paginate_resource(_fake_source(_records(3)), limit=0))


def test_whole_path_fake_fetch_to_kcc_pairs(tmp_path: Path):
    """fake fetch → paginate → write CSV → read_kcc_rows → iter_kcc_pairs."""

    # One genuine agri Q->A and one junk row, in the source schema the crawl
    # writes and the KCC reader reads.
    source_records = [
        {
            "StateName": "Punjab",
            "DistrictName": "Ludhiana",
            "Crop": "Paddy",
            "QueryType": "Nutrient",
            "QueryText": "leaf yellow paddy",
            "KccAns": "Apply nitrogen fertiliser and ensure proper drainage in the field.",
            "CreatedOn": "2024-06-01",
        },
        {
            "StateName": "Punjab",
            "DistrictName": "Ludhiana",
            "Crop": "Paddy",
            "QueryType": "General",
            "QueryText": "test",
            "KccAns": "some sufficiently long answer that will be dropped for a junk query",
            "CreatedOn": "2024-06-01",
        },
    ]

    out_csv = tmp_path / "kcc.csv.gz"
    stats = FetchStatistics()

    written = write_resource_csv(
        paginate_resource(_fake_source(source_records), limit=5, stats=stats),
        out_csv,
        fieldnames=KCC_CSV_FIELDNAMES,
    )

    assert written == 2
    assert stats.assembled == 2

    # The CSV the crawl wrote is exactly what the offline reader consumes.
    pairs = list(iter_kcc_pairs(read_kcc_rows(out_csv)))

    assert len(pairs) == 1  # junk query dropped by the filter
    assert pairs[0].anchor == "leaf yellow paddy"
    assert pairs[0].kind == "kcc_qa"

    # And the CSV is real gzip with the expected header.
    with gzip.open(out_csv, "rt", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    assert header == list(KCC_CSV_FIELDNAMES)


def test_fetch_statistics_to_dict_is_json_friendly():
    stats = FetchStatistics(pages=2, assembled=6, total_reported=10, stopped_reason="empty_page")
    payload = json.loads(json.dumps(stats.to_dict()))
    assert payload["shortfall"] == 4
    assert payload["stopped_reason"] == "empty_page"
