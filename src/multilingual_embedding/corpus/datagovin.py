"""
Fetching a data.gov.in resource into a local CSV — the pure, tested core.

This is the **online front door** for any data.gov.in resource (the Kisan Call
Centre transcripts first; import/export and finance resources can reuse it). It
is split deliberately in two so the fragile part is the tested part:

- **here (library, offline, tested):** the *pagination assembly* — walk a
  resource's ``offset``/``limit`` pages and reconcile what arrived against the
  total the server reports. This is the logic where a silent off-by-one quietly
  truncates a corpus, so per the oracle-diff / silent-defect house rule it lives
  Mac-side behind an *injected* page-fetcher and is pinned by tests, never
  buried in an untested box script.
- **the box edge (:mod:`.kcc_crawl`, online, untested):** the thin live layer —
  it reads the API key from the environment, builds the real HTTP page fetcher,
  and runs on the GPU box under an explicit go. Nothing here opens a socket; the
  transport is a callable the caller supplies.

The data.gov.in v1 resource API answers
``GET https://api.data.gov.in/resource/{id}?api-key=…&format=json&offset=O&limit=L``
with a ``{"total", "count", "offset", "limit", "records": [...]}`` envelope. A
:class:`ResourcePage` is exactly that answer, and :func:`paginate_resource`
drives it.

Three pagination bugs this core is written to make impossible, each pinned by a
test:

1. **Stepping by the requested limit** when the server caps a page smaller than
   asked — silently skips rows. We advance by the number of records a page
   *actually* returned, never by the requested ``limit``.
2. **Stopping at the first short page** — a page shorter than the limit is the
   server's page size, not the end of the resource. Only an *empty* page or
   reaching the reported total ends the walk.
3. **A truncated pull passing as complete** — if the server serves fewer rows
   than it reported as ``total``, :attr:`FetchStatistics.shortfall` is non-zero
   and names the gap loudly instead of letting a partial corpus look whole.
"""

from __future__ import annotations

import csv
import gzip
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from multilingual_embedding.core.exceptions import MultilingualEmbeddingError
from multilingual_embedding.core.logging import get_logger

__all__ = [
    "DataGovInError",
    "FetchStatistics",
    "PageFetcher",
    "ResourcePage",
    "paginate_resource",
    "write_resource_csv",
]

_logger = get_logger(__name__)


class DataGovInError(MultilingualEmbeddingError):
    """Raised when a data.gov.in request is malformed (e.g. a non-positive limit)."""


@dataclass(slots=True)
class ResourcePage:
    """
    One page of a data.gov.in resource response.

    Mirrors the API's JSON envelope: ``records`` is this page's rows, ``total``
    is the resource-wide row count the server reports, and ``offset``/``limit``
    echo the request. ``total`` is what the assembly reconciles against; a page
    that reports ``total <= 0`` is treated as "count unknown", and then only an
    empty page ends the walk.
    """

    records: list[dict[str, Any]]

    total: int

    offset: int = 0

    limit: int = 0


# The injected transport boundary: given (offset, limit), return that page. The
# live HTTP implementation is built on the box (:mod:`.kcc_crawl`); tests pass a
# fake in-memory source. Keeping this a plain callable is what lets the assembly
# be tested without a network.
PageFetcher = Callable[[int, int], ResourcePage]


@dataclass(slots=True)
class FetchStatistics:
    """
    A reconciling ledger of a paginated fetch.

    ``assembled`` is how many records the walk yielded; ``total_reported`` is
    what the server claimed the resource holds. They should agree — when they do
    not, :attr:`shortfall` (reported minus assembled) is positive and names a
    truncated pull, so a partial corpus cannot quietly pass as complete.
    ``stopped_reason`` records *why* the walk ended (``complete`` /
    ``empty_page`` / ``max_records``).
    """

    pages: int = 0

    assembled: int = 0

    total_reported: int = 0

    stopped_reason: str = ""

    @property
    def shortfall(self) -> int:
        """Reported total minus what we assembled; positive means rows are missing."""

        return self.total_reported - self.assembled

    def to_dict(self) -> dict[str, Any]:
        """A JSON-friendly view for a build report."""

        return {
            "pages": self.pages,
            "assembled": self.assembled,
            "total_reported": self.total_reported,
            "shortfall": self.shortfall,
            "stopped_reason": self.stopped_reason,
        }


def paginate_resource(
    fetch_page: PageFetcher,
    *,
    limit: int = 1000,
    max_records: int | None = None,
    stats: FetchStatistics | None = None,
) -> Iterator[dict[str, Any]]:
    """
    Walk a resource's pages and yield every record, exactly once, in order.

    ``fetch_page(offset, limit)`` returns a :class:`ResourcePage`; this drives it
    from ``offset = 0`` upward, advancing by the number of records a page
    *actually* returned (never by the requested ``limit``), so a server that
    caps pages below the ask does not cause skipped rows. The walk ends only on
    an empty page, on reaching the server-reported ``total``, or on
    ``max_records`` — a short-but-non-empty page is the server's page size, not
    the end.

    ``max_records`` is the hard safety bound (the analogue of the PIB crawl's
    fetch cap); the box edge always sets it so a misreported total cannot spin.
    Every stop is recorded in ``stats`` with a reason, and ``stats.shortfall``
    reconciles what was assembled against the reported total.
    """

    if limit <= 0:
        raise DataGovInError("page limit must be positive", limit=limit)

    stats = stats if stats is not None else FetchStatistics()

    offset = 0

    while True:
        page = fetch_page(offset, limit)

        stats.pages += 1

        # Trust the server's latest total for the reconciliation. It should not
        # move mid-crawl; if a resource grows underneath us, the newest figure
        # is the honest denominator for the shortfall.
        stats.total_reported = page.total

        # An empty page is the only unambiguous "no more rows" signal, so it
        # ends the walk even when the reported total was never reached — and the
        # shortfall then makes that gap loud.
        if not page.records:
            stats.stopped_reason = "empty_page"
            return

        for record in page.records:
            if max_records is not None and stats.assembled >= max_records:
                stats.stopped_reason = "max_records"
                return

            stats.assembled += 1

            yield record

        # Advance by what actually arrived — NOT by the requested limit — so a
        # server-capped short page does not skip the rows between the cap and
        # the ask.
        offset += len(page.records)

        # Reached (or passed) the reported total: done, and without paying for
        # one more fetch that would return an empty page. Guarded on a positive
        # total so a "count unknown" resource falls through to the empty-page
        # terminator instead of stopping after the first page.
        if page.total > 0 and offset >= page.total:
            stats.stopped_reason = "complete"
            return


def write_resource_csv(
    records: Iterable[dict[str, Any]],
    output_path: str | Path,
    *,
    fieldnames: Iterable[str],
) -> int:
    """
    Write assembled records to the CSV the domain reader consumes.

    ``fieldnames`` fixes the header and column order (for KCC, pass
    ``KCC_CSV_FIELDNAMES`` so :func:`.kcc.read_kcc_rows`'s case-insensitive
    column resolve finds ``QueryText`` / ``KccAns`` / …). A key a record is
    missing is written blank; an extra key is ignored, so a resource that adds a
    column does not break the write. Streams row-by-row (so it composes with the
    lazy :func:`paginate_resource` at bounded memory) and writes gzip when
    ``output_path`` ends in ``.gz``. Returns the row count written.
    """

    header = list(fieldnames)

    output = Path(output_path).expanduser()

    output.parent.mkdir(parents=True, exist_ok=True)

    opener = gzip.open if output.suffix == ".gz" else open

    written = 0

    with opener(output, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")

        writer.writeheader()

        for record in records:
            writer.writerow({key: record.get(key, "") for key in header})

            written += 1

    _logger.info("Wrote data.gov.in resource CSV", extra={"rows": written, "path": str(output)})

    return written
