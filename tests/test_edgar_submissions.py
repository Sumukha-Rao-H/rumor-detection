"""Submissions paging tests — no network.

`filings.recent` is not a company's whole history. SEC keeps the most recent
1,000 filings *or* one year there, whichever is larger. JPMorgan files enough
that one year fills 25,937 records, so its `recent` block begins 2025-08-29
while the study window opens 2024-09-01. Reading `recent` alone would drop
eleven months for exactly the companies that file the most, with no error
anywhere — the company would simply look like it had no events.

The page list below is JPMorgan's real one, trimmed.
"""

import pytest

from src.collectors.edgar import (
    EdgarRequestError, fetch_company_filings, pages_to_fetch, records_from_block,
)
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts


# JPMorgan (CIK 0000019617), fetched 2026-08-29. 69 pages; the first 13 here.
JPM_PAGES = [
    {"name": "CIK0000019617-submissions-001.json", "filingCount": 2028,
     "filingFrom": "2025-07-29", "filingTo": "2025-08-27"},
    {"name": "CIK0000019617-submissions-002.json", "filingCount": 2000,
     "filingFrom": "2025-06-24", "filingTo": "2025-07-27"},
    {"name": "CIK0000019617-submissions-003.json", "filingCount": 2096,
     "filingFrom": "2025-05-13", "filingTo": "2025-06-22"},
    {"name": "CIK0000019617-submissions-004.json", "filingCount": 2030,
     "filingFrom": "2025-04-04", "filingTo": "2025-05-11"},
    {"name": "CIK0000019617-submissions-005.json", "filingCount": 2227,
     "filingFrom": "2025-03-04", "filingTo": "2025-04-02"},
    {"name": "CIK0000019617-submissions-006.json", "filingCount": 2057,
     "filingFrom": "2025-01-29", "filingTo": "2025-03-02"},
    {"name": "CIK0000019617-submissions-007.json", "filingCount": 2031,
     "filingFrom": "2024-12-17", "filingTo": "2025-01-27"},
    {"name": "CIK0000019617-submissions-008.json", "filingCount": 2042,
     "filingFrom": "2024-11-13", "filingTo": "2024-12-15"},
    {"name": "CIK0000019617-submissions-009.json", "filingCount": 2057,
     "filingFrom": "2024-10-08", "filingTo": "2024-11-11"},
    {"name": "CIK0000019617-submissions-010.json", "filingCount": 2090,
     "filingFrom": "2024-09-03", "filingTo": "2024-10-06"},
    {"name": "CIK0000019617-submissions-011.json", "filingCount": 2056,
     "filingFrom": "2024-07-30", "filingTo": "2024-09-01"},
    {"name": "CIK0000019617-submissions-012.json", "filingCount": 2048,
     "filingFrom": "2024-06-25", "filingTo": "2024-07-28"},
    {"name": "CIK0000019617-submissions-069.json", "filingCount": 903,
     "filingFrom": "1994-01-20", "filingTo": "2002-02-09"},
]

# Apple's only older page: entirely before the window.
AAPL_PAGES = [
    {"name": "CIK0000320193-submissions-001.json", "filingCount": 1242,
     "filingFrom": "1994-01-26", "filingTo": "2015-06-08"},
]


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


#: A window that reaches back PAST JPMorgan's `recent` block, so paging is
#: required. Deliberately explicit rather than read from config: this file
#: tests `pages_to_fetch`, and tying it to whatever the study window happens to
#: be today would make the tests silently stop exercising the paging logic the
#: moment the window moves. It moved on 2026-08-29 (shortened to 2025-09-01),
#: which is exactly how this was found.
PAGING_WINDOW = (date_str_to_ts("2024-09-01"), date_str_to_ts("2026-08-01"))


@pytest.fixture(scope="module")
def window() -> tuple[int, int]:
    return PAGING_WINDOW


def block(**fields) -> dict:
    return fields


class FakeClient:
    """Serves a submissions payload and a dict of page name -> payload."""

    def __init__(self, submissions, pages=None):
        self.submissions = submissions
        self.pages = pages or {}
        self.urls: list[str] = []

    def submissions_url(self, cik) -> str:
        return f"https://data.sec.gov/submissions/CIK{cik}.json"

    def submissions_page_url(self, name) -> str:
        return f"https://data.sec.gov/submissions/{name}"

    def get_json(self, url, force=False):
        self.urls.append(url)
        name = url.rsplit("/", 1)[-1]
        if name in self.pages:
            # An older-filings page is the bare block, with no `filings`
            # wrapper — verified against the real API.
            return self.pages[name]
        return self.submissions


# -- page selection --------------------------------------------------------

def test_pages_before_the_window_are_skipped(window):
    assert pages_to_fetch(AAPL_PAGES, *window) == [], (
        "Apple's recent block already reaches back past the window start")


def test_pages_overlapping_the_window_are_fetched(window):
    wanted = pages_to_fetch(JPM_PAGES, *window)
    assert wanted == [p["name"] for p in JPM_PAGES[:11]], (
        "eleven months of JPMorgan's window live in pages 001-011")
    assert "CIK0000019617-submissions-069.json" not in wanted
    assert "CIK0000019617-submissions-012.json" not in wanted


def test_a_page_ending_exactly_on_the_window_start_is_kept(window):
    """Page 011 ends 2024-09-01 — the window's first day. Off-by-one territory."""
    assert "CIK0000019617-submissions-011.json" in pages_to_fetch(JPM_PAGES, *window)


def test_missing_date_bounds_are_fetched_not_guessed(window):
    pages = [{"name": "page-a.json"},
             {"name": "page-b.json", "filingFrom": "", "filingTo": ""}]
    assert pages_to_fetch(pages, *window) == ["page-a.json", "page-b.json"], (
        "a wrong skip is invisible in the output; an extra fetch is not")


def test_a_page_without_a_name_is_ignored(window):
    assert pages_to_fetch([{"filingFrom": "2025-01-01", "filingTo": "2025-02-01"}],
                          *window) == []


def test_no_files_block_is_not_an_error(window):
    assert pages_to_fetch([], *window) == []
    assert pages_to_fetch(None, *window) == []


# -- record shaping --------------------------------------------------------

def test_records_from_block_zips_the_parallel_arrays():
    recs = records_from_block(block(
        accessionNumber=["a-1", "a-2"], form=["8-K", "4"],
        filingDate=["2025-01-02", "2025-01-03"]))
    assert recs == [
        {"accessionNumber": "a-1", "form": "8-K", "filingDate": "2025-01-02"},
        {"accessionNumber": "a-2", "form": "4", "filingDate": "2025-01-03"},
    ]


def test_ragged_arrays_raise():
    """zip() would truncate to the shortest and misalign every later field."""
    with pytest.raises(EdgarRequestError, match="ragged"):
        records_from_block(block(accessionNumber=["a-1", "a-2"], form=["8-K"]))


def test_empty_recent_returns_nothing():
    assert records_from_block({}) == []
    assert records_from_block(block(accessionNumber=[], form=[])) == []


# -- the whole fetch -------------------------------------------------------

def test_filings_from_recent_and_pages_are_combined(cfg):
    """The Done-when: a heavy filer reaches back past the window start."""
    submissions = {"filings": {
        "recent": block(accessionNumber=["new-1"], form=["8-K"],
                        filingDate=["2025-09-01"]),
        "files": JPM_PAGES,
    }}
    pages = {
        p["name"]: block(accessionNumber=[f"old-{i}"], form=["8-K"],
                         filingDate=[p["filingFrom"]])
        for i, p in enumerate(JPM_PAGES)
    }
    client = FakeClient(submissions, pages)
    records = fetch_company_filings(cfg, client, "0000019617", *PAGING_WINDOW)

    dates = sorted(r["filingDate"] for r in records)
    assert dates[0] < "2024-09-01" <= dates[-1], (
        "paging must reach earlier than the recent block's own start")
    assert len(client.urls) == 12, "1 submissions call + 11 pages, not 70"


def test_a_short_history_costs_one_request(cfg):
    submissions = {"filings": {
        "recent": block(accessionNumber=["a-1"], form=["8-K"],
                        filingDate=["2025-01-01"]),
        "files": AAPL_PAGES,
    }}
    client = FakeClient(submissions)
    assert len(fetch_company_filings(cfg, client, "0000320193")) == 1
    assert len(client.urls) == 1


def test_duplicate_accession_numbers_are_dropped(cfg):
    """Consecutive pages share an edge date, so a filing can arrive twice."""
    submissions = {"filings": {
        "recent": block(accessionNumber=["dup"], form=["8-K"],
                        filingDate=["2024-09-03"]),
        "files": [JPM_PAGES[9]],
    }}
    pages = {JPM_PAGES[9]["name"]: block(
        accessionNumber=["dup", "other"], form=["8-K", "8-K"],
        filingDate=["2024-09-03", "2024-09-04"])}
    records = fetch_company_filings(cfg, FakeClient(submissions, pages), "x",
                                    *PAGING_WINDOW)
    assert sorted(r["accessionNumber"] for r in records) == ["dup", "other"]


def test_a_company_that_has_filed_nothing_is_not_an_error(cfg):
    client = FakeClient({"filings": {"recent": {}, "files": []}})
    assert fetch_company_filings(cfg, client, "0000000001") == []


def test_page_urls_sit_beside_the_submissions_file(cfg):
    import requests

    from src.collectors.edgar import EdgarClient
    # A session is injected because this only exercises URL building; omitting
    # one would signal a live SEC client, which now demands a real contact
    # address (see `require_sec_user_agent`).
    c = EdgarClient(cfg, session=requests.Session())
    assert c.submissions_page_url("CIK0000019617-submissions-011.json") == (
        f"{cfg['edgar']['submissions_base']}/CIK0000019617-submissions-011.json")


def test_a_wrapped_block_is_named_not_a_keyerror():
    """The main file wraps the block; a page does not. Confusing the two should say so."""
    with pytest.raises(EdgarRequestError, match="parallel arrays"):
        records_from_block({"filings": {"recent": {"form": ["8-K"]}}})


# --------------------------------------------------------------------------
# What the shortened study window did to paging (2026-08-29)
# --------------------------------------------------------------------------

def test_the_current_window_needs_fewer_pages_but_not_none_for_long(cfg):
    """Paging is not dead code just because the window shrank.

    `recent` holds the most recent 1,000 filings OR one year, whichever is
    larger. The window now opens 2025-09-01, and one year back from today is
    2025-08-29 — **three days of margin**. Every week that passes, `recent`'s
    one-year floor moves a week closer to the window start, and heavy filers
    need their pages again.

    So this asserts the mechanism still selects correctly against the live
    config, without asserting a page count that is true only this week.
    """
    start = date_str_to_ts(cfg["study_window"]["start"])
    end = date_str_to_ts(cfg["study_window"]["end"])

    # A page that ends after the window opens is always required.
    overlapping = [{"name": "p-late.json", "filingFrom": "2025-08-01",
                    "filingTo": "2025-09-15"}]
    assert pages_to_fetch(overlapping, start, end) == ["p-late.json"]

    # One that ends before it never is.
    old = [{"name": "p-old.json", "filingFrom": "2024-01-01",
            "filingTo": "2024-06-01"}]
    assert pages_to_fetch(old, start, end) == []
