"""8-K parsing tests — the two traps that do not raise anything.

1. Item codes are strings. As a float, "1.01" still prints 1.01 and looks
   fine — but "1.10" becomes 1.1, and so does "1.1", merging two different
   item codes into one. The codes are the event taxonomy the study splits on.

2. `acceptanceDateTime` is UTC. Read as local time, every t0 shifts by four or
   five hours, and by a *different* amount either side of a daylight-saving
   change, so the error is not even constant.

The acceptance-time cases use Apple's real Q3 earnings 8-K, accession
0000320193-26-000018, fetched from EDGAR on 2026-08-29.
"""

import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src import db
from src.collectors.edgar import EdgarRequestError, filing_rows, normalise_items
from src.utils.config import load_config


# Apple's Q3 FY26 earnings 8-K, verbatim from data.sec.gov.
APPLE_8K = {
    "accessionNumber": "0000320193-26-000018",
    "form": "8-K",
    "items": "2.02,9.01",
    "acceptanceDateTime": "2026-07-30T20:30:28.000Z",
    "filingDate": "2026-07-30",
    "reportDate": "2026-07-30",
    "primaryDocument": "aapl-20260730.htm",
}
APPLE_ACCEPTANCE_UTC = 1785443428   # hand-checked below

FORM_4 = {
    "accessionNumber": "0001140361-26-034741", "form": "4", "items": "",
    "acceptanceDateTime": "2026-08-27T22:30:30.000Z",
    "filingDate": "2026-08-27", "reportDate": "2026-08-25",
    "primaryDocument": "xslF345X06/form4.xml",
}


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


@pytest.fixture
def conn(tmp_path):
    c = db.get_conn(tmp_path / "test.db")
    yield c
    c.close()


def rows(cfg, *records, cik="0000320193", ticker="AAPL"):
    return filing_rows(cfg, list(records), cik, ticker)


# -- trap 1: item codes ----------------------------------------------------

def test_item_codes_survive_the_round_trip_as_strings(cfg, conn):
    """The Done-when: out of SQLite as "1.01", not 1.01."""
    db.upsert_filings(conn, rows(cfg, {**APPLE_8K, "items": "1.01"}))
    value = conn.execute("SELECT items FROM filings").fetchone()["items"]
    assert value == "1.01"
    assert isinstance(value, str)


def test_one_point_ten_and_one_point_one_stay_different(cfg, conn):
    """Stated explicitly because this is the whole reason for the rule."""
    db.upsert_filings(conn, rows(
        cfg,
        {**APPLE_8K, "accessionNumber": "a-1", "items": "1.10"},
        {**APPLE_8K, "accessionNumber": "a-2", "items": "1.1"},
    ))
    stored = {r["items"] for r in conn.execute("SELECT items FROM filings")}
    assert stored == {"1.10", "1.1"}
    assert float("1.10") == float("1.1"), "which is exactly what must not happen"


def test_a_numeric_item_code_raises(cfg):
    with pytest.raises(EdgarRequestError, match="must be strings"):
        rows(cfg, {**APPLE_8K, "items": 1.01})


def test_whitespace_in_items_is_normalised():
    assert normalise_items("2.02, 9.01") == "2.02,9.01"
    assert normalise_items(" 5.02 ") == "5.02"
    assert normalise_items("2.02,,9.01") == "2.02,9.01"


def test_empty_items_is_an_empty_string_not_null():
    """Some 8-Ks carry no item codes. A LIKE filter must still behave."""
    assert normalise_items("") == ""
    assert normalise_items(None) == ""


# -- trap 2: acceptance time -----------------------------------------------

def test_acceptance_time_is_utc_not_local(cfg):
    """The Done-when, against a filing that can be looked up on EDGAR by hand."""
    row = rows(cfg, APPLE_8K)[0]
    assert row["acceptance_utc"] == APPLE_ACCEPTANCE_UTC


def test_acceptance_time_hand_checked_against_eastern(cfg):
    """20:30:28 UTC is 16:30:28 in New York — half an hour after the close.

    Exactly where an earnings 8-K belongs, and the reason acceptance time alone
    is not t0: the press release went out before it.
    """
    row = rows(cfg, APPLE_8K)[0]
    eastern = datetime.fromtimestamp(row["acceptance_utc"],
                                     ZoneInfo("America/New_York"))
    assert (eastern.hour, eastern.minute) == (16, 30)
    assert eastern.utcoffset().total_seconds() == -4 * 3600  # EDT in July


def test_blank_acceptance_time_keeps_the_row_with_null(cfg):
    """Lose a timestamp, never an event. P2-10 counts these."""
    row = rows(cfg, {**APPLE_8K, "acceptanceDateTime": ""})[0]
    assert row["acceptance_utc"] is None
    assert row["accession_no"] == "0000320193-26-000018"


def test_blank_report_date_is_null_not_zero(cfg):
    """0 would read as 1 January 1970 — the oldest event in the study."""
    row = rows(cfg, {**APPLE_8K, "reportDate": ""})[0]
    assert row["report_date_utc"] is None


def test_dates_are_utc_midnight(cfg):
    row = rows(cfg, APPLE_8K)[0]
    assert row["filing_date_utc"] == 1785369600  # 2026-07-30 00:00 UTC


# -- form selection --------------------------------------------------------

def test_only_configured_forms_are_kept(cfg):
    kept = rows(cfg, APPLE_8K, FORM_4, {**APPLE_8K, "accessionNumber": "b",
                                        "form": "10-Q"})
    assert [r["form"] for r in kept] == ["8-K"]


def test_amendments_are_kept(cfg):
    kept = rows(cfg, {**APPLE_8K, "accessionNumber": "amend", "form": "8-K/A"})
    assert len(kept) == 1, "an 8-K/A is a real filing with its own acceptance time"


def test_8k12b_is_not_matched_by_prefix(cfg):
    """startswith('8-K') would swallow a different form entirely."""
    kept = rows(cfg, {**APPLE_8K, "accessionNumber": "c", "form": "8-K12B"})
    assert kept == []


# -- the row shape ---------------------------------------------------------

def test_cik_and_ticker_are_carried_onto_every_row(cfg):
    row = rows(cfg, APPLE_8K)[0]
    assert (row["cik"], row["ticker"]) == ("0000320193", "AAPL")
    assert row["primary_doc"] == "aapl-20260730.htm"
    assert row["fetched_utc"] > 0


def test_a_company_with_no_ticker_still_stores_its_filings(cfg):
    row = rows(cfg, APPLE_8K, ticker=None)[0]
    assert row["ticker"] is None


def test_rerun_adds_no_duplicate_filings(cfg, conn):
    assert db.upsert_filings(conn, rows(cfg, APPLE_8K)) == 1
    assert db.upsert_filings(conn, rows(cfg, APPLE_8K)) == 0
    assert conn.execute("SELECT COUNT(*) FROM filings").fetchone()[0] == 1


def test_the_stored_column_type_is_text(cfg, conn):
    """SQLite is loosely typed; assert what actually came back out."""
    db.upsert_filings(conn, rows(cfg, APPLE_8K))
    r = conn.execute("SELECT typeof(items), typeof(acceptance_utc) FROM filings"
                     ).fetchone()
    assert tuple(r) == ("text", "integer")
