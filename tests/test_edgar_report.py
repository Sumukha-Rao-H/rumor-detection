"""Sanity-report tests — on a fixture whose answers are known by hand.

The acceptance-hour histogram is not decoration. It is the empirical
justification for the whole t0 correction: the plan asserts 8-Ks cluster after
the US close while the press release that moved the market went out earlier. If
this table came out flat, the project's headline contribution would have no
basis.
"""

import copy

import pytest

from src import db
from src.collectors.edgar import (
    acceptance_hour_histogram, ciks_with_no_history_before_the_window,
    filings_per_company, filings_report, item_code_counts, new_york_label,
    print_filings_report,
)
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts, iso_utc_to_ts


WINDOW_START = "2025-09-01"
WINDOW_END = "2026-08-01"


@pytest.fixture(scope="module")
def cfg() -> dict:
    c = copy.deepcopy(load_config())
    c["study_window"]["start"] = WINDOW_START
    c["study_window"]["end"] = WINDOW_END
    return c


def filing(acc, when, items="8.01", cik="0000320193", ticker="AAPL") -> dict:
    return {
        "accession_no": acc, "cik": cik, "ticker": ticker, "form": "8-K",
        "items": items, "acceptance_utc": iso_utc_to_ts(when),
        "filing_date_utc": None, "report_date_utc": None,
        "primary_doc": "x.htm", "fetched_utc": 1,
    }


@pytest.fixture
def conn(tmp_path, cfg):
    c = db.get_conn(tmp_path / "r.db")
    db.upsert_filings(c, [
        # four after the close, two in the morning — a known 4:2 split
        filing("a-1", "2025-10-01T20:30:00Z", items="2.02,9.01"),
        filing("a-2", "2025-11-04T20:45:00Z", items="2.02,9.01"),
        filing("a-3", "2026-01-06T21:05:00Z", items="8.01"),
        filing("a-4", "2026-03-10T20:15:00Z", items="1.01,9.01"),
        filing("m-1", "2025-12-02T12:00:00Z", items="5.02",
               cik="0000789019", ticker="MSFT"),
        filing("m-2", "2026-02-02T12:30:00Z", items="5.07",
               cik="0000789019", ticker="MSFT"),
        # outside the window, kept on purpose
        filing("old-1", "2019-05-05T20:30:00Z"),
    ])
    yield c
    c.close()


# -- the Done-when ---------------------------------------------------------

def test_acceptance_hour_histogram_counts_correctly(conn, cfg):
    hours = acceptance_hour_histogram(conn, date_str_to_ts(WINDOW_START),
                                      date_str_to_ts(WINDOW_END))
    counts = {h["hour_utc"]: h["n"] for h in hours}
    assert counts == {12: 2, 20: 3, 21: 1}
    assert sum(h["pct"] for h in hours) == pytest.approx(100.0)


def test_the_out_of_window_filing_is_not_counted(conn, cfg):
    hours = acceptance_hour_histogram(conn, date_str_to_ts(WINDOW_START),
                                      date_str_to_ts(WINDOW_END))
    assert sum(h["n"] for h in hours) == 6, "the 2019 row must stay out"


# -- UI-context rule 7: a bare hour is a bug -------------------------------

def test_the_histogram_reports_local_time_alongside_utc(conn, cfg):
    hours = acceptance_hour_histogram(conn, date_str_to_ts(WINDOW_START),
                                      date_str_to_ts(WINDOW_END))
    for row in hours:
        assert "EDT" in row["new_york"] and "EST" in row["new_york"]


def test_new_york_label_shows_both_sides_of_daylight_saving():
    """The same UTC hour is 16:00 in July and 15:00 in January.

    Printing one would be wrong for half the study window.
    """
    assert new_york_label(20) == "16:00 EDT / 15:00 EST"
    assert new_york_label(21) == "17:00 EDT / 16:00 EST"


def test_twenty_utc_is_after_the_close():
    """The whole argument in one assertion: 20:00 UTC is past 16:00 ET."""
    label = new_york_label(20)
    assert label.startswith("16:00"), (
        "if this is not at or after the close, the t0 correction's premise "
        "is wrong")


# -- item codes ------------------------------------------------------------

def test_item_codes_are_split_not_pooled(conn, cfg):
    """`items` holds '2.02,9.01'; grouping on the raw string counts
    combinations and hides how often each event type occurs."""
    counts = {r["item"]: r["n"] for r in item_code_counts(
        cfg, conn, date_str_to_ts(WINDOW_START), date_str_to_ts(WINDOW_END))}
    assert counts["9.01"] == 3
    assert counts["2.02"] == 2
    assert counts["1.01"] == 1


def test_item_table_marks_the_scheduled_codes(conn, cfg):
    """UI-context rule 4: scheduled is never pooled with unscheduled, and an
    unmarked table invites exactly that."""
    rows = {r["item"]: r for r in item_code_counts(
        cfg, conn, date_str_to_ts(WINDOW_START), date_str_to_ts(WINDOW_END))}
    assert rows["2.02"]["scheduled"] is True
    assert rows["8.01"]["scheduled"] is False
    assert rows["9.01"]["excluded"] is True


# -- per company -----------------------------------------------------------

def test_filings_per_company_summary(conn):
    summary = filings_per_company(conn, date_str_to_ts(WINDOW_START),
                                  date_str_to_ts(WINDOW_END))
    assert summary["companies"] == 2
    assert (summary["min"], summary["max"]) == (2, 4)


# -- the split-CIK detector ------------------------------------------------

def test_a_cik_with_no_prior_history_is_flagged(tmp_path, cfg):
    """The ExxonMobil shape: a CIK that did not exist before the window."""
    c = db.get_conn(tmp_path / "x.db")
    db.upsert_filings(c, [
        filing("old", "2019-05-05T20:30:00Z", cik="0000034088", ticker=None),
        filing("new", "2026-07-07T20:30:00Z", cik="0002115436", ticker="XOM"),
    ])
    flagged = ciks_with_no_history_before_the_window(
        c, date_str_to_ts(WINDOW_START))
    assert [r["cik"] for r in flagged] == ["0002115436"]
    c.close()


def test_a_company_filing_since_before_the_window_is_not_flagged(conn, cfg):
    """Apple has a 2019 filing, so it has history; Microsoft here does not."""
    flagged = {r["cik"] for r in ciks_with_no_history_before_the_window(
        conn, date_str_to_ts(WINDOW_START))}
    assert "0000320193" not in flagged
    assert "0000789019" in flagged


def test_the_detector_is_sharper_than_quiet_for_ninety_days(conn, cfg):
    """A first draft flagged anyone quiet for 90 days — 502 companies, mostly
    ordinary firms filing a few times a year. Asking whether the CIK existed at
    all is the version that cannot miss a reorganisation."""
    quiet_but_established = filing("a-late", "2026-06-01T20:30:00Z")
    db.upsert_filings(conn, [quiet_but_established])
    flagged = {r["cik"] for r in ciks_with_no_history_before_the_window(
        conn, date_str_to_ts(WINDOW_START))}
    assert "0000320193" not in flagged


# -- the whole report ------------------------------------------------------

def test_window_and_all_time_counts_are_both_reported(conn, cfg):
    """`filings` keeps out-of-window rows on purpose, so showing one figure
    without the other would mislead in either direction."""
    report = filings_report(cfg, conn)
    assert report["total_rows"] == 7
    assert report["in_window"] == 6
    assert report["outside_window"] == 1


def test_rows_with_no_acceptance_time_are_counted_not_dropped(conn, cfg):
    row = filing("null-1", "2026-01-01T20:00:00Z")
    row["acceptance_utc"] = None
    db.upsert_filings(conn, [row])
    assert filings_report(cfg, conn)["no_acceptance_time"] == 1


def test_an_empty_filings_table_says_so(tmp_path, cfg, capsys):
    c = db.get_conn(tmp_path / "empty.db")
    print_filings_report(filings_report(cfg, c))
    assert "EMPTY" in capsys.readouterr().out
    c.close()


def test_the_report_never_implies_intent(conn, cfg, capsys):
    """UI-context rule 2: footprint, never insider trading — in every line a
    person reads, report figures included."""
    print_filings_report(filings_report(cfg, conn))
    out = capsys.readouterr().out.lower()
    for banned in ("insider", "suspicious", "illegal", "fraud"):
        assert banned not in out
