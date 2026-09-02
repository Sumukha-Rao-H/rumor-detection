"""P3-05 — does every event have the price history the model would have seen?

An event whose stock has no bars across its padded span cannot be scored: its
features come back NaN, and NaNs that reach a feature matrix get imputed,
dropped, or silently treated as zero somewhere downstream. These tests pin each
failure cause, and pin the two decisions that make the audit honest — the span
is widened by the t0 lookback so it cannot pass what t0 would fail, and
coverage is counted in exchange sessions rather than bars.
"""

import copy

import pandas as pd
import pytest

from src import db
from src.pipeline.coverage import (Verdict, audit, print_report,
                                   required_span, session_count)
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts, get_market_calendar


DAY = 86400
HOUR = 3600


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def conn(tmp_path):
    return db.get_conn(tmp_path / "cov.db")


def add_universe_company(conn, ticker):
    db.upsert_companies(conn, [{"cik": f"CIK{ticker}", "ticker": ticker,
                                "name": ticker, "in_universe": 1}])


def add_filing(conn, ticker, acceptance_utc, accession=None):
    db.upsert_filings(conn, [{
        "accession_no": accession or f"{ticker}-{acceptance_utc}",
        "cik": f"CIK{ticker}", "ticker": ticker, "form": "8-K", "items": "1.01",
        "acceptance_utc": acceptance_utc, "filing_date_utc": acceptance_utc,
    }])


def add_session_bars(conn, ticker, start_ts, end_ts, interval="60m",
                     skip=lambda d: False):
    """One bar at 14:30 UTC on every exchange session in the span.

    One bar per session is all the audit counts, so this mirrors reality
    without generating seven bars a day.
    """
    from src.utils.timeutils import get_market_calendar
    cal = get_market_calendar("XNYS")
    a = pd.Timestamp(start_ts, unit="s", tz="UTC").normalize().tz_localize(None)
    b = pd.Timestamp(end_ts, unit="s", tz="UTC").normalize().tz_localize(None)
    rows = []
    for session in cal.sessions_in_range(a, b):
        if skip(session):
            continue
        ts = int(pd.Timestamp(session, tz="UTC").timestamp()) + 14 * HOUR + 1800
        rows.append((ticker, ts, 1.0, 1.0, 1.0, 1.0, 1000.0, interval))
    db.upsert_bars(conn, rows)
    return len(rows)


def mid_window(cfg, offset_days=120):
    """An acceptance time comfortably inside the window, pad included."""
    return date_str_to_ts(cfg["study_window"]["start"]) + offset_days * DAY


def outcomes(cfg, conn):
    return {v.ticker: v.outcome for v in audit(cfg, conn)}


# --------------------------------------------------------------------------
# the two decisions that make the audit honest
# --------------------------------------------------------------------------

def test_anchor_widens_by_the_t0_lookback(cfg):
    """t0 lands before acceptance, so the audit must look further back.

    Otherwise an event could pass here and fail once the real t0 is computed.
    """
    acc = mid_window(cfg)
    start, end, _ = required_span(cfg, acc)
    expected = (acc - cfg["market"]["pad_days_before"] * DAY
                - cfg["news"]["t0_lookback_hours"] * HOUR)
    assert start == expected
    assert end == acc + cfg["market"]["pad_days_after"] * DAY


def test_pad_is_clamped_at_the_window_start(cfg):
    """An early event's pad reaches where we deliberately collected nothing."""
    window_start = date_str_to_ts(cfg["study_window"]["start"])
    start, _, clamped = required_span(cfg, window_start + 2 * DAY)
    assert clamped and start == window_start


def test_sessions_not_bar_counts(cfg, conn):
    """A fully covered event scores 1.0, not the 1.077 a bar ratio would give.

    Seven bars per 6.5-hour session is why this is counted in sessions.
    """
    acc = mid_window(cfg)
    start, end, _ = required_span(cfg, acc)
    add_universe_company(conn, "AAA")
    add_filing(conn, "AAA", acc)
    written = add_session_bars(conn, "AAA", start, end)
    v = audit(cfg, conn)[0]
    assert v.sessions_present == v.sessions_expected == written
    assert v.outcome == "ok"


def test_weekend_dated_bars_do_not_inflate_coverage(cfg, conn):
    """A bar planted on a non-session date must never substitute for a real
    trading-day gap.

    Before the calendar intersection, `audit()` counted "distinct calendar
    dates with a bar" directly, so padding a real gap with bars dated on
    Saturdays inside the same span could turn a genuine `gaps` verdict into
    a false `ok`. This reproduces exactly that padding and asserts it has no
    effect.
    """
    acc = mid_window(cfg)
    start, end, _ = required_span(cfg, acc)
    hole_lo = pd.Timestamp(start + 6 * DAY, unit="s", tz="UTC").tz_localize(None)
    hole_hi = pd.Timestamp(start + 16 * DAY, unit="s", tz="UTC").tz_localize(None)
    add_universe_company(conn, "WKND")
    add_filing(conn, "WKND", acc)
    add_session_bars(conn, "WKND", start, end,
                     skip=lambda d: hole_lo <= d <= hole_hi)

    # Without padding, the missing real sessions are a genuine gap.
    v = audit(cfg, conn)[0]
    assert v.outcome == "gaps"
    real_present = v.sessions_present

    # Pad the hole with bars dated on Saturdays in the same date range — a
    # real yfinance pull would never legitimately produce these for a 60m
    # interval, but nothing in `audit()` used to check for it.
    cal = get_market_calendar(cfg["market"]["calendar"])
    rows = []
    d = hole_lo
    while d <= hole_hi:
        saturday = d + pd.Timedelta(days=(5 - d.dayofweek) % 7)
        if saturday <= hole_hi:
            ts = int(saturday.tz_localize("UTC").timestamp()) + 14 * HOUR
            rows.append(("WKND", ts, 1.0, 1.0, 1.0, 1.0, 1000.0,
                        cfg["market"]["interval"]))
        d += pd.Timedelta(days=7)
    assert rows, "test setup: no Saturdays landed inside the hole"
    db.upsert_bars(conn, rows)

    v2 = audit(cfg, conn)[0]
    assert v2.sessions_present == real_present   # weekend bars must not count
    assert v2.outcome == "gaps"                  # still a real gap, not "ok"


def test_zero_expected_sessions_is_reported_explicitly(conn):
    """A required span with no exchange sessions must never fall through to
    `ok` on the strength of a stray bar.

    Not reachable with today's config (`pad_days_before=30` alone guarantees
    `expected > 0`), but the guard used to be a bare `if expected and ...`
    truthiness check, which would have skipped the coverage-ratio check
    entirely and reported `ok` here. This pins the explicit `no_sessions`
    outcome instead.
    """
    cfg = copy.deepcopy(load_config())
    cfg["market"]["pad_days_before"] = 0
    cfg["market"]["pad_days_after"] = 0
    cfg["news"]["t0_lookback_hours"] = 0
    # A Saturday: with every pad zeroed, the required span is this single
    # non-session date, so `session_count` must be 0.
    acc = date_str_to_ts("2025-09-06") + 12 * HOUR
    start, end, clamped = required_span(cfg, acc)
    assert not clamped
    assert session_count(cfg, start, end) == 0

    add_universe_company(conn, "WKEND0")
    add_filing(conn, "WKEND0", acc)
    db.upsert_bars(conn, [("WKEND0", acc, 1.0, 1.0, 1.0, 1.0, 1000.0,
                          cfg["market"]["interval"])])

    v = audit(cfg, conn)[0]
    assert v.sessions_expected == 0
    assert v.outcome == "no_sessions"


def test_ratio_exactly_at_min_session_coverage_is_ok(conn):
    """The ratio check is strict `<`: exactly at the threshold is still ok.

    Two full trading weeks (10 sessions, no holiday in range) with one
    mid-span session missing gives present/expected == 9/10 == 0.9, which is
    today's configured `min_session_coverage` exactly.
    """
    cfg = copy.deepcopy(load_config())
    assert cfg["market"]["min_session_coverage"] == 0.9
    cfg["market"]["pad_days_before"] = 0
    cfg["market"]["pad_days_after"] = 13
    cfg["news"]["t0_lookback_hours"] = 0

    acc = date_str_to_ts("2026-02-02")  # Monday, no holiday until Feb 16
    start, end, clamped = required_span(cfg, acc)
    assert not clamped

    cal = get_market_calendar(cfg["market"]["calendar"])
    a = pd.Timestamp(start, unit="s", tz="UTC").normalize().tz_localize(None)
    b = pd.Timestamp(end, unit="s", tz="UTC").normalize().tz_localize(None)
    sessions = list(cal.sessions_in_range(a, b))
    assert len(sessions) == 10
    drop = sessions[4]                  # a mid-span session, not an edge one

    add_universe_company(conn, "EDGE")
    add_filing(conn, "EDGE", acc)
    add_session_bars(conn, "EDGE", start, end, skip=lambda d: d == drop)

    v = audit(cfg, conn)[0]
    assert v.sessions_expected == 10
    assert v.sessions_present == 9
    assert v.outcome == "ok"


# --------------------------------------------------------------------------
# each outcome
# --------------------------------------------------------------------------

def test_a_boundary_session_is_not_lost_to_the_span_edge(cfg, conn):
    """Regression: comparing exact timestamps instead of dates faked gaps.

    The span boundaries land mid-day, so an edge session's bars could fall
    outside the span while `session_count` still counted the session — up to
    two days lost at each end. On a clamped five-session span that alone drops
    the ratio to 0.8, and the first real run reported 59 "gaps" of which 54
    were this artefact on perfectly healthy large-caps.
    """
    window_start = date_str_to_ts(cfg["study_window"]["start"])
    acc = window_start + 3 * DAY                      # forces a clamped span
    start, end, clamped = required_span(cfg, acc)
    assert clamped
    add_universe_company(conn, "AON")
    add_filing(conn, "AON", acc)
    add_session_bars(conn, "AON", start, end)
    v = audit(cfg, conn)[0]
    assert v.sessions_present == v.sessions_expected
    assert v.outcome == "ok"


def test_a_fully_covered_event_passes(cfg, conn):
    acc = mid_window(cfg)
    start, end, _ = required_span(cfg, acc)
    add_universe_company(conn, "AAA")
    add_filing(conn, "AAA", acc)
    add_session_bars(conn, "AAA", start, end)
    assert outcomes(cfg, conn) == {"AAA": "ok"}


def test_event_with_no_bars_is_reported(cfg, conn):
    add_universe_company(conn, "NONE")
    add_filing(conn, "NONE", mid_window(cfg))
    assert outcomes(cfg, conn) == {"NONE": "no_bars"}


def test_hourly_history_starting_inside_the_span_is_starts_late(cfg, conn):
    """Issue 25: 13 real tickers have daily history but late hourly history."""
    acc = mid_window(cfg)
    start, end, _ = required_span(cfg, acc)
    add_universe_company(conn, "CALY")
    add_filing(conn, "CALY", acc)
    add_session_bars(conn, "CALY", start + 20 * DAY, end)
    assert outcomes(cfg, conn) == {"CALY": "starts_late"}


def test_delisted_before_the_pad_ends_is_ends_early(cfg, conn):
    acc = mid_window(cfg)
    start, end, _ = required_span(cfg, acc)
    add_universe_company(conn, "GONE")
    add_filing(conn, "GONE", acc)
    add_session_bars(conn, "GONE", start, end - 20 * DAY)
    assert outcomes(cfg, conn) == {"GONE": "ends_early"}


def test_a_hole_in_the_middle_is_gaps(cfg, conn):
    """Both ends present, the middle missing — a halt a bar count would hide."""
    acc = mid_window(cfg)
    start, end, _ = required_span(cfg, acc)
    hole_lo = pd.Timestamp(start + 6 * DAY, unit="s", tz="UTC").tz_localize(None)
    hole_hi = pd.Timestamp(start + 26 * DAY, unit="s", tz="UTC").tz_localize(None)
    add_universe_company(conn, "HALT")
    add_filing(conn, "HALT", acc)
    add_session_bars(conn, "HALT", start, end,
                     skip=lambda d: hole_lo <= d <= hole_hi)
    assert outcomes(cfg, conn) == {"HALT": "gaps"}


def test_starts_late_is_reported_before_gaps(cfg, conn):
    """Cause, not symptom: late history is also technically a gap."""
    acc = mid_window(cfg)
    start, end, _ = required_span(cfg, acc)
    add_universe_company(conn, "LATE")
    add_filing(conn, "LATE", acc)
    add_session_bars(conn, "LATE", start + 25 * DAY, end)   # both late AND short
    assert outcomes(cfg, conn) == {"LATE": "starts_late"}


# --------------------------------------------------------------------------
# scope and guards
# --------------------------------------------------------------------------

def test_only_in_universe_tickers_are_audited(cfg, conn):
    acc = mid_window(cfg)
    start, end, _ = required_span(cfg, acc)
    add_universe_company(conn, "IN")
    add_filing(conn, "IN", acc)
    add_session_bars(conn, "IN", start, end)
    db.upsert_companies(conn, [{"cik": "CIKOUT", "ticker": "OUT",
                                "in_universe": 0}])
    add_filing(conn, "OUT", acc)
    assert set(outcomes(cfg, conn)) == {"IN"}


def test_filings_outside_the_window_are_not_audited(cfg, conn):
    add_universe_company(conn, "AAA")
    add_filing(conn, "AAA", date_str_to_ts(cfg["study_window"]["end"]) + 30 * DAY)
    with pytest.raises(SystemExit, match="nothing to"):
        audit(cfg, conn)


def test_raises_when_the_universe_is_empty(cfg, conn):
    db.upsert_companies(conn, [{"cik": "C", "ticker": "AAA", "in_universe": 0}])
    add_filing(conn, "AAA", mid_window(cfg))
    with pytest.raises(SystemExit, match="in_universe"):
        audit(cfg, conn)


def test_session_count_matches_the_calendar(cfg):
    """A quiet week in the middle of the window is five sessions."""
    start = date_str_to_ts("2025-10-06")          # Monday
    end = date_str_to_ts("2025-10-10")            # Friday
    assert session_count(cfg, start, end) == 5


# --------------------------------------------------------------------------
# print_report — the acceptance check itself
# --------------------------------------------------------------------------

def _verdict(ticker, outcome, accession=None, expected=10, present=10,
            clamped=False):
    return Verdict(accession or f"{ticker}-{outcome}-{present}", ticker,
                   0, outcome, expected, present, clamped)


def test_print_report_summarises_counts_and_percentage(cfg, capsys):
    verdicts = [
        _verdict("AAA", "ok"),
        _verdict("AAA", "ok", accession="AAA-2"),
        _verdict("BBB", "no_bars", present=0),
        _verdict("BBB", "starts_late", accession="BBB-2"),
        _verdict("CCC", "gaps", present=5),
    ]
    print_report(cfg, verdicts)
    out = capsys.readouterr().out
    assert "events audited : 5" in out
    assert "ok           : 2  (40.0%)" in out
    assert "no_bars      : 1" in out
    assert "starts_late  : 1" in out
    assert "gaps         : 1" in out
    # no_sessions never fired, so it must not clutter a report that has none.
    assert "no_sessions" not in out


def test_print_report_worst_offenders_label_matches_ticker_outcome_pairs(cfg, capsys):
    """Regression: the header used to read "Failing tickers — worst 25 of 27"
    on the real DB when the true count was 14 distinct tickers — `worst` is
    keyed by (ticker, outcome), so one ticker failing two different ways (BBB
    here) inflated the "tickers" count without the label saying so.
    """
    verdicts = [
        _verdict("AAA", "ok"),
        _verdict("BBB", "no_bars", present=0),
        _verdict("BBB", "starts_late", accession="BBB-2"),
        _verdict("CCC", "gaps", present=5),
    ]
    print_report(cfg, verdicts, list_failures=True)
    out = capsys.readouterr().out
    # 3 (ticker, outcome) pairs from 2 distinct failing tickers (BBB, CCC) —
    # not "3 failing tickers".
    assert "Worst offenders — 3 of 3 (ticker, outcome) pairs, 2 distinct tickers:" in out
    assert "All 3 failing events:" in out


def test_print_report_runs_with_no_failures(cfg, capsys):
    """The all-clear path: no "Worst offenders" section when nothing failed."""
    verdicts = [_verdict("AAA", "ok"), _verdict("BBB", "ok")]
    print_report(cfg, verdicts, list_failures=True)
    out = capsys.readouterr().out
    assert "ok           : 2  (100.0%)" in out
    assert "Worst offenders" not in out
    assert "All 0 failing events:" in out
