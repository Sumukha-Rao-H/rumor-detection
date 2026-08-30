"""P3-05 — does every event have the price history the model would have seen?

An event whose stock has no bars across its padded span cannot be scored: its
features come back NaN, and NaNs that reach a feature matrix get imputed,
dropped, or silently treated as zero somewhere downstream. These tests pin each
failure cause, and pin the two decisions that make the audit honest — the span
is widened by the t0 lookback so it cannot pass what t0 would fail, and
coverage is counted in exchange sessions rather than bars.
"""

import pandas as pd
import pytest

from src import db
from src.pipeline.coverage import audit, required_span, session_count
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts


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
