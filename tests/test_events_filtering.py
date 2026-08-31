"""P4-03 — item codes become labels, and non-events are dropped with a reason.

Three properties have to hold together: nothing survives on an excluded code
alone, every survivor carries a scheduled flag, and the events with no usable
price history are excluded rather than left to arrive in the feature matrix as
NaNs. The last one had no task assigned to it until this one.
"""

import pytest

from src import db
from src.pipeline.events import (
    apply_filters, classify_items, parse_items, write_filters,
)
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts, iso_utc_to_ts


DAY = 86400


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def conn(tmp_path):
    return db.get_conn(tmp_path / "ev.db")


def mid_window(cfg, offset_days=120):
    return date_str_to_ts(cfg["study_window"]["start"]) + offset_days * DAY


def seed_event(conn, cfg, event_id, items, ticker="AAA", acceptance=None,
               with_bars=True):
    """An event plus, by default, the price coverage it needs to pass P3-05."""
    acceptance = acceptance or mid_window(cfg)
    db.upsert_companies(conn, [{"cik": f"CIK{ticker}", "ticker": ticker,
                                "in_universe": 1}])
    db.upsert_filings(conn, [{
        "accession_no": event_id, "cik": f"CIK{ticker}", "ticker": ticker,
        "form": "8-K", "items": items, "acceptance_utc": acceptance,
        "filing_date_utc": acceptance,
    }])
    db.upsert_events(conn, [{
        "event_id": event_id, "accession_no": event_id, "ticker": ticker,
        "items": items, "t0_filing_utc": acceptance, "t0_utc": acceptance,
        "t0_source": "filing", "usable": 0,
    }])
    if with_bars:
        _add_full_coverage(conn, cfg, ticker, acceptance)
    return acceptance


def _add_full_coverage(conn, cfg, ticker, acceptance):
    """One bar on every exchange session across the event's required span."""
    import pandas as pd
    from src.pipeline.coverage import required_span
    from src.utils.timeutils import get_market_calendar
    start, end, _ = required_span(cfg, acceptance)
    cal = get_market_calendar(cfg["market"]["calendar"])
    a = pd.Timestamp(start, unit="s", tz="UTC").normalize().tz_localize(None)
    b = pd.Timestamp(end, unit="s", tz="UTC").normalize().tz_localize(None)
    db.upsert_bars(conn, [
        (ticker, int(pd.Timestamp(s, tz="UTC").timestamp()) + 14 * 3600 + 1800,
         1.0, 1.0, 1.0, 1.0, 1000.0, cfg["market"]["interval"])
        for s in cal.sessions_in_range(a, b)
    ])


def reasons(rows):
    return {r["event_id"]: r["exclude_reason"] for r in rows}


# --------------------------------------------------------------------------
# the item rules
# --------------------------------------------------------------------------

def test_an_event_of_only_excluded_items_is_dropped(cfg):
    """Done-when, part 1. 9.01 is an attachment marker, not an event."""
    assert classify_items(cfg, "9.01").exclude_reason == "only_excluded_items"
    assert classify_items(cfg, "9.01,5.07").exclude_reason == "only_excluded_items"


def test_excluded_codes_are_stripped_but_the_event_survives(cfg):
    """The common case: 2.02 with a 9.01 attachment riding along."""
    label = classify_items(cfg, "2.02,9.01")
    assert label.exclude_reason is None
    assert label.kept == ["2.02"] and label.is_scheduled == 1


def test_any_scheduled_code_makes_the_event_scheduled(cfg):
    """Earnings plus a surprise is still a filing whose timing was known.

    4,945 of 16,842 events carry more than one code, so this decides a lot of
    them. Calling it unscheduled would overstate the surprise.
    """
    assert classify_items(cfg, "2.02,8.01").is_scheduled == 1
    assert classify_items(cfg, "8.01,5.02").is_scheduled == 0


def test_a_code_in_no_list_is_kept_and_unscheduled(cfg):
    """5.03 and 3.02 are in none of the three lists.

    `unscheduled_focus` is a reporting-emphasis list, not a filter; treating
    absence from it as exclusion would silently shrink the study.
    """
    for code in ("5.03", "3.02", "1.02"):
        label = classify_items(cfg, code)
        assert label.exclude_reason is None
        assert label.kept == [code] and label.is_scheduled == 0


def test_item_codes_are_compared_as_strings(cfg):
    """`1.10` and `1.1` are different codes; a float would merge them."""
    assert parse_items("1.10") == ["1.10"]
    assert classify_items(cfg, "1.10").kept == ["1.10"]
    assert classify_items(cfg, "1.1").kept == ["1.1"]


def test_missing_item_codes_are_excluded(cfg):
    for empty in (None, "", "  ", ",,"):
        assert classify_items(cfg, empty).exclude_reason == "no_item_codes"


# --------------------------------------------------------------------------
# the whole pass
# --------------------------------------------------------------------------

def test_every_surviving_event_carries_a_scheduled_flag(cfg, conn):
    """Done-when, part 2."""
    seed_event(conn, cfg, "e-sched", "2.02")
    seed_event(conn, cfg, "e-unsched", "8.01", ticker="BBB")
    seed_event(conn, cfg, "e-dropped", "9.01", ticker="CCC")

    write_filters(cfg, conn)
    kept = conn.execute(
        "SELECT event_id, is_scheduled FROM events "
        "WHERE exclude_reason IS NULL").fetchall()
    assert {r["event_id"]: r["is_scheduled"] for r in kept} == {
        "e-sched": 1, "e-unsched": 0}


def test_coverage_failures_are_excluded_with_a_reason(cfg, conn):
    """Done-when, part 3 — issue 25, which had no owning task until now."""
    seed_event(conn, cfg, "e-ok", "8.01")
    seed_event(conn, cfg, "e-nobars", "8.01", ticker="GONE", with_bars=False)

    got = reasons(apply_filters(cfg, conn))
    assert got["e-ok"] is None
    assert got["e-nobars"].startswith("no_price_coverage:")


def test_items_rule_wins_over_coverage_rule(cfg, conn):
    """"This is not an event" is prior to "we cannot measure it"."""
    seed_event(conn, cfg, "e-both", "9.01", ticker="GONE", with_bars=False)
    seed_event(conn, cfg, "e-ok", "8.01")
    assert reasons(apply_filters(cfg, conn))["e-both"] == "only_excluded_items"


def test_rerunning_rewrites_a_stale_exclude_reason(cfg, conn):
    """Reasons are rewritten each pass, never merged, so none can go stale."""
    seed_event(conn, cfg, "e-1", "9.01")
    seed_event(conn, cfg, "e-2", "8.01", ticker="BBB")
    write_filters(cfg, conn)
    assert conn.execute("SELECT exclude_reason FROM events WHERE event_id='e-1'"
                        ).fetchone()[0] == "only_excluded_items"

    conn.execute("UPDATE events SET items = '8.01' WHERE event_id = 'e-1'")
    conn.commit()
    write_filters(cfg, conn)
    assert conn.execute("SELECT exclude_reason FROM events WHERE event_id='e-1'"
                        ).fetchone()[0] is None


def test_usable_is_left_for_the_materiality_pass(cfg, conn):
    """P4-04 is the last gate and owns the final verdict."""
    seed_event(conn, cfg, "e-1", "8.01")
    write_filters(cfg, conn)
    assert conn.execute(
        "SELECT usable FROM events WHERE event_id='e-1'").fetchone()[0] == 0


def test_raises_if_every_event_is_excluded(cfg, conn):
    seed_event(conn, cfg, "e-1", "9.01")
    seed_event(conn, cfg, "e-2", "5.07", ticker="BBB")
    with pytest.raises(SystemExit, match="EVERY one of"):
        apply_filters(cfg, conn)


def test_raises_when_there_are_no_events(cfg, conn):
    with pytest.raises(SystemExit, match="no events to filter"):
        apply_filters(cfg, conn)
