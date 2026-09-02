"""P4-03 — item codes become labels, and non-events are dropped with a reason.

Three properties have to hold together: nothing survives on an excluded code
alone, every survivor carries a scheduled flag, and the events with no usable
price history are excluded rather than left to arrive in the feature matrix as
NaNs. The last one had no task assigned to it until this one.

The `cfg` fixture pins a fabricated `items:` block rather than reading the
production one. These tests are about the BEHAVIOUR of the rules; when they
read live config, editing config silently changes what they assert — a test
for "a code in no list is kept" stops testing anything the day that code is
listed. The production config is checked separately, once, for the invariants
the pipeline depends on.
"""

import pytest

from src import db
from src.pipeline.events import (
    apply_filters, classify_items, items_config, parse_items, print_report,
    unknown_codes, write_filters,
)
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts


DAY = 86400

# Real 8-K codes, but frozen here so config edits cannot change these tests'
# meaning: 9.01 is an attachment marker, 5.07 routine paperwork, 2.02 earnings.
FIXTURE_ITEMS = {
    "exclude": ["9.01", "5.07"],
    "scheduled": ["2.02"],
    "unscheduled_focus": ["1.01", "5.02", "8.01"],
}


@pytest.fixture
def cfg():
    cfg = load_config()
    cfg["items"] = {k: list(v) for k, v in FIXTURE_ITEMS.items()}
    return cfg


@pytest.fixture
def conn(tmp_path):
    return db.get_conn(tmp_path / "ev.db")


def mid_window(cfg, offset_days=120):
    return date_str_to_ts(cfg["study_window"]["start"]) + offset_days * DAY


def seed_event(conn, cfg, event_id, items, ticker="AAA", acceptance=None,
               with_bars=True, form="8-K", in_universe=1):
    """An event plus, by default, the price coverage it needs to pass P3-05."""
    acceptance = acceptance or mid_window(cfg)
    db.upsert_companies(conn, [{"cik": f"CIK{ticker}", "ticker": ticker,
                                "in_universe": in_universe}])
    db.upsert_filings(conn, [{
        "accession_no": event_id, "cik": f"CIK{ticker}", "ticker": ticker,
        "form": form, "items": items, "acceptance_utc": acceptance,
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


def stored(conn, event_id):
    return dict(conn.execute(
        "SELECT usable, exclude_reason, is_scheduled FROM events "
        "WHERE event_id = ?", (event_id,)).fetchone())


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


def test_an_excluded_code_never_makes_an_event_scheduled(cfg):
    """The 5.07 case, decided: excluded wins, and the label says so.

    5.07 sat in `exclude` AND `scheduled`. Exclusion runs first, so 153 usable
    events carrying 5.07 were labelled unscheduled while `print_report`
    announced 5.07 as a scheduled code. The plan's item table settles it —
    5.07 is "routine paperwork, exclude" — so 5.07 is an exclusion only, and a
    5.02 riding with it stays the surprise it is.
    """
    label = classify_items(cfg, "5.02,5.07,9.01")
    assert label.kept == ["5.02"] and label.is_scheduled == 0
    assert label.exclude_reason is None


def test_a_code_in_both_lists_is_refused_rather_than_silently_resolved(cfg):
    """Two incompatible intents. Picking one silently is how 5.07 got lost."""
    cfg["items"]["scheduled"] = ["2.02", "5.07"]
    with pytest.raises(SystemExit, match="both list 5.07"):
        classify_items(cfg, "2.02")
    with pytest.raises(SystemExit, match="Pick one list"):
        items_config(cfg)


def test_the_production_config_satisfies_the_invariant():
    """The one thing these tests must read the real config for."""
    real = load_config()
    assert not set(real["items"]["exclude"]) & set(real["items"]["scheduled"])
    assert "5.07" in real["items"]["exclude"]
    assert "5.07" not in real["items"]["scheduled"]


def test_a_code_in_no_list_is_kept_and_unscheduled(cfg):
    """`unscheduled_focus` is a reporting-emphasis list, not a filter.

    Treating absence from it as exclusion would silently shrink the study.
    """
    for code in ("5.03", "3.02", "1.02"):
        label = classify_items(cfg, code)
        assert label.exclude_reason is None
        assert label.kept == [code] and label.is_scheduled == 0


def test_codes_in_no_list_are_counted_as_unknown(cfg):
    """Kept, but never silently: a shape change in EDGAR's items field would
    turn every code unknown and collapse the split to "all unscheduled"."""
    assert unknown_codes(cfg, ["8.01", "2.02", "9.01"]) == []
    assert unknown_codes(cfg, ["5.03", "8.01"]) == ["5.03"]
    # A garbled code is kept — a free label this config has not listed is far
    # likelier than a broken row — but it is counted.
    assert classify_items(cfg, "ITEM 2.02").kept == ["ITEM 2.02"]
    assert unknown_codes(cfg, ["ITEM 2.02"]) == ["ITEM 2.02"]


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


def test_an_event_outside_the_study_window_is_excluded(cfg, conn):
    """The coverage audit only sees in-window filings, so a stale row used to
    collect no reason at all and pass the gate.

    The window really was shortened once (2024-09-01 -> 2025-09-01) and the
    event builder upserts without deleting, so this is a live shape, not a
    hypothetical.
    """
    seed_event(conn, cfg, "e-ok", "8.01")
    seed_event(conn, cfg, "e-old", "8.01", ticker="BBB", with_bars=False,
               acceptance=mid_window(cfg) - 900 * DAY)
    assert reasons(apply_filters(cfg, conn))["e-old"] == "outside_study_window"


def test_an_event_outside_the_universe_is_excluded(cfg, conn):
    """Same hole, other half: the audit only covers in-universe tickers."""
    seed_event(conn, cfg, "e-ok", "8.01")
    seed_event(conn, cfg, "e-gone", "8.01", ticker="ZZZ", with_bars=False,
               in_universe=0)
    assert reasons(apply_filters(cfg, conn))["e-gone"] == "not_in_universe"


def test_an_amendment_is_labelled_like_any_other_filing(cfg, conn):
    """8-K/A carries item codes and is classified on them alone.

    Whether an amendment and the filing it amends are one positive is a study
    design question owned upstream; this pass must at least not mislabel it.
    """
    seed_event(conn, cfg, "e-amd", "5.02,9.01", form="8-K/A")
    assert reasons(apply_filters(cfg, conn))["e-amd"] is None


def test_rerunning_rewrites_a_stale_exclude_reason(cfg, conn):
    """Reasons are rewritten each pass, never merged, so none can go stale."""
    seed_event(conn, cfg, "e-1", "9.01")
    seed_event(conn, cfg, "e-2", "8.01", ticker="BBB")
    write_filters(cfg, conn)
    assert stored(conn, "e-1")["exclude_reason"] == "only_excluded_items"

    conn.execute("UPDATE events SET items = '8.01' WHERE event_id = 'e-1'")
    conn.commit()
    write_filters(cfg, conn)
    assert stored(conn, "e-1")["exclude_reason"] is None


def test_a_newly_excluded_event_stops_being_usable(cfg, conn):
    """`usable` is what db.usable_events reads, so a reason alone is not enough.

    Before this, a config change that dropped an event left `usable = 1` and
    the feature matrix went on consuming it.
    """
    seed_event(conn, cfg, "e-1", "8.01")
    seed_event(conn, cfg, "e-2", "8.01", ticker="BBB")
    conn.execute("UPDATE events SET usable = 1")
    conn.commit()

    conn.execute("UPDATE events SET items = '9.01' WHERE event_id = 'e-1'")
    conn.commit()
    write_filters(cfg, conn)

    assert stored(conn, "e-1") == {"usable": 0, "is_scheduled": 0,
                                   "exclude_reason": "only_excluded_items"}
    assert [r["event_id"] for r in db.usable_events(conn)] == ["e-2"]


def test_a_later_passs_exclusion_is_preserved(cfg, conn):
    """Materiality owns `immaterial`; this pass must not NULL it.

    Re-running the module's default command used to wipe 8,937 materiality
    verdicts while leaving `usable = 0`, so the census funnel printed
    16,842 - 1,168 -> 6,737 and nothing warned.
    """
    seed_event(conn, cfg, "e-1", "8.01")
    seed_event(conn, cfg, "e-2", "8.01", ticker="BBB")
    write_filters(cfg, conn)
    conn.execute("UPDATE events SET exclude_reason = 'immaterial', usable = 0 "
                 "WHERE event_id = 'e-1'")
    conn.execute("UPDATE events SET usable = 1 WHERE event_id = 'e-2'")
    conn.commit()

    write_filters(cfg, conn)
    assert stored(conn, "e-1")["exclude_reason"] == "immaterial"
    assert stored(conn, "e-2") == {"usable": 1, "is_scheduled": 0,
                                   "exclude_reason": None}


def test_the_funnel_adds_up_after_a_write(cfg, conn):
    """events == usable + excluded + not-yet-measured, and never overlapping."""
    seed_event(conn, cfg, "e-ok", "8.01")
    seed_event(conn, cfg, "e-drop", "9.01", ticker="BBB")
    seed_event(conn, cfg, "e-nobars", "8.01", ticker="GONE", with_bars=False)
    write_filters(cfg, conn)

    rows = conn.execute("SELECT usable, exclude_reason FROM events").fetchall()
    excluded = sum(1 for r in rows if r["exclude_reason"])
    usable = sum(1 for r in rows if r["usable"])
    pending = sum(1 for r in rows if not r["usable"] and not r["exclude_reason"])
    assert len(rows) == usable + excluded + pending
    assert not [r for r in rows if r["usable"] and r["exclude_reason"]]


def test_usable_is_left_for_the_materiality_pass(cfg, conn):
    """P4-04 is the last gate and owns the positive verdict."""
    seed_event(conn, cfg, "e-1", "8.01")
    write_filters(cfg, conn)
    assert stored(conn, "e-1")["usable"] == 0


def test_raises_if_every_event_is_excluded(cfg, conn):
    seed_event(conn, cfg, "e-1", "9.01")
    seed_event(conn, cfg, "e-2", "5.07", ticker="BBB")
    with pytest.raises(SystemExit, match="check items.exclude"):
        apply_filters(cfg, conn)


def test_the_every_event_excluded_error_names_the_real_cause(cfg, conn):
    """Both codes are legitimate and kept; the cause is missing price data.

    The message used to send the reader to items.exclude regardless, which is
    the wrong file when a market-data outage is what emptied the study.
    """
    seed_event(conn, cfg, "nb-1", "8.01", with_bars=False)
    seed_event(conn, cfg, "nb-2", "5.02", ticker="BBB", with_bars=False)
    with pytest.raises(SystemExit, match="price history"):
        apply_filters(cfg, conn)


def test_raises_when_there_are_no_events(cfg, conn):
    with pytest.raises(SystemExit, match="no events to filter"):
        apply_filters(cfg, conn)


def test_raises_when_there_are_no_events_at_all(cfg, conn):
    with pytest.raises(SystemExit, match="no events to filter"):
        print_report(cfg, conn)


def test_the_report_prints_the_split_and_the_unknown_tally(cfg, conn, capsys):
    """The printed report is the acceptance check, and was never exercised.

    Its "scheduled codes" line has to agree with what the filter actually did:
    it used to announce 5.07 as scheduled while dropping every 5.07 filing.
    """
    seed_event(conn, cfg, "e-sched", "2.02,9.01")
    seed_event(conn, cfg, "e-unsched", "8.01", ticker="BBB")
    seed_event(conn, cfg, "e-odd", "5.03", ticker="CCC")
    seed_event(conn, cfg, "e-drop", "9.01", ticker="DDD")
    print_report(cfg, conn)
    out = capsys.readouterr().out

    assert "scheduled codes: 2.02\n" in out       # not "2.02, 5.07"
    assert "events in      : 4" in out
    assert "excluded only_excluded_items" in out
    assert "(scheduled 0 / unscheduled 1)" in out          # rule 5, split
    assert "scheduled   :      1" in out and "unscheduled :      2" in out
    assert "5.03 (1)" in out                               # the unknown tally
    assert "--census" in out                               # materiality warning
