"""P4-04 — did the market actually react?

Without this filter, routine dividend declarations and credit-facility
amendments become positives and the model learns paperwork. The tests that
matter most are the benchmark subtraction (a stock rising with the market did
nothing) and the Friday rule (8.1% of events have no trading in the next 24
wall-clock hours, and dropping them would delete the "bad news on Friday
afternoon" category wholesale).

This module is on the LABEL path, so it looks at prices after t0 on purpose.
That is correct here and would be leakage anywhere near a feature.
"""

import pytest

from src import db
from src.pipeline.materiality import (
    BarSeries, apply_filter, interval_seconds, measure, print_report, write_filter,
)
from src.utils.config import load_config
from src.utils.timeutils import iso_utc_to_ts


HOUR = 3600


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def conn(tmp_path):
    return db.get_conn(tmp_path / "mat.db")


def series_of(*bars, interval_s: int = HOUR) -> BarSeries:
    """Build a BarSeries from `(ts, price)` (flat bar: open == close == price)
    or `(ts, open, close)` triples — the latter for bars whose open and close
    differ, needed to exercise the straddling case."""
    rows = [(b[0], b[1], b[1]) if len(b) == 2 else b for b in bars]
    return BarSeries(sorted(rows), interval_s)


T0 = iso_utc_to_ts("2026-02-25T20:00:00Z")


def two_point(cfg, start_price, end_price, t0=T0, bench_start=100.0,
              bench_end=100.0, horizon_offset=None):
    """A stock with one bar before t0 and one after the horizon, plus SPY."""
    h = horizon_offset or cfg["materiality"]["window_hours"] * HOUR
    ts_before, ts_after = t0 - HOUR, t0 + h + HOUR
    return {
        "AAA": series_of((ts_before, start_price), (ts_after, end_price)),
        cfg["market"]["benchmark"]: series_of((ts_before, bench_start),
                                              (ts_after, bench_end)),
    }


# --------------------------------------------------------------------------
# the move itself
# --------------------------------------------------------------------------

def test_a_big_move_is_material(cfg):
    m = measure(cfg, two_point(cfg, 100.0, 110.0), "AAA", T0)
    assert m.reason is None and m.benchmark_applied
    assert m.adjusted == pytest.approx(0.10)


def test_a_flat_move_is_not_material(cfg):
    """Routine filings — a dividend declaration moves nothing."""
    m = measure(cfg, two_point(cfg, 100.0, 100.2), "AAA", T0)
    assert abs(m.adjusted) < cfg["materiality"]["min_abs_return"]


def test_a_move_matched_by_the_market_is_not_material(cfg):
    """THE benchmark test: up 5% on a day the market rose 5% is not news."""
    m = measure(cfg, two_point(cfg, 100.0, 105.0, bench_start=100.0,
                               bench_end=105.0), "AAA", T0)
    assert m.raw == pytest.approx(0.05)
    assert m.adjusted == pytest.approx(0.0, abs=1e-12)


def test_a_move_against_the_market_is_material(cfg):
    """Flat while the market fell 4% is a 4% relative move."""
    m = measure(cfg, two_point(cfg, 100.0, 100.0, bench_start=100.0,
                               bench_end=96.0), "AAA", T0)
    assert m.adjusted == pytest.approx(0.04)


def test_a_large_fall_is_material(cfg, conn):
    """Materiality is about magnitude, not direction."""
    m = measure(cfg, two_point(cfg, 100.0, 88.0), "AAA", T0)
    assert m.adjusted == pytest.approx(-0.12)
    assert abs(m.adjusted) >= cfg["materiality"]["min_abs_return"]


def test_benchmark_relative_false_uses_the_raw_return(cfg):
    off = {**cfg, "materiality": {**cfg["materiality"],
                                  "benchmark_relative": False}}
    m = measure(off, two_point(cfg, 100.0, 105.0, bench_start=100.0,
                               bench_end=105.0), "AAA", T0)
    assert m.adjusted == pytest.approx(0.05) and not m.benchmark_applied


def test_missing_benchmark_bar_falls_back_and_records_it(cfg):
    """A raw return labelled benchmark-relative would misstate the measurement."""
    s = two_point(cfg, 100.0, 110.0)
    del s[cfg["market"]["benchmark"]]
    m = measure(cfg, s, "AAA", T0)
    assert m.adjusted == pytest.approx(0.10) and m.benchmark_applied is False


def test_negative_benchmark_price_falls_back_to_raw(cfg):
    """A negative benchmark price is not a valid pre-news price either — the
    old `not b_start` falsiness check let it through (only 0/None are falsy)."""
    m = measure(cfg, two_point(cfg, 100.0, 110.0, bench_start=-5.0), "AAA", T0)
    assert m.benchmark_applied is False and m.adjusted == pytest.approx(0.10)


def test_non_positive_end_price_is_reported_not_silently_immaterial(cfg):
    """The `start` guard has always rejected <= 0; `end` must too (LOW finding)."""
    assert measure(cfg, two_point(cfg, 100.0, 0.0), "AAA", T0).reason == "no_post_t0_bar"
    assert measure(cfg, two_point(cfg, 100.0, -5.0), "AAA", T0).reason == "no_post_t0_bar"


def test_nan_pre_t0_close_is_reported_not_silently_immaterial(cfg):
    """`NaN <= 0` is False in Python — a `> 0` check is needed to catch it."""
    s = two_point(cfg, float("nan"), 110.0)
    assert measure(cfg, s, "AAA", T0).reason == "no_pre_t0_bar"


def test_nan_post_horizon_close_is_reported_not_silently_immaterial(cfg):
    s = two_point(cfg, 100.0, float("nan"))
    assert measure(cfg, s, "AAA", T0).reason == "no_post_t0_bar"


# --------------------------------------------------------------------------
# the pre-news baseline — a bar straddling t0 must not leak post-t0 price
# --------------------------------------------------------------------------

def test_interval_seconds_parses_the_configured_bar_duration():
    assert interval_seconds("60m") == HOUR
    assert interval_seconds("1d") == 24 * HOUR
    assert interval_seconds("2h") == 2 * HOUR


def test_a_bar_that_straddles_t0_uses_its_own_open_not_its_close(cfg):
    """The confirmed-most-severe bug: t0 lands strictly inside a bar's window
    (open < t0 < open + interval). That bar's CLOSE reflects trading AFTER
    t0 and must not be used as the pre-news price — but reaching back to an
    earlier bar would be reaching back further than necessary. The bar's own
    OPEN is the last price known at or before t0 and is what must be used.

    Real-DB repro this guards against: 42.8% of events have t0 strictly
    inside a bar's window; a true -4.4% move was computing as 0.0% and being
    dropped as immaterial before this fix (open=100.0, close=95.6, t0 mid-bar,
    flat afterwards).
    """
    h = cfg["materiality"]["window_hours"] * HOUR
    straddles_t0 = T0 - HOUR // 2   # window [T0-1800, T0+1800): contains t0
    after = T0 + h + HOUR
    s = {
        "AAA": series_of((straddles_t0, 100.0, 95.6), (after, 95.6)),
        cfg["market"]["benchmark"]: series_of((straddles_t0, 100.0), (after, 100.0)),
    }
    m = measure(cfg, s, "AAA", T0)
    assert m.reason is None
    # True pre-news price is this bar's own OPEN (100.0), not its CLOSE
    # (95.6), which already contains the reaction.
    assert m.raw == pytest.approx(95.6 / 100.0 - 1.0)
    assert m.raw == pytest.approx(-0.044)
    assert abs(m.adjusted) >= cfg["materiality"]["min_abs_return"]  # was 0.0%, dropped


def test_t0_exactly_at_a_bars_open_uses_that_bars_own_open(cfg):
    """A bar opening exactly at t0 has window [t0, t0+interval) — entirely
    at/after t0 — so its own CLOSE is not a valid pre-news price. But its
    OPEN (the price at the exact instant t0 occurs) is the correct, most
    recent pre-news price — reaching back to an earlier bar's close would be
    needlessly stale.
    """
    h = cfg["materiality"]["window_hours"] * HOUR
    prior = T0 - HOUR              # an earlier, unrelated bar
    opens_at_t0 = T0                # window [T0, T0+HOUR): open == t0 exactly
    after = T0 + h + HOUR
    s = {
        "AAA": series_of((prior, 50.0), (opens_at_t0, 100.0, 95.6), (after, 95.6)),
        cfg["market"]["benchmark"]: series_of((prior, 100.0), (opens_at_t0, 100.0),
                                              (after, 100.0)),
    }
    m = measure(cfg, s, "AAA", T0)
    # start must be 100.0 (this bar's own open) — not 50.0 (the prior bar's
    # close) and not 95.6 (this bar's own close).
    assert m.raw == pytest.approx(95.6 / 100.0 - 1.0)


# --------------------------------------------------------------------------
# the horizon — the Friday rule
# --------------------------------------------------------------------------

def test_friday_event_measures_to_the_next_open_session(cfg):
    """8.1% of events, 1,138 of them Fridays.

    t0 is Friday evening; the 24-hour mark lands on Saturday when nothing
    trades. A hard cutoff would see no bar, score a zero move, and drop the
    event — deleting the "bad news on Friday afternoon" category wholesale.
    """
    friday = iso_utc_to_ts("2026-02-27T21:00:00Z")
    monday = iso_utc_to_ts("2026-03-02T20:00:00Z")
    s = {"AAA": series_of((friday - HOUR, 100.0), (monday, 112.0)),
         cfg["market"]["benchmark"]: series_of((friday - HOUR, 100.0),
                                               (monday, 100.0))}
    m = measure(cfg, s, "AAA", friday)
    assert m.reason is None
    assert m.adjusted == pytest.approx(0.12)


def test_the_horizon_is_a_minimum_not_a_window(cfg):
    """The first close AT OR AFTER the horizon, however far away it is."""
    s = two_point(cfg, 100.0, 110.0, horizon_offset=10 * 24 * HOUR)
    assert measure(cfg, s, "AAA", T0).adjusted == pytest.approx(0.10)


def test_no_bar_before_t0_is_reported(cfg):
    s = {"AAA": series_of((T0 + 48 * HOUR, 100.0))}
    assert measure(cfg, s, "AAA", T0).reason == "no_pre_t0_bar"


def test_no_bar_after_the_horizon_is_reported(cfg):
    s = {"AAA": series_of((T0 - HOUR, 100.0))}
    assert measure(cfg, s, "AAA", T0).reason == "no_post_t0_bar"


def test_exactly_the_threshold_is_material(cfg, conn):
    """Inclusive. A boundary decided silently is one someone later disputes."""
    thr = cfg["materiality"]["min_abs_return"]
    m = measure(cfg, two_point(cfg, 100.0, 100.0 * (1 + thr)), "AAA", T0)
    assert m.adjusted == pytest.approx(thr)
    seed_event(conn, cfg, "e-edge", "AAA", 100.0, 100.0 * (1 + thr))
    assert apply_filter(cfg, conn)[0]["is_material"] == 1


# --------------------------------------------------------------------------
# the final gate
# --------------------------------------------------------------------------

def seed_event(conn, cfg, event_id, ticker, start_price, end_price,
               t0=T0, exclude_reason=None, is_scheduled=0):
    h = cfg["materiality"]["window_hours"] * HOUR
    iv = cfg["market"]["interval"]
    bench = cfg["market"]["benchmark"]
    db.upsert_bars(conn, [
        (ticker, t0 - HOUR, 0, 0, 0, start_price, 1.0, iv),
        (ticker, t0 + h + HOUR, 0, 0, 0, end_price, 1.0, iv),
        (bench, t0 - HOUR, 0, 0, 0, 100.0, 1.0, iv),
        (bench, t0 + h + HOUR, 0, 0, 0, 100.0, 1.0, iv),
    ])
    db.upsert_filings(conn, [{
        "accession_no": event_id, "cik": f"CIK{ticker}", "ticker": ticker,
        "form": "8-K", "items": "8.01", "acceptance_utc": t0,
        "filing_date_utc": t0,
    }])
    db.upsert_events(conn, [{
        "event_id": event_id, "accession_no": event_id, "ticker": ticker,
        "items": "8.01", "t0_filing_utc": t0, "t0_utc": t0,
        "t0_source": "filing", "is_scheduled": is_scheduled,
        "exclude_reason": exclude_reason, "usable": 0,
    }])


def test_usable_requires_both_material_and_unexcluded(cfg, conn):
    seed_event(conn, cfg, "e-big", "AAA", 100.0, 115.0)
    seed_event(conn, cfg, "e-small", "BBB", 100.0, 100.1)
    write_filter(cfg, conn)
    got = dict(conn.execute("SELECT event_id, usable FROM events"))
    assert got == {"e-big": 1, "e-small": 0}
    assert conn.execute(
        "SELECT exclude_reason FROM events WHERE event_id='e-small'"
    ).fetchone()[0] == "immaterial"


def test_already_excluded_events_are_not_measured(cfg, conn):
    """No price move turns an attachment-only filing into an event."""
    seed_event(conn, cfg, "e-big", "AAA", 100.0, 115.0)
    seed_event(conn, cfg, "e-dropped", "BBB", 100.0, 150.0,
               exclude_reason="only_excluded_items")
    write_filter(cfg, conn)
    r = conn.execute("SELECT usable, abs_return, exclude_reason FROM events "
                     "WHERE event_id='e-dropped'").fetchone()
    assert r["usable"] == 0 and r["abs_return"] is None
    assert r["exclude_reason"] == "only_excluded_items"


def test_rerunning_recomputes_rather_than_accumulates(cfg, conn):
    seed_event(conn, cfg, "e-big", "AAA", 100.0, 115.0)
    assert write_filter(cfg, conn) == 1
    assert write_filter(cfg, conn) == 1
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1


def test_raises_when_nothing_is_material(cfg, conn):
    seed_event(conn, cfg, "e-1", "AAA", 100.0, 100.1)
    with pytest.raises(SystemExit, match="NO event"):
        apply_filter(cfg, conn)


def test_raises_when_there_are_no_events(cfg, conn):
    with pytest.raises(SystemExit, match="no events"):
        apply_filter(cfg, conn)


# --------------------------------------------------------------------------
# --report
# --------------------------------------------------------------------------

def test_report_raises_the_same_clear_error_as_apply_filter_on_empty_events(cfg, conn):
    """Previously a bare ZeroDivisionError; must match apply_filter's message."""
    with pytest.raises(SystemExit, match="no events"):
        print_report(cfg, conn)


def test_report_separates_no_price_data_from_exclusions_made_earlier(cfg, conn, capsys):
    """`no_bars`/`no_pre_t0_bar`/`no_post_t0_bar` come from this module's own
    bar lookup and must be reported separately from `only_excluded_items` and
    friends, which are decided by universe.py/events.py/t0.py before this
    module ever runs."""
    seed_event(conn, cfg, "e-usable", "AAA", 100.0, 115.0)
    seed_event(conn, cfg, "e-upstream", "BBB", 100.0, 150.0,
               exclude_reason="only_excluded_items")
    seed_event(conn, cfg, "e-no-bars", "CCC", 100.0, 150.0)
    write_filter(cfg, conn)
    # CCC has no bars of its own: give it a distinct ticker with none inserted.
    conn.execute("DELETE FROM bars WHERE ticker = 'CCC'")
    write_filter(cfg, conn)

    print_report(cfg, conn)
    out = capsys.readouterr().out
    assert "excluded earlier: 1" in out       # only e-upstream
    assert "no price data   : 1" in out       # only e-no-bars
    assert "no_bars" in out
    assert conn.execute(
        "SELECT exclude_reason FROM events WHERE event_id='e-no-bars'"
    ).fetchone()[0] == "no_bars"


def test_a_reason_this_module_owns_is_re_measured_not_frozen(cfg, conn):
    """`immaterial` is a verdict about config, not a permanent fact.

    A first run marks a 0.1% move `immaterial`. `min_abs_return` is then
    retuned downward. Re-running must re-measure and let that event become
    usable again — the old code skipped every row that already carried a
    reason, so it re-persisted the first run's verdict forever and silently
    froze the positive count against a config that had since changed.
    """
    seed_event(conn, cfg, "e-borderline", "AAA", 100.0, 102.0)  # 2%: under 3%, over 1%
    seed_event(conn, cfg, "e-mover", "BBB", 100.0, 130.0)  # keeps the
    write_filter(cfg, conn)                                # loud-guard quiet
    stored = conn.execute(
        "SELECT usable, exclude_reason FROM events WHERE event_id = 'e-borderline'"
    ).fetchone()
    assert (stored["usable"], stored["exclude_reason"]) == (0, "immaterial")

    relaxed = {**cfg, "materiality": {**cfg["materiality"], "min_abs_return": 0.01}}
    rows = {r["event_id"]: r for r in apply_filter(relaxed, conn)}
    assert rows["e-borderline"]["usable"] == 1, (
        "a stored `immaterial` verdict survived a config change that should "
        "have overturned it")
    assert rows["e-borderline"]["exclude_reason"] is None


def test_a_reason_an_earlier_pass_owns_is_preserved(cfg, conn):
    """The other half of the rule: this module does not overturn upstream.

    `not_in_universe` belongs to events.py/universe.py. Re-measuring price for
    such an event and clearing its reason would silently readmit something an
    earlier gate deliberately dropped.
    """
    seed_event(conn, cfg, "e-upstream", "AAA", 100.0, 130.0,
               exclude_reason="not_in_universe")
    seed_event(conn, cfg, "e-mover", "BBB", 100.0, 130.0)  # keeps the loud guard quiet
    rows = {r["event_id"]: r for r in apply_filter(cfg, conn)}
    assert rows["e-upstream"]["exclude_reason"] == "not_in_universe"
    assert rows["e-upstream"]["usable"] == 0, (
        "an upstream exclusion was overturned by a price measurement")
