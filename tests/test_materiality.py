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
from src.pipeline.materiality import BarSeries, apply_filter, measure, write_filter
from src.utils.config import load_config
from src.utils.timeutils import iso_utc_to_ts


HOUR = 3600


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def conn(tmp_path):
    return db.get_conn(tmp_path / "mat.db")


def series_of(*pairs) -> BarSeries:
    return BarSeries(sorted(pairs))


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
