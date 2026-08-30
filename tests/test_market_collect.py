"""P3-01 — the daily-bar pull over the whole candidate list.

The Done-when is `test_interrupted_run_then_resume_covers_every_ticker`: a
6,000-ticker run that takes hours must survive being killed. Everything else
here protects a detail that would make that true in a test and false in
practice — state namespaced per interval so the hourly run does not inherit the
daily run's, zero bars recorded as `empty` rather than a permanent `ok`, and
the run-level guard actually able to fire on a database that already has rows.
"""

import pandas as pd
import pytest

from src import db
from src.collectors import market
from src.collectors.market import (
    collect_many, coverage_report, default_start_ts, fetch_source,
)
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts


DAY = 86400
TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA"]


@pytest.fixture(scope="module")
def cfg():
    """The real config, with the inter-request pause removed.

    A 1 s pause per ticker is right against Yahoo and pointless against a fake:
    it turned this file into a 40-second test. `RateLimiter` itself is covered
    in `test_utils.py`, and where the limiter sits in the fetch path — after the
    cache check, before the request — has its own test below.
    """
    cfg = load_config()
    cfg["market"] = {**cfg["market"], "min_interval_s": 0.0}
    return cfg


def fresh_db(tmp_path, name):
    return db.get_conn(tmp_path / name)


def frame(start: str, days: int) -> pd.DataFrame:
    """A daily OHLCV frame with a naive index, the shape yfinance returns."""
    idx = pd.DatetimeIndex(pd.date_range(start, periods=days, freq="D"))
    return pd.DataFrame(
        {"Open": [10.0] * days, "High": [11.0] * days, "Low": [9.0] * days,
         "Close": [10.5] * days, "Volume": [1_000_000] * days},
        index=idx,
    )


class FakeYF:
    """Stands in for `yf.Ticker`; can be told to blow up on the Nth call."""

    def __init__(self, by_ticker, fail_after=None, error=None):
        self.by_ticker = by_ticker
        self.fail_after = fail_after
        self.error = error or KeyboardInterrupt("^C")
        self.calls: list[str] = []

    def __call__(self, ticker):
        self.calls.append(ticker)
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise self.error
        df = self.by_ticker.get(ticker, pd.DataFrame())
        return type("T", (), {"history": lambda _self, **kw: df})()


def patch_yf(monkeypatch, fake):
    monkeypatch.setattr(market.yf, "Ticker", fake)
    return fake


def run(cfg, conn, tickers=TICKERS, interval="1d", resume=False,
        start="2024-09-01", end="2026-08-01"):
    return collect_many(cfg, conn, tickers, date_str_to_ts(start),
                        date_str_to_ts(end), interval, resume=resume)


# --------------------------------------------------------------------------
# the candidate list
# --------------------------------------------------------------------------

def test_candidate_tickers_dedupes_predecessor_rows(tmp_path):
    """P2-11's predecessor rows carry the successor's ticker."""
    conn = fresh_db(tmp_path, "cand.db")
    db.upsert_companies(conn, [
        {"cik": "0000034088", "ticker": "XOM", "name": "Exxon Mobil"},
        {"cik": "0000093410", "ticker": "XOM", "name": "Chevron (predecessor)",
         "successor_cik": "0000034088"},
        {"cik": "0000320193", "ticker": "AAPL", "name": "Apple"},
    ])
    assert db.candidate_tickers(conn) == ["AAPL", "XOM"]


def test_candidate_tickers_ignores_in_universe(tmp_path):
    """The list must work before the liquidity filter has ever run."""
    conn = fresh_db(tmp_path, "cand2.db")
    db.upsert_companies(conn, [
        {"cik": "1", "ticker": "AAPL", "in_universe": 0},
        {"cik": "2", "ticker": "MSFT", "in_universe": 0},
    ])
    assert db.universe_tickers(conn) == []
    assert db.candidate_tickers(conn) == ["AAPL", "MSFT"]


# --------------------------------------------------------------------------
# the derived start date
# --------------------------------------------------------------------------

def test_daily_start_defaults_to_min_history_before_window(cfg):
    """The liquidity filter needs history BEFORE the window to rank on."""
    window = date_str_to_ts(cfg["study_window"]["start"])
    expected = window - cfg["universe"]["min_history_days"] * DAY
    assert default_start_ts(cfg, cfg["market"]["daily_interval"]) == expected
    assert default_start_ts(cfg, "1d") < window


def test_hourly_start_is_the_window_start(cfg):
    """The hourly pull is unchanged — its ceiling is the rolling window."""
    assert (default_start_ts(cfg, cfg["market"]["interval"])
            == date_str_to_ts(cfg["study_window"]["start"]))


# --------------------------------------------------------------------------
# resume
# --------------------------------------------------------------------------

def test_interrupted_run_then_resume_covers_every_ticker(cfg, tmp_path,
                                                         monkeypatch):
    """THE Done-when: kill the run, resume, land on a clean run's row count."""
    data = {t: frame("2024-09-02", 30) for t in TICKERS}

    clean = fresh_db(tmp_path, "clean.db")
    patch_yf(monkeypatch, FakeYF(data))
    run(cfg, clean)
    expected = clean.execute("SELECT COUNT(*) FROM bars").fetchone()[0]

    killed = fresh_db(tmp_path, "killed.db")
    patch_yf(monkeypatch, FakeYF(data, fail_after=2))
    with pytest.raises(KeyboardInterrupt):
        run(cfg, killed)
    partial = killed.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
    assert 0 < partial < expected

    resumed = patch_yf(monkeypatch, FakeYF(data))
    run(cfg, killed, resume=True)
    assert killed.execute("SELECT COUNT(*) FROM bars").fetchone()[0] == expected
    # and it did not re-fetch what it already had
    assert set(resumed.calls) == set(TICKERS[2:])


def test_resume_skips_ok_and_retries_empty(cfg, tmp_path, monkeypatch):
    """A delisted symbol is worth another go; a collected one is not.

    This is where prices differ from EDGAR: 'no 8-Ks' is a permanent, honest
    `ok`, while 'no bars at all' cannot be told apart from a Yahoo hiccup.
    """
    data = {t: frame("2024-09-02", 10) for t in TICKERS[:3]}  # TSLA returns nothing
    patch_yf(monkeypatch, FakeYF(data))
    conn = fresh_db(tmp_path, "empty.db")
    run(cfg, conn)

    states = dict(conn.execute(
        "SELECT key, status FROM fetch_state WHERE source = ?",
        (fetch_source("1d"),)).fetchall())
    assert states["AAPL"] == "ok" and states["TSLA"] == "empty"

    second = patch_yf(monkeypatch, FakeYF(data))
    run(cfg, conn, resume=True)
    assert second.calls == ["TSLA"]


def test_resume_retries_a_failed_ticker(cfg, tmp_path, monkeypatch):
    """A transient 429 from Yahoo must not be mistaken for a finished ticker."""
    data = {t: frame("2024-09-02", 10) for t in TICKERS}
    conn = fresh_db(tmp_path, "retry.db")
    patch_yf(monkeypatch, FakeYF(data, fail_after=1,
                                 error=RuntimeError("429 Too Many Requests")))
    run(cfg, conn)
    assert conn.execute(
        "SELECT status FROM fetch_state WHERE key = 'MSFT'").fetchone()[0] == "failed"

    second = patch_yf(monkeypatch, FakeYF(data))
    run(cfg, conn, resume=True)
    assert "MSFT" in second.calls


def test_state_is_namespaced_by_interval(cfg, tmp_path, monkeypatch):
    """Otherwise the hourly run skips all 6,000 tickers on its first --resume."""
    data = {t: frame("2024-09-02", 10) for t in TICKERS}
    conn = fresh_db(tmp_path, "ns.db")
    patch_yf(monkeypatch, FakeYF(data))
    run(cfg, conn, interval="1d")

    assert db.completed_keys(conn, fetch_source("1d")) == set(TICKERS)
    assert db.completed_keys(conn, fetch_source("60m")) == set()

    hourly = patch_yf(monkeypatch, FakeYF(data))
    run(cfg, conn, interval="60m", resume=True, start="2025-09-01")
    assert set(hourly.calls) == set(TICKERS)


def test_bars_are_written_before_state(cfg, tmp_path, monkeypatch):
    """A crash between the two re-fetches one ticker; the reverse loses rows."""
    seen = {}
    real_set = db.set_fetch_state

    def spy(conn, source, key, status, **kw):
        seen[key] = conn.execute(
            "SELECT COUNT(*) FROM bars WHERE ticker = ?", (key,)).fetchone()[0]
        return real_set(conn, source, key, status, **kw)

    monkeypatch.setattr(market.db, "set_fetch_state", spy)
    patch_yf(monkeypatch, FakeYF({t: frame("2024-09-02", 10) for t in TICKERS}))
    run(cfg, fresh_db(tmp_path, "order.db"))
    assert all(n == 10 for n in seen.values()), seen


def test_keyboard_interrupt_stops_the_run(cfg, tmp_path, monkeypatch):
    """`except Exception` does not catch it — correct today, easy to break."""
    conn = fresh_db(tmp_path, "kb.db")
    fake = patch_yf(monkeypatch, FakeYF({}, fail_after=0))
    with pytest.raises(KeyboardInterrupt):
        run(cfg, conn)
    assert fake.calls == ["AAPL"]


def test_cache_covered_ticker_is_not_refetched(cfg, tmp_path, monkeypatch):
    """The incremental path: a second run with no new days issues no request.

    The seeded frame's last bar is 2024-09-11, so a window ending there is
    already covered and `collect_ticker` returns without touching the network.
    """
    conn = fresh_db(tmp_path, "cache.db")
    patch_yf(monkeypatch, FakeYF({t: frame("2024-09-02", 10) for t in TICKERS}))
    run(cfg, conn, end="2024-09-11")
    second = patch_yf(monkeypatch, FakeYF({}))
    run(cfg, conn, end="2024-09-11")
    assert second.calls == []


def test_limiter_waits_once_per_request_and_never_for_a_cached_ticker(
        cfg, tmp_path, monkeypatch):
    """A resume that skips thousands of tickers must not sleep for each of them."""
    waits = []
    limiter = type("L", (), {"wait": lambda _self: waits.append(1)})()
    conn = fresh_db(tmp_path, "lim.db")
    patch_yf(monkeypatch, FakeYF({"AAPL": frame("2024-09-02", 10)}))

    start, end = date_str_to_ts("2024-09-01"), date_str_to_ts("2024-09-11")
    market.collect_ticker(conn, "AAPL", start, end, "1d", limiter=limiter)
    assert len(waits) == 1
    market.collect_ticker(conn, "AAPL", start, end, "1d", limiter=limiter)
    assert len(waits) == 1  # second call was served by the cache


# --------------------------------------------------------------------------
# the zero-record guard
# --------------------------------------------------------------------------

def test_guard_fires_when_every_attempted_ticker_parsed_zero(cfg, tmp_path,
                                                             monkeypatch):
    patch_yf(monkeypatch, FakeYF({}))
    with pytest.raises(SystemExit, match="ZERO bars"):
        run(cfg, fresh_db(tmp_path, "guard.db"))


def test_guard_silent_on_a_populated_db_with_one_empty_ticker(cfg, tmp_path,
                                                              monkeypatch):
    """The bug being fixed: the old guard only compared against COUNT(*)."""
    data = {t: frame("2024-09-02", 10) for t in TICKERS[:3]}
    patch_yf(monkeypatch, FakeYF(data))
    run(cfg, fresh_db(tmp_path, "ok.db"))  # no raise


def test_guard_silent_on_a_resume_mop_up_of_known_empty_tickers(cfg, tmp_path,
                                                                monkeypatch):
    """The mop-up run attempts only symbols already known to return nothing.

    Without excluding those from the guard's denominator, every `--resume` after
    a completed pull would exit non-zero on a perfectly healthy run — and a
    guard that cries wolf is a guard people switch off.
    """
    data = {t: frame("2024-09-02", 10) for t in TICKERS[:3]}  # TSLA is dead
    conn = fresh_db(tmp_path, "mopup.db")
    patch_yf(monkeypatch, FakeYF(data))
    run(cfg, conn)

    second = patch_yf(monkeypatch, FakeYF(data))
    assert run(cfg, conn, resume=True) == 0  # no raise
    assert second.calls == ["TSLA"]


def test_guard_still_fires_when_a_known_empty_run_includes_a_fresh_ticker(
        cfg, tmp_path, monkeypatch):
    """Exempting known-empty tickers must not disarm the guard entirely."""
    conn = fresh_db(tmp_path, "mixed.db")
    patch_yf(monkeypatch, FakeYF({t: frame("2024-09-02", 10) for t in TICKERS[:3]}))
    run(cfg, conn)                                   # TSLA recorded empty
    patch_yf(monkeypatch, FakeYF({}))                # now yfinance breaks
    with pytest.raises(SystemExit, match="ZERO bars"):
        run(cfg, conn, tickers=["TSLA", "AMZN"])     # AMZN is fresh ground


def test_guard_silent_when_everything_was_skipped(cfg, tmp_path, monkeypatch):
    """Resuming a finished run attempted nothing, so it failed at nothing."""
    conn = fresh_db(tmp_path, "skip.db")
    patch_yf(monkeypatch, FakeYF({t: frame("2024-09-02", 10) for t in TICKERS}))
    run(cfg, conn)
    patch_yf(monkeypatch, FakeYF({}))
    assert run(cfg, conn, resume=True) == 0  # no raise


# --------------------------------------------------------------------------
# the coverage report — how the Done-when gets checked
# --------------------------------------------------------------------------

def seed_bars(conn, ticker, first: str, last: str, interval="1d"):
    db.upsert_bars(conn, [
        (ticker, date_str_to_ts(first), 1.0, 1.0, 1.0, 1.0, 1.0, interval),
        (ticker, date_str_to_ts(last), 1.0, 1.0, 1.0, 1.0, 1.0, interval),
    ])


def test_report_separates_missing_late_start_and_covered(cfg, tmp_path):
    conn = fresh_db(tmp_path, "rep.db")
    required_start = default_start_ts(cfg, "1d")
    from src.utils.timeutils import ts_to_iso
    start_iso = ts_to_iso(required_start)[:10]
    seed_bars(conn, "AAPL", start_iso, cfg["study_window"]["end"])
    seed_bars(conn, "NEWCO", "2026-01-05", cfg["study_window"]["end"])
    seed_bars(conn, "GONE", start_iso, "2025-11-01")
    db.set_fetch_state(conn, fetch_source("1d"), "DEAD", "empty",
                       error="yfinance returned no bars for this window")

    rep = coverage_report(cfg, conn, ["AAPL", "NEWCO", "GONE", "DEAD"], "1d")
    assert rep["covered"] == ["AAPL"]
    assert [t for t, _ in rep["late_start"]] == ["NEWCO"]
    assert [t for t, _ in rep["early_end"]] == ["GONE"]
    assert rep["missing"] == [("DEAD", "empty",
                               "yfinance returned no bars for this window")]


def test_report_tolerates_a_boundary_in_a_holiday_week(cfg, tmp_path):
    """The required start can land on a weekend and legitimately have no bar."""
    conn = fresh_db(tmp_path, "tol.db")
    from src.utils.timeutils import ts_to_iso
    tol_days = cfg["market"]["coverage_tolerance_days"]
    late = ts_to_iso(default_start_ts(cfg, "1d") + (tol_days - 1) * DAY)[:10]
    seed_bars(conn, "AAPL", late, cfg["study_window"]["end"])
    rep = coverage_report(cfg, conn, ["AAPL"], "1d")
    assert rep["covered"] == ["AAPL"] and not rep["late_start"]


def test_report_flags_a_ticker_never_fetched(cfg, tmp_path):
    conn = fresh_db(tmp_path, "never.db")
    rep = coverage_report(cfg, conn, ["AAPL"], "1d")
    assert rep["missing"] == [("AAPL", "never fetched", None)]
