"""Market collector tests — DataFrame conversion and window logic, no network."""

import pandas as pd

from src.collectors import market
from src.collectors.market import HOURLY_MAX_LOOKBACK_S, clamp_start, df_to_rows


def make_df(index, tz=None):
    idx = pd.DatetimeIndex(index, tz=tz)
    return pd.DataFrame(
        {"Open": [1.0] * len(idx), "High": [2.0] * len(idx),
         "Low": [0.5] * len(idx), "Close": [1.5] * len(idx),
         "Volume": [1000] * len(idx)},
        index=idx,
    )


def test_df_to_rows_converts_tz_aware_to_utc_epoch():
    df = make_df(["2025-06-02 09:30:00"], tz="America/New_York")
    rows = df_to_rows(df, "TSLA", "60m")
    assert len(rows) == 1
    ticker, ts, o, h, l, c, v, interval = rows[0]
    assert ticker == "TSLA" and interval == "60m"
    assert ts == int(pd.Timestamp("2025-06-02 13:30:00", tz="UTC").timestamp())


def test_df_to_rows_localizes_naive_index_as_utc():
    df = make_df(["2025-06-02"])
    rows = df_to_rows(df, "TSLA", "1d")
    assert rows[0][1] == int(pd.Timestamp("2025-06-02", tz="UTC").timestamp())


def test_df_to_rows_skips_nan_close_and_empty():
    df = make_df(["2025-06-02", "2025-06-03"])
    df.loc[df.index[0], "Close"] = float("nan")
    assert len(df_to_rows(df, "TSLA", "1d")) == 1
    assert df_to_rows(pd.DataFrame(), "TSLA", "1d") == []
    assert df_to_rows(None, "TSLA", "1d") == []


def test_clamp_start_only_for_hourly():
    now = 2_000_000_000
    old = now - 3 * 365 * 86400
    assert clamp_start(old, "60m", now) == now - HOURLY_MAX_LOOKBACK_S
    assert clamp_start(old, "1d", now) == old
    recent = now - 86400
    assert clamp_start(recent, "60m", now) == recent


class FakeTicker:
    """Records the window yfinance would have been asked for."""

    asked: list[tuple] = []

    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, start, end, interval, auto_adjust):
        FakeTicker.asked.append((self.symbol, start, end))
        return pd.DataFrame()


def _run(monkeypatch, resume, cached_ts, start_ts, end_ts):
    FakeTicker.asked = []
    monkeypatch.setattr(market.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(market.db, "latest_bar_ts", lambda *a, **k: cached_ts)
    monkeypatch.setattr(market.db, "upsert_bars", lambda conn, rows: len(rows))
    monkeypatch.setattr(market, "utc_now_ts", lambda: end_ts)
    market.collect_ticker(None, "TSLA", start_ts, end_ts, "1d", resume=resume)
    return FakeTicker.asked


def test_resume_ignores_a_window_that_starts_before_the_cache(monkeypatch):
    """The incremental path only moves forward — earlier bars are never fetched."""
    now = 2_000_000_000
    cached = now - 30 * 86400
    asked = _run(monkeypatch, True, cached, now - 365 * 86400, now)
    assert asked and asked[0][1] == market.ts_to_dt(cached + 1)


def test_backfill_requests_the_full_window(monkeypatch):
    now = 2_000_000_000
    cached = now - 30 * 86400
    start = now - 365 * 86400
    asked = _run(monkeypatch, False, cached, start, now)
    assert asked and asked[0][1] == market.ts_to_dt(start)
