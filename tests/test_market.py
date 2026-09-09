"""Market collector tests — DataFrame conversion and window logic, no network."""

import pandas as pd
import pytest

from src.collectors.market import clamp_start, df_to_rows
from src.utils.config import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def make_df(index, tz="UTC"):
    """yfinance always hands back a tz-aware index, so the default is one.

    Passing `tz=None` builds the naive frame `df_to_rows` now refuses; every
    other test here is about NaN handling or duplicate stamps and wants an
    ordinary frame.
    """
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


def test_a_tz_naive_frame_is_refused_rather_than_stored_shifted():
    """A naive index from yfinance is EXCHANGE-LOCAL time, not UTC.

    It used to be localized as UTC with a warning, which stored every bar four
    or five hours off. `upsert_bars` sets OHLCV unconditionally on conflict, so
    that guess would rewrite the frozen snapshot with wrong prices and nothing
    in the data to show it. Refuse instead, the way `timeutils.iso_utc_to_ts`
    refuses — a warning nobody reads is not a guard.
    """
    df = make_df(["2025-06-02"], tz=None)
    with pytest.raises(ValueError, match="tz-naive"):
        df_to_rows(df, "TSLA", "1d")


def test_df_to_rows_handles_a_real_tz_aware_daily_frame():
    """The actual production shape: yfinance daily bars localized to NYSE time."""
    df = make_df(["2025-06-02"], tz="America/New_York")
    rows = df_to_rows(df, "TSLA", "1d")
    assert rows[0][1] == int(pd.Timestamp("2025-06-02", tz="America/New_York").timestamp())


def test_df_to_rows_skips_nan_close_and_empty():
    df = make_df(["2025-06-02", "2025-06-03"])
    df.loc[df.index[0], "Close"] = float("nan")
    assert len(df_to_rows(df, "TSLA", "1d")) == 1
    assert df_to_rows(pd.DataFrame(), "TSLA", "1d") == []
    assert df_to_rows(None, "TSLA", "1d") == []


def test_df_to_rows_skips_nan_open_high_or_low_even_with_a_valid_close(caplog):
    """A valid Close with a NaN Open/High/Low used to pass straight through
    and land as NULL in `bars`. Any NaN in OHLC now drops the whole bar."""
    df = make_df(["2025-06-02", "2025-06-03", "2025-06-04"])
    df.loc[df.index[0], "Open"] = float("nan")
    df.loc[df.index[1], "High"] = float("nan")
    df.loc[df.index[2], "Low"] = float("nan")
    with caplog.at_level("WARNING"):
        rows = df_to_rows(df, "TSLA", "1d")
    assert rows == []
    assert "skipped 3 bar(s)" in caplog.text


def test_df_to_rows_dedupes_duplicate_timestamps_and_warns(caplog):
    """Duplicate index entries in one fetch must not silently pick a winner."""
    df = make_df(["2025-06-02", "2025-06-02"])
    df.loc[df.index[1], "Close"] = 9.0
    with caplog.at_level("WARNING"):
        rows = df_to_rows(df, "TSLA", "1d")
    assert len(rows) == 1
    assert rows[0][5] == 9.0  # last occurrence wins, matching upsert's ON CONFLICT
    assert "duplicate" in caplog.text


def test_clamp_start_only_for_hourly(cfg):
    now = 2_000_000_000
    lookback = cfg["market"]["hourly_max_lookback_days"] * 86400
    old = now - 3 * 365 * 86400
    assert clamp_start(cfg, old, "60m", now) == now - lookback
    assert clamp_start(cfg, old, "1d", now) == old
    recent = now - 86400
    assert clamp_start(cfg, recent, "60m", now) == recent


def test_clamp_start_follows_market_interval_rather_than_a_literal(cfg):
    """The clamp used to key on the literal "60m", so editing
    `market.interval` alone switched it off without a word — silently asking
    yfinance for intraday history it does not serve."""
    now = 2_000_000_000
    old = now - 3 * 365 * 86400
    moved = {**cfg, "market": {**cfg["market"], "interval": "30m"}}
    assert clamp_start(moved, old, "60m", now) == old, (
        "60m is no longer the configured intraday interval")
    lookback = cfg["market"]["hourly_max_lookback_days"] * 86400
    assert clamp_start(moved, old, "30m", now) == now - lookback, (
        "the clamp must follow whatever market.interval says")
