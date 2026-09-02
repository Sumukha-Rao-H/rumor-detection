"""Market collector tests — DataFrame conversion and window logic, no network."""

import pandas as pd

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


def test_df_to_rows_localizes_naive_index_as_utc(caplog):
    """Defensive fallback only: verified against the installed yfinance 1.5.2
    that both '1d' and '60m' — the only intervals this collector uses — always
    come back tz-aware, so this path is not expected to fire in production.
    It must still warn loudly if it ever does, since naive-as-UTC is a guess
    (the naive value would really be exchange-local time)."""
    df = make_df(["2025-06-02"])
    with caplog.at_level("WARNING"):
        rows = df_to_rows(df, "TSLA", "1d")
    assert rows[0][1] == int(pd.Timestamp("2025-06-02", tz="UTC").timestamp())
    assert "tz-naive" in caplog.text


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


def test_clamp_start_only_for_hourly():
    now = 2_000_000_000
    old = now - 3 * 365 * 86400
    assert clamp_start(old, "60m", now) == now - HOURLY_MAX_LOOKBACK_S
    assert clamp_start(old, "1d", now) == old
    recent = now - 86400
    assert clamp_start(recent, "60m", now) == recent
