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
