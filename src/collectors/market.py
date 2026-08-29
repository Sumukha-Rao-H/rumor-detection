"""Market data collector — yfinance OHLCV into the `bars` table.

Backtest granularity is hourly ('60m') bars: yfinance serves ~730 days of
those, matching the hourly decision step. Daily bars are supplementary
context. 1m/5m/15m/30m bars are NOT used (30–60 day history is useless
for backtests).

The hourly window is ROLLING — bars available today silently disappear later —
so coverage must be downloaded broadly and early, then frozen. Broad coverage
(not just event windows) is required because negative sampling and trailing
z-scores both need continuous history. Budget days of wall-clock time for a
full 1,500-ticker pull, and record the download date with --stamp-snapshot.

Fetches are incremental: each run resumes from the latest cached bar per
(ticker, interval), so historical bars are fetched once and re-runs are cheap.
All bar timestamps are stored as UTC epoch seconds of the bar's open.

Usage:
  python -m src.collectors.market --tickers TSLA,AAPL --start 2025-09-01 --end 2026-08-01
  python -m src.collectors.market --universe              # every liquid ticker + benchmark
  python -m src.collectors.market --universe --interval 1d
  python -m src.collectors.market --universe --stamp-snapshot
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
import yfinance as yf

from src import db
from src.utils.config import load_config
from src.utils.ratelimit import RateLimiter
from src.utils.timeutils import (
    date_str_to_ts, ts_to_dt, ts_to_iso, utc_now_ts,
)

log = logging.getLogger(__name__)

HOURLY_MAX_LOOKBACK_S = 729 * 86400  # yfinance serves ~730 days of 60m bars


def df_to_rows(df: pd.DataFrame, ticker: str, interval: str) -> list[tuple]:
    """History DataFrame -> bars rows. Index is converted to UTC epoch secs."""
    if df is None or df.empty:
        return []
    idx = df.index
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")  # daily bars can come back naive
    else:
        idx = idx.tz_convert("UTC")
    rows = []
    for ts, row in zip(idx, df.itertuples(index=False)):
        if pd.isna(row.Close):
            continue
        rows.append((
            ticker,
            int(ts.timestamp()),
            float(row.Open),
            float(row.High),
            float(row.Low),
            float(row.Close),
            float(row.Volume) if not pd.isna(row.Volume) else 0.0,
            interval,
        ))
    return rows


def clamp_start(start_ts: int, interval: str, now_ts: int) -> int:
    """Enforce yfinance's history window for intraday bars."""
    if interval == "60m":
        floor = now_ts - HOURLY_MAX_LOOKBACK_S
        if start_ts < floor:
            log.warning("60m bars only go back ~730 days — clamping start "
                        "%s -> %s", ts_to_iso(start_ts), ts_to_iso(floor))
            return floor
    return start_ts


def collect_ticker(conn, ticker: str, start_ts: int, end_ts: int,
                   interval: str) -> int:
    """Fetch and upsert bars for one ticker, resuming from the cache."""
    now = utc_now_ts()
    start_ts = clamp_start(start_ts, interval, now)
    cached = db.latest_bar_ts(conn, ticker, interval)
    if cached is not None and cached >= start_ts:
        start_ts = cached + 1  # incremental: refetch nothing we already have
    if start_ts >= end_ts:
        log.info("%s [%s]: cache already covers window", ticker, interval)
        return 0
    hist = yf.Ticker(ticker).history(
        start=ts_to_dt(start_ts), end=ts_to_dt(end_ts),
        interval=interval, auto_adjust=True,
    )
    rows = df_to_rows(hist, ticker, interval)
    n = db.upsert_bars(conn, rows)
    log.info("%s [%s]: %d bars upserted (%s -> %s)", ticker, interval,
             len(rows), ts_to_iso(start_ts), ts_to_iso(end_ts))
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tickers", help="comma-separated, e.g. TSLA,AAPL")
    group.add_argument("--universe", action="store_true",
                       help="every ticker in the liquid universe, plus the benchmark")
    parser.add_argument("--start", help="YYYY-MM-DD (default: study_window.start)")
    parser.add_argument("--end", help="YYYY-MM-DD (default: now)")
    parser.add_argument("--interval", help="60m (default) or 1d")
    parser.add_argument("--stamp-snapshot", action="store_true",
                        help="record this run's date as the frozen snapshot date")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config()
    mcfg = cfg["market"]
    interval = args.interval or mcfg["interval"]
    start_ts = date_str_to_ts(args.start or cfg["study_window"]["start"])
    end_ts = date_str_to_ts(args.end) if args.end else utc_now_ts()

    conn = db.get_conn(cfg["paths"]["db"])
    if args.universe:
        tickers = db.universe_tickers(conn)
        if not tickers:
            raise SystemExit(
                "companies table is empty — run `python -m src.collectors.edgar "
                "--build-universe` first."
            )
        if mcfg["benchmark"] not in tickers:
            tickers.append(mcfg["benchmark"])  # SPY market control
    else:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    limiter = RateLimiter(mcfg["min_interval_s"])
    total = 0
    failed = 0
    for ticker in tickers:
        limiter.wait()
        try:
            total += collect_ticker(conn, ticker, start_ts, end_ts, interval)
        except Exception:
            failed += 1
            log.exception("failed to collect %s — continuing", ticker)

    n_bars = conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
    log.info("Done. %d tickers processed (%d failed); bars table now has %d rows.",
             len(tickers), failed, n_bars)

    # Silent-failure guard: an empty pull must never pass quietly.
    if cfg["logging"]["fail_on_zero_records"] and total == 0 and n_bars == 0:
        raise SystemExit(
            f"ZERO bars written for {len(tickers)} tickers — yfinance is "
            f"returning nothing. Do not treat this run as successful."
        )

    if args.stamp_snapshot:
        stamp = ts_to_iso(utc_now_ts())
        db.set_meta(conn, f"snapshot_frozen_{interval}", stamp, utc_now_ts())
        log.info("Snapshot date for %s bars recorded as %s. Do not re-download.",
                 interval, stamp)


if __name__ == "__main__":
    main()
