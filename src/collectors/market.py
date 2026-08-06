"""Market data collector — yfinance OHLCV into the `bars` table (plan §5.3).

Backtest granularity is hourly ('60m') bars: yfinance serves ~730 days of
those, matching the hourly decision step. Daily bars are supplementary
context. 1m/5m/15m/30m bars are NOT used (30–60 day history is useless
for backtests).

Fetches are incremental: each run resumes from the latest cached bar per
(ticker, interval), so historical bars are fetched once and re-runs are cheap.
All bar timestamps are stored as UTC epoch seconds of the bar's open.

Usage:
  python -m src.collectors.market --tickers TSLA,AAPL --start 2025-01-01 --end 2025-06-01
  python -m src.collectors.market --from-db            # all tickers seen in posts + SPY
  python -m src.collectors.market --from-db --interval 1d
  python -m src.collectors.market --tickers TSLA --start 2024-12-01 --backfill
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
import yfinance as yf

from src import db
from src.utils.config import load_config
from src.utils.ratelimit import RateLimiter
from src.utils.timeutils import date_str_to_ts, ts_to_dt, ts_to_iso, utc_now_ts

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
                   interval: str, resume: bool = True) -> int:
    """Fetch and upsert bars for one ticker, resuming from the cache.

    `resume` is the incremental path and only ever moves *forward*: it starts
    from the newest cached bar, so a window that begins earlier than what is
    already stored fetches nothing. Pass resume=False to widen coverage
    backwards — upserts are idempotent, so re-requesting the overlap is
    wasted bandwidth but never duplicate rows.
    """
    now = utc_now_ts()
    start_ts = clamp_start(start_ts, interval, now)
    cached = db.latest_bar_ts(conn, ticker, interval) if resume else None
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
    group.add_argument("--from-db", action="store_true",
                       help="every ticker in post_tickers, plus the benchmark")
    parser.add_argument("--start", help="YYYY-MM-DD (default: backtest_window.start)")
    parser.add_argument("--end", help="YYYY-MM-DD (default: now)")
    parser.add_argument("--interval", help="60m (default) or 1d")
    parser.add_argument("--backfill", action="store_true",
                        help="re-request the whole window instead of resuming "
                             "from the newest cached bar — needed to extend "
                             "coverage backwards")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config()
    mcfg = cfg["market"]
    interval = args.interval or mcfg["interval"]
    start_ts = date_str_to_ts(args.start or cfg["backtest_window"]["start"])
    end_ts = date_str_to_ts(args.end) if args.end else utc_now_ts()

    conn = db.get_conn(cfg["paths"]["db"])
    if args.from_db:
        tickers = db.distinct_post_tickers(conn)
        if mcfg["benchmark"] not in tickers:
            tickers.append(mcfg["benchmark"])  # SPY market control (plan §7.3)
    else:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    limiter = RateLimiter(mcfg["min_interval_s"])
    total = 0
    for ticker in tickers:
        limiter.wait()
        try:
            total += collect_ticker(conn, ticker, start_ts, end_ts, interval,
                                    resume=not args.backfill)
        except Exception:
            log.exception("failed to collect %s — continuing", ticker)
    n_bars = conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
    log.info("Done. %d tickers processed; bars table now has %d rows.",
             len(tickers), n_bars)


if __name__ == "__main__":
    main()
