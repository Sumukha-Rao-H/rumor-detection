"""Ground-truth labeling — a rule-based stand-in for plan §6.4.

The full plan calls for LLM triage (Gemini) proposing a label, with a human
reviewing every single one before it's trusted. That step isn't wired in
yet. This module implements a deterministic version of the same rule so
there's a real, working label pipeline today — every label it writes has
`human_reviewed = 0` and a `label_source` ending in `_auto`, so nothing here
pretends to be reviewed ground truth. Swap in the LLM + review step later
without touching the schema.

Rule, matching plan §6.4:
  TRUE  — a whitelisted-domain article exists within [t0, t0 + horizon]
          -> label 1, t_official = that article's earliest seen_utc.
  FALSE — no whitelisted article in that window, AND (only when hourly bars
          are cached for the ticker) the return over the window is < 4% and
          the volume z-score vs. a prior baseline is < 2
          -> label 0, t_official = t0 + horizon.
  Ambiguous — no whitelisted article, but price bars show an abnormal move
          -> left NULL, flagged for human/LLM review.
  Also NULL when there's no news AND not enough cached bars to judge either
          way -> left NULL, label_source records why.

This fetches missing news (GDELT + Finnhub) and price bars per event ticker
automatically — you don't need to run the news/market collectors by hand
first for tickers that show up as events.

Usage:
  python -m src.pipeline.labeling
"""

from __future__ import annotations

import argparse
import logging
import os
import statistics

from src import db
from src.collectors import market, news
from src.utils.config import load_config
from src.utils.ratelimit import RateLimiter

log = logging.getLogger(__name__)

RETURN_THRESHOLD = 0.04
ZSCORE_THRESHOLD = 2.0
MIN_BASELINE_BARS = 5 * 24  # ~5 trading days of hourly bars needed for a baseline


def price_move_abnormal(
    conn, ticker: str, t0: int, horizon_s: int, interval: str,
) -> bool | None:
    """None if there isn't enough cached bar history to judge either way."""
    baseline = conn.execute(
        """SELECT close, volume FROM bars WHERE ticker = ? AND interval = ? AND ts_utc < ?
           ORDER BY ts_utc DESC LIMIT 720""",  # ~30 days of 60m bars
        (ticker, interval, t0),
    ).fetchall()
    window = conn.execute(
        """SELECT ts_utc, close, volume FROM bars WHERE ticker = ? AND interval = ?
           AND ts_utc BETWEEN ? AND ? ORDER BY ts_utc ASC""",
        (ticker, interval, t0, t0 + horizon_s),
    ).fetchall()
    if len(baseline) < MIN_BASELINE_BARS or len(window) < 1:
        return None

    # Compare against the last pre-event close, not the window's own first bar —
    # a jump that happens exactly at t0 (the informed-trading footprint we
    # actually care about) would otherwise cancel out against itself.
    pre_event_close = baseline[0]["close"]  # baseline is DESC: index 0 = latest before t0
    ret = (window[-1]["close"] - pre_event_close) / pre_event_close
    vols = [r["volume"] for r in baseline]
    mean_v, std_v = statistics.fmean(vols), statistics.pstdev(vols) or 1.0
    window_mean_v = statistics.fmean([r["volume"] for r in window])
    z = (window_mean_v - mean_v) / std_v
    return abs(ret) >= RETURN_THRESHOLD or z >= ZSCORE_THRESHOLD


def label_event(conn, cfg: dict, event: dict) -> tuple[int | None, int | None, str]:
    ticker, t0 = event["ticker"], event["t0_utc"]
    horizon_s = cfg["event"]["label_horizon_hours"] * 3600
    whitelist = set(cfg["news"]["whitelist"])

    articles = conn.execute(
        """SELECT seen_utc, source_domain FROM news WHERE ticker = ?
           AND seen_utc BETWEEN ? AND ? ORDER BY seen_utc ASC""",
        (ticker, t0, t0 + horizon_s),
    ).fetchall()
    confirming = [a for a in articles if a["source_domain"] in whitelist]
    if confirming:
        return 1, confirming[0]["seen_utc"], "news_match_auto"

    abnormal = price_move_abnormal(conn, ticker, t0, horizon_s, cfg["market"]["interval"])
    if abnormal is False:
        return 0, t0 + horizon_s, "no_news_no_price_move_auto"
    if abnormal is True:
        return None, None, "ambiguous_price_moved_no_news"
    return None, None, "ambiguous_insufficient_price_history"


def ensure_ground_truth_fetched(conn, cfg: dict, events: list[dict]) -> None:
    """Pulls news + bars for every event ticker's window, once per ticker.

    Rate limiters are created ONCE here and shared across every ticker in
    the loop — a fresh RateLimiter per call only enforces spacing within
    that one call, so looping many tickers with no shared limiter hammers
    GDELT/yfinance back-to-back with no real throttling between them."""
    horizon_s = cfg["event"]["label_horizon_hours"] * 3600
    by_ticker: dict[str, tuple[int, int]] = {}
    for e in events:
        lo, hi = e["t0_utc"] - 86400, e["t0_utc"] + horizon_s
        cur = by_ticker.get(e["ticker"])
        by_ticker[e["ticker"]] = (min(lo, cur[0]), max(hi, cur[1])) if cur else (lo, hi)

    ncfg, mcfg = cfg["news"], cfg["market"]
    gdelt_limiter = RateLimiter(ncfg["gdelt_min_interval_s"])
    finnhub_limiter = RateLimiter(ncfg["finnhub_min_interval_s"])
    market_limiter = RateLimiter(mcfg["min_interval_s"])

    has_finnhub_key = bool(os.getenv("FINNHUB_API_KEY", "").strip())
    apis = ["gdelt", "finnhub"] if has_finnhub_key else ["gdelt"]
    if not has_finnhub_key:
        log.warning("FINNHUB_API_KEY not set in .env — using GDELT only for ground-truth news")

    for ticker, (start_ts, end_ts) in by_ticker.items():
        try:
            news.collect(cfg, conn, ticker, query=None, start_ts=start_ts,
                        end_ts=end_ts, apis=apis,
                        gdelt_limiter=gdelt_limiter, finnhub_limiter=finnhub_limiter)
        except Exception:
            log.exception("news fetch failed for %s — continuing", ticker)
        try:
            market_limiter.wait()
            market.collect_ticker(conn, ticker, start_ts, end_ts, mcfg["interval"])
        except Exception:
            log.exception("bars fetch failed for %s — continuing", ticker)


def label_all(conn, cfg: dict) -> dict[str, int]:
    events = [dict(r) for r in db.unlabeled_events(conn)]
    ensure_ground_truth_fetched(conn, cfg, events)

    counts = {"true": 0, "false": 0, "ambiguous": 0}
    for event in events:
        label, t_official, source = label_event(conn, cfg, event)
        db.set_event_label(conn, event["event_id"], label, t_official, source)
        key = "true" if label == 1 else "false" if label == 0 else "ambiguous"
        counts[key] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    counts = label_all(conn, cfg)
    log.info("Done. %s", counts)


if __name__ == "__main__":
    main()
