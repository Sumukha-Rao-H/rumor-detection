"""Per-event news retrieval — plan §6.4, stage 1 of ground-truth labeling.

§6.4 needs, for every rumor event, the headlines published in
[t₀ − 24h, t₀ + 72h] so the next stage can ask whether any of them confirms or
denies the claim. This module fetches exactly that and nothing else; deciding
what the headlines *mean* is stage 2's job.

Two measurements (2026-08-05) shape the design:

  coverage  GDELT's DOC API serves the whole 19-month window, so it is the
            only source that can label the older half of the dataset. Finnhub's
            free tier stops at ~12 months — queried before that it returns 0
            rows, cheerfully and with HTTP 200. Spans older than
            `news.finnhub_lookback_days` are therefore skipped rather than
            wasted, and the run is honest about which events GDELT alone covers.
  pacing    GDELT throttles harder than its documented "one call every 5
            seconds" — at 10s spacing it still 429s under load. The collector
            paces at `news.gdelt_min_interval_s` and backs off on top of that,
            so the practical cost is hours, not minutes. Everything is
            therefore resumable: `news_spans` records each completed query and
            a re-run skips it.

Windows are merged per ticker before querying. Events cluster (one ticker often
has several rumors in a week), and merging overlapping windows turns 1,170
event windows into ~1,010 queries — at GDELT's pacing that is over an hour
saved for free.

The query is the company name, not the claim: a merged span can cover several
events with different claim types, and over-narrow queries would starve the
labeler of the denial articles that produce FALSE labels. Filtering to what is
relevant is stage 2's job, where the claim text is available.

Usage:
  python -m src.pipeline.event_news --dry-run     # what would be fetched
  python -m src.pipeline.event_news --apis finnhub    # the fast 12 months
  python -m src.pipeline.event_news                   # everything (hours)
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import requests

from src import db
from src.collectors.news import (
    NewsUnavailable,
    fetch_finnhub,
    fetch_gdelt,
    finnhub_items_to_rows,
    gdelt_articles_to_rows,
)
from src.pipeline.tickers import load_universe
from src.utils.config import load_config, require_env
from src.utils.ratelimit import RateLimiter
from src.utils.timeutils import utc_now_ts

log = logging.getLogger(__name__)

HOUR = 3600


@dataclass(frozen=True)
class Window:
    """One news query: a ticker and the time span to fetch for it."""
    ticker: str
    start_utc: int
    end_utc: int
    n_events: int          # how many events this merged span serves

    def key(self) -> tuple:
        return (self.ticker, self.start_utc, self.end_utc)


def event_windows(conn, cfg: dict, unlabeled_only: bool = True) -> list[Window]:
    """Merged [t₀−pre, t₀+post] spans, one group per ticker.

    Merging is what keeps this affordable: overlapping windows on the same
    ticker answer the same query, and GDELT charges by the call.
    """
    pre = int(cfg["news"]["event_pre_hours"]) * HOUR
    post = int(cfg["event"]["label_horizon_hours"]) * HOUR
    sql = ("SELECT ticker, t0_utc FROM events WHERE n_rumor_posts > 0"
           + (" AND label IS NULL" if unlabeled_only else "")
           + " ORDER BY ticker, t0_utc")

    windows: list[Window] = []
    by_ticker: dict[str, list[int]] = {}
    for ticker, t0 in conn.execute(sql):
        by_ticker.setdefault(ticker, []).append(t0)

    for ticker, t0s in sorted(by_ticker.items()):
        start = end = None
        count = 0
        for t0 in sorted(t0s):
            lo, hi = t0 - pre, t0 + post
            if end is not None and lo <= end:
                end = max(end, hi)
                count += 1
                continue
            if end is not None:
                windows.append(Window(ticker, start, end, count))
            start, end, count = lo, hi, 1
        if end is not None:
            windows.append(Window(ticker, start, end, count))
    return windows


def finnhub_covers(window: Window, cfg: dict, now: int | None = None) -> bool:
    """Whether Finnhub's free tier still holds this window.

    Its archive is a rolling ~12 months and an out-of-range request returns an
    empty list with HTTP 200 — indistinguishable from "no news happened" unless
    we check the date ourselves.
    """
    now = utc_now_ts() if now is None else now
    horizon = now - int(cfg["news"]["finnhub_lookback_days"]) * 86400
    return window.start_utc >= horizon


def collect(conn, cfg: dict, apis: list[str], limit: int | None = None,
            dry_run: bool = False, gdelt_scope: str = "gap") -> dict:
    """Fetch every outstanding window. Resumable: done spans are skipped.

    `gdelt_scope="gap"` (the default) asks GDELT only for windows Finnhub
    cannot serve. Measured 2026-08-05, GDELT completes roughly one query in
    three even at 20s spacing, so covering all 1,010 windows costs the best
    part of a day while covering the 413 Finnhub misses costs a few hours —
    for windows Finnhub does serve it returns ~150 curated headlines already.
    Use "all" for belt-and-braces source diversity if there is time to spare.
    """
    windows = event_windows(conn, cfg)
    universe = load_universe(cfg["tickers"]["universe_csv"])
    session = requests.Session()
    session.headers["User-Agent"] = cfg["reddit"]["user_agent"]
    stats: dict[str, int] = {"windows": len(windows)}

    for api in apis:
        done = db.fetched_news_spans(conn, api)
        todo = [w for w in windows if w.key() not in done]
        if api == "finnhub":
            skipped = [w for w in todo if not finnhub_covers(w, cfg)]
            todo = [w for w in todo if finnhub_covers(w, cfg)]
            stats["finnhub_too_old"] = len(skipped)
        elif api == "gdelt" and gdelt_scope == "gap":
            todo = [w for w in todo if not finnhub_covers(w, cfg)]
        if limit is not None:
            todo = todo[:limit]
        stats[f"{api}_todo"] = len(todo)
        log.info("%s: %d windows outstanding (%d already fetched)",
                 api, len(todo), len(done))
        if dry_run or not todo:
            continue

        limiter = RateLimiter(cfg["news"][f"{api}_min_interval_s"])
        key = require_env("FINNHUB_API_KEY") if api == "finnhub" else None
        for i, window in enumerate(todo, 1):
            limiter.wait()
            try:
                rows = _fetch(api, cfg, session, window, universe, key)
            except (requests.RequestException, NewsUnavailable) as exc:
                # Deliberately not marked done: an unanswered query must stay
                # outstanding, or §6.4 would read our throttling as "no news"
                # and label the event FALSE.
                log.warning("%s %s failed, will retry next run: %s",
                            api, window.ticker, exc)
                stats[f"{api}_failed"] = stats.get(f"{api}_failed", 0) + 1
                continue
            new = db.upsert_news(conn, rows)
            # Recorded even when empty: "we asked and there was nothing" is a
            # real answer, and re-asking it costs the same as asking.
            db.mark_news_span(conn, window.ticker, window.start_utc,
                              window.end_utc, api, len(rows), utc_now_ts())
            stats[f"{api}_rows"] = stats.get(f"{api}_rows", 0) + new
            stats[f"{api}_done"] = stats.get(f"{api}_done", 0) + 1
            if i % 25 == 0:
                log.info("%s %d/%d windows (%d new headlines)", api, i,
                         len(todo), stats.get(f"{api}_rows", 0))
    return stats


def _fetch(api: str, cfg: dict, session, window: Window, universe: dict,
           api_key: str | None) -> list[tuple]:
    if api == "gdelt":
        name = universe.get(window.ticker)
        query = f'"{name}"' if name else window.ticker
        articles = fetch_gdelt(cfg, session, query, window.start_utc,
                               window.end_utc)
        return gdelt_articles_to_rows(articles, window.ticker)
    if api == "finnhub":
        items = fetch_finnhub(session, api_key, window.ticker,
                              window.start_utc, window.end_utc)
        return finnhub_items_to_rows(items, window.ticker,
                                     cfg["news"].get("finnhub_source_domains"))
    raise ValueError(f"unknown api {api!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apis", default="finnhub,gdelt",
                        help="comma-separated subset of finnhub,gdelt")
    parser.add_argument("--limit", type=int, help="fetch at most N windows per API")
    parser.add_argument("--gdelt-scope", choices=("gap", "all"), default="gap",
                        help="'gap' (default) queries GDELT only where Finnhub "
                             "cannot reach; 'all' queries every window")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the workload without calling anything")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    stats = collect(conn, cfg, [a.strip() for a in args.apis.split(",")],
                    limit=args.limit, dry_run=args.dry_run,
                    gdelt_scope=args.gdelt_scope)
    log.info("windows: %s", stats)
    covered = conn.execute(
        """SELECT COUNT(*) FROM events e WHERE e.n_rumor_posts > 0 AND EXISTS
           (SELECT 1 FROM news n WHERE n.ticker = e.ticker
            AND n.seen_utc BETWEEN e.t0_utc - ? AND e.t0_utc + ?)""",
        (int(cfg["news"]["event_pre_hours"]) * HOUR,
         int(cfg["event"]["label_horizon_hours"]) * HOUR),
    ).fetchone()[0]
    total = conn.execute(
        "SELECT COUNT(*) FROM events WHERE n_rumor_posts > 0").fetchone()[0]
    log.info("events with >=1 headline in window: %d/%d", covered, total)


if __name__ == "__main__":
    main()
