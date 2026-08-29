"""News collector — Finnhub (primary) + GDELT (breadth).

This is LABEL infrastructure, not a feature source, and it runs from week 1.
Companies wire a press release first and file the 8-K with that release
attached minutes-to-hours later, so `acceptanceDateTime` alone overstates the
warning window. The corrected clock is

    t0 = min(8-K acceptanceDateTime, earliest article for that ticker/event)

Finnhub is the workhorse: clean per-ticker headlines with exact UNIX
timestamps, ~60 calls/min free. GDELT adds breadth from smaller outlets but is
rate-limited and unreliable under load — treat it as optional, never as a
blocking dependency. Never scrape Reuters/Bloomberg directly; an aggregator's
headline plus timestamp is all the t0 correction needs.

The credibility whitelist (config `news.whitelist`) is applied at t0-resolution
time, not here — the collector stores everything it sees, deduped by URL.

Usage:
  python -m src.collectors.news --ticker TSLA --start 2025-01-01 --end 2025-01-08
  python -m src.collectors.news --ticker TSLA --apis finnhub \
      --start 2025-01-01 --end 2025-01-08
"""

from __future__ import annotations

import argparse
import logging
import re
from urllib.parse import urlparse

import requests

from src import db
from src.utils.config import load_config, require_env
from src.utils.ratelimit import Backoff, RateLimiter
from src.utils.timeutils import (
    date_str_to_ts, gdelt_to_ts, ts_to_dt, ts_to_gdelt, utc_now_ts,
)

log = logging.getLogger(__name__)




def domain_of(url: str) -> str:
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def publisher_stem(identifier: str | None) -> str:
    """Normalise a publisher identifier so a domain and a name compare equal.

    Lowercase, drop the TLD, strip everything non-alphanumeric:

        'reuters.com'    -> 'reuters'      'Reuters'      -> 'reuters'
        'prnewswire.com' -> 'prnewswire'   'PR Newswire'  -> 'prnewswire'

    The two APIs identify publishers differently — GDELT by domain, Finnhub by
    display name — and this is what lets one whitelist cover both without a
    hand-maintained mapping table that would need an entry per publisher.

    Weakness worth knowing: a very short stem could collide ('ft.com' -> 'ft').
    The whitelist is small and curated, so a collision shows up in review.
    """
    if not identifier:
        return ""
    text = identifier.strip().lower()
    if "." in text and " " not in text:      # looks like a domain
        text = text.rsplit(".", 1)[0]
    return re.sub(r"[^a-z0-9]", "", text)


def tier_of(cfg: dict, source_domain: str | None,
            source_name: str | None) -> int | None:
    """Credibility tier for a publisher, or None if it is on neither list.

    Both identifiers are checked against both tiers: the tier is a property of
    the publisher, not of which API happened to supply the row. If Finnhub ever
    returns "Reuters", that is tier 1.
    """
    ncfg = cfg["news"]
    stems = {publisher_stem(source_domain), publisher_stem(source_name)} - {""}
    if not stems:
        return None
    for tier, key in ((1, "whitelist_tier1"), (2, "whitelist_tier2")):
        allowed = {publisher_stem(entry) for entry in ncfg.get(key, [])}
        if stems & allowed:
            return tier
    return None


def gdelt_articles_to_rows(cfg: dict, articles: list[dict],
                           ticker: str) -> list[dict]:
    """GDELT artlist entries -> news rows.

    GDELT gives a publisher DOMAIN and a CRAWL time (`seendate`) — when its
    crawler found the article, not when the publisher published it. So
    `seen_utc` is filled and `published_utc` left NULL rather than pretending
    the two are the same. See `finnhub_items_to_rows` for the other half.
    """
    fetched = utc_now_ts()
    rows = []
    for art in articles:
        url, seendate = art.get("url"), art.get("seendate")
        if not url or not seendate:
            continue
        try:
            seen_utc = gdelt_to_ts(seendate)
        except ValueError:
            continue
        domain = art.get("domain") or domain_of(url)
        rows.append({
            "url": url, "ticker": ticker, "title": art.get("title") or "",
            "source_domain": domain, "source_name": None,
            "source_tier": tier_of(cfg, domain, None),
            "published_utc": None,          # GDELT does not report it
            "seen_utc": seen_utc,           # crawl time — an upper bound
            "fetched_utc": fetched, "api": "gdelt",
        })
    return rows


def finnhub_items_to_rows(cfg: dict, items: list[dict], ticker: str) -> list[tuple]:
    """Finnhub /company-news entries -> news rows.

    The publisher comes from the `source` field ("Benzinga", "CNBC"), NOT from
    the URL. Every Finnhub `url` is a redirect wrapper on finnhub.io, so
    deriving a domain from it labelled every article `finnhub.io` and made the
    t0 whitelist unusable.

    Finnhub gives a display NAME, not a domain, so `source_name` is filled and
    `source_domain` left NULL rather than guessing a domain from the name.

    Its `datetime` field is the PUBLICATION time, which is exactly what t0
    needs, so it goes in `published_utc`. `seen_utc` (crawl time) is left NULL:
    this API does not report one.
    """
    fetched = utc_now_ts()
    rows = []
    for item in items:
        url, ts = item.get("url"), item.get("datetime")
        if not url or not ts:
            continue
        name = (item.get("source") or "").strip() or None
        rows.append({
            "url": url, "ticker": ticker, "title": item.get("headline") or "",
            "source_domain": None, "source_name": name,
            "source_tier": tier_of(cfg, None, name),
            "published_utc": int(ts),       # Finnhub's `datetime` IS publication
            "seen_utc": None,               # no crawl time from this API
            "fetched_utc": fetched, "api": "finnhub",
        })
    return rows


def fetch_gdelt(cfg: dict, session: requests.Session, query: str,
                start_ts: int, end_ts: int) -> list[dict]:
    ncfg = cfg["news"]
    backoff = Backoff(base_s=15)
    for _ in range(5):
        try:
            resp = session.get(ncfg["gdelt_base"], params={
                "query": query,
                "mode": "artlist",
                "maxrecords": ncfg["max_records"],
                "format": "json",
                "startdatetime": ts_to_gdelt(start_ts),
                "enddatetime": ts_to_gdelt(end_ts),
            }, timeout=60)
            if resp.status_code in (429, 503):
                backoff.sleep(f"HTTP {resp.status_code} from GDELT")
                continue
            resp.raise_for_status()
        except requests.RequestException as exc:
            backoff.sleep(f"GDELT request failed: {exc}")
            continue
        try:
            return resp.json().get("articles", [])
        except ValueError:  # GDELT returns plain-text errors with HTTP 200
            log.error("GDELT non-JSON response: %s", resp.text[:200])
            return []
    log.error("GDELT: giving up after repeated failures for %r", query)
    return []


def fetch_finnhub(cfg: dict, session: requests.Session, api_key: str,
                  ticker: str, start_ts: int, end_ts: int) -> list[dict]:
    base = cfg["news"]["finnhub_base"]
    resp = session.get(f"{base}/company-news", params={
        "symbol": ticker,
        "from": ts_to_dt(start_ts).strftime("%Y-%m-%d"),
        "to": ts_to_dt(end_ts).strftime("%Y-%m-%d"),
        "token": api_key,
    }, timeout=30)
    resp.raise_for_status()
    return resp.json()


def default_gdelt_query(conn, ticker: str) -> str:
    """'"Company Name"' if the companies table knows it, else the bare ticker.
    The name comes from SEC company_tickers_exchange.json via edgar.py."""
    name = db.company_name(conn, ticker)
    return f'"{name}"' if name else ticker


def date_windows(cfg: dict, start_ts: int, end_ts: int) -> list[tuple[int, int]]:
    """Split [start_ts, end_ts] into windows of `news.max_window_days`.

    MEASURED 2026-08-29: `/company-news` caps a response at ~245 articles and
    returns the NEWEST ones. TSLA for 2025-09-01..2026-08-29 came back with 242
    articles, every one of them from 2026-08-11 onward — a year requested, a
    fortnight delivered, HTTP 200, no warning. The same shape as the redirect
    HTML the collectors already guard against: the response looks fine and the
    data is missing.

    Chunked rather than warned about, because a warning inside a 6,000-company
    run is not read and the loss is invisible in the output.
    """
    span = cfg["news"]["max_window_days"] * 86400
    windows = []
    cursor = start_ts
    while cursor < end_ts:
        stop = min(cursor + span, end_ts)
        windows.append((cursor, stop))
        cursor = stop
    return windows or [(start_ts, end_ts)]


def universe_tickers_for_news(conn) -> list[str]:
    """Every company in `companies`, or the liquid subset once Phase 3 sets it.

    `in_universe` is NULL for all 6,054 rows until the Phase 3 liquidity filter
    runs, so falling back to the whole table is the difference between
    collecting news and collecting nothing.
    """
    liquid = db.universe_tickers(conn)
    if liquid:
        return liquid
    return [r[0] for r in conn.execute(
        "SELECT ticker FROM companies WHERE ticker IS NOT NULL ORDER BY ticker")]


def collect(cfg: dict, conn, ticker: str, query: str | None,
            start_ts: int, end_ts: int, apis: list[str],
            gdelt_limiter: RateLimiter | None = None,
            finnhub_limiter: RateLimiter | None = None) -> int:
    """Fetch and store headlines for one ticker over [start_ts, end_ts].
    Returns the number of records parsed.

    gdelt_limiter/finnhub_limiter: pass limiters that persist across calls when
    collecting for many tickers in one process — a fresh RateLimiter() only
    enforces spacing *within* a single call, so back-to-back calls with no
    shared limiter don't actually throttle against each other. The CLI (one
    ticker per process) doesn't need this, hence the defaults."""
    ncfg = cfg["news"]
    session = requests.Session()
    session.headers["User-Agent"] = cfg["http"]["user_agent"]
    parsed = 0

    if "finnhub" in apis:
        api_key = require_env("FINNHUB_API_KEY")
        (finnhub_limiter or RateLimiter(ncfg["finnhub_min_interval_s"])).wait()
        items = fetch_finnhub(cfg, session, api_key, ticker, start_ts, end_ts)
        rows = finnhub_items_to_rows(cfg, items, ticker)
        parsed += len(rows)
        n = db.upsert_news(conn, rows)
        log.info("Finnhub %s: %d items, %d new", ticker, len(items), n)

    if "gdelt" in apis:
        (gdelt_limiter or RateLimiter(ncfg["gdelt_min_interval_s"])).wait()
        q = query or default_gdelt_query(conn, ticker)
        articles = fetch_gdelt(cfg, session, q, start_ts, end_ts)
        rows = gdelt_articles_to_rows(cfg, articles, ticker)
        parsed += len(rows)
        n = db.upsert_news(conn, rows)
        log.info("GDELT %r: %d articles, %d new", q, len(articles), n)

    # Zero for ONE ticker is ordinary — a small company can genuinely have no
    # news in a quiet week. Zero across an ENTIRE run is the broken-endpoint
    # signal, and `collect_many` raises on that. Aborting here instead would
    # kill a watchlist run on its first quiet company.
    if parsed == 0:
        log.warning("no records for %s over %s -> %s via %s",
                    ticker, ts_to_dt(start_ts).date(), ts_to_dt(end_ts).date(), apis)
    return parsed


def collect_many(cfg: dict, conn, tickers: list[str], start_ts: int, end_ts: int,
                 apis: list[str], query: str | None = None) -> int:
    """Collect for several tickers in one process, then apply the zero guard.

    One RateLimiter is shared across the whole run: a fresh limiter per call
    only spaces requests *within* that call, so back-to-back calls would not
    throttle against each other and the free tier's ~60/min would be breached.

    Raises if the whole run parsed nothing and `logging.fail_on_zero_records`
    is set. That is the guard AGENTS.md rule 8 requires — an HTTP 200 carrying
    redirect HTML or empty JSON makes a broken collector look healthy while it
    writes nothing for days, and a process that exits 0 lets a scheduler report
    success.
    """
    ncfg = cfg["news"]
    gdelt_limiter = RateLimiter(ncfg["gdelt_min_interval_s"])
    finnhub_limiter = RateLimiter(ncfg["finnhub_min_interval_s"])

    windows = date_windows(cfg, start_ts, end_ts)
    if len(windows) > 1:
        log.info("range split into %d window(s) of %d day(s) — a single long "
                 "request returns only the newest ~245 articles",
                 len(windows), ncfg["max_window_days"])

    total, failed = 0, 0
    for ticker in tickers:
        for win_start, win_end in windows:
            try:
                total += collect(cfg, conn, ticker, query, win_start, win_end,
                                 apis, gdelt_limiter=gdelt_limiter,
                                 finnhub_limiter=finnhub_limiter)
            except Exception:
                failed += 1
                log.exception("failed to collect %s %s -> %s — continuing",
                              ticker, ts_to_dt(win_start).date(),
                              ts_to_dt(win_end).date())

    log.info("Done. %d ticker(s) x %d window(s) processed (%d failed); "
             "%d record(s) parsed.", len(tickers), len(windows), failed, total)

    if cfg["logging"]["fail_on_zero_records"] and total == 0:
        raise SystemExit(
            f"ZERO records parsed across all {len(tickers)} ticker(s) via "
            f"{apis} for {ts_to_dt(start_ts).date()} -> {ts_to_dt(end_ts).date()}. "
            f"One quiet ticker is normal; all of them means the endpoint is "
            f"broken. Do not treat this run as successful."
        )
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", help="a single ticker")
    parser.add_argument("--tickers", help="comma-separated tickers")
    parser.add_argument("--watchlist", action="store_true",
                        help="use news.seed_watchlist from config")
    parser.add_argument("--universe", action="store_true",
                        help="every company in `companies` (the liquid subset "
                             "once Phase 3 sets in_universe)")
    parser.add_argument("--retier", action="store_true",
                        help="recompute source_tier for every stored row from "
                             "the current whitelist, then exit. Use after "
                             "editing news.whitelist_tier1/tier2.")
    parser.add_argument("--query", help="GDELT query override, e.g. '\"Tesla\"'")
    parser.add_argument("--start", help="YYYY-MM-DD")
    parser.add_argument("--end", help="YYYY-MM-DD (default: now)")
    parser.add_argument("--apis", default="finnhub,gdelt",
                        help="comma-separated subset of finnhub,gdelt")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])

    if args.retier:
        changed = db.retier_news(conn, cfg)
        log.info("Re-tiered from current config: %d row(s) changed.", changed)
        return

    if args.universe:
        tickers = universe_tickers_for_news(conn)
        if not tickers:
            raise SystemExit(
                "companies table is empty — run "
                "`python -m src.collectors.edgar --build-universe` first.")
    elif args.watchlist:
        tickers = list(cfg["news"]["seed_watchlist"])
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    elif args.ticker:
        tickers = [args.ticker.upper()]
    else:
        parser.error("give one of --ticker, --tickers, --watchlist or "
                     "--universe (or --retier)")

    # A daily run needs no dates: the last `default_lookback_days` is exactly
    # the "accumulating" this collector exists to do.
    end_ts = date_str_to_ts(args.end) if args.end else utc_now_ts()
    start_ts = (date_str_to_ts(args.start) if args.start
                else end_ts - cfg["news"]["default_lookback_days"] * 86400)

    collect_many(
        cfg, conn, tickers,
        start_ts=start_ts,
        end_ts=end_ts,
        apis=[a.strip() for a in args.apis.split(",")],
        query=args.query,
    )
    total = conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
    log.info("Done. news table now has %d rows.", total)


if __name__ == "__main__":
    main()
