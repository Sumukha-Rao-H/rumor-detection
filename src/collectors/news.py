"""Ground-truth news collector — GDELT 2.0 DOC API + Finnhub (plan §5.4).

Never scrape Reuters/Bloomberg directly: aggregator headline + timestamp is
sufficient for labeling. GDELT is free with no key (15-min update latency);
its `seendate` is the t_official candidate. Finnhub free tier provides clean
per-ticker headlines with UNIX timestamps (FINNHUB_API_KEY in .env).

The source-credibility whitelist (config news.whitelist) is applied at
labeling time (§6.4), not here — the collector stores everything it sees,
deduped by URL.

Usage:
  python -m src.collectors.news --ticker TSLA --start 2025-01-01 --end 2025-01-08
  python -m src.collectors.news --ticker TSLA --query '"Tesla" (merger OR acquisition)' \
      --start 2025-01-01 --end 2025-01-08 --apis gdelt
"""

from __future__ import annotations

import argparse
import logging
from urllib.parse import urlparse

import requests

from src import db
from src.pipeline.tickers import load_universe
from src.utils.config import load_config, require_env
from src.utils.ratelimit import Backoff, RateLimiter
from src.utils.timeutils import (
    date_str_to_ts, gdelt_to_ts, ts_to_dt, ts_to_gdelt, utc_now_ts,
)

log = logging.getLogger(__name__)

FINNHUB_BASE = "https://finnhub.io/api/v1"


def domain_of(url: str) -> str:
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def gdelt_articles_to_rows(articles: list[dict], ticker: str) -> list[tuple]:
    """GDELT artlist entries -> news rows (url, ticker, title, domain, ts, api)."""
    rows = []
    for art in articles:
        url, seendate = art.get("url"), art.get("seendate")
        if not url or not seendate:
            continue
        try:
            seen_utc = gdelt_to_ts(seendate)
        except ValueError:
            continue
        rows.append((url, ticker, art.get("title") or "",
                     art.get("domain") or domain_of(url), seen_utc, "gdelt"))
    return rows


def finnhub_items_to_rows(items: list[dict], ticker: str) -> list[tuple]:
    """Finnhub /company-news entries -> news rows."""
    rows = []
    for item in items:
        url, ts = item.get("url"), item.get("datetime")
        if not url or not ts:
            continue
        rows.append((url, ticker, item.get("headline") or "",
                     domain_of(url), int(ts), "finnhub"))
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


def fetch_finnhub(session: requests.Session, api_key: str, ticker: str,
                  start_ts: int, end_ts: int) -> list[dict]:
    resp = session.get(f"{FINNHUB_BASE}/company-news", params={
        "symbol": ticker,
        "from": ts_to_dt(start_ts).strftime("%Y-%m-%d"),
        "to": ts_to_dt(end_ts).strftime("%Y-%m-%d"),
        "token": api_key,
    }, timeout=30)
    resp.raise_for_status()
    return resp.json()


def default_gdelt_query(cfg: dict, ticker: str) -> str:
    """'"Company Name"' if the universe knows it, else the bare ticker."""
    universe = load_universe(cfg["tickers"]["universe_csv"])
    name = universe.get(ticker)
    return f'"{name}"' if name else ticker


def collect(cfg: dict, conn, ticker: str, query: str | None,
            start_ts: int, end_ts: int, apis: list[str]) -> None:
    ncfg = cfg["news"]
    session = requests.Session()
    session.headers["User-Agent"] = cfg["reddit"]["user_agent"]

    if "gdelt" in apis:
        RateLimiter(ncfg["gdelt_min_interval_s"]).wait()
        q = query or default_gdelt_query(cfg, ticker)
        articles = fetch_gdelt(cfg, session, q, start_ts, end_ts)
        n = db.upsert_news(conn, gdelt_articles_to_rows(articles, ticker))
        log.info("GDELT %r: %d articles, %d new", q, len(articles), n)

    if "finnhub" in apis:
        api_key = require_env("FINNHUB_API_KEY")
        RateLimiter(ncfg["finnhub_min_interval_s"]).wait()
        items = fetch_finnhub(session, api_key, ticker, start_ts, end_ts)
        n = db.upsert_news(conn, finnhub_items_to_rows(items, ticker))
        log.info("Finnhub %s: %d items, %d new", ticker, len(items), n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--query", help="GDELT query override, e.g. "
                        "'\"Tesla\" (merger OR acquisition)'")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", help="YYYY-MM-DD (default: now)")
    parser.add_argument("--apis", default="gdelt,finnhub",
                        help="comma-separated subset of gdelt,finnhub")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    collect(
        cfg, conn,
        ticker=args.ticker.upper(),
        query=args.query,
        start_ts=date_str_to_ts(args.start),
        end_ts=date_str_to_ts(args.end) if args.end else utc_now_ts(),
        apis=[a.strip() for a in args.apis.split(",")],
    )
    total = conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
    log.info("Done. news table now has %d rows.", total)


if __name__ == "__main__":
    main()
