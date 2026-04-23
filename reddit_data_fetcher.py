"""
Reddit + Stock Market Data Collector
=====================================
Fetches posts from financial subreddits and correlates them with
stock price data using yfinance. Stores everything in PostgreSQL.

Install dependencies:
    pip install requests yfinance psycopg2-binary python-dotenv schedule
"""

import os
import re
import time
import logging
import requests
import yfinance as yf
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

SUBREDDITS = ["wallstreetbets", "stocks", "investing", "StockMarket"]
SORT_TYPES = ["hot", "new", "top"]
POST_LIMIT  = 100          # max per request (Reddit cap)
FETCH_DELAY = 2            # seconds between Reddit requests (rate limit)

# Common stock tickers to look for in post text
TICKER_PATTERN = re.compile(r'\b([A-Z]{1,5})\b')
KNOWN_TICKERS  = {
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "TSLA", "META",
    "NVDA", "AMD", "INTC", "NFLX", "BABA", "GME", "AMC",
    "SPY", "QQQ", "PLTR", "RIVN", "LCID", "NIO", "COIN",
    "JPM", "BAC", "GS", "WFC", "BRK", "V", "MA", "PYPL",
}

HEADERS = {"User-Agent": "StockRumorDetector/1.0 (research project)"}

DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     os.getenv("DB_PORT",     "5432"),
    "dbname":   os.getenv("DB_NAME",     "stock_rumors"),
    "user":     os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────

def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def init_db():
    """Create tables if they don't exist."""
    ddl = """
    CREATE TABLE IF NOT EXISTS reddit_posts (
        id              TEXT PRIMARY KEY,
        subreddit       TEXT        NOT NULL,
        title           TEXT        NOT NULL,
        body            TEXT,
        author          TEXT,
        score           INTEGER,
        upvote_ratio    FLOAT,
        num_comments    INTEGER,
        url             TEXT,
        permalink       TEXT,
        flair           TEXT,
        is_self         BOOLEAN,
        created_utc     TIMESTAMPTZ NOT NULL,
        fetched_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        tickers_found   TEXT[]      DEFAULT '{}'
    );

    CREATE TABLE IF NOT EXISTS stock_prices (
        id              SERIAL PRIMARY KEY,
        ticker          TEXT        NOT NULL,
        price_date      DATE        NOT NULL,
        open            FLOAT,
        high            FLOAT,
        low             FLOAT,
        close           FLOAT,
        volume          BIGINT,
        fetched_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (ticker, price_date)
    );

    CREATE TABLE IF NOT EXISTS post_ticker_links (
        post_id         TEXT        REFERENCES reddit_posts(id) ON DELETE CASCADE,
        ticker          TEXT        NOT NULL,
        PRIMARY KEY (post_id, ticker)
    );

    CREATE INDEX IF NOT EXISTS idx_posts_created   ON reddit_posts (created_utc);
    CREATE INDEX IF NOT EXISTS idx_posts_subreddit ON reddit_posts (subreddit);
    CREATE INDEX IF NOT EXISTS idx_posts_tickers   ON reddit_posts USING GIN (tickers_found);
    CREATE INDEX IF NOT EXISTS idx_prices_ticker   ON stock_prices (ticker, price_date);
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()
    log.info("Database schema ready.")


# ─────────────────────────────────────────────
# REDDIT DATA FETCHER CLASS
# ─────────────────────────────────────────────

class RedditDataFetcher:
    """Fetches Reddit posts and stock prices for analysis."""
    
    def __init__(self):
        pass
    
    def fetch_recent_posts(self, query: str, limit: int = 10):
        """Fetch recent posts from financial subreddits matching the query."""
        all_posts = []
        for subreddit in SUBREDDITS:
            raw_posts = fetch_subreddit(subreddit, sort="new", limit=limit)
            for raw in raw_posts:
                parsed = parse_post(raw, subreddit)
                all_posts.append({
                    "id": parsed["id"],
                    "title": parsed["title"],
                    "content": parsed["body"],
                    "ticker": parsed["tickers"][0] if parsed["tickers"] else "UNKNOWN",
                    "timestamp": parsed["created_utc"].isoformat(),
                })
        return all_posts
    
    def fetch_stock_prices(self, ticker: str, start_date: str, end_date: str):
        """Fetch stock price data for a given ticker."""
        try:
            data = yf.download(ticker, period="1d", interval="1d", progress=False, auto_adjust=True)
            if data.empty:
                return {
                    "current_price": 0.0,
                    "current_volume": 0.0,
                    "price_change_24h": 0.0,
                    "volume_change_24h": 0.0
                }
            
            # Get the most recent row
            latest = data.iloc[-1]
            if len(data) > 1:
                prev = data.iloc[-2]
                price_change = (latest["Close"] - prev["Close"]) / prev["Close"]
                volume_change = (latest["Volume"] - prev["Volume"]) / prev["Volume"] if prev["Volume"] > 0 else 0.0
            else:
                price_change = 0.0
                volume_change = 0.0
            
            return {
                "current_price": float(latest["Close"]),
                "current_volume": int(latest["Volume"]),
                "price_change_24h": price_change,
                "volume_change_24h": volume_change
            }
        except Exception as e:
            log.error(f"Error fetching stock prices for {ticker}: {e}")
            return {
                "current_price": 0.0,
                "current_volume": 0.0,
                "price_change_24h": 0.0,
                "volume_change_24h": 0.0
            }


# ─────────────────────────────────────────────
# REDDIT FETCHER (Legacy Functions)
# ─────────────────────────────────────────────

def fetch_subreddit(subreddit: str, sort: str = "hot", limit: int = POST_LIMIT) -> list[dict]:
    """Fetch posts from a subreddit using the .json trick."""
    url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
    params = {"limit": limit, "raw_json": 1}

    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        posts = data["data"]["children"]
        log.info(f"  Fetched {len(posts)} posts from r/{subreddit} [{sort}]")
        return [p["data"] for p in posts]
    except Exception as e:
        log.error(f"  Failed to fetch r/{subreddit}/{sort}: {e}")
        return []


def extract_tickers(text: str) -> list[str]:
    """Extract known stock tickers mentioned in post text."""
    if not text:
        return []
    candidates = TICKER_PATTERN.findall(text)
    return list({t for t in candidates if t in KNOWN_TICKERS})


def parse_post(raw: dict, subreddit: str) -> dict:
    """Normalize a raw Reddit post dict."""
    body = raw.get("selftext", "") or ""
    title = raw.get("title", "")
    tickers = extract_tickers(f"{title} {body}")

    return {
        "id":           raw["id"],
        "subreddit":    subreddit,
        "title":        title,
        "body":         body[:10000],   # cap at 10k chars
        "author":       raw.get("author"),
        "score":        raw.get("score", 0),
        "upvote_ratio": raw.get("upvote_ratio", 0.0),
        "num_comments": raw.get("num_comments", 0),
        "url":          raw.get("url"),
        "permalink":    raw.get("permalink"),
        "flair":        raw.get("link_flair_text"),
        "is_self":      raw.get("is_self", False),
        "created_utc":  datetime.fromtimestamp(raw["created_utc"], tz=timezone.utc),
        "tickers":      tickers,
    }


def save_posts(posts: list[dict]):
    """Upsert posts into PostgreSQL."""
    if not posts:
        return

    rows = [
        (
            p["id"], p["subreddit"], p["title"], p["body"],
            p["author"], p["score"], p["upvote_ratio"], p["num_comments"],
            p["url"], p["permalink"], p["flair"], p["is_self"],
            p["created_utc"], datetime.now(tz=timezone.utc), p["tickers"],
        )
        for p in posts
    ]

    sql = """
        INSERT INTO reddit_posts
            (id, subreddit, title, body, author, score, upvote_ratio,
             num_comments, url, permalink, flair, is_self, created_utc,
             fetched_at, tickers_found)
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            score        = EXCLUDED.score,
            num_comments = EXCLUDED.num_comments,
            upvote_ratio = EXCLUDED.upvote_ratio,
            fetched_at   = EXCLUDED.fetched_at
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows)

            # Save post ↔ ticker links
            link_rows = [
                (p["id"], ticker)
                for p in posts
                for ticker in p["tickers"]
            ]
            if link_rows:
                execute_values(cur, """
                    INSERT INTO post_ticker_links (post_id, ticker)
                    VALUES %s ON CONFLICT DO NOTHING
                """, link_rows)

        conn.commit()

    log.info(f"  Saved {len(posts)} posts to DB.")


# ─────────────────────────────────────────────
# STOCK PRICE FETCHER
# ─────────────────────────────────────────────

def fetch_stock_prices(tickers: set[str], period: str = "7d"):
    """
    Fetch OHLCV data for all found tickers using yfinance.
    period: '1d', '5d', '7d', '1mo', '3mo', etc.
    """
    if not tickers:
        return

    log.info(f"Fetching stock data for: {', '.join(tickers)}")
    rows = []

    for ticker in tickers:
        try:
            data = yf.download(ticker, period=period, interval="1d",
                               progress=False, auto_adjust=True)
            if data.empty:
                log.warning(f"  No data for {ticker}")
                continue

            for date, row in data.iterrows():
                rows.append((
                    ticker,
                    date.date(),
                    float(row["Open"].iloc[0])   if hasattr(row["Open"],   "iloc") else float(row["Open"]),
                    float(row["High"].iloc[0])   if hasattr(row["High"],   "iloc") else float(row["High"]),
                    float(row["Low"].iloc[0])    if hasattr(row["Low"],    "iloc") else float(row["Low"]),
                    float(row["Close"].iloc[0])  if hasattr(row["Close"],  "iloc") else float(row["Close"]),
                    int(row["Volume"].iloc[0])   if hasattr(row["Volume"], "iloc") else int(row["Volume"]),
                    datetime.now(tz=timezone.utc),
                ))

            log.info(f"  {ticker}: {len(data)} trading days fetched")
            time.sleep(0.5)

        except Exception as e:
            log.error(f"  yfinance error for {ticker}: {e}")

    if not rows:
        return

    sql = """
        INSERT INTO stock_prices
            (ticker, price_date, open, high, low, close, volume, fetched_at)
        VALUES %s
        ON CONFLICT (ticker, price_date) DO UPDATE SET
            open       = EXCLUDED.open,
            high       = EXCLUDED.high,
            low        = EXCLUDED.low,
            close      = EXCLUDED.close,
            volume     = EXCLUDED.volume,
            fetched_at = EXCLUDED.fetched_at
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows)
        conn.commit()

    log.info(f"  Saved {len(rows)} stock price rows to DB.")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline():
    log.info("=" * 50)
    log.info("Starting data collection pipeline...")
    log.info("=" * 50)

    all_posts   = []
    all_tickers = set()

    for subreddit in SUBREDDITS:
        for sort in SORT_TYPES:
            log.info(f"Fetching r/{subreddit} [{sort}]...")
            raw_posts = fetch_subreddit(subreddit, sort=sort)
            parsed    = [parse_post(p, subreddit) for p in raw_posts]
            all_posts.extend(parsed)

            for p in parsed:
                all_tickers.update(p["tickers"])

            time.sleep(FETCH_DELAY)

    # Deduplicate by post ID
    seen = set()
    unique_posts = []
    for p in all_posts:
        if p["id"] not in seen:
            seen.add(p["id"])
            unique_posts.append(p)

    log.info(f"\nTotal unique posts: {len(unique_posts)}")
    log.info(f"Tickers found: {all_tickers or 'none'}")

    save_posts(unique_posts)
    fetch_stock_prices(all_tickers, period="7d")

    log.info("Pipeline complete.")


if __name__ == "__main__":
    init_db()
    run_pipeline()