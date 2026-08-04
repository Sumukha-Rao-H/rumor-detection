"""SQLite storage layer — schema from implementation_plan.md Appendix C.

Four tables: posts, bars, news, events. All timestamps are UTC epoch seconds.
All writes are idempotent upserts so re-running any collector never duplicates
rows. Plain sqlite3, no ORM.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
  id TEXT PRIMARY KEY, subreddit TEXT, title TEXT, selftext TEXT,
  author TEXT, created_utc INTEGER, score INTEGER, upvote_ratio REAL,
  num_comments INTEGER, flair TEXT, url TEXT, source TEXT,  -- 'arctic'|'live'
  fetched_utc INTEGER, score_6h INTEGER, score_24h INTEGER
);
CREATE INDEX IF NOT EXISTS idx_posts_created ON posts (created_utc);
CREATE INDEX IF NOT EXISTS idx_posts_subreddit ON posts (subreddit);

CREATE TABLE IF NOT EXISTS post_tickers (
  post_id TEXT REFERENCES posts(id) ON DELETE CASCADE,
  ticker TEXT NOT NULL,
  PRIMARY KEY (post_id, ticker)
);
CREATE INDEX IF NOT EXISTS idx_post_tickers_ticker ON post_tickers (ticker);

CREATE TABLE IF NOT EXISTS bars (
  ticker TEXT, ts_utc INTEGER, open REAL, high REAL, low REAL,
  close REAL, volume REAL, interval TEXT, PRIMARY KEY (ticker, ts_utc, interval)
);

CREATE TABLE IF NOT EXISTS news (
  url TEXT PRIMARY KEY, ticker TEXT, title TEXT, source_domain TEXT,
  seen_utc INTEGER, api TEXT  -- 'gdelt'|'finnhub'|'yf'
);
CREATE INDEX IF NOT EXISTS idx_news_ticker ON news (ticker, seen_utc);

CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY, ticker TEXT, t0_utc INTEGER, claim_summary TEXT,
  claim_type TEXT, post_ids TEXT, label INTEGER,  -- 1 TRUE, 0 FALSE, NULL unverified
  t_official_utc INTEGER, label_source TEXT, human_reviewed INTEGER DEFAULT 0,
  n_posts INTEGER, subreddits TEXT                -- plan §6.3 stores both
);
CREATE INDEX IF NOT EXISTS idx_events_ticker ON events (ticker, t0_utc);
"""

# Columns added after the first DBs were created (plan §6.3). ALTER is the only
# way to reach a table that CREATE TABLE IF NOT EXISTS silently skips.
MIGRATIONS: dict[str, dict[str, str]] = {
    "events": {"n_posts": "INTEGER", "subreddits": "TEXT"},
}

POST_COLUMNS = (
    "id", "subreddit", "title", "selftext", "author", "created_utc", "score",
    "upvote_ratio", "num_comments", "flair", "url", "source", "fetched_utc",
    "score_6h", "score_24h",
)


def get_conn(db_path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) the project DB with the schema applied."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns missing from DBs created by an earlier schema version."""
    for table, columns in MIGRATIONS.items():
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, decl in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
    conn.commit()


def upsert_posts(conn: sqlite3.Connection, posts: list[dict]) -> int:
    """Insert posts; on conflict refresh the mutable engagement fields.
    Returns the number of genuinely new rows."""
    if not posts:
        return 0
    before = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    conn.executemany(
        f"""
        INSERT INTO posts ({", ".join(POST_COLUMNS)})
        VALUES ({", ".join(":" + c for c in POST_COLUMNS)})
        ON CONFLICT(id) DO UPDATE SET
          score = excluded.score,
          upvote_ratio = excluded.upvote_ratio,
          num_comments = excluded.num_comments,
          fetched_utc = excluded.fetched_utc
        """,
        [{c: p.get(c) for c in POST_COLUMNS} for p in posts],
    )
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    return after - before


def link_post_tickers(conn: sqlite3.Connection, links: list[tuple[str, str]]) -> None:
    """links: (post_id, ticker) pairs."""
    if not links:
        return
    conn.executemany(
        "INSERT OR IGNORE INTO post_tickers (post_id, ticker) VALUES (?, ?)", links
    )
    conn.commit()


def set_post_score_snapshot(
    conn: sqlite3.Connection, post_id: str, column: str, score: int
) -> None:
    """Record the delayed engagement snapshot (score_6h / score_24h)."""
    if column not in ("score_6h", "score_24h"):
        raise ValueError(f"Bad snapshot column: {column}")
    conn.execute(f"UPDATE posts SET {column} = ? WHERE id = ?", (score, post_id))
    conn.commit()


def upsert_bars(conn: sqlite3.Connection, rows: list[tuple]) -> int:
    """rows: (ticker, ts_utc, open, high, low, close, volume, interval)."""
    if not rows:
        return 0
    cur = conn.executemany(
        """
        INSERT INTO bars (ticker, ts_utc, open, high, low, close, volume, interval)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, ts_utc, interval) DO UPDATE SET
          open=excluded.open, high=excluded.high, low=excluded.low,
          close=excluded.close, volume=excluded.volume
        """,
        rows,
    )
    conn.commit()
    return cur.rowcount


def latest_bar_ts(conn: sqlite3.Connection, ticker: str, interval: str) -> int | None:
    row = conn.execute(
        "SELECT MAX(ts_utc) FROM bars WHERE ticker = ? AND interval = ?",
        (ticker, interval),
    ).fetchone()
    return row[0]


def upsert_news(conn: sqlite3.Connection, rows: list[tuple]) -> int:
    """rows: (url, ticker, title, source_domain, seen_utc, api)."""
    if not rows:
        return 0
    before = conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
    conn.executemany(
        """
        INSERT OR IGNORE INTO news (url, ticker, title, source_domain, seen_utc, api)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
    return after - before


EVENT_COLUMNS = (
    "event_id", "ticker", "t0_utc", "claim_summary", "claim_type", "post_ids",
    "n_posts", "subreddits",
)


def upsert_events(conn: sqlite3.Connection, events: list[dict]) -> int:
    """Insert/refresh clustered events. Returns the number of new rows.

    Labeling fields (label, t_official_utc, label_source, human_reviewed) are
    never touched: re-clustering must not throw away human review work
    (plan §6.4 — "do not skip human review"). claim_summary/claim_type are only
    overwritten when the caller supplies them, so re-running the clustering
    pass does not wipe LLM triage output either.
    """
    if not events:
        return 0
    before = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    conn.executemany(
        f"""
        INSERT INTO events ({", ".join(EVENT_COLUMNS)})
        VALUES ({", ".join(":" + c for c in EVENT_COLUMNS)})
        ON CONFLICT(event_id) DO UPDATE SET
          t0_utc = excluded.t0_utc,
          post_ids = excluded.post_ids,
          n_posts = excluded.n_posts,
          subreddits = excluded.subreddits,
          claim_summary = COALESCE(excluded.claim_summary, claim_summary),
          claim_type = COALESCE(excluded.claim_type, claim_type)
        """,
        [{c: e.get(c) for c in EVENT_COLUMNS} for e in events],
    )
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    return after - before


def prune_events(conn: sqlite3.Connection, keep_ids: set[str]) -> int:
    """Delete events the current clustering no longer produces.

    Anything a human has already touched (label set, or human_reviewed) is kept
    regardless — re-running the pipeline must never destroy review work.
    """
    with conn:
        conn.execute("CREATE TEMP TABLE IF NOT EXISTS _keep (event_id TEXT PRIMARY KEY)")
        conn.execute("DELETE FROM _keep")
        conn.executemany("INSERT OR IGNORE INTO _keep VALUES (?)",
                         [(i,) for i in keep_ids])
        cur = conn.execute(
            """
            DELETE FROM events
            WHERE event_id NOT IN (SELECT event_id FROM _keep)
              AND human_reviewed = 0 AND label IS NULL
            """
        )
    return cur.rowcount


def known_post_ids(conn: sqlite3.Connection, ids: list[str]) -> set[str]:
    """Which of these post ids are already stored? (for cheap dedupe)"""
    if not ids:
        return set()
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT id FROM posts WHERE id IN ({placeholders})", ids
    ).fetchall()
    return {r[0] for r in rows}


def distinct_post_tickers(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT ticker FROM post_tickers ORDER BY ticker"
    ).fetchall()
    return [r[0] for r in rows]
