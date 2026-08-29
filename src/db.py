"""SQLite storage layer.

Seven tables:

  companies  the study universe, dated at the START of the window so the
             ticker->CIK map is not survivorship-biased (review §7.5).
  filings    raw 8-K rows straight from EDGAR — one row per accession number.
  events     the analysis unit: one usable filing plus its corrected t0,
             materiality and scheduled/unscheduled split.
  bars       yfinance OHLCV, hourly and daily.
  news       headlines with timestamps — this is LABEL infrastructure, not a
             feature source, because t0 = min(acceptance, earliest article).
  meta       key/value provenance, e.g. when the price snapshot was frozen.
  fetch_state per-item collector progress, so an interrupted run resumes.

All timestamps are UTC epoch seconds. All writes are idempotent upserts so
re-running any collector never duplicates rows. Plain sqlite3, no ORM.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
  cik TEXT PRIMARY KEY,          -- zero-padded 10-digit CIK
  ticker TEXT NOT NULL,
  name TEXT,
  exchange TEXT,
  sic TEXT,
  in_universe INTEGER DEFAULT 0, -- passed the liquidity filter
  adv_usd REAL,                  -- average daily traded value used by that filter
  last_price REAL,
  universe_as_of INTEGER         -- the window-start date the filter was applied at
);
CREATE INDEX IF NOT EXISTS idx_companies_ticker ON companies (ticker);

CREATE TABLE IF NOT EXISTS filings (
  accession_no TEXT PRIMARY KEY,
  cik TEXT,
  ticker TEXT,
  form TEXT,                     -- '8-K', '8-K/A'
  items TEXT,                    -- comma-separated item codes, e.g. '1.01,9.01'
  acceptance_utc INTEGER,        -- acceptanceDateTime — NOT t0 on its own
  filing_date_utc INTEGER,
  report_date_utc INTEGER,
  primary_doc TEXT,
  fetched_utc INTEGER
);
CREATE INDEX IF NOT EXISTS idx_filings_ticker ON filings (ticker, acceptance_utc);
CREATE INDEX IF NOT EXISTS idx_filings_acceptance ON filings (acceptance_utc);

CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY,     -- accession number with punctuation stripped
  accession_no TEXT REFERENCES filings(accession_no) ON DELETE CASCADE,
  ticker TEXT,
  items TEXT,
  t0_filing_utc INTEGER,         -- SEC acceptance time
  t0_news_utc INTEGER,           -- earliest matching article, NULL if none found
  t0_utc INTEGER,                -- min of the two — the real "public" moment
  t0_source TEXT,                -- 'filing' | 'news'
  is_scheduled INTEGER,          -- 1 for item 2.02 / 5.07 style known-in-advance
  abs_return REAL,               -- post-announcement move, for the materiality filter
  is_material INTEGER,
  usable INTEGER DEFAULT 0,      -- survived item-code + materiality + coverage filters
  exclude_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ticker ON events (ticker, t0_utc);
CREATE INDEX IF NOT EXISTS idx_events_usable ON events (usable, is_scheduled);

CREATE TABLE IF NOT EXISTS bars (
  ticker TEXT, ts_utc INTEGER, open REAL, high REAL, low REAL,
  close REAL, volume REAL, interval TEXT,
  PRIMARY KEY (ticker, ts_utc, interval)
);

CREATE TABLE IF NOT EXISTS news (
  url TEXT PRIMARY KEY, ticker TEXT, title TEXT,
  -- Publisher identity. The two APIs give DIFFERENT kinds of identifier and
  -- they get different columns, so nothing downstream has to consult `api` to
  -- know what it is holding:
  source_domain TEXT,            -- GDELT: 'reuters.com'. NULL for Finnhub.
  source_name TEXT,              -- Finnhub: 'Benzinga'. NULL for GDELT.
  source_tier INTEGER,           -- 1 wire/top-tier, 2 fast republisher, NULL not credible
  -- THREE distinct times. They were one column until P1-14, which meant t0
  -- silently mixed publication time with crawl time depending on the source:
  published_utc INTEGER,         -- when the PUBLISHER published it. What t0 needs.
                                 -- Finnhub gives this; NULL for GDELT.
  seen_utc INTEGER,              -- when the AGGREGATOR's crawler found it.
                                 -- GDELT gives this; NULL for Finnhub.
                                 -- Later than publication by an unknown amount,
                                 -- so it is an UPPER BOUND on publication.
  fetched_utc INTEGER,           -- when WE pulled the row. Provenance only.
  api TEXT                       -- 'finnhub' | 'gdelt'
);
CREATE INDEX IF NOT EXISTS idx_news_ticker ON news (ticker, seen_utc);

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY, value TEXT, updated_utc INTEGER
);

-- Per-item collector progress, so an interrupted run continues instead of
-- restarting. Deliberately NOT a column on `companies`: that table is an
-- as-of snapshot dated at the window start, and a mutable progress counter
-- does not belong inside it. Records the OUTCOME, not just the attempt --
-- 'ok' with rows_written = 0 means "fetched, genuinely files no 8-Ks, do not
-- come back", which the filings table alone cannot express.
CREATE TABLE IF NOT EXISTS fetch_state (
  source TEXT,                   -- 'edgar'
  key TEXT,                      -- the CIK for edgar
  status TEXT,                   -- 'ok' | 'failed'
  records INTEGER,               -- what the fetch returned
  rows_written INTEGER,          -- what was stored from it
  error TEXT,
  updated_utc INTEGER,
  PRIMARY KEY (source, key)
);
"""


#: Columns added after a table first shipped. `CREATE TABLE IF NOT EXISTS`
#: leaves an existing database untouched, so a new column has to be added
#: explicitly or every dev keeps an old schema without noticing.
MIGRATIONS: tuple[tuple[str, str, str], ...] = (
    ("news", "source_name", "TEXT"),
    ("news", "source_tier", "INTEGER"),
    ("news", "published_utc", "INTEGER"),
    ("news", "fetched_utc", "INTEGER"),
)


#: One-shot data repairs, guarded by a key in `meta` so each runs exactly once.
#: Distinct from MIGRATIONS, which only add columns.
DATA_MIGRATIONS: tuple[tuple[str, str], ...] = (
    (
        "p1_14_move_finnhub_publication_time",
        # Before P1-14 the news table had ONE timestamp column, `seen_utc`, and
        # the Finnhub collector wrote the article's PUBLICATION time into it.
        # `seen_utc` now means crawl time, so those legacy values are sitting in
        # a column that means something else — worse than missing, because they
        # would be read as an upper bound rather than the exact time they are.
        # Their old meaning is known exactly, so move them rather than discard.
        """UPDATE news SET published_utc = seen_utc, seen_utc = NULL
           WHERE api = 'finnhub'
             AND published_utc IS NULL AND seen_utc IS NOT NULL""",
    ),
    (
        "p1_14_clear_finnhub_crawl_time",
        # Finnhub reports no crawl time at all, so ANY value in seen_utc on a
        # finnhub row is a legacy publication time. The migration above misses
        # rows that were re-fetched first (the re-fetch filled published_utc, so
        # the WHERE clause skipped them) and left the stale copy behind.
        "UPDATE news SET seen_utc = NULL WHERE api = 'finnhub' AND seen_utc IS NOT NULL",
    ),
)


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Add missing columns, then run any pending data repairs.

    Idempotent — safe on every connect. Column adds check `PRAGMA table_info`;
    data repairs are guarded by a key in `meta`.
    """
    for table, column, decl in MIGRATIONS:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    conn.commit()

    for key, sql in DATA_MIGRATIONS:
        done = conn.execute(
            "SELECT 1 FROM meta WHERE key = ?", (f"migration:{key}",)
        ).fetchone()
        if done:
            continue
        cursor = conn.execute(sql)
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value, updated_utc) VALUES (?, ?, ?)",
            (f"migration:{key}", str(cursor.rowcount), int(time.time())),
        )
    conn.commit()


def get_conn(db_path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) the project DB with the schema applied."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    _apply_migrations(conn)
    return conn


# --------------------------------------------------------------------------
# companies
# --------------------------------------------------------------------------

COMPANY_COLUMNS = (
    "cik", "ticker", "name", "exchange", "sic", "in_universe", "adv_usd",
    "last_price", "universe_as_of",
)


def upsert_companies(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """Insert/refresh company rows. Returns the number of genuinely new ones."""
    if not rows:
        return 0
    before = conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
    conn.executemany(
        f"""
        INSERT INTO companies ({", ".join(COMPANY_COLUMNS)})
        VALUES ({", ".join(":" + c for c in COMPANY_COLUMNS)})
        ON CONFLICT(cik) DO UPDATE SET
          ticker = excluded.ticker,
          name = COALESCE(excluded.name, companies.name),
          exchange = COALESCE(excluded.exchange, companies.exchange),
          sic = COALESCE(excluded.sic, companies.sic),
          -- COALESCE, not a plain overwrite: a collector that has no opinion
          -- about liquidity passes NULL, and a universe rebuild must not wipe
          -- the flags the Phase 3 filter set. Same failure as the one P1-14
          -- fixed in upsert_news — a re-run losing a column another stage
          -- filled. The filter still writes 0 and 1 explicitly.
          in_universe = COALESCE(excluded.in_universe, companies.in_universe),
          adv_usd = COALESCE(excluded.adv_usd, companies.adv_usd),
          last_price = COALESCE(excluded.last_price, companies.last_price),
          universe_as_of = COALESCE(excluded.universe_as_of, companies.universe_as_of)
        """,
        [{c: r.get(c) for c in COMPANY_COLUMNS} for r in rows],
    )
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
    return after - before


def universe_tickers(conn: sqlite3.Connection) -> list[str]:
    """Tickers that passed the liquidity filter."""
    rows = conn.execute(
        "SELECT DISTINCT ticker FROM companies WHERE in_universe = 1 ORDER BY ticker"
    ).fetchall()
    return [r[0] for r in rows]


def company_name(conn: sqlite3.Connection, ticker: str) -> str | None:
    row = conn.execute(
        "SELECT name FROM companies WHERE ticker = ? AND name IS NOT NULL LIMIT 1",
        (ticker,),
    ).fetchone()
    return row[0] if row else None


# --------------------------------------------------------------------------
# filings
# --------------------------------------------------------------------------

FILING_COLUMNS = (
    "accession_no", "cik", "ticker", "form", "items", "acceptance_utc",
    "filing_date_utc", "report_date_utc", "primary_doc", "fetched_utc",
)


def upsert_filings(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """Insert filings; existing rows are left alone (EDGAR data is immutable).
    Returns the number of genuinely new rows."""
    if not rows:
        return 0
    before = conn.execute("SELECT COUNT(*) FROM filings").fetchone()[0]
    conn.executemany(
        f"""INSERT OR IGNORE INTO filings ({", ".join(FILING_COLUMNS)})
            VALUES ({", ".join(":" + c for c in FILING_COLUMNS)})""",
        [{c: r.get(c) for c in FILING_COLUMNS} for r in rows],
    )
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM filings").fetchone()[0]
    return after - before


def companies_for_collection(conn: sqlite3.Connection,
                            tickers: list[str] | None = None) -> list[sqlite3.Row]:
    """Companies to fetch filings for: the whole map, or a named subset.

    Not `universe_tickers` — that returns only what the Phase 3 liquidity
    filter has approved, which is nothing until Phase 3 runs.
    """
    if tickers:
        marks = ",".join("?" * len(tickers))
        return conn.execute(
            f"SELECT cik, ticker FROM companies WHERE ticker IN ({marks}) "
            f"ORDER BY ticker", tickers
        ).fetchall()
    return conn.execute("SELECT cik, ticker FROM companies ORDER BY ticker").fetchall()


def latest_filing_ts(conn: sqlite3.Connection, cik: str) -> int | None:
    row = conn.execute(
        "SELECT MAX(acceptance_utc) FROM filings WHERE cik = ?", (cik,)
    ).fetchone()
    return row[0]


# --------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------

EVENT_COLUMNS = (
    "event_id", "accession_no", "ticker", "items", "t0_filing_utc",
    "t0_news_utc", "t0_utc", "t0_source", "is_scheduled", "abs_return",
    "is_material", "usable", "exclude_reason",
)


def upsert_events(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """Insert/refresh events. Derived columns are recomputed on conflict so the
    event builder can be re-run after a config change. Returns new-row count."""
    if not rows:
        return 0
    before = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    updatable = [c for c in EVENT_COLUMNS if c not in ("event_id", "accession_no")]
    conn.executemany(
        f"""
        INSERT INTO events ({", ".join(EVENT_COLUMNS)})
        VALUES ({", ".join(":" + c for c in EVENT_COLUMNS)})
        ON CONFLICT(event_id) DO UPDATE SET
          {", ".join(f"{c} = excluded.{c}" for c in updatable)}
        """,
        [{c: r.get(c) for c in EVENT_COLUMNS} for r in rows],
    )
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    return after - before


def usable_events(conn: sqlite3.Connection, scheduled: int | None = None):
    """Events that survived filtering. `scheduled=0/1` restricts to the
    unscheduled/scheduled half — every headline number is reported split."""
    if scheduled is None:
        return conn.execute(
            "SELECT * FROM events WHERE usable = 1 ORDER BY t0_utc"
        ).fetchall()
    return conn.execute(
        "SELECT * FROM events WHERE usable = 1 AND is_scheduled = ? ORDER BY t0_utc",
        (scheduled,),
    ).fetchall()


# --------------------------------------------------------------------------
# bars
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# news
# --------------------------------------------------------------------------

NEWS_COLUMNS = (
    "url", "ticker", "title", "source_domain", "source_name", "source_tier",
    "published_utc", "seen_utc", "fetched_utc", "api",
)


def upsert_news(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """Insert/refresh news rows. Returns the number of genuinely new ones.

    Takes **dicts**, not tuples. The row reached ten fields in P1-14 and
    positional tuples had already caused two rounds of silent test breakage
    when a column was inserted; `companies` uses the same named-column pattern.

    `ON CONFLICT DO UPDATE` with COALESCE rather than `INSERT OR IGNORE`: a
    re-fetch now **fills in** fields that were NULL, instead of skipping the row
    entirely. That was issue #14 — rows collected before a collector fix could
    not be repaired by re-running.
    """
    if not rows:
        return 0
    before = conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
    conn.executemany(
        f"""
        INSERT INTO news ({", ".join(NEWS_COLUMNS)})
        VALUES ({", ".join(":" + c for c in NEWS_COLUMNS)})
        -- COALESCE(existing, new): fill gaps, never overwrite. A timestamp
        -- already recorded must not silently move on a re-fetch — that is the
        -- kind of drift that makes a result impossible to reproduce. Deliberate
        -- re-tiering goes through retier_news(), which UPDATEs directly.
        ON CONFLICT(url) DO UPDATE SET
          title = COALESCE(news.title, excluded.title),
          source_domain = COALESCE(news.source_domain, excluded.source_domain),
          source_name = COALESCE(news.source_name, excluded.source_name),
          source_tier = COALESCE(news.source_tier, excluded.source_tier),
          published_utc = COALESCE(news.published_utc, excluded.published_utc),
          seen_utc = COALESCE(news.seen_utc, excluded.seen_utc),
          fetched_utc = COALESCE(news.fetched_utc, excluded.fetched_utc)
        """,
        [{c: r.get(c) for c in NEWS_COLUMNS} for r in rows],
    )
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
    return after - before


def earliest_news_ts(
    conn: sqlite3.Connection, ticker: str, lo_utc: int, hi_utc: int,
    max_tier: int | None = 2, allow_crawl_time: bool = False,
) -> int | None:
    """Earliest credible article timestamp for a ticker in [lo, hi].

    The second half of the t0 correction: companies wire a press release before
    filing the 8-K, so acceptance time alone overstates the warning window.

    `max_tier` says how far down the credibility tiers to look:

      1     wires and top-tier outlets only — the release itself
      2     also fast republishers of wire copy (default)
      None  anything at all, including untiered publishers

    `allow_crawl_time` falls back to the aggregator's crawl time when the
    publisher's own timestamp is unknown (GDELT rows). Off by default: crawl
    time lags publication, so it pushes t0 later and understates lead time.
    Conservative, but not the same measurement.

    Phase 4 calls this with 1 and with 2 and reports both. That comparison IS
    the sensitivity analysis: Finnhub's free tier carries no wire services, so
    a tier-1-only t0 falls back to filing time for nearly every event.
    """
    # published_utc is the only column that means one thing. seen_utc is the
    # aggregator's CRAWL time, which lags publication by an unknown amount.
    # Falling back to it can only push t0 LATER, which UNDERSTATES lead time —
    # the safe direction for a claim, but not the default.
    time_expr = ("COALESCE(published_utc, seen_utc)" if allow_crawl_time
                 else "published_utc")
    tier_clause = "" if max_tier is None else \
        " AND source_tier IS NOT NULL AND source_tier <= ?"
    args: list = [ticker, lo_utc, hi_utc]
    if max_tier is not None:
        args.append(max_tier)

    row = conn.execute(
        f"""SELECT {time_expr} AS ts FROM news
            WHERE ticker = ? AND {time_expr} BETWEEN ? AND ?{tier_clause}
            ORDER BY ts ASC LIMIT 1""",
        args,
    ).fetchone()
    return row["ts"] if row else None


def retier_news(conn: sqlite3.Connection, cfg: dict) -> int:
    """Recompute `source_tier` for every row from the CURRENT config.

    Stored tiers go stale the moment the whitelist is tuned — the same silent
    drift as an unpinned dependency. This is the escape hatch, and a test
    asserts it actually moves rows after a whitelist change rather than leaving
    it an untested promise.

    Returns the number of rows whose tier changed.
    """
    from src.collectors.news import tier_of  # local: db must not import collectors at module level

    changed = 0
    for row in conn.execute(
        "SELECT url, source_domain, source_name, source_tier FROM news"
    ).fetchall():
        tier = tier_of(cfg, row["source_domain"], row["source_name"])
        if tier != row["source_tier"]:
            conn.execute("UPDATE news SET source_tier = ? WHERE url = ?",
                         (tier, row["url"]))
            changed += 1
    conn.commit()
    return changed


# --------------------------------------------------------------------------
# meta
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# fetch_state — collector progress
# --------------------------------------------------------------------------

def set_fetch_state(conn: sqlite3.Connection, source: str, key: str,
                    status: str, records: int = 0, rows_written: int = 0,
                    error: str | None = None) -> None:
    """Record one item's outcome and COMMIT immediately.

    Committed per item on purpose: state buffered to the end of a run is
    worthless, because surviving a kill is the entire point.
    """
    conn.execute(
        """INSERT INTO fetch_state
             (source, key, status, records, rows_written, error, updated_utc)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(source, key) DO UPDATE SET
             status = excluded.status,
             records = excluded.records,
             rows_written = excluded.rows_written,
             error = excluded.error,
             updated_utc = excluded.updated_utc""",
        (source, key, status, records, rows_written, error, int(time.time())),
    )
    conn.commit()


def completed_keys(conn: sqlite3.Connection, source: str) -> set[str]:
    """Keys this source finished successfully — what `--resume` skips.

    Only 'ok'. A failure is usually a transient 503 or a dropped connection,
    and picking those up is the reason to resume after an outage.
    """
    return {
        row[0] for row in conn.execute(
            "SELECT key FROM fetch_state WHERE source = ? AND status = 'ok'",
            (source,),
        )
    }


def set_meta(conn: sqlite3.Connection, key: str, value: str, ts_utc: int) -> None:
    """Provenance. Used to stamp the price-snapshot freeze date, because
    yfinance's hourly window rolls and bars silently disappear over time."""
    conn.execute(
        """INSERT INTO meta (key, value, updated_utc) VALUES (?, ?, ?)
           ON CONFLICT(key) DO UPDATE SET value=excluded.value,
                                          updated_utc=excluded.updated_utc""",
        (key, value, ts_utc),
    )
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None
