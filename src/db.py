"""SQLite storage layer.

Five tables:

  companies  the study universe, dated at the START of the window so the
             ticker->CIK map is not survivorship-biased (review §7.5).
  filings    raw 8-K rows straight from EDGAR — one row per accession number.
  events     the analysis unit: one usable filing plus its corrected t0,
             materiality and scheduled/unscheduled split.
  bars       yfinance OHLCV, hourly and daily.
  news       headlines with timestamps — this is LABEL infrastructure, not a
             feature source, because t0 = min(acceptance, earliest article).
  meta       key/value provenance, e.g. when the price snapshot was frozen.

All timestamps are UTC epoch seconds. All writes are idempotent upserts so
re-running any collector never duplicates rows. Plain sqlite3, no ORM.
"""

from __future__ import annotations

import sqlite3
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
  seen_utc INTEGER, api TEXT     -- 'finnhub' | 'gdelt'
);
CREATE INDEX IF NOT EXISTS idx_news_ticker ON news (ticker, seen_utc);

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY, value TEXT, updated_utc INTEGER
);
"""


#: Columns added after a table first shipped. `CREATE TABLE IF NOT EXISTS`
#: leaves an existing database untouched, so a new column has to be added
#: explicitly or every dev keeps an old schema without noticing.
MIGRATIONS: tuple[tuple[str, str, str], ...] = (
    ("news", "source_name", "TEXT"),
)


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Add any missing columns. Idempotent — safe on every connect."""
    for table, column, decl in MIGRATIONS:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
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
          in_universe = excluded.in_universe,
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

def upsert_news(conn: sqlite3.Connection, rows: list[tuple]) -> int:
    """rows: (url, ticker, title, source_domain, source_name, seen_utc, api).

    `source_domain` and `source_name` are deliberately separate: GDELT gives a
    domain, Finnhub gives a display name, and collapsing them into one column
    would mean every reader had to check `api` to know which kind it had.
    """
    if not rows:
        return 0
    before = conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
    conn.executemany(
        """
        INSERT OR IGNORE INTO news
          (url, ticker, title, source_domain, source_name, seen_utc, api)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
    return after - before


def earliest_news_ts(
    conn: sqlite3.Connection, ticker: str, lo_utc: int, hi_utc: int,
    domains: set[str] | None = None,
) -> int | None:
    """Earliest article timestamp for a ticker in [lo, hi]. This is the second
    half of the t0 correction: companies wire a press release before filing the
    8-K, so acceptance time alone overstates the warning window."""
    rows = conn.execute(
        """SELECT seen_utc, source_domain FROM news
           WHERE ticker = ? AND seen_utc BETWEEN ? AND ?
           ORDER BY seen_utc ASC""",
        (ticker, lo_utc, hi_utc),
    ).fetchall()
    for row in rows:
        if domains is None or row["source_domain"] in domains:
            return row["seen_utc"]
    return None


# --------------------------------------------------------------------------
# meta
# --------------------------------------------------------------------------

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
