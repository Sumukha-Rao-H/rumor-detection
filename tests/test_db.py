"""Storage-layer tests: schema, idempotent upserts, t0 lookup, provenance."""

import pytest

from src import db


@pytest.fixture()
def conn(tmp_path):
    c = db.get_conn(tmp_path / "test.db")
    yield c
    c.close()


def make_company(cik="0000320193", **over):
    row = {
        "cik": cik, "ticker": "AAPL", "name": "Apple Inc.", "exchange": "Nasdaq",
        "sic": "3571", "in_universe": 1, "adv_usd": 1e10, "last_price": 200.0,
        "universe_as_of": 1_725_148_800,
    }
    row.update(over)
    return row


def make_filing(acc="0000320193-26-000073", **over):
    row = {
        "accession_no": acc, "cik": "0000320193", "ticker": "AAPL", "form": "8-K",
        "items": "2.02,9.01", "acceptance_utc": 1_753_907_428,
        "filing_date_utc": 1_753_833_600, "report_date_utc": 1_753_833_600,
        "primary_doc": "aapl-20260730.htm", "fetched_utc": 1_753_910_000,
    }
    row.update(over)
    return row


def make_event(eid="000032019326000073", **over):
    row = {
        "event_id": eid, "accession_no": "0000320193-26-000073", "ticker": "AAPL",
        "items": "2.02", "t0_filing_utc": 1_753_907_428, "t0_news_utc": None,
        "t0_utc": 1_753_907_428, "t0_source": "filing", "is_scheduled": 1,
        "abs_return": 0.05, "is_material": 1, "usable": 1, "exclude_reason": None,
    }
    row.update(over)
    return row


def test_companies_upsert_and_universe(conn):
    assert db.upsert_companies(conn, [make_company()]) == 1
    assert db.upsert_companies(conn, [make_company(last_price=222.0)]) == 0
    row = conn.execute("SELECT last_price, name FROM companies").fetchone()
    assert row["last_price"] == 222.0
    assert row["name"] == "Apple Inc."  # not clobbered by the refresh

    db.upsert_companies(conn, [make_company(cik="0000789019", ticker="MSFT",
                                            name="Microsoft", in_universe=0)])
    assert db.universe_tickers(conn) == ["AAPL"]
    assert db.company_name(conn, "AAPL") == "Apple Inc."
    assert db.company_name(conn, "NOPE") is None


def test_a_refresh_with_no_liquidity_opinion_keeps_in_universe(conn):
    """Regression guard (P2-02).

    `upsert_companies` used to write `in_universe = excluded.in_universe` with
    no COALESCE. The universe build has no opinion about liquidity and passes
    NULL, so rebuilding the map after the Phase 3 filter had run would empty
    every flag — and `market.py --universe` would then download nothing while
    reporting success.
    """
    db.upsert_companies(conn, [make_company(in_universe=1)])
    db.upsert_companies(conn, [make_company(in_universe=None)])
    assert db.universe_tickers(conn) == ["AAPL"]

    # An explicit 0 still means what it says.
    db.upsert_companies(conn, [make_company(in_universe=0)])
    assert db.universe_tickers(conn) == []


def test_filings_are_immutable_once_stored(conn):
    assert db.upsert_filings(conn, [make_filing()]) == 1
    assert db.upsert_filings(conn, [make_filing(items="1.01")]) == 0
    row = conn.execute("SELECT items FROM filings").fetchone()
    assert row["items"] == "2.02,9.01"  # EDGAR data never rewritten
    assert db.latest_filing_ts(conn, "0000320193") == 1_753_907_428
    assert db.latest_filing_ts(conn, "0000000000") is None


def test_events_recompute_derived_columns_on_rerun(conn):
    db.upsert_filings(conn, [make_filing()])
    assert db.upsert_events(conn, [make_event()]) == 1
    # Re-running the builder after a config change must update, not duplicate.
    assert db.upsert_events(conn, [make_event(usable=0,
                                              exclude_reason="below materiality")]) == 0
    row = conn.execute("SELECT usable, exclude_reason FROM events").fetchone()
    assert row["usable"] == 0 and row["exclude_reason"] == "below materiality"


def test_usable_events_split_by_scheduled(conn):
    db.upsert_filings(conn, [make_filing(), make_filing(acc="0000320193-26-000074")])
    db.upsert_events(conn, [
        make_event(),
        make_event(eid="000032019326000074",
                   accession_no="0000320193-26-000074",
                   items="1.01", is_scheduled=0),
    ])
    assert len(db.usable_events(conn)) == 2
    assert [r["items"] for r in db.usable_events(conn, scheduled=0)] == ["1.01"]
    assert [r["items"] for r in db.usable_events(conn, scheduled=1)] == ["2.02"]


def test_bars_upsert_and_latest(conn):
    rows = [("TSLA", 1_750_000_000, 1, 2, 0.5, 1.5, 1000, "60m"),
            ("TSLA", 1_750_003_600, 1.5, 2, 1, 1.8, 900, "60m")]
    db.upsert_bars(conn, rows)
    db.upsert_bars(conn, rows)  # idempotent
    assert conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0] == 2
    assert db.latest_bar_ts(conn, "TSLA", "60m") == 1_750_003_600
    assert db.latest_bar_ts(conn, "AAPL", "60m") is None


def test_news_dedupe_by_url(conn):
    row = {"url": "https://reuters.com/a", "ticker": "TSLA",
           "title": "Tesla acquires X", "source_domain": "reuters.com",
           "source_name": None, "source_tier": 1,
           "published_utc": 1_750_000_000, "seen_utc": None,
           "fetched_utc": 1_750_000_100, "api": "finnhub"}
    assert db.upsert_news(conn, [row]) == 1
    assert db.upsert_news(conn, [row]) == 0


def test_earliest_news_ts_drives_the_t0_correction(conn):
    """Tiers decide what counts as "the news is public".

    A stock-screener blog picks it up first, then a republisher, then the wire
    itself. Which timestamp becomes t0 depends entirely on how far down the
    tiers you are willing to look — which is why Phase 4 reports both.
    """
    def row(url, tier, published, **kw):
        base = {"url": url, "ticker": "AAPL", "title": "t",
                "source_domain": None, "source_name": None, "source_tier": tier,
                "published_utc": published, "seen_utc": None,
                "fetched_utc": 0, "api": "finnhub"}
        base.update(kw)
        return base

    db.upsert_news(conn, [
        row("https://smallblog.example/a", None, 1_750_000_000),   # untiered
        row("https://finnhub.io/api/news?id=b", 2, 1_750_000_500),  # tier 2
        row("https://businesswire.com/c", 1, 1_750_001_000),        # tier 1
    ])
    lo, hi = 1_749_900_000, 1_750_010_000

    # tier 1 only — the release itself
    assert db.earliest_news_ts(conn, "AAPL", lo, hi, max_tier=1) == 1_750_001_000
    # tier 1+2 (default) — a republisher counts, so t0 moves earlier
    assert db.earliest_news_ts(conn, "AAPL", lo, hi) == 1_750_000_500
    # anything at all — the blog wins, which is why this is not the default
    assert db.earliest_news_ts(conn, "AAPL", lo, hi, max_tier=None) == 1_750_000_000

    assert db.earliest_news_ts(conn, "TSLA", lo, hi) is None


def test_retier_after_a_whitelist_change(conn):
    """The escape hatch for stale stored tiers.

    Stored tiers go stale the moment the whitelist is tuned. This asserts the
    fix actually moves rows rather than being an untested promise.
    """
    from src.utils.config import load_config

    cfg = load_config()
    db.upsert_news(conn, [{
        "url": "https://x/1", "ticker": "MSFT", "title": "t",
        "source_domain": None, "source_name": "ChartMill", "source_tier": None,
        "published_utc": 1_750_000_000, "seen_utc": None, "fetched_utc": 0,
        "api": "finnhub"}])
    assert conn.execute(
        "SELECT source_tier FROM news WHERE url='https://x/1'").fetchone()[0] is None

    cfg["news"]["whitelist_tier2"] = list(cfg["news"]["whitelist_tier2"]) + ["ChartMill"]
    assert db.retier_news(conn, cfg) == 1
    assert conn.execute(
        "SELECT source_tier FROM news WHERE url='https://x/1'").fetchone()[0] == 2

    # and back again — retiering is not one-way
    cfg["news"]["whitelist_tier2"] = [
        e for e in cfg["news"]["whitelist_tier2"] if e != "ChartMill"]
    assert db.retier_news(conn, cfg) == 1
    assert conn.execute(
        "SELECT source_tier FROM news WHERE url='https://x/1'").fetchone()[0] is None


def test_retier_is_a_noop_when_nothing_changed(conn):
    from src.utils.config import load_config

    db.upsert_news(conn, [{
        "url": "https://y/1", "ticker": "MSFT", "title": "t",
        "source_domain": "reuters.com", "source_name": None, "source_tier": 1,
        "published_utc": None, "seen_utc": 1_750_000_000, "fetched_utc": 0,
        "api": "gdelt"}])
    cfg = load_config()
    db.retier_news(conn, cfg)
    assert db.retier_news(conn, cfg) == 0


def test_meta_records_snapshot_provenance(conn):
    assert db.get_meta(conn, "snapshot_frozen_60m") is None
    db.set_meta(conn, "snapshot_frozen_60m", "2026-09-01", 1_756_684_800)
    db.set_meta(conn, "snapshot_frozen_60m", "2026-09-02", 1_756_771_200)
    assert db.get_meta(conn, "snapshot_frozen_60m") == "2026-09-02"


def test_migration_adds_source_name_to_an_old_database(tmp_path):
    """`CREATE TABLE IF NOT EXISTS` leaves an existing database alone, so a new
    column has to be added explicitly or every dev silently keeps an old
    schema. Simulates a pre-P1-13 database and checks the upgrade path.
    """
    import sqlite3

    path = tmp_path / "old.db"
    raw = sqlite3.connect(path)
    raw.executescript("""
        CREATE TABLE news (
          url TEXT PRIMARY KEY, ticker TEXT, title TEXT, source_domain TEXT,
          seen_utc INTEGER, api TEXT
        );
        INSERT INTO news VALUES ('u1','TSLA','old row','finnhub.io',1,'finnhub');
    """)
    raw.commit()
    raw.close()

    conn = db.get_conn(path)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(news)")}
    assert "source_name" in cols
    assert conn.execute("SELECT COUNT(*) FROM news").fetchone()[0] == 1, \
        "the migration must not lose existing rows"
    assert conn.execute("SELECT source_name FROM news").fetchone()[0] is None
    conn.close()


def test_migration_is_idempotent(tmp_path):
    """get_conn runs it on every connect."""
    path = tmp_path / "m.db"
    for _ in range(3):
        conn = db.get_conn(path)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(news)")]
        assert cols.count("source_name") == 1
        conn.close()


def _news(url, **kw):
    base = {"url": url, "ticker": "AAPL", "title": "t", "source_domain": None,
            "source_name": None, "source_tier": 1, "published_utc": None,
            "seen_utc": None, "fetched_utc": 0, "api": "finnhub"}
    base.update(kw)
    return base


def test_t0_uses_publication_time_not_crawl_time(conn):
    """The point of P1-14.

    A GDELT row crawled at 12:00 and a Finnhub row published at 13:00. The crawl
    happened earlier on the clock, but it says nothing about when its article was
    published — so by default t0 must come from the publication time it can
    actually trust.
    """
    db.upsert_news(conn, [
        _news("https://g/1", api="gdelt", seen_utc=1_750_000_000,
              source_domain="reuters.com"),
        _news("https://f/1", published_utc=1_750_003_600, source_name="CNBC",
              source_tier=2),
    ])
    lo, hi = 1_749_000_000, 1_751_000_000

    # default: only rows whose publication time is known
    assert db.earliest_news_ts(conn, "AAPL", lo, hi, max_tier=2) == 1_750_003_600

    # opting in to crawl time reaches the earlier GDELT row
    assert db.earliest_news_ts(conn, "AAPL", lo, hi, max_tier=2,
                               allow_crawl_time=True) == 1_750_000_000


def test_crawl_time_fallback_can_only_push_t0_later_or_equal(conn):
    """Using crawl time is conservative: it understates lead time rather than
    overstating it. Safe direction for a claim, but a different measurement."""
    db.upsert_news(conn, [
        _news("https://f/2", published_utc=1_750_005_000, source_tier=1),
        _news("https://g/2", api="gdelt", seen_utc=1_750_009_000,
              source_domain="reuters.com"),
    ])
    lo, hi = 1_749_000_000, 1_751_000_000
    strict = db.earliest_news_ts(conn, "AAPL", lo, hi)
    loose = db.earliest_news_ts(conn, "AAPL", lo, hi, allow_crawl_time=True)
    assert loose <= strict


def test_rows_with_no_publication_time_are_invisible_by_default(conn):
    db.upsert_news(conn, [
        _news("https://g/3", api="gdelt", seen_utc=1_750_000_000,
              source_domain="reuters.com"),
    ])
    lo, hi = 1_749_000_000, 1_751_000_000
    assert db.earliest_news_ts(conn, "AAPL", lo, hi) is None
    assert db.earliest_news_ts(conn, "AAPL", lo, hi, allow_crawl_time=True) is not None


def test_refetch_backfills_missing_fields(conn):
    """Fixes issue #14.

    upsert_news used INSERT OR IGNORE, so a row collected before a collector fix
    could never be repaired by re-running — the re-fetch simply skipped it. It
    now fills NULLs while leaving existing values alone.
    """
    db.upsert_news(conn, [_news("https://x/9", source_name=None,
                                source_tier=None, published_utc=None)])
    assert db.upsert_news(conn, [_news("https://x/9", source_name="CNBC",
                                       source_tier=2, published_utc=123)]) == 0
    row = conn.execute(
        "SELECT source_name, source_tier, published_utc FROM news "
        "WHERE url='https://x/9'").fetchone()
    assert (row["source_name"], row["source_tier"], row["published_utc"]) == (
        "CNBC", 2, 123)


def test_refetch_does_not_overwrite_existing_values(conn):
    db.upsert_news(conn, [_news("https://x/10", title="original",
                                published_utc=100)])
    db.upsert_news(conn, [_news("https://x/10", title="rewritten",
                                published_utc=999)])
    row = conn.execute(
        "SELECT title, published_utc FROM news WHERE url='https://x/10'").fetchone()
    assert row["published_utc"] == 100, "an existing timestamp must not move"


def test_data_migration_moves_legacy_publication_times(tmp_path):
    """Before P1-14 the news table had one timestamp column and the Finnhub
    collector wrote PUBLICATION time into it. `seen_utc` now means crawl time,
    so those values sit in a column that means something else — worse than
    missing, because they read as an upper bound rather than an exact time.
    Their old meaning is known exactly, so they are moved, not discarded.
    """
    import sqlite3

    path = tmp_path / "legacy.db"
    raw = sqlite3.connect(path)
    raw.executescript("""
        CREATE TABLE news (
          url TEXT PRIMARY KEY, ticker TEXT, title TEXT, source_domain TEXT,
          seen_utc INTEGER, api TEXT
        );
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT, updated_utc INTEGER);
        INSERT INTO news VALUES ('f1','TSLA','a','finnhub.io',1700000000,'finnhub');
        INSERT INTO news VALUES ('g1','TSLA','b','reuters.com',1700000500,'gdelt');
    """)
    raw.commit(); raw.close()

    conn = db.get_conn(path)
    finn = conn.execute("SELECT * FROM news WHERE url='f1'").fetchone()
    gdelt = conn.execute("SELECT * FROM news WHERE url='g1'").fetchone()

    assert finn["published_utc"] == 1700000000, "moved to the column that means it"
    assert finn["seen_utc"] is None, "Finnhub reports no crawl time"
    # GDELT's value really was a crawl time — it stays where it is
    assert gdelt["seen_utc"] == 1700000500
    assert gdelt["published_utc"] is None
    conn.close()


def test_data_migrations_run_once(tmp_path):
    """Guarded by a key in `meta`, so reconnecting does not re-run them."""
    path = tmp_path / "once.db"
    conn = db.get_conn(path)
    db.upsert_news(conn, [_news("https://z/1", api="finnhub",
                                published_utc=500, seen_utc=None)])
    conn.close()

    conn = db.get_conn(path)          # reconnect: migrations must be no-ops
    row = conn.execute("SELECT published_utc, seen_utc FROM news").fetchone()
    assert (row["published_utc"], row["seen_utc"]) == (500, None)
    keys = [r[0] for r in conn.execute(
        "SELECT key FROM meta WHERE key LIKE 'migration:%'")]
    assert len(keys) == len(set(keys)) == len(db.DATA_MIGRATIONS)
    conn.close()
