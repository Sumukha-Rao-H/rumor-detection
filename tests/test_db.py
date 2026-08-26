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
    row = ("https://reuters.com/a", "TSLA", "Tesla acquires X", "reuters.com",
           None, 1, 1_750_000_000, "finnhub")
    assert db.upsert_news(conn, [row]) == 1
    assert db.upsert_news(conn, [row]) == 0


def test_earliest_news_ts_drives_the_t0_correction(conn):
    """Tiers decide what counts as "the news is public".

    A stock-screener blog picks it up first, then a republisher, then the wire
    itself. Which timestamp becomes t0 depends entirely on how far down the
    tiers you are willing to look — which is why Phase 4 reports both.
    """
    db.upsert_news(conn, [
        ("https://smallblog.example/a", "AAPL", "chatter", "smallblog.example",
         None, None, 1_750_000_000, "gdelt"),            # untiered
        ("https://finnhub.io/api/news?id=b", "AAPL", "pickup", None,
         "Benzinga", 2, 1_750_000_500, "finnhub"),        # tier 2
        ("https://businesswire.com/c", "AAPL", "press release", "businesswire.com",
         None, 1, 1_750_001_000, "gdelt"),                # tier 1
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
    db.upsert_news(conn, [
        ("https://x/1", "MSFT", "t", None, "ChartMill", None, 1_750_000_000, "finnhub"),
    ])
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

    db.upsert_news(conn, [
        ("https://y/1", "MSFT", "t", "reuters.com", None, 1, 1_750_000_000, "gdelt"),
    ])
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
