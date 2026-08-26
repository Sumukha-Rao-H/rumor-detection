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
           1_750_000_000, "finnhub")
    assert db.upsert_news(conn, [row]) == 1
    assert db.upsert_news(conn, [row]) == 0


def test_earliest_news_ts_drives_the_t0_correction(conn):
    db.upsert_news(conn, [
        # A blog picks it up first, then the wire, then the 8-K is accepted.
        ("https://smallblog.example/a", "AAPL", "chatter", "smallblog.example",
         1_750_000_000, "gdelt"),
        ("https://businesswire.com/b", "AAPL", "press release", "businesswire.com",
         1_750_001_000, "finnhub"),
    ])
    lo, hi = 1_749_900_000, 1_750_010_000
    assert db.earliest_news_ts(conn, "AAPL", lo, hi) == 1_750_000_000
    # Restricting to credible domains skips the blog.
    assert db.earliest_news_ts(
        conn, "AAPL", lo, hi, domains={"businesswire.com"}) == 1_750_001_000
    assert db.earliest_news_ts(conn, "AAPL", lo, hi, domains={"reuters.com"}) is None
    assert db.earliest_news_ts(conn, "TSLA", lo, hi) is None


def test_meta_records_snapshot_provenance(conn):
    assert db.get_meta(conn, "snapshot_frozen_60m") is None
    db.set_meta(conn, "snapshot_frozen_60m", "2026-09-01", 1_756_684_800)
    db.set_meta(conn, "snapshot_frozen_60m", "2026-09-02", 1_756_771_200)
    assert db.get_meta(conn, "snapshot_frozen_60m") == "2026-09-02"
