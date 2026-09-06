"""P7-04 fix — the live monitor must advance its own filing horizon.

`test_without_fresh_filings_every_alert_stays_deferred_forever` is the reason
this module exists. It reproduces the bug as it stood: the monitor fetched
price bars but never new filings, so `backfill`'s horizon never moved, and
every alert it raised sat unanswerable for ever. The monitor would have run for
weeks, raised hundreds of alerts, and learned nothing from any of them.
"""

import pytest

from src import db
from src.live import Alert, append, backfill, data_horizon
from src.live.monitor import fetch_recent_filings
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts

HOUR = 3600
BASE = date_str_to_ts("2026-08-10")


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def conn(tmp_path):
    c = db.get_conn(tmp_path / "filings.db")
    db.upsert_companies(c, [{"cik": "0000000001", "ticker": "AAA",
                             "in_universe": 1}])
    db.upsert_filings(c, [{"accession_no": "0001", "cik": "0000000001",
                           "ticker": "AAA", "form": "8-K", "items": "8.01",
                           "acceptance_utc": BASE, "filing_date_utc": BASE}])
    return c


class FakeClient:
    """Returns one submissions payload, and records whether the cache was
    bypassed — the single detail the whole fix depends on."""

    def __init__(self, records):
        self.records = records
        self.forced = []

    def submissions_url(self, cik):
        return f"https://example/{cik}.json"

    def submissions_page_url(self, name):
        return f"https://example/{name}"

    def get_json(self, url, force=False):
        self.forced.append(force)
        return {"filings": {"recent": self.records, "files": []}}


def payload(accessions, acceptance_ts, form="8-K", items="8.01"):
    n = len(accessions)
    return {"accessionNumber": list(accessions),
            "form": [form] * n,
            "items": [items] * n,
            "acceptanceDateTime": [
                __import__("src.utils.timeutils", fromlist=["ts_to_iso"])
                .ts_to_iso(t) for t in acceptance_ts],
            "filingDate": ["2026-08-12"] * n,
            "reportDate": ["2026-08-12"] * n,
            "primaryDocument": ["a.htm"] * n}


# --------------------------------------------------------------------------
# The bug this fixes
# --------------------------------------------------------------------------
def test_without_fresh_filings_every_alert_stays_deferred_forever(cfg, conn):
    """The state P7-04 shipped in. The horizon never moves, so an alert raised
    after it can never be scored — not now, and not in a month either."""
    append(conn, [Alert(ts_utc=BASE + 10 * HOUR, ticker="AAA",
                        detector="cusum", score=3.0, threshold=1.0,
                        features={})])

    first = backfill(cfg, conn)
    assert first["pending"] == 1 and first["scored"] == 0

    # Time passes, more bars arrive, the job runs again — and nothing changes,
    # because only bars were ever fetched.
    again = backfill(cfg, conn)
    assert again["pending"] == 1 and again["scored"] == 0
    assert data_horizon(conn) == BASE          # frozen


def test_fetching_filings_advances_the_horizon_and_unblocks_scoring(cfg, conn):
    """The fix. One fetch moves the horizon past the alert's window, and the
    same alert that was stuck now resolves."""
    append(conn, [Alert(ts_utc=BASE + 10 * HOUR, ticker="AAA",
                        detector="cusum", score=3.0, threshold=1.0,
                        features={})])
    assert backfill(cfg, conn)["pending"] == 1

    client = FakeClient(payload(["0002"], [BASE + 200 * HOUR]))
    result = fetch_recent_filings(cfg, conn, tickers=["AAA"], client=client,
                                  now_ts=BASE + 300 * HOUR)

    assert result["new_filings"] == 1
    assert data_horizon(conn) == BASE + 200 * HOUR
    assert backfill(cfg, conn)["scored"] == 1


# --------------------------------------------------------------------------
# The three details it depends on
# --------------------------------------------------------------------------
def test_the_cache_is_bypassed(cfg, conn):
    """EdgarClient is cache-first by design, which is right for a fixed
    historical window and fatal here: the cached submissions file would come
    back unchanged and no new filing would ever be seen."""
    client = FakeClient(payload(["0002"], [BASE + 100 * HOUR]))
    fetch_recent_filings(cfg, conn, tickers=["AAA"], client=client,
                         now_ts=BASE + 200 * HOUR)
    assert client.forced and all(client.forced), \
        "every EDGAR read in the live path must bypass the cache"


def test_the_window_starts_before_the_horizon(cfg, conn):
    """Deliberate overlap: acceptance times are not strictly ordered against
    visibility, and amendments arrive late. Re-reading is free; missing one at
    the boundary silently costs an outcome."""
    client = FakeClient(payload(["0002"], [BASE + 100 * HOUR]))
    result = fetch_recent_filings(cfg, conn, tickers=["AAA"], client=client,
                                  now_ts=BASE + 200 * HOUR)
    overlap = cfg["live"]["filing_overlap_hours"] * HOUR
    assert result["since_utc"] == BASE - overlap


def test_refetching_the_same_filing_is_idempotent(cfg, conn):
    """The overlap re-reads filings already held. That must cost nothing."""
    client = FakeClient(payload(["0002"], [BASE + 100 * HOUR]))
    a = fetch_recent_filings(cfg, conn, tickers=["AAA"], client=client,
                             now_ts=BASE + 200 * HOUR)
    b = fetch_recent_filings(cfg, conn, tickers=["AAA"], client=client,
                             now_ts=BASE + 200 * HOUR)
    assert a["new_filings"] == 1
    assert b["new_filings"] == 0


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------
def test_zero_records_across_every_company_is_refused(cfg, conn):
    """A company with no new 8-K is ordinary; nothing at all from any company
    means the endpoint is broken and the run must not report success."""
    class Empty(FakeClient):
        def get_json(self, url, force=False):
            return {"filings": {"recent": {}, "files": []}}

    with pytest.raises(SystemExit, match="zero records"):
        fetch_recent_filings(cfg, conn, tickers=["AAA"], client=Empty({}),
                             now_ts=BASE + 200 * HOUR)


def test_one_company_failing_does_not_stop_the_rest(cfg, conn):
    """One 404 must not cost every other company, exactly as the Phase 2
    collector reasons about it."""
    db.upsert_companies(conn, [{"cik": "0000000002", "ticker": "BBB",
                                "in_universe": 1}])

    class Flaky(FakeClient):
        def get_json(self, url, force=False):
            if "0000000001" in url:
                raise RuntimeError("404")
            return {"filings": {"recent": self.records, "files": []}}

    result = fetch_recent_filings(
        cfg, conn, tickers=["AAA", "BBB"],
        client=Flaky(payload(["0003"], [BASE + 100 * HOUR])),
        now_ts=BASE + 200 * HOUR)
    assert result["failed"] == 1
    assert result["new_filings"] == 1


def test_fetching_with_no_history_is_refused(cfg, tmp_path):
    """This appends to a history; it does not build one."""
    empty = db.get_conn(tmp_path / "empty.db")
    db.upsert_companies(empty, [{"cik": "0000000001", "ticker": "AAA",
                                 "in_universe": 1}])
    with pytest.raises(SystemExit, match="no filings stored"):
        fetch_recent_filings(cfg, empty, tickers=["AAA"])
