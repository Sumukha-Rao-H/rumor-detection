"""Resumability — an interrupted 6,000-company run must continue, not restart.

The Done-when is `test_interrupted_run_then_resume_matches_an_uninterrupted_run`:
kill a run midway, resume, and land on exactly the row count a clean run
produces. Everything else here protects a detail that would make that true in a
test and false in practice — state committed per company rather than at the
end, written after the filings rather than before, and Ctrl-C not swallowed.
"""

import pytest

from src import db
from src.collectors.edgar import EdgarRequestError, collect_many
from src.utils.config import load_config


APPLE_8K = {
    "accessionNumber": "0000320193-26-000018", "form": "8-K",
    "items": "2.02,9.01", "acceptanceDateTime": "2026-07-30T20:30:28.000Z",
    "filingDate": "2026-07-30", "reportDate": "2026-07-30",
    "primaryDocument": "aapl-20260730.htm",
}
FORM_4 = {"accessionNumber": "f4-1", "form": "4", "items": "",
          "acceptanceDateTime": "2026-01-02T20:00:00.000Z",
          "filingDate": "2026-01-02", "reportDate": "2026-01-02",
          "primaryDocument": "f.xml"}

CIKS = ["0000019617", "0000320193", "0000789019", "0001045810", "0001318605"]
TICKERS = ["JPM", "AAPL", "MSFT", "NVDA", "TSLA"]


def submissions(*records) -> dict:
    fields = sorted({f for r in records for f in r})
    return {"filings": {
        "recent": {f: [r.get(f, "") for r in records] for f in fields},
        "files": [],
    }}


def payload_for(cik: str) -> dict:
    """Two 8-Ks per company, with accession numbers unique to that company."""
    return submissions(
        {**APPLE_8K, "accessionNumber": f"{cik}-a"},
        {**APPLE_8K, "accessionNumber": f"{cik}-b"},
    )


class FakeClient:
    """Serves per-CIK payloads; can be told to blow up on the Nth call."""

    def __init__(self, by_cik, fail_after=None, error=None):
        self.by_cik = by_cik
        self.fail_after = fail_after
        self.error = error or KeyboardInterrupt("^C")
        self.calls: list[str] = []

    def submissions_url(self, cik) -> str:
        return f"https://data.sec.gov/submissions/CIK{cik}.json"

    def submissions_page_url(self, name) -> str:
        return f"https://data.sec.gov/submissions/{name}"

    def get_json(self, url, force=False):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise self.error
        cik = url.rsplit("CIK", 1)[-1].removesuffix(".json")
        self.calls.append(cik)
        result = self.by_cik[cik]
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


def fresh_db(tmp_path, name):
    conn = db.get_conn(tmp_path / name)
    db.upsert_companies(conn, [
        {"cik": c, "ticker": t, "name": t, "exchange": "NYSE"}
        for c, t in zip(CIKS, TICKERS)
    ])
    return conn


def count(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM filings").fetchone()[0]


# -- the Done-when ---------------------------------------------------------

def test_interrupted_run_then_resume_matches_an_uninterrupted_run(cfg, tmp_path):
    payloads = {cik: payload_for(cik) for cik in CIKS}

    clean = fresh_db(tmp_path, "clean.db")
    collect_many(cfg, clean, client=FakeClient(payloads))
    expected = count(clean)
    assert expected == 10, "five companies, two filings each"

    killed = fresh_db(tmp_path, "killed.db")
    with pytest.raises(KeyboardInterrupt):
        collect_many(cfg, killed, client=FakeClient(payloads, fail_after=2))
    partial = count(killed)
    assert 0 < partial < expected, "the kill has to land mid-run to prove anything"

    resumed = FakeClient(payloads)
    collect_many(cfg, killed, client=resumed, resume=True)

    assert count(killed) == expected
    assert len(resumed.calls) == len(CIKS) - 2, (
        "resume must skip exactly the companies already finished")

    clean_rows = {r[0] for r in clean.execute("SELECT accession_no FROM filings")}
    killed_rows = {r[0] for r in killed.execute("SELECT accession_no FROM filings")}
    assert clean_rows == killed_rows
    clean.close(); killed.close()


# -- the details that make it true in practice -----------------------------

def test_state_is_committed_per_company_not_at_the_end(cfg, tmp_path):
    """State buffered to the end of a run is worthless — the point is the kill."""
    conn = fresh_db(tmp_path, "s.db")
    payloads = {cik: payload_for(cik) for cik in CIKS}
    with pytest.raises(KeyboardInterrupt):
        collect_many(cfg, conn, client=FakeClient(payloads, fail_after=3))
    assert len(db.completed_keys(conn, "edgar")) == 3
    conn.close()


def test_filings_are_written_before_state(cfg, tmp_path):
    """A crash between the two must re-fetch a company, never skip one.

    Marking a company done whose rows never landed loses filings silently;
    re-fetching one is free, because every write is an idempotent upsert.
    """
    conn = fresh_db(tmp_path, "order.db")
    payloads = {cik: payload_for(cik) for cik in CIKS}
    with pytest.raises(KeyboardInterrupt):
        collect_many(cfg, conn, client=FakeClient(payloads, fail_after=2))
    done = db.completed_keys(conn, "edgar")
    stored = {r[0] for r in conn.execute("SELECT DISTINCT cik FROM filings")}
    assert done <= stored, "no CIK may be marked done without its rows present"
    conn.close()


def test_keyboard_interrupt_stops_the_run(cfg, tmp_path):
    """`except Exception` does not catch it — true today, easy to break later."""
    conn = fresh_db(tmp_path, "kb.db")
    payloads = {cik: payload_for(cik) for cik in CIKS}
    client = FakeClient(payloads, fail_after=1)
    with pytest.raises(KeyboardInterrupt):
        collect_many(cfg, conn, client=client)
    assert len(client.calls) == 1, "Ctrl-C must not be swallowed and carried on"
    conn.close()


# -- what state records ----------------------------------------------------

def test_state_records_why_a_company_has_no_rows(cfg, tmp_path):
    conn = fresh_db(tmp_path, "why.db")
    payloads = {c: payload_for(c) for c in CIKS}
    payloads["0000789019"] = submissions(FORM_4)          # quiet, but fine
    payloads["0001045810"] = EdgarRequestError("HTTP 404 — not retried")
    collect_many(cfg, conn, client=FakeClient(payloads))

    rows = {r["key"]: r for r in conn.execute("SELECT * FROM fetch_state")}
    quiet, broken = rows["0000789019"], rows["0001045810"]
    assert (quiet["status"], quiet["rows_written"]) == ("ok", 0)
    assert quiet["records"] == 1, "it was fetched; it simply files no 8-Ks"
    assert broken["status"] == "failed"
    assert "404" in broken["error"]
    conn.close()


def test_resume_retries_a_company_that_failed(cfg, tmp_path):
    """A 503 or a dropped connection is worth another go — that is why you resume."""
    conn = fresh_db(tmp_path, "retry.db")
    payloads = {c: payload_for(c) for c in CIKS}
    payloads["0001045810"] = EdgarRequestError("HTTP 503 from EDGAR")
    collect_many(cfg, conn, client=FakeClient(payloads))

    payloads["0001045810"] = payload_for("0001045810")    # SEC is back
    second = FakeClient(payloads)
    collect_many(cfg, conn, client=second, resume=True)
    assert second.calls == ["0001045810"]
    assert db.completed_keys(conn, "edgar") == set(CIKS)
    conn.close()


def test_state_upsert_is_idempotent(tmp_path):
    conn = db.get_conn(tmp_path / "idem.db")
    for _ in range(3):
        db.set_fetch_state(conn, "edgar", "0000320193", "ok", records=5,
                           rows_written=2)
    assert conn.execute("SELECT COUNT(*) FROM fetch_state").fetchone()[0] == 1
    db.set_fetch_state(conn, "edgar", "0000320193", "failed", error="boom")
    row = conn.execute("SELECT status, error FROM fetch_state").fetchone()
    assert (row["status"], row["error"]) == ("failed", "boom")
    conn.close()


def test_resume_ignores_state_from_another_source(tmp_path):
    """The `source` column is what lets Phase 3's price download reuse this."""
    conn = db.get_conn(tmp_path / "src.db")
    db.set_fetch_state(conn, "market", "0000320193", "ok")
    assert db.completed_keys(conn, "edgar") == set()
    assert db.completed_keys(conn, "market") == {"0000320193"}
    conn.close()
