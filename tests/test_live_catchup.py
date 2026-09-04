"""P7-04 — the unattended cycle, and the recovery path it depends on.

The scheduled job has no operator. That makes two things load-bearing: the
whole cycle must be one command with one exit code, and the alert log must
survive the database being thrown away, because in CI it will be.
"""

import pytest

from src import db
from src.live import Alert, append, export_csv, import_csv, verify_chain
from src.live.catchup import run
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts

HOUR = 3600
BASE = date_str_to_ts("2026-08-10")


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def conn(tmp_path, cfg):
    c = db.get_conn(tmp_path / "catchup.db")
    iv = cfg["market"]["interval"]
    start = date_str_to_ts("2025-09-01")
    for ticker in ("AAA", cfg["market"]["benchmark"]):
        db.upsert_bars(c, [
            (ticker, start + i * HOUR, 100.0, 101.0, 99.0,
             100.0 + (i % 7) * 0.1, 1_000_000 + (i % 13) * 1000, iv)
            for i in range(900)])
    db.upsert_companies(c, [{"cik": "C1", "ticker": "AAA", "in_universe": 1}])
    db.upsert_filings(c, [{"accession_no": "0001", "cik": "C1",
                           "ticker": "AAA", "form": "8-K", "items": "8.01",
                           "acceptance_utc": start + 899 * HOUR,
                           "filing_date_utc": start + 899 * HOUR}])
    return c


def make(offset=0, detector="cusum"):
    return Alert(ts_utc=BASE + offset * HOUR, ticker="AAA", detector=detector,
                 score=3.0, threshold=1.0, features={"volume_z": 3.0})


# --------------------------------------------------------------------------
# One command, one exit code
# --------------------------------------------------------------------------
def test_the_whole_cycle_runs_in_one_call(cfg, conn, tmp_path):
    """A four-step shell pipeline fails in four ways and three are silent."""
    out = tmp_path / "alerts.csv"
    result = run(cfg, conn, fetch=False, log_csv=str(out))

    for key in ("bars_scored", "alerts_found", "alerts_new", "outcomes",
                "chain", "log", "elapsed_s"):
        assert key in result
    assert out.exists()


def test_it_verifies_the_chain_every_run(cfg, conn, tmp_path):
    """A break found weeks later is a break nobody can date."""
    append(conn, [make(offset=i) for i in range(3)])
    result = run(cfg, conn, fetch=False, log_csv=str(tmp_path / "a.csv"))
    assert result["chain"]["cusum"] is True


def test_a_broken_chain_is_reported_not_swallowed(cfg, conn, tmp_path):
    append(conn, [make(offset=i) for i in range(3)])
    conn.execute("UPDATE alerts SET score = 99.0 WHERE seq = 1")
    conn.commit()

    result = run(cfg, conn, fetch=False, log_csv=str(tmp_path / "a.csv"))
    assert result["chain"]["cusum"] is False


def test_outcomes_are_backfilled_after_alerts_are_logged(cfg, conn, tmp_path):
    """Ordering matters: an alert raised today and a filing that lands in the
    same run must both be accounted for."""
    result = run(cfg, conn, fetch=False, log_csv=str(tmp_path / "a.csv"))
    assert "outcomes" in result
    assert set(result["outcomes"]) >= {"scored", "filed", "pending"}


def test_no_fetch_downloads_nothing(cfg, conn, tmp_path):
    result = run(cfg, conn, fetch=False, log_csv=str(tmp_path / "a.csv"))
    assert result["bars_appended"] == 0


# --------------------------------------------------------------------------
# The log survives losing the database
# --------------------------------------------------------------------------
def test_the_log_round_trips_through_csv(cfg, conn, tmp_path, monkeypatch):
    """The CI cache is evictable and the database is disposable. This is the
    path that makes that safe."""
    append(conn, [make(offset=i) for i in range(5)])
    out = tmp_path / "alerts.csv"
    assert export_csv(conn, out) == 5

    fresh = db.get_conn(tmp_path / "rebuilt.db")
    assert import_csv(fresh, out) == 5
    assert fresh.execute("SELECT COUNT(*) FROM alerts").fetchone()[0] == 5


def test_a_restored_log_still_verifies(cfg, conn, tmp_path):
    """Hashes are restored as written, never recomputed — recomputing would
    make any corruption verify perfectly, which is the one thing the chain
    exists to prevent."""
    append(conn, [make(offset=i) for i in range(4)])
    out = tmp_path / "alerts.csv"
    export_csv(conn, out)

    fresh = db.get_conn(tmp_path / "rebuilt.db")
    import_csv(fresh, out)
    assert verify_chain(fresh)["cusum"]["ok"] is True


def test_a_corrupted_csv_fails_verification_after_import(cfg, conn, tmp_path):
    """Proves the point above: tampering with the file is caught on restore,
    not laundered by it."""
    append(conn, [make(offset=i) for i in range(4)])
    out = tmp_path / "alerts.csv"
    export_csv(conn, out)
    text = out.read_text().replace("3.0", "99.0", 1)
    out.write_text(text)

    fresh = db.get_conn(tmp_path / "rebuilt.db")
    import_csv(fresh, out)
    assert verify_chain(fresh)["cusum"]["ok"] is False


def test_importing_twice_adds_nothing(cfg, conn, tmp_path):
    """Every scheduled run imports the committed log; it must be a no-op when
    the cached database already holds those rows."""
    append(conn, [make(offset=i) for i in range(3)])
    out = tmp_path / "alerts.csv"
    export_csv(conn, out)

    fresh = db.get_conn(tmp_path / "rebuilt.db")
    assert import_csv(fresh, out) == 3
    assert import_csv(fresh, out) == 0


def test_the_export_is_byte_stable(cfg, conn, tmp_path):
    """A committed file that churned on every run would bury real changes in
    noise and make the git history useless as a record."""
    append(conn, [make(offset=i) for i in range(4)])
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    export_csv(conn, a)
    export_csv(conn, b)
    assert a.read_bytes() == b.read_bytes()
