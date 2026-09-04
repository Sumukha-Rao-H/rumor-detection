"""P7-02 — the append-only log, and three ways of proving it is one.

The tests that matter here deliberately tamper with the log and demand the
verifier notices. A log that merely *says* it is append-only is worth nothing;
one whose edits are detectable in a single command is evidence.
"""

import json

import pytest

from src import db
from src.live import Alert, alert_id, append, summary, unscored, verify_chain
from src.utils.timeutils import date_str_to_ts

HOUR = 3600
BASE = date_str_to_ts("2026-08-10")


@pytest.fixture
def conn(tmp_path):
    return db.get_conn(tmp_path / "alerts.db")


def make(detector="cusum", ticker="AAA", offset=0, score=3.0):
    return Alert(ts_utc=BASE + offset * HOUR, ticker=ticker,
                 detector=detector, score=score, threshold=1.0,
                 features={"volume_z": score, "volatility": 0.02})


# --------------------------------------------------------------------------
# Append-only
# --------------------------------------------------------------------------
def test_alerts_are_written_with_everything_needed_to_audit_them(conn):
    """Timestamp, score, and the feature values that triggered it — the task's
    three requirements."""
    append(conn, [make()], raised_utc=BASE + 10)
    row = conn.execute("SELECT * FROM alerts").fetchone()

    assert row["ts_utc"] == BASE
    assert row["ticker"] == "AAA"
    assert row["detector"] == "cusum"
    assert row["score"] == pytest.approx(3.0)
    assert row["threshold"] == pytest.approx(1.0)
    assert row["raised_utc"] == BASE + 10
    assert json.loads(row["features"])["volume_z"] == pytest.approx(3.0)


def test_the_first_write_wins(conn):
    """Everywhere else in this project the LAST write wins, which is what makes
    the collectors idempotent. Here it must be the first, or a detector whose
    threshold was retuned could quietly restate its own history."""
    append(conn, [make(score=3.0)])
    append(conn, [make(score=99.0)])          # same bar, same detector

    rows = conn.execute("SELECT score FROM alerts").fetchall()
    assert len(rows) == 1
    assert rows[0]["score"] == pytest.approx(3.0)


def test_append_reports_only_new_rows(conn):
    """So a caller can tell "nothing fired" from "everything fired again"."""
    alerts = [make(offset=i) for i in range(3)]
    assert append(conn, alerts) == 3
    assert append(conn, alerts) == 0


def test_different_detectors_on_the_same_bar_are_separate_alerts(conn):
    append(conn, [make(detector="cusum"), make(detector="volume_zscore")])
    assert conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0] == 2


def test_ids_are_deterministic(conn):
    """Re-running the monitor must produce the same id, which is what lets the
    insert be a no-op rather than a duplicate."""
    assert alert_id("cusum", "AAA", BASE) == alert_id("cusum", "AAA", BASE)
    assert alert_id("cusum", "AAA", BASE) != alert_id("cusum", "AAA", BASE + 1)


# --------------------------------------------------------------------------
# The chain catches tampering
# --------------------------------------------------------------------------
def test_a_clean_log_verifies(conn):
    append(conn, [make(offset=i) for i in range(5)])
    report = verify_chain(conn)
    assert report["cusum"]["ok"] is True
    assert report["cusum"]["rows"] == 5


def test_editing_a_row_is_detected(conn):
    """The realistic failure: someone improves a score after seeing the
    outcome."""
    append(conn, [make(offset=i) for i in range(5)])
    conn.execute("UPDATE alerts SET score = 99.0 WHERE seq = 2")
    conn.commit()

    report = verify_chain(conn)["cusum"]
    assert report["ok"] is False
    assert report["broken_at"] == 2
    assert "edited" in report["reason"]


def test_deleting_a_row_is_detected(conn):
    """An inconvenient alert quietly removed."""
    append(conn, [make(offset=i) for i in range(5)])
    conn.execute("DELETE FROM alerts WHERE seq = 2")
    conn.commit()

    report = verify_chain(conn)["cusum"]
    assert report["ok"] is False
    assert "removed" in report["reason"] or "seq gap" in report["reason"]


def test_editing_the_features_is_detected(conn):
    """The features are chained too — an alert whose stated cause could be
    rewritten would not be auditable."""
    append(conn, [make(offset=i) for i in range(3)])
    conn.execute("UPDATE alerts SET features = ? WHERE seq = 1",
                 (json.dumps({"volume_z": 42.0}),))
    conn.commit()
    assert verify_chain(conn)["cusum"]["ok"] is False


def test_one_detectors_tampering_does_not_implicate_another(conn):
    """Chains are per detector, so a break is localised rather than condemning
    the whole log."""
    append(conn, [make(detector="cusum", offset=i) for i in range(3)])
    append(conn, [make(detector="volume_zscore", offset=i) for i in range(3)])
    conn.execute("UPDATE alerts SET score = 99.0 "
                 "WHERE detector = 'cusum' AND seq = 1")
    conn.commit()

    report = verify_chain(conn)
    assert report["cusum"]["ok"] is False
    assert report["volume_zscore"]["ok"] is True


def test_verification_survives_an_empty_log(conn):
    assert verify_chain(conn) == {}


# --------------------------------------------------------------------------
# Outcomes live elsewhere
# --------------------------------------------------------------------------
def test_outcomes_are_a_separate_table_so_the_log_stays_immutable(conn):
    """Recording what happened must not mean updating what was said."""
    append(conn, [make()])
    row = conn.execute("SELECT alert_id, row_sha FROM alerts").fetchone()

    conn.execute(
        "INSERT INTO alert_outcomes (alert_id, checked_utc, filed, "
        "accession_no, item_code, t0_utc, lead_trading_h) "
        "VALUES (?,?,?,?,?,?,?)",
        (row["alert_id"], BASE + 99, 1, "0001-25-000001", "8.01",
         BASE + 20 * HOUR, 12.5))
    conn.commit()

    after = conn.execute("SELECT row_sha FROM alerts").fetchone()
    assert after["row_sha"] == row["row_sha"]      # the log did not move
    assert verify_chain(conn)["cusum"]["ok"] is True


def test_unscored_is_the_backfill_queue(conn):
    append(conn, [make(offset=i) for i in range(3)])
    assert len(unscored(conn)) == 3

    first = conn.execute("SELECT alert_id FROM alerts ORDER BY seq").fetchone()
    conn.execute("INSERT INTO alert_outcomes (alert_id, checked_utc, filed) "
                 "VALUES (?,?,?)", (first["alert_id"], BASE, 0))
    conn.commit()
    assert len(unscored(conn)) == 2


def test_summary_counts_per_detector(conn):
    append(conn, [make(detector="cusum", offset=i) for i in range(4)])
    append(conn, [make(detector="volume_zscore", offset=0)])
    s = summary(conn)
    assert s["cusum"]["alerts"] == 4
    assert s["volume_zscore"]["alerts"] == 1
    assert s["cusum"]["first_utc"] == BASE
