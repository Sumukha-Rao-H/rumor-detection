"""P7-03 — scoring the alert log, and the third state that keeps it honest.

`test_an_alert_past_the_data_horizon_is_deferred` is the one that matters. The
obvious implementation records yes or no; that counts absence of data as
absence of an event, and would score a whole day of recent alerts as false
positives on no evidence at all.
"""

import pytest

from src import db
from src.live import Alert, append, backfill, data_horizon, hit_rates, unscored
from src.live.outcomes import first_filing_after, item_breakdown, window_seconds
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts

HOUR = 3600
BASE = date_str_to_ts("2026-08-10")


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def conn(tmp_path):
    c = db.get_conn(tmp_path / "outcomes.db")
    db.upsert_companies(c, [{"cik": "C1", "ticker": "AAA", "in_universe": 1},
                            {"cik": "C2", "ticker": "BBB", "in_universe": 1}])
    return c


def add_filing(conn, ticker, ts, accession, items="8.01", cik="C1"):
    db.upsert_filings(conn, [{"accession_no": accession, "cik": cik,
                              "ticker": ticker, "form": "8-K", "items": items,
                              "acceptance_utc": ts, "filing_date_utc": ts}])


def make(ticker="AAA", offset=0, detector="cusum"):
    return Alert(ts_utc=BASE + offset * HOUR, ticker=ticker, detector=detector,
                 score=3.0, threshold=1.0, features={"volume_z": 3.0})


# --------------------------------------------------------------------------
# The third state
# --------------------------------------------------------------------------
def test_an_alert_past_the_data_horizon_is_deferred(cfg, conn):
    """Not "checked and clean" — not answerable yet. An alert two hours before
    the end of the filing feed has not been tested against 48 hours of
    evidence, and calling it a miss would count missing data as a missing
    event."""
    add_filing(conn, "AAA", BASE + 1 * HOUR, "0001")      # horizon = BASE + 1h
    append(conn, [make(offset=0)])                        # window ends BASE+48h

    result = backfill(cfg, conn)
    assert result["pending"] == 1
    assert result["scored"] == 0
    assert len(unscored(conn)) == 1        # stays queued for next time


def test_a_deferred_alert_is_scored_once_the_data_catches_up(cfg, conn):
    """Self-healing: no manual retry, no state to reconcile."""
    add_filing(conn, "AAA", BASE + 1 * HOUR, "0001")
    append(conn, [make(offset=0)])
    assert backfill(cfg, conn)["pending"] == 1

    add_filing(conn, "BBB", BASE + 60 * HOUR, "0002", cik="C2")   # horizon moves
    result = backfill(cfg, conn)
    assert result["scored"] == 1
    assert unscored(conn) == []


def test_the_horizon_is_the_newest_filing_not_the_clock(cfg, conn):
    """However long ago an alert was raised, the answer depends on data that
    has been collected, not on time having passed."""
    assert data_horizon(conn) is None
    add_filing(conn, "AAA", BASE + 5 * HOUR, "0001")
    assert data_horizon(conn) == BASE + 5 * HOUR


def test_scoring_without_any_filings_is_refused(cfg, conn):
    append(conn, [make()])
    with pytest.raises(SystemExit, match="no filings stored"):
        backfill(cfg, conn)


# --------------------------------------------------------------------------
# Hits and misses
# --------------------------------------------------------------------------
def test_a_filing_inside_the_window_is_a_hit(cfg, conn):
    append(conn, [make(offset=0)])
    add_filing(conn, "AAA", BASE + 10 * HOUR, "0001", items="1.01")
    add_filing(conn, "BBB", BASE + 100 * HOUR, "0009", cik="C2")   # moves horizon

    result = backfill(cfg, conn)
    assert result["filed"] == 1
    row = conn.execute("SELECT * FROM alert_outcomes").fetchone()
    assert row["filed"] == 1
    assert row["accession_no"] == "0001"
    assert row["item_code"] == "1.01"
    assert row["t0_utc"] == BASE + 10 * HOUR


def test_a_filing_outside_the_window_is_a_miss(cfg, conn):
    append(conn, [make(offset=0)])
    add_filing(conn, "AAA", BASE + 60 * HOUR, "0001")     # past 48h
    add_filing(conn, "BBB", BASE + 200 * HOUR, "0009", cik="C2")

    backfill(cfg, conn)
    row = conn.execute("SELECT * FROM alert_outcomes").fetchone()
    assert row["filed"] == 0
    assert row["accession_no"] is None


def test_a_filing_by_another_company_does_not_count(cfg, conn):
    append(conn, [make(ticker="AAA", offset=0)])
    add_filing(conn, "BBB", BASE + 5 * HOUR, "0001", cik="C2")
    add_filing(conn, "BBB", BASE + 200 * HOUR, "0009", cik="C2")

    backfill(cfg, conn)
    assert conn.execute("SELECT filed FROM alert_outcomes").fetchone()[0] == 0


def test_a_filing_in_the_same_second_is_not_predicted_by_the_alert(cfg, conn):
    """Strictly after: a filing accepted the instant the alert fired was not
    anticipated by it."""
    append(conn, [make(offset=0)])
    add_filing(conn, "AAA", BASE, "0001")
    add_filing(conn, "BBB", BASE + 200 * HOUR, "0009", cik="C2")

    backfill(cfg, conn)
    assert conn.execute("SELECT filed FROM alert_outcomes").fetchone()[0] == 0


def test_the_earliest_qualifying_filing_wins(cfg, conn):
    """The one the alert would have been anticipating."""
    add_filing(conn, "AAA", BASE + 30 * HOUR, "0002", items="5.02")
    add_filing(conn, "AAA", BASE + 10 * HOUR, "0001", items="1.01")
    add_filing(conn, "BBB", BASE + 200 * HOUR, "0009", cik="C2")
    append(conn, [make(offset=0)])

    backfill(cfg, conn)
    row = conn.execute("SELECT * FROM alert_outcomes").fetchone()
    assert row["accession_no"] == "0001"


def test_the_corrected_t0_is_preferred_when_an_event_row_exists(cfg, conn):
    """min(acceptance, earliest matched news) — the same fallback sampling.py
    uses, so live and offline agree on what "when it became public" means."""
    add_filing(conn, "AAA", BASE + 10 * HOUR, "0001")
    db.upsert_events(conn, [{"event_id": "E1", "accession_no": "0001",
                             "ticker": "AAA", "items": "8.01",
                             "t0_filing_utc": BASE + 10 * HOUR,
                             "t0_utc": BASE + 8 * HOUR, "t0_source": "news",
                             "is_scheduled": 0, "usable": 1,
                             "exclude_reason": None}])
    add_filing(conn, "BBB", BASE + 200 * HOUR, "0009", cik="C2")
    append(conn, [make(offset=0)])

    backfill(cfg, conn)
    assert conn.execute("SELECT t0_utc FROM alert_outcomes").fetchone()[0] \
        == BASE + 8 * HOUR


# --------------------------------------------------------------------------
# The two clocks
# --------------------------------------------------------------------------
def test_the_outcome_window_is_wall_clock(cfg):
    """48 wall-clock hours, NOT decision.horizon_hours which is 48 BARS. A
    company can file overnight or at a weekend."""
    assert window_seconds(cfg) == 48 * 3600


def test_the_recorded_lead_is_in_trading_hours(cfg, conn):
    """Rule 3. A number that changed units between the offline table and the
    live log would be unreadable."""
    from src.utils.timeutils import trading_hours_between

    append(conn, [make(offset=0)])
    add_filing(conn, "AAA", BASE + 10 * HOUR, "0001")
    add_filing(conn, "BBB", BASE + 200 * HOUR, "0009", cik="C2")
    backfill(cfg, conn)

    lead = conn.execute("SELECT lead_trading_h FROM alert_outcomes").fetchone()[0]
    assert lead == pytest.approx(trading_hours_between(BASE, BASE + 10 * HOUR))
    assert lead < 10          # strictly fewer than the 10 wall-clock hours


# --------------------------------------------------------------------------
# The log stays immutable
# --------------------------------------------------------------------------
def test_scoring_does_not_touch_the_alert_log(cfg, conn):
    from src.live import verify_chain

    append(conn, [make(offset=i) for i in range(3)])
    before = [r["row_sha"] for r in
              conn.execute("SELECT row_sha FROM alerts ORDER BY seq")]
    add_filing(conn, "AAA", BASE + 200 * HOUR, "0009")
    backfill(cfg, conn)

    after = [r["row_sha"] for r in
             conn.execute("SELECT row_sha FROM alerts ORDER BY seq")]
    assert before == after
    assert verify_chain(conn)["cusum"]["ok"] is True


def test_rescoring_is_idempotent(cfg, conn):
    append(conn, [make(offset=0)])
    add_filing(conn, "AAA", BASE + 10 * HOUR, "0001")
    add_filing(conn, "BBB", BASE + 200 * HOUR, "0009", cik="C2")

    assert backfill(cfg, conn)["scored"] == 1
    assert backfill(cfg, conn)["scored"] == 0        # already answered
    assert conn.execute("SELECT COUNT(*) FROM alert_outcomes").fetchone()[0] == 1


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def test_hit_rates_report_pending_alongside_scored(cfg, conn):
    """So a reader can see how much of the log is still unanswerable rather
    than assuming the scored part is all of it."""
    add_filing(conn, "AAA", BASE + 10 * HOUR, "0001")
    add_filing(conn, "BBB", BASE + 200 * HOUR, "0009", cik="C2")
    append(conn, [make(offset=0), make(offset=190)])   # the second is pending

    backfill(cfg, conn)
    r = hit_rates(conn)["cusum"]
    assert r["scored"] == 1 and r["filed"] == 1
    assert r["hit_rate"] == pytest.approx(1.0)
    assert r["pending"] == 1


def test_item_breakdown_counts_what_was_caught(cfg, conn):
    add_filing(conn, "AAA", BASE + 5 * HOUR, "0001", items="1.01")
    add_filing(conn, "BBB", BASE + 5 * HOUR, "0002", items="5.02", cik="C2")
    add_filing(conn, "AAA", BASE + 300 * HOUR, "0009")
    append(conn, [make(ticker="AAA", offset=0), make(ticker="BBB", offset=0)])

    backfill(cfg, conn)
    items = item_breakdown(conn)
    assert items["1.01"] == 1 and items["5.02"] == 1
