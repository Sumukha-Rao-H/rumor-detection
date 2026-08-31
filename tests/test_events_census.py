"""P4-05 — how many real positives does the study actually have?

The count is the deliverable's denominator, and the plan calls a shortfall a
stop-and-tell rather than something to absorb quietly. The clustering tests
carry issue 28: Capricor filed twice in one day about the same news, and
counting that as two positives would inflate the total and let a model be
credited twice for the same detection.
"""

import pytest

from src import db
from src.pipeline.events import census, distinct_announcements
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts


HOUR = 3600


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def conn(tmp_path):
    return db.get_conn(tmp_path / "cen.db")


def seed(conn, event_id, ticker, t0, usable=1, is_scheduled=0, items="8.01",
         exclude_reason=None):
    db.upsert_companies(conn, [{"cik": f"CIK{ticker}", "ticker": ticker,
                                "in_universe": 1}])
    db.upsert_filings(conn, [{
        "accession_no": event_id, "cik": f"CIK{ticker}", "ticker": ticker,
        "form": "8-K", "items": items, "acceptance_utc": t0,
        "filing_date_utc": t0,
    }])
    db.upsert_events(conn, [{
        "event_id": event_id, "accession_no": event_id, "ticker": ticker,
        "items": items, "t0_filing_utc": t0, "t0_utc": t0,
        "t0_source": "filing", "is_scheduled": is_scheduled, "usable": usable,
        "exclude_reason": exclude_reason,
    }])


def base(cfg, days=100):
    return date_str_to_ts(cfg["study_window"]["start"]) + days * 86400


# --------------------------------------------------------------------------
# clustering — issue 28
# --------------------------------------------------------------------------

def test_two_events_within_the_gap_are_one_announcement():
    """The Capricor case: two 8-Ks hours apart about the same news."""
    t = 1_800_000_000
    pairs = [("CAPR", t), ("CAPR", t + 2 * HOUR)]
    assert distinct_announcements(pairs, 6 * HOUR) == 1


def test_events_far_apart_stay_separate():
    t = 1_800_000_000
    pairs = [("CAPR", t), ("CAPR", t + 20 * HOUR)]
    assert distinct_announcements(pairs, 6 * HOUR) == 2


def test_three_consecutive_events_collapse_to_one():
    """Clusters are transitive along consecutive events, not pairwise.

    Three filings an hour apart are one announcement, not two.
    """
    t = 1_800_000_000
    pairs = [("A", t), ("A", t + HOUR), ("A", t + 2 * HOUR)]
    assert distinct_announcements(pairs, 6 * HOUR) == 1


def test_a_chain_longer_than_the_gap_still_collapses():
    """Each step is inside the gap even though the ends are far apart.

    Recorded because it is a genuine judgement: the alternative would split the
    chain, and neither is obviously right. Transitive is chosen because a
    rolling story is still one announcement.
    """
    t = 1_800_000_000
    pairs = [("A", t + i * 5 * HOUR) for i in range(4)]   # spans 15h
    assert distinct_announcements(pairs, 6 * HOUR) == 1


def test_clusters_do_not_span_tickers():
    t = 1_800_000_000
    pairs = [("A", t), ("B", t + HOUR)]
    assert distinct_announcements(pairs, 6 * HOUR) == 2


def test_simultaneous_events_are_one_announcement():
    t = 1_800_000_000
    assert distinct_announcements([("A", t), ("A", t)], 6 * HOUR) == 1


def test_wider_gaps_never_increase_the_count():
    """A sanity property: collapsing more can only ever collapse more."""
    t = 1_800_000_000
    pairs = [("A", t + i * 7 * HOUR) for i in range(6)]
    counts = [distinct_announcements(pairs, h * HOUR) for h in (6, 24, 72)]
    assert counts == sorted(counts, reverse=True)


def test_no_events_is_zero_announcements():
    assert distinct_announcements([], 6 * HOUR) == 0


# --------------------------------------------------------------------------
# the funnel
# --------------------------------------------------------------------------

def test_census_counts_the_funnel(cfg, conn):
    seed(conn, "e1", "AAA", base(cfg))
    seed(conn, "e2", "BBB", base(cfg, 110), usable=0,
         exclude_reason="immaterial")
    c = census(cfg, conn)
    assert c["events"] == 2 and c["usable"] == 1
    assert c["reasons"]["immaterial"] == 1


def test_census_splits_scheduled_and_unscheduled(cfg, conn):
    seed(conn, "e1", "AAA", base(cfg), is_scheduled=1, items="2.02")
    seed(conn, "e2", "BBB", base(cfg, 110), is_scheduled=0)
    c = census(cfg, conn)
    assert c["scheduled"] == 1 and c["usable"] - c["scheduled"] == 1


def test_census_ignores_excluded_events_in_the_item_tally(cfg, conn):
    seed(conn, "e1", "AAA", base(cfg), items="8.01")
    seed(conn, "e2", "BBB", base(cfg, 110), items="1.01", usable=0,
         exclude_reason="immaterial")
    assert dict(census(cfg, conn)["top_items"]) == {"8.01": 1}


def test_excluded_codes_are_not_counted_in_the_tally(cfg, conn):
    """9.01 rides along on a usable event but is not what happened."""
    seed(conn, "e1", "AAA", base(cfg), items="8.01,9.01")
    assert dict(census(cfg, conn)["top_items"]) == {"8.01": 1}


def test_events_per_stock_per_month_uses_the_configured_window(cfg, conn):
    """Derived from study_window, so it stays correct if the window moves."""
    for i in range(12):
        seed(conn, f"e{i}", "AAA", base(cfg, 100 + i * 2))
    c = census(cfg, conn)
    assert c["tickers"] == 1
    assert c["per_stock_month"] == pytest.approx(12 / c["months"])
    assert 10 < c["months"] < 12          # the 2025-09-01 -> 2026-08-01 window
