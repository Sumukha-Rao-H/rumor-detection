"""P4-05 — how many real positives does the study actually have?

The count is the deliverable's denominator, and the plan calls a shortfall a
stop-and-tell rather than something to absorb quietly. The clustering tests
carry issue 28: Capricor filed twice in one day about the same news, and
counting that as two positives would inflate the total and let a model be
credited twice for the same detection.

The `cfg` fixture pins a fabricated `items:` block for the same reason as
tests/test_events_filtering.py: these tests must not change meaning when the
config is edited.
"""

import pytest

from src import db
from src.pipeline.events import census, distinct_announcements, print_census
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts


HOUR = 3600
DAY = 86400

FIXTURE_ITEMS = {
    "exclude": ["9.01", "5.07"],
    "scheduled": ["2.02"],
    "unscheduled_focus": ["1.01", "5.02", "8.01"],
}


@pytest.fixture
def cfg():
    cfg = load_config()
    cfg["items"] = {k: list(v) for k, v in FIXTURE_ITEMS.items()}
    return cfg


@pytest.fixture
def conn(tmp_path):
    return db.get_conn(tmp_path / "cen.db")


def seed(conn, event_id, ticker, t0, usable=1, is_scheduled=0, items="8.01",
         exclude_reason=None, form="8-K"):
    db.upsert_companies(conn, [{"cik": f"CIK{ticker}", "ticker": ticker,
                                "in_universe": 1}])
    db.upsert_filings(conn, [{
        "accession_no": event_id, "cik": f"CIK{ticker}", "ticker": ticker,
        "form": form, "items": items, "acceptance_utc": t0,
        "filing_date_utc": t0,
    }])
    db.upsert_events(conn, [{
        "event_id": event_id, "accession_no": event_id, "ticker": ticker,
        "items": items, "t0_filing_utc": t0, "t0_utc": t0,
        "t0_source": "filing", "is_scheduled": is_scheduled, "usable": usable,
        "exclude_reason": exclude_reason,
    }])


def base(cfg, days=100):
    return date_str_to_ts(cfg["study_window"]["start"]) + days * DAY


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


def test_the_gap_boundary_is_exclusive():
    """Exactly `gap` apart is one announcement. Undocumented until now."""
    t = 1_800_000_000
    assert distinct_announcements([("A", t), ("A", t + 6 * HOUR)], 6 * HOUR) == 1
    assert distinct_announcements([("A", t), ("A", t + 6 * HOUR + 1)],
                                  6 * HOUR) == 2


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


def test_unsorted_input_is_clustered_correctly():
    """The census passes rows in table order, not time order."""
    t = 1_800_000_000
    pairs = [("A", t + 2 * HOUR), ("A", t), ("A", t + HOUR)]
    assert distinct_announcements(pairs, 6 * HOUR) == 1


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
    assert c["reasons"]["immaterial"] == (1, 0, 1)   # total, sched, unsched


def test_the_exclusion_tally_is_split_scheduled_and_unscheduled(cfg, conn):
    """Rule 5: every number split, never pooled — the exclusions too.

    Pooled, you cannot see that immaterial drops land on one half.
    """
    seed(conn, "keep", "AAA", base(cfg))
    seed(conn, "e1", "BBB", base(cfg, 101), usable=0, is_scheduled=1,
         items="2.02", exclude_reason="immaterial")
    seed(conn, "e2", "CCC", base(cfg, 102), usable=0, is_scheduled=0,
         exclude_reason="immaterial")
    assert census(cfg, conn)["reasons"]["immaterial"] == (2, 1, 1)


def test_the_funnel_adds_up(cfg, conn):
    """events == usable + excluded + not-yet-filtered."""
    seed(conn, "e1", "AAA", base(cfg))
    seed(conn, "e2", "BBB", base(cfg, 101), usable=0,
         exclude_reason="immaterial")
    seed(conn, "e3", "CCC", base(cfg, 102), usable=0)     # not filtered yet
    c = census(cfg, conn)
    excluded = sum(n for n, _s, _u in c["reasons"].values())
    assert c["events"] == c["usable"] + excluded + c["pending"]
    assert c["pending"] == 1


def test_census_splits_scheduled_and_unscheduled(cfg, conn):
    seed(conn, "e1", "AAA", base(cfg), is_scheduled=1, items="2.02")
    seed(conn, "e2", "BBB", base(cfg, 110), is_scheduled=0)
    c = census(cfg, conn)
    assert c["scheduled"] == 1 and c["unscheduled"] == 1


def test_census_ignores_excluded_events_in_the_item_tally(cfg, conn):
    seed(conn, "e1", "AAA", base(cfg), items="8.01")
    seed(conn, "e2", "BBB", base(cfg, 110), items="1.01", usable=0,
         exclude_reason="immaterial")
    assert dict(census(cfg, conn)["top_items"]) == {"8.01": 1}


def test_excluded_codes_are_not_counted_in_the_tally(cfg, conn):
    """9.01 rides along on a usable event but is not what happened."""
    seed(conn, "e1", "AAA", base(cfg), items="8.01,9.01")
    assert dict(census(cfg, conn)["top_items"]) == {"8.01": 1}


def test_codes_in_no_config_list_are_tallied_separately(cfg, conn):
    """Kept and counted as unscheduled — but the count is visible."""
    seed(conn, "e1", "AAA", base(cfg), items="5.03,8.01")
    c = census(cfg, conn)
    assert dict(c["unknown_items"]) == {"5.03": 1}


def test_an_event_outside_the_window_is_not_a_positive(cfg, conn):
    """A stale row left behind by a shortened window used to be counted.

    The window really was shortened once (2024-09-01 -> 2025-09-01), and the
    `months` divisor comes from config, so a stale row inflates both the
    positive count and the per-stock-month rate the alert budget is judged on.
    """
    seed(conn, "e1", "AAA", base(cfg))
    seed(conn, "old", "BBB", base(cfg) - 900 * DAY)
    c = census(cfg, conn)
    assert c["events"] == 2 and c["usable"] == 1 and c["out_of_window"] == 1


def test_amendments_are_counted_and_named(cfg, conn):
    """8-K/A positives are separate rows from the filing they amend."""
    seed(conn, "e1", "AAA", base(cfg))
    seed(conn, "e2", "AAA", base(cfg, 110), form="8-K/A")
    assert dict(census(cfg, conn)["forms"]) == {"8-K": 1, "8-K/A": 1}


def test_events_per_stock_per_month_uses_the_configured_window(cfg, conn):
    """Derived from study_window, so it stays correct if the window moves."""
    for i in range(12):
        seed(conn, f"e{i}", "AAA", base(cfg, 100 + i * 2))
    c = census(cfg, conn)
    assert c["tickers"] == 1
    assert c["per_stock_month"] == pytest.approx(12 / c["months"])
    assert 10 < c["months"] < 12          # the 2025-09-01 -> 2026-08-01 window


def test_the_clustering_gaps_come_from_config(cfg, conn):
    """Rule 7: thresholds live in config, not in the report's source."""
    seed(conn, "e1", "AAA", base(cfg))
    cfg["census"]["cluster_gap_hours"] = [1, 12]
    assert sorted(census(cfg, conn)["announcements"]) == [1, 12]


# --------------------------------------------------------------------------
# the zero cases — a stop-and-tell, not a traceback
# --------------------------------------------------------------------------

def test_zero_usable_events_stops_loudly(cfg, conn):
    """This is the module's own acceptance check; it used to divide by zero."""
    seed(conn, "e1", "AAA", base(cfg), usable=0,
         exclude_reason="only_excluded_items")
    with pytest.raises(SystemExit, match="0 of 1 events are usable"):
        census(cfg, conn)
    with pytest.raises(SystemExit, match="stop-and-tell"):
        print_census(cfg, conn)


def test_no_events_at_all_stops_loudly(cfg, conn):
    with pytest.raises(SystemExit, match="no events"):
        census(cfg, conn)


def test_every_usable_event_outside_the_window_stops_loudly(cfg, conn):
    """Zero in-window positives is the same stop-and-tell, not a 0.0% report."""
    seed(conn, "old", "AAA", base(cfg) - 900 * DAY)
    with pytest.raises(SystemExit, match="usable"):
        census(cfg, conn)


def test_the_census_prints_the_funnel_and_the_verdict(cfg, conn, capsys):
    """The printed census was never exercised by a test until now."""
    for i in range(3):
        seed(conn, f"s{i}", f"T{i}", base(cfg, 100 + i), is_scheduled=1,
             items="2.02")
    seed(conn, "u1", "TU", base(cfg, 120), items="5.02")
    seed(conn, "x1", "TX", base(cfg, 130), usable=0, is_scheduled=1,
         items="2.02", exclude_reason="immaterial")
    seed(conn, "a1", "TA", base(cfg, 140), form="8-K/A", items="5.02")

    print_census(cfg, conn)
    out = capsys.readouterr().out
    assert "events built            : 6" in out
    assert "excluded immaterial" in out and "(scheduled 1 / unscheduled 0)" in out
    assert "USABLE POSITIVES" in out and "5" in out
    assert "scheduled   : 3" in out and "unscheduled : 2" in out
    assert "BELOW the estimate" in out          # 5 positives, plan wants 3,000+
    assert "8-K/A 1" in out                     # amendments named
    assert "collapsing events within" in out
