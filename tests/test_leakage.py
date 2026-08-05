"""Split leakage tests (plan §6.4 acceptance criteria).

The whole project rests on the agent not having seen the future. These assert
the two ways a split can hand it the answer anyway: overlapping time ranges,
and one ticker's events straddling a boundary.
"""

import json

import pytest

from src import db
from src.pipeline.dataset import assign_splits, describe, export, labeled_events

DAY = 86400
T0 = 1_700_000_000


def _cfg(quarantine_days=7):
    return {"split": {"train": 0.70, "val": 0.15, "test": 0.15,
                      "method": "temporal", "quarantine_days": quarantine_days,
                      "min_events": 300}}


def _event(conn, event_id, ticker, t0, label=1, reviewed=1):
    db.upsert_events(conn, [{
        "event_id": event_id, "ticker": ticker, "t0_utc": t0,
        "claim_summary": "c", "claim_type": "merger",
        "post_ids": json.dumps(["p"]), "n_posts": 1,
        "subreddits": json.dumps(["stocks"]),
    }])
    conn.execute("""UPDATE events SET label = ?, human_reviewed = ?,
                    n_rumor_posts = 1 WHERE event_id = ?""",
                 (label, reviewed, event_id))
    conn.commit()


def _spread(conn, n, ticker_prefix="T", start=T0, step=10 * DAY):
    """n events, one per ticker, evenly spaced in time."""
    for i in range(n):
        _event(conn, f"E{i:03d}", f"{ticker_prefix}{i:03d}", start + i * step,
               label=i % 2)


# --- the plan's stated criterion --------------------------------------------

def test_train_ends_before_val_begins(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    _spread(conn, 100)
    frame = assign_splits(labeled_events(conn), _cfg())
    train, val = frame[frame.split == "train"], frame[frame.split == "val"]
    assert train.t0_utc.max() < val.t0_utc.min()


def test_val_ends_before_test_begins(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    _spread(conn, 100)
    frame = assign_splits(labeled_events(conn), _cfg())
    val, test = frame[frame.split == "val"], frame[frame.split == "test"]
    assert val.t0_utc.max() < test.t0_utc.min()


def test_every_event_lands_in_exactly_one_split(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    _spread(conn, 60)
    frame = assign_splits(labeled_events(conn), _cfg())
    assert set(frame.split) <= {"train", "val", "test"}
    assert frame.event_id.is_unique


def test_the_splits_respect_the_configured_proportions(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    _spread(conn, 100)
    frame = assign_splits(labeled_events(conn), _cfg(quarantine_days=0))
    counts = frame.split.value_counts()
    assert counts["train"] == 70 and counts["val"] == 15 and counts["test"] == 15


# --- ticker quarantine ------------------------------------------------------

def test_a_ticker_never_straddles_a_boundary(tmp_path):
    """One company's January rumor must not teach its February one."""
    conn = db.get_conn(tmp_path / "t.db")
    _spread(conn, 100)
    # Put two AAA events either side of the train/val cut, a day apart.
    cut = T0 + 70 * 10 * DAY
    _event(conn, "AAA-early", "AAA", cut - DAY)
    _event(conn, "AAA-late", "AAA", cut + DAY)
    frame = assign_splits(labeled_events(conn), _cfg())
    for _, group in frame.groupby("ticker"):
        assert group.split.nunique() == 1


def test_straddling_events_are_dropped_not_reassigned(tmp_path):
    """Moving them relocates the leak; only removal ends it."""
    conn = db.get_conn(tmp_path / "t.db")
    _spread(conn, 100)
    cut = T0 + 70 * 10 * DAY
    _event(conn, "AAA-early", "AAA", cut - DAY)
    _event(conn, "AAA-late", "AAA", cut + DAY)
    frame = assign_splits(labeled_events(conn), _cfg())
    assert "AAA" not in set(frame.ticker)


def test_a_ticker_far_from_the_boundary_survives(tmp_path):
    """Quarantine must not delete a company merely for recurring."""
    conn = db.get_conn(tmp_path / "t.db")
    _spread(conn, 100)
    _event(conn, "BBB-1", "BBB", T0 + DAY)
    _event(conn, "BBB-2", "BBB", T0 + 3 * DAY)
    frame = assign_splits(labeled_events(conn), _cfg())
    assert set(frame[frame.ticker == "BBB"].split) == {"train"}


def test_quarantine_window_is_configurable(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    _spread(conn, 100)
    cut = T0 + 70 * 10 * DAY
    _event(conn, "AAA-early", "AAA", cut - 5 * DAY)
    _event(conn, "AAA-late", "AAA", cut + 5 * DAY)
    assert "AAA" not in set(assign_splits(labeled_events(conn), _cfg(7)).ticker)
    assert "AAA" in set(assign_splits(labeled_events(conn), _cfg(1)).ticker)


# --- what may be exported at all --------------------------------------------

def test_unreviewed_events_never_reach_the_dataset(tmp_path):
    """A machine proposal is not a label; this is the last line of defence."""
    conn = db.get_conn(tmp_path / "t.db")
    _event(conn, "REVIEWED", "AAA", T0, reviewed=1)
    _event(conn, "MACHINE", "BBB", T0 + DAY, reviewed=0)
    assert list(labeled_events(conn).event_id) == ["REVIEWED"]


def test_unlabeled_events_never_reach_the_dataset(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    _event(conn, "LABELED", "AAA", T0, label=1)
    _event(conn, "UNVERIFIED", "BBB", T0 + DAY, label=None)
    assert list(labeled_events(conn).event_id) == ["LABELED"]


def test_an_empty_dataset_does_not_crash_the_export(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    frame, stats = export(conn, _cfg(), path=None)
    assert frame.empty and stats["rows"] == 0


def test_export_writes_parquet_that_round_trips(tmp_path):
    import pandas as pd

    conn = db.get_conn(tmp_path / "t.db")
    _spread(conn, 40)
    path = tmp_path / "events.parquet"
    frame, _ = export(conn, _cfg(), path=path)
    assert list(pd.read_parquet(path).event_id) == list(frame.event_id)


def test_stats_report_class_balance_per_split(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    _spread(conn, 100)
    stats = describe(assign_splits(labeled_events(conn), _cfg()))
    assert 0 <= stats["minority_share"] <= 0.5
    assert stats["train"]["rows"] and stats["test"]["rows"]


def test_a_random_split_is_refused(tmp_path):
    conn = db.get_conn(tmp_path / "t.db")
    _spread(conn, 20)
    cfg = _cfg()
    cfg["split"]["method"] = "random"
    with pytest.raises(SystemExit):
        assign_splits(labeled_events(conn), cfg)
