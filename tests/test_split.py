"""P4-14 — the temporal split, and the seal that makes a peek an error.

The seal is the point of this task. A boundary nobody enforces is a comment;
these tests exist to prove `assert_not_test` actually stops test data reaching
train/val work, and that opening it is deliberate and recorded.
"""

import pytest

from src import db
from src.pipeline.split import (
    LIVE, META_SEALED, TEST, TRAIN, VAL, assert_not_test, boundaries, counts,
    is_sealed, seal, split_of, unseal,
)
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts

HOUR = 3600


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def conn(tmp_path):
    return db.get_conn(tmp_path / "split.db")


def test_boundaries_partition_the_window_with_no_gap_or_overlap(cfg):
    lo = date_str_to_ts(cfg["study_window"]["start"])
    hi = date_str_to_ts(cfg["study_window"]["end"])
    train_end, val_end = boundaries(cfg)

    assert lo < train_end < val_end < hi
    # Half-open and contiguous: the instant a period ends is the instant the
    # next begins, so no hour is in two splits and none is in none.
    assert split_of(cfg, train_end - 1) == TRAIN
    assert split_of(cfg, train_end) == VAL
    assert split_of(cfg, val_end - 1) == VAL
    assert split_of(cfg, val_end) == TEST


def test_the_test_period_is_the_last_15_percent_by_date(cfg):
    lo = date_str_to_ts(cfg["study_window"]["start"])
    hi = date_str_to_ts(cfg["study_window"]["end"])
    _, val_end = boundaries(cfg)
    share = (hi - val_end) / (hi - lo)
    assert share == pytest.approx(cfg["split"]["test"], abs=0.01)


def test_fractions_that_do_not_sum_to_one_are_refused(cfg):
    bad = {**cfg, "split": {**cfg["split"], "train": 0.70, "val": 0.15,
                            "test": 0.30}}
    with pytest.raises(SystemExit, match="must sum to 1.0"):
        boundaries(bad)


def test_a_fresh_database_is_sealed_by_default(conn):
    """Fail closed. A DB that was never sealed, or lost its meta, must not
    behave as though the test set were open."""
    assert is_sealed(conn) is True


def test_the_guard_raises_on_a_test_timestamp_and_names_the_boundary(cfg, conn):
    seal(cfg, conn)
    _, val_end = boundaries(cfg)

    with pytest.raises(SystemExit, match="SEALED TEST SET"):
        assert_not_test(cfg, conn, [val_end], context="a baseline")

    # The message has to be actionable, not just angry.
    try:
        assert_not_test(cfg, conn, [val_end + 5 * HOUR], context="a baseline")
    except SystemExit as e:
        assert "a baseline" in str(e)
        assert "--unseal" in str(e)


def test_the_guard_passes_train_and_val_timestamps(cfg, conn):
    seal(cfg, conn)
    train_end, val_end = boundaries(cfg)
    lo = date_str_to_ts(cfg["study_window"]["start"])
    assert_not_test(cfg, conn, [lo, train_end - 1, train_end, val_end - 1])


def test_the_guard_accepts_a_bare_timestamp_not_only_an_iterable(cfg, conn):
    seal(cfg, conn)
    _, val_end = boundaries(cfg)
    assert_not_test(cfg, conn, val_end - 1)
    with pytest.raises(SystemExit, match="SEALED TEST SET"):
        assert_not_test(cfg, conn, val_end)


def test_unsealing_requires_a_reason_and_is_recorded(cfg, conn):
    seal(cfg, conn)
    with pytest.raises(SystemExit, match="needs a reason"):
        unseal(cfg, conn, "   ")

    unseal(cfg, conn, "final evaluation, Phase 10")
    assert is_sealed(conn) is False
    _, val_end = boundaries(cfg)
    assert_not_test(cfg, conn, [val_end])          # now permitted
    assert db.get_meta(conn, "split:unseal_reason") == "final evaluation, Phase 10"


def test_seal_stores_the_boundary_so_a_later_reader_resolves_the_same_split(cfg, conn):
    """The boundary is stamped, not recomputed. A config edit after sealing
    must not silently move the wall the study was already evaluated against."""
    seal(cfg, conn)
    train_end, val_end = boundaries(cfg)
    assert db.get_meta(conn, "split:train_end_utc") == str(train_end)
    assert db.get_meta(conn, "split:val_end_utc") == str(val_end)
    assert db.get_meta(conn, META_SEALED) == "1"


def test_counts_report_every_split_and_lose_no_events(cfg, conn):
    """Events are reported per period, never optimised — but they must all be
    accounted for, or the split is dropping data silently."""
    lo = date_str_to_ts(cfg["study_window"]["start"])
    train_end, val_end = boundaries(cfg)
    iv = cfg["market"]["interval"]
    db.upsert_companies(conn, [{"cik": "C1", "ticker": "AAA",
                                "in_universe": 1}])
    placed = {TRAIN: lo + HOUR, VAL: train_end + HOUR, TEST: val_end + HOUR}
    for i, (name, ts) in enumerate(placed.items()):
        db.upsert_bars(conn, [("AAA", ts, 0, 0, 0, 100.0, 1e6, iv)])
        db.upsert_filings(conn, [{
            "accession_no": f"A-{i}", "cik": "C1", "ticker": "AAA",
            "form": "8-K", "items": "8.01", "acceptance_utc": ts,
            "filing_date_utc": ts}])
        db.upsert_events(conn, [{
            "event_id": f"E{i}", "accession_no": f"A-{i}", "ticker": "AAA",
            "items": "8.01", "t0_filing_utc": ts, "t0_utc": ts,
            "t0_source": "filing", "is_scheduled": 0, "usable": 1,
            "exclude_reason": None}])

    c = counts(cfg, conn)
    assert c[TRAIN]["events"] == 1
    assert c[VAL]["events"] == 1
    assert c[TEST]["events"] == 1
    assert sum(c[n]["events"] for n in (TRAIN, VAL, TEST)) == 3


# --- the seal is bounded ABOVE too (P7-01) --------------------------------


def test_data_after_the_study_window_is_live_not_test(cfg):
    """`split_of` used to return TEST for everything at or after val_end,
    unbounded, so today's bars counted as sealed test data. `boundaries()`
    already documented the test period as bounded — "test is [val_end, end]" —
    and the live monitor needs that to be true in the code as well."""
    hi = date_str_to_ts(cfg["study_window"]["end"])
    assert split_of(cfg, hi - HOUR) == TEST
    assert split_of(cfg, hi) == LIVE
    assert split_of(cfg, hi + 30 * 24 * HOUR) == LIVE


def test_the_guard_still_refuses_the_whole_test_period(cfg, conn):
    """Narrowing the upper bound must not open the actual test set."""
    seal(cfg, conn)
    _, val_end = boundaries(cfg)
    hi = date_str_to_ts(cfg["study_window"]["end"])

    with pytest.raises(SystemExit, match="SEALED TEST SET"):
        assert_not_test(cfg, conn, [val_end])
    with pytest.raises(SystemExit, match="SEALED TEST SET"):
        assert_not_test(cfg, conn, [hi - HOUR])


def test_the_guard_permits_data_after_the_study_window(cfg, conn):
    """What P7-01 needs: the live monitor scores bars that postdate the study
    entirely, and refusing them would protect nothing."""
    seal(cfg, conn)
    hi = date_str_to_ts(cfg["study_window"]["end"])
    assert_not_test(cfg, conn, [hi, hi + 7 * 24 * HOUR])


def test_the_seal_guards_EVALUATION_not_only_training(cfg, conn):
    """The gap found during the Phase 10 pre-flight (2026-09-07).

    `assert_not_test` was wired into the three TRAINING paths — `rl/train.py`,
    `base.py`, `gradient_boosting.py` — which stops a model being fitted on the
    test set, the worse leak. But nothing consulted the seal on the way IN to
    an evaluation, so `compare --split test` built the test frame and would
    have scored on it. Evaluating once is the entire thing the seal rations, so
    the guard belongs where the bounds are handed out.
    """
    from src.pipeline.evalset import split_bounds
    from src.pipeline.split import seal, unseal

    seal(cfg, conn)

    # Bounds WITHOUT a connection stay available: reporting split sizes or
    # drawing the calendar reads no rows and must not need unsealing.
    lo, hi = split_bounds(cfg, "test")
    assert lo < hi

    # Bounds WITH a connection are a declaration of intent to read.
    with pytest.raises(SystemExit, match="SEALED TEST SET"):
        split_bounds(cfg, "test", conn=conn)

    # train/val are never gated.
    for split in ("train", "val"):
        assert split_bounds(cfg, split, conn=conn)[0] < split_bounds(
            cfg, split, conn=conn)[1]

    # And unsealing deliberately lets the final run through.
    unseal(cfg, conn, "test: the final evaluation")
    assert split_bounds(cfg, "test", conn=conn)[0] < hi
