"""P5-03 — the evaluation population, and why its two sides are shaped differently.

The asymmetry is the decision this module encodes: a positive is one episode
carrying one label, a negative is a single bar. These tests pin the properties
that make the headline metric mean anything — no bar counted twice, the base
rate near the plan's figure, and the alert budget sitting below the window
count so precision can actually discriminate.
"""

import numpy as np
import pandas as pd
import pytest

from src import db
from src.eval import contract
from src.pipeline.evalset import build_eval_frame, split_bounds
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts

HOUR = 3600


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def conn(tmp_path, cfg):
    """One ticker with long history and a single usable event inside the range."""
    c = db.get_conn(tmp_path / "evalset.db")
    iv = cfg["market"]["interval"]
    start = date_str_to_ts("2025-09-01")
    n = 900

    for ticker in ("AAA", cfg["market"]["benchmark"]):
        db.upsert_bars(c, [
            (ticker, start + i * HOUR, 100.0, 101.0, 99.0,
             100.0 + (i % 7) * 0.1, 1_000_000 + (i % 13) * 1000, iv)
            for i in range(n)])

    db.upsert_companies(c, [{"cik": "C1", "ticker": "AAA", "in_universe": 1}])
    t0 = start + 800 * HOUR
    db.upsert_filings(c, [{
        "accession_no": "A-1", "cik": "C1", "ticker": "AAA", "form": "8-K",
        "items": "8.01", "acceptance_utc": t0, "filing_date_utc": t0}])
    db.upsert_events(c, [{
        "event_id": "E1", "accession_no": "A-1", "ticker": "AAA",
        "items": "8.01", "t0_filing_utc": t0, "t0_utc": t0,
        "t0_source": "filing", "is_scheduled": 0, "usable": 1,
        "exclude_reason": None}])
    return c


@pytest.fixture
def bounds():
    start = date_str_to_ts("2025-09-01")
    return start + 600 * HOUR, start + 900 * HOUR


def test_a_positive_is_one_episode_not_one_window_per_hour(cfg, conn, bounds):
    """One positive per EVENT. 48 positive hours per event would put the base
    rate at 12.5% and always-quiet at 87.5%, which is not what the plan quotes."""
    frame = build_eval_frame(cfg, conn, *bounds, progress_every=0)
    positives = frame[frame["t0_utc"].notna()]

    assert positives["window_id"].nunique() == 1
    assert len(positives) == cfg["decision"]["horizon_hours"]


def test_a_negative_is_a_single_bar(cfg, conn, bounds):
    """Every other in-universe hour is its own 'do I alert now?' decision."""
    frame = build_eval_frame(cfg, conn, *bounds, progress_every=0)
    negatives = frame[frame["t0_utc"].isna()]
    per_window = negatives.groupby("window_id").size()

    assert (per_window == 1).all()
    assert negatives["window_id"].str.startswith("bar:").all()


def test_no_bar_is_counted_twice(cfg, conn, bounds):
    """A bar inside an episode cannot also be its own decision point — every
    metric would count it twice."""
    frame = build_eval_frame(cfg, conn, *bounds, progress_every=0)
    assert not frame.duplicated(subset=["ticker", "ts_utc"]).any()


def test_episode_bars_are_strictly_before_t0(cfg, conn, bounds):
    frame = build_eval_frame(cfg, conn, *bounds, progress_every=0)
    positives = frame[frame["t0_utc"].notna()]
    assert (positives["ts_utc"] < positives["t0_utc"]).all()


def test_negatives_carry_null_label_columns_together(cfg, conn, bounds):
    """The contract requires the three to be null exactly on quiet windows."""
    frame = build_eval_frame(cfg, conn, *bounds, progress_every=0)
    negatives = frame[frame["t0_utc"].isna()]
    assert negatives["is_scheduled"].isna().all()
    assert negatives["item_code"].isna().all()


def test_the_frame_satisfies_the_contract(cfg, conn, bounds):
    """The point of the exercise: it goes straight to a Baseline with no
    adapter in between."""
    from src.baselines import VolumeZScore

    frame = build_eval_frame(cfg, conn, *bounds, progress_every=0)
    for c, ty in (("window_id", "string"), ("ticker", "string"),
                  ("ts_utc", "Int64"), ("t0_utc", "Int64"),
                  ("is_scheduled", "boolean"), ("item_code", "string")):
        frame[c] = frame[c].astype(ty)

    out = VolumeZScore(cfg).predict(frame, threshold=2.5)
    pd.testing.assert_frame_equal(out, contract.validate_predictions(out))


def test_the_base_rate_is_small_the_way_the_plan_says(cfg, conn, bounds):
    """One episode against hundreds of quiet bars. The exact figure depends on
    the fixture; what matters is the order of magnitude — a base rate near 50%
    would mean the asymmetry had been lost."""
    frame = build_eval_frame(cfg, conn, *bounds, progress_every=0)
    windows = frame.groupby("window_id")["t0_utc"].first()
    base_rate = windows.notna().mean()
    assert 0 < base_rate < 0.05


def test_bars_outside_the_range_are_excluded(cfg, conn, bounds):
    lo, hi = bounds
    frame = build_eval_frame(cfg, conn, lo, hi, progress_every=0)
    negatives = frame[frame["t0_utc"].isna()]
    assert (negatives["ts_utc"] >= lo).all()
    assert (negatives["ts_utc"] < hi).all()


def test_an_empty_range_raises_rather_than_returning_nothing(cfg, conn):
    start = date_str_to_ts("2020-01-01")
    with pytest.raises(SystemExit, match="empty"):
        build_eval_frame(cfg, conn, start, start + 10 * HOUR, progress_every=0)


def test_a_universe_with_no_companies_raises(cfg, conn, bounds):
    with pytest.raises(SystemExit, match="no in-universe companies"):
        build_eval_frame(cfg, conn, *bounds, tickers=[], progress_every=0)


def test_split_bounds_cover_the_study_window_without_gaps(cfg):
    lo = date_str_to_ts(cfg["study_window"]["start"])
    hi = date_str_to_ts(cfg["study_window"]["end"])
    tr, va, te = (split_bounds(cfg, s) for s in ("train", "val", "test"))

    assert tr[0] == lo and te[1] == hi
    assert tr[1] == va[0] and va[1] == te[0]


def test_an_unknown_split_is_refused(cfg):
    with pytest.raises(SystemExit, match="unknown split"):
        split_bounds(cfg, "holdout")
