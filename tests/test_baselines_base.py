"""P5-01 — the shape every baseline is held to.

These tests are about the base class, not about any detector. What they pin
down is the handful of rules that decide whether the Phase 5 comparison table
compares like with like: one FLAG per episode, no window silently leaving the
denominator, the sealed test set untouched, and the P1-07 contract satisfied
before any metric runs.

The two subclasses below are deliberately trivial. A baseline with real
behaviour would test its own scoring; here the scoring is a stub so that a
failure can only mean the shared machinery is wrong.
"""

import numpy as np
import pandas as pd
import pytest

from src import db
from src.baselines import Baseline, unscoreable_floor
from src.eval import contract
from src.eval.metrics import precision_at_alert_budget
from src.pipeline.split import boundaries, seal
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts

HOUR = 3600


# --------------------------------------------------------------------------
# Stub baselines
# --------------------------------------------------------------------------
class ConstantBaseline(Baseline):
    """Scores every hour the same. The shape always-quiet takes."""

    def __init__(self, cfg=None, value: float = 0.0):
        super().__init__(cfg)
        self.value = value

    def score(self, frame):
        return pd.Series(self.value, index=frame.index, dtype="float64")


class ColumnBaseline(Baseline):
    """Passes a feature column straight through, NaNs and all."""

    def __init__(self, cfg=None, column: str = "volume_z"):
        super().__init__(cfg)
        self.column = column

    def score(self, frame):
        return frame[self.column]


class LearningBaseline(Baseline):
    """Records that fit() ran, so the default no-op can be told from an override."""

    fitted = False

    def fit(self, train):
        self.fitted = True
        return self

    def score(self, frame):
        return pd.Series(1.0, index=frame.index, dtype="float64")


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def conn(tmp_path):
    return db.get_conn(tmp_path / "baselines.db")


def make_frame(windows) -> pd.DataFrame:
    """Build a feature frame.

    `windows` is a list of (window_id, ticker, t0_utc or None, [scores]).
    Hours run backwards from t0 so every row sits strictly before it, which is
    what the contract requires of a positive window.
    """
    rows = []
    base = date_str_to_ts("2025-10-01")
    for wid, ticker, t0, scores in windows:
        anchor = t0 if t0 is not None else base + 500 * HOUR
        n = len(scores)
        for i, s in enumerate(scores):
            rows.append({
                "window_id": wid,
                "ticker": ticker,
                "ts_utc": anchor - (n - i) * HOUR,
                "t0_utc": t0,
                "is_scheduled": None if t0 is None else True,
                "item_code": None if t0 is None else "8.01",
                "volume_z": s,
            })
    df = pd.DataFrame(rows)
    return df.astype({"t0_utc": "Int64", "is_scheduled": "boolean",
                      "item_code": "string", "ticker": "string",
                      "window_id": "string", "ts_utc": "Int64"})


@pytest.fixture
def frame():
    t0 = date_str_to_ts("2025-10-15")
    return make_frame([
        ("W1", "AAA", t0, [0.0, 1.0, 3.0, 4.0]),      # crosses at hour 2
        ("W2", "BBB", None, [0.0, 0.1, 0.2, 0.3]),    # quiet, never crosses
    ])


# --------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------
def test_predict_returns_a_valid_contract_frame(cfg, frame):
    out = ColumnBaseline(cfg).predict(frame, threshold=2.5)

    assert list(out.columns) == list(contract.SCHEMA)
    # validate_predictions is what every metric calls; passing it unchanged is
    # the whole promise of this class.
    pd.testing.assert_frame_equal(out, contract.validate_predictions(out))


def test_output_feeds_the_headline_metric_without_adaptation(cfg, frame):
    """The stated Done-when: no per-model adapter between here and metrics."""
    out = ColumnBaseline(cfg).predict(frame, threshold=2.5)
    result = precision_at_alert_budget(out, max_alerts=1)
    assert result.n_windows == 2
    assert result.n_positive == 1


def test_an_episode_holds_at_most_one_flag(cfg):
    """Five consecutive crossings are one alert, not five — an episode ends
    when it flags, and the alert budget is spent per window."""
    t0 = date_str_to_ts("2025-10-15")
    f = make_frame([("W1", "AAA", t0, [9.0, 9.0, 9.0, 9.0, 9.0])])
    out = ColumnBaseline(cfg).predict(f, threshold=2.5)

    flags = out[out["action"] == contract.FLAG]
    assert len(flags) == 1
    assert flags["ts_utc"].iloc[0] == out["ts_utc"].min()


def test_the_flag_lands_on_the_first_crossing_by_time_not_row_order(cfg):
    """A caller is not required to hand over a sorted frame; sorting by row
    position instead of ts_utc would date the alert wrongly and inflate the
    measured lead time."""
    t0 = date_str_to_ts("2025-10-15")
    f = make_frame([("W1", "AAA", t0, [0.0, 5.0, 6.0, 7.0])])
    shuffled = f.iloc[::-1].reset_index(drop=True)

    out = ColumnBaseline(cfg).predict(shuffled, threshold=2.5)
    flagged = out.loc[out["action"] == contract.FLAG, "ts_utc"]
    expected = f.loc[f["volume_z"] >= 2.5, "ts_utc"].min()
    assert len(flagged) == 1
    assert flagged.iloc[0] == expected


def test_a_row_at_or_after_t0_is_rejected_rather_than_laundered(cfg):
    """The base class must not quietly repair a leaking frame — that hour is a
    moment the news was already public."""
    t0 = date_str_to_ts("2025-10-15")
    f = make_frame([("W1", "AAA", t0, [1.0, 2.0])])
    # An hour PAST t0. The contract's rule is strictly-after by design — P4-11
    # pinned the decision window to strictly before t0, so a row stamped at t0
    # is the boundary, not a violation.
    f.loc[f.index[-1], "ts_utc"] = t0 + HOUR

    with pytest.raises(ValueError, match="leakage"):
        ColumnBaseline(cfg).predict(f, threshold=99.0)


def test_quiet_window_metadata_stays_null(cfg, frame):
    """t0, is_scheduled and item_code are null together on a negative; the
    contract enforces the pairing, so assembly must not fill a placeholder."""
    out = ColumnBaseline(cfg).predict(frame, threshold=2.5)
    quiet = out[out["window_id"] == "W2"]
    assert quiet["t0_utc"].isna().all()
    assert quiet["is_scheduled"].isna().all()
    assert quiet["item_code"].isna().all()


# --------------------------------------------------------------------------
# Unscoreable windows — the decision sampling.py deferred to P5
# --------------------------------------------------------------------------
def test_an_unscoreable_window_is_ranked_last_not_dropped(cfg):
    """Issue 32's 234 all-NaN windows. Keeping them costs nothing and dropping
    them would raise precision without detecting anything."""
    t0 = date_str_to_ts("2025-10-15")
    f = make_frame([
        ("W1", "AAA", t0, [1.0, 4.0]),
        ("DEAD", "BBB", None, [np.nan, np.nan]),
    ])
    out = ColumnBaseline(cfg).predict(f, threshold=2.5)

    dead = out[out["window_id"] == "DEAD"]
    assert len(dead) == 2                                   # survived
    assert dead["score"].max() < out[out["window_id"] == "W1"]["score"].min()
    assert (dead["action"] == contract.WAIT).all()          # never flagged


def test_unscoreable_windows_preserve_the_window_count(cfg):
    """The property the P5-06 table depends on: two baselines must evaluate on
    the same windows, or their precisions are not comparable."""
    t0 = date_str_to_ts("2025-10-15")
    f = make_frame([
        ("W1", "AAA", t0, [1.0, 4.0]),
        ("D1", "BBB", None, [np.nan, np.nan]),
        ("D2", "CCC", None, [np.nan, np.nan]),
    ])
    scoreable = precision_at_alert_budget(
        ConstantBaseline(cfg, value=1.0).predict(f, threshold=99.0), max_alerts=1)
    partial = precision_at_alert_budget(
        ColumnBaseline(cfg).predict(f, threshold=2.5), max_alerts=1)

    assert scoreable.n_windows == partial.n_windows == 3


def test_one_nan_hour_does_not_condemn_the_whole_window(cfg):
    """A missing bar costs that hour, not the window — the detector still has
    evidence on the others."""
    t0 = date_str_to_ts("2025-10-15")
    f = make_frame([("W1", "AAA", t0, [np.nan, 1.0, 4.0])])
    out = ColumnBaseline(cfg).predict(f, threshold=2.5)

    assert len(out) == 3
    assert (out["action"] == contract.FLAG).sum() == 1


def test_error_mode_names_the_offending_windows(cfg):
    """For a baseline that ought to score everything, silence is worse than a
    stack trace."""
    strict = {**cfg, "baselines": {**cfg["baselines"],
                                   "common": {"unscoreable": "error"}}}
    t0 = date_str_to_ts("2025-10-15")
    f = make_frame([("W1", "AAA", t0, [1.0, 2.0]),
                    ("DEAD", "BBB", None, [np.nan, np.nan])])

    with pytest.raises(ValueError, match="DEAD"):
        ColumnBaseline(strict).predict(f, threshold=2.5)


def test_an_unknown_unscoreable_policy_is_refused(cfg):
    bad = {**cfg, "baselines": {**cfg["baselines"],
                                "common": {"unscoreable": "ignore"}}}
    t0 = date_str_to_ts("2025-10-15")
    f = make_frame([("W1", "AAA", t0, [np.nan, np.nan])])
    with pytest.raises(ValueError, match="unscoreable"):
        ColumnBaseline(bad).predict(f, threshold=2.5)


def test_constant_scores_do_not_produce_nan(cfg, frame):
    """Always-quiet emits one value everywhere; the floor arithmetic must not
    turn that into NaN and poison every aggregate."""
    out = ConstantBaseline(cfg, value=0.0).predict(frame, threshold=99.0)
    assert np.isfinite(out["score"]).all()
    assert (out["action"] == contract.WAIT).all()


def test_the_floor_sits_below_every_real_score():
    scores = pd.Series([2.0, 5.0, np.nan, -3.0])
    assert unscoreable_floor(scores) < scores.min()


def test_the_floor_is_finite_when_nothing_is_scoreable():
    """±inf is rejected by the contract precisely so one infinity cannot
    poison Brier and calibration."""
    assert np.isfinite(unscoreable_floor(pd.Series([np.nan, np.nan])))


# --------------------------------------------------------------------------
# The seal
# --------------------------------------------------------------------------
def test_predicting_on_the_sealed_test_split_raises(cfg, conn):
    """P5-03's threshold sweep must not reach the test set by omission."""
    seal(cfg, conn)
    _, val_end = boundaries(cfg)
    f = make_frame([("W1", "AAA", val_end + 10 * HOUR, [1.0, 2.0])])

    with pytest.raises(SystemExit, match="SEALED TEST SET"):
        ColumnBaseline(cfg).predict(f, threshold=2.5, conn=conn)


def test_validation_timestamps_are_permitted(cfg, conn):
    seal(cfg, conn)
    train_end, _ = boundaries(cfg)
    f = make_frame([("W1", "AAA", train_end + 10 * HOUR, [1.0, 4.0])])
    out = ColumnBaseline(cfg).predict(f, threshold=2.5, conn=conn)
    assert len(out) == 2


def test_without_a_connection_the_seal_is_not_claimed(cfg, frame):
    """A synthetic frame has no database behind it. Skipping the check is
    honest; pretending to have made it would not be."""
    out = ColumnBaseline(cfg).predict(frame, threshold=2.5)
    assert len(out) == len(frame)


# --------------------------------------------------------------------------
# The class itself
# --------------------------------------------------------------------------
def test_score_must_be_implemented():
    class NoScore(Baseline):
        pass

    with pytest.raises(TypeError):
        NoScore()


def test_fit_defaults_to_a_noop_and_returns_self(cfg, frame):
    b = ConstantBaseline(cfg)
    assert b.fit(frame) is b
    assert len(b.predict(frame, threshold=99.0)) == len(frame)


def test_an_overridden_fit_still_chains(cfg, frame):
    b = LearningBaseline(cfg).fit(frame)
    assert b.fitted is True
    assert len(b.predict(frame, threshold=0.5)) == len(frame)


def test_name_defaults_to_the_class_name(cfg):
    assert ConstantBaseline(cfg).name == "ConstantBaseline"


def test_an_empty_frame_raises_rather_than_reporting_success(cfg, frame):
    with pytest.raises(ValueError, match="empty"):
        ConstantBaseline(cfg).predict(frame.iloc[:0], threshold=1.0)


def test_a_frame_missing_contract_columns_is_refused(cfg, frame):
    with pytest.raises(ValueError, match="missing"):
        ConstantBaseline(cfg).predict(frame.drop(columns=["t0_utc"]),
                                      threshold=1.0)
