"""P5-05 — the learned baseline, and the guards that keep it honest.

Two assertions carry most of the weight. `test_label_columns_are_not_features`
keeps `is_scheduled` and `item_code` out of the model — both describe the event
being predicted and would be leakage of the plainest kind. And
`test_single_class_training_frame_raises` stops a model that has never seen a
positive from silently becoming always-quiet under another name.
"""

import numpy as np
import pandas as pd
import pytest

from src import db
from src.baselines import FEATURES, GradientBoosting, label_rows
from src.eval import contract
from src.eval.metrics import precision_at_alert_budget
from src.pipeline.split import boundaries, seal
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts

HOUR = 3600


@pytest.fixture
def cfg():
    """Small and fast — 30 trees is plenty to learn a constructed signal."""
    base = load_config()
    return {**base, "baselines": {**base["baselines"],
                                  "gradient_boosting": {
                                      **base["baselines"]["gradient_boosting"],
                                      "n_estimators": 30, "n_jobs": 1}}}


@pytest.fixture
def conn(tmp_path):
    return db.get_conn(tmp_path / "gb.db")


def make_frame(n_pos: int = 40, n_neg: int = 120, hours: int = 4,
               separable: bool = True, seed: int = 0) -> pd.DataFrame:
    """Positives carry an elevated signal across every feature when separable."""
    rng = np.random.default_rng(seed)
    base = date_str_to_ts("2025-10-01")
    rows = []
    for i in range(n_pos + n_neg):
        positive = i < n_pos
        anchor = base + i * 200 * HOUR
        shift = 2.0 if (positive and separable) else 0.0
        for h in range(hours):
            row = {
                "window_id": f"{'P' if positive else 'N'}{i}",
                "ticker": f"T{i}",
                "ts_utc": anchor - (hours - h) * HOUR,
                "t0_utc": anchor if positive else None,
                "is_scheduled": True if positive else None,
                "item_code": "8.01" if positive else None,
            }
            for f in FEATURES:
                row[f] = float(rng.normal(shift, 1.0))
            rows.append(row)
    return pd.DataFrame(rows).astype({
        "window_id": "string", "ticker": "string", "ts_utc": "Int64",
        "t0_utc": "Int64", "is_scheduled": "boolean", "item_code": "string"})


@pytest.fixture
def frame():
    return make_frame()


# --------------------------------------------------------------------------
# Leakage guards
# --------------------------------------------------------------------------
def test_label_columns_are_not_features():
    """`is_scheduled` and `item_code` describe the event being predicted and
    are known only once it has happened."""
    assert "is_scheduled" not in FEATURES
    assert "item_code" not in FEATURES
    assert "t0_utc" not in FEATURES
    assert "window_id" not in FEATURES


def test_single_class_training_frame_raises(cfg):
    """A model that has never seen a positive scores everything identically —
    always-quiet under another name, but reported as a learned result."""
    only_negatives = make_frame(n_pos=0, n_neg=20)
    with pytest.raises(ValueError, match="one class"):
        GradientBoosting(cfg).fit(only_negatives)


def test_fit_never_touches_the_sealed_test_split(cfg, conn):
    """Fitting on test data is the one leak no later check could catch."""
    seal(cfg, conn)
    _, val_end = boundaries(cfg)
    f = make_frame(n_pos=4, n_neg=8)
    f["ts_utc"] = pd.array([val_end + i * HOUR for i in range(len(f))],
                           dtype="Int64")

    with pytest.raises(SystemExit, match="SEALED TEST SET"):
        GradientBoosting(cfg).fit(f, conn=conn)


# --------------------------------------------------------------------------
# Fit / predict discipline
# --------------------------------------------------------------------------
def test_predict_before_fit_raises(cfg, frame):
    """Better a named error than a plausible-looking number that means
    nothing."""
    with pytest.raises(ValueError, match="not fitted"):
        GradientBoosting(cfg).predict(frame, threshold=0.5)


def test_fit_returns_self_so_it_chains(cfg, frame):
    model = GradientBoosting(cfg)
    assert model.fit(frame) is model


def test_refuses_a_frame_missing_a_feature(cfg, frame):
    model = GradientBoosting(cfg).fit(frame)
    with pytest.raises(ValueError, match="missing feature column"):
        model.predict(frame.drop(columns=["volume_z"]), threshold=0.5)


# --------------------------------------------------------------------------
# Behaviour
# --------------------------------------------------------------------------
def test_scores_are_probabilities_in_range(cfg, frame):
    """Unlike the other three, this baseline emits probabilities — so Brier
    and ECE legitimately apply to it."""
    out = GradientBoosting(cfg).fit(frame).predict(frame, threshold=0.5)
    assert out["score"].between(0.0, 1.0).all()


def test_it_learns_a_separable_signal(cfg, frame):
    """A sanity check on the model, not on the market: if it cannot separate
    constructed data, every later number is noise."""
    model = GradientBoosting(cfg).fit(frame)
    out = model.predict(frame, threshold=0.5)
    r = precision_at_alert_budget(out, max_alerts=40)
    assert r.precision > r.base_rate * 2


def test_it_does_not_invent_signal_in_noise(cfg):
    """The other half: on unseparable data it must not beat the base rate on
    held-out rows."""
    train = make_frame(separable=False, seed=1)
    held_out = make_frame(separable=False, seed=2)
    model = GradientBoosting(cfg).fit(train)
    r = precision_at_alert_budget(model.predict(held_out, threshold=0.5),
                                  max_alerts=40)
    assert r.precision < r.base_rate * 2


def test_is_deterministic(cfg, frame):
    """random_state is fixed so a rerun reproduces the reported number."""
    a = GradientBoosting(cfg).fit(frame).predict(frame, threshold=0.5)
    b = GradientBoosting(cfg).fit(frame).predict(frame, threshold=0.5)
    pd.testing.assert_series_equal(a["score"], b["score"])


def test_feature_order_does_not_matter(cfg, frame):
    """The model selects its columns by name from a fixed list, so a reordered
    matrix cannot be scored against the wrong columns."""
    model = GradientBoosting(cfg).fit(frame)
    shuffled = frame[list(reversed(frame.columns))]
    a = model.predict(frame, threshold=0.5)["score"]
    b = model.predict(shuffled, threshold=0.5)["score"]
    pd.testing.assert_series_equal(a, b)


def test_nan_rows_still_get_a_score(cfg, frame):
    """xgboost routes NaN by a learned default direction, so this is the one
    baseline with an opinion about every window — no unscoreable fallback."""
    model = GradientBoosting(cfg).fit(frame)
    with_nan = frame.copy()
    with_nan.loc[with_nan.index[:8], "volume_z"] = np.nan
    out = model.predict(with_nan, threshold=0.5)
    assert np.isfinite(out["score"]).all()
    assert out["window_id"].nunique() == with_nan["window_id"].nunique()


def test_labels_come_from_t0_presence(cfg, frame):
    """`t0_utc` IS the label — the contract keeps no separate label column,
    because a redundant one can disagree with what it duplicates."""
    y = label_rows(frame)
    assert set(np.unique(y)) == {0, 1}
    assert y.sum() == frame["t0_utc"].notna().sum()


def test_output_passes_the_contract(cfg, frame):
    out = GradientBoosting(cfg).fit(frame).predict(frame, threshold=0.5)
    pd.testing.assert_frame_equal(out, contract.validate_predictions(out))


def test_name_is_stable_for_the_report(cfg):
    assert GradientBoosting(cfg).name == "gradient_boosting"


# --------------------------------------------------------------------------
# The contaminated features
# --------------------------------------------------------------------------
def test_the_sampling_contaminated_features_are_excluded_by_default(cfg):
    """Not a hyperparameter choice. quiet_gap_hours=168 means a training
    negative is never within 7 days of a filing (0.1% of them, against 18.9%
    of training positives), so "recent 8-K" separates the TRAINING classes
    almost perfectly. At evaluation 31.5% of negatives have a recent 8-K, so
    the learned rule is inverted rather than merely useless."""
    model = GradientBoosting(cfg)
    assert "days_since_last_8k" not in model.features
    assert "days_since_last_earnings" not in model.features
    # Added 2026-09-09: a positive window ends at t0 and most 8-Ks are accepted
    # after the close, so this encodes how the window was CUT. Unlike the two
    # above it points the SAME way at evaluation, so the model is rewarded for
    # learning it rather than merely misled — 5.67x lift on its own.
    assert "trading_hours_to_close" not in model.features
    assert "volume_z" in model.features


def test_the_exclusion_is_reproducible_both_ways(cfg, frame):
    """Setting exclude_features to [] reproduces the contaminated run, so the
    number in the report can be checked rather than taken on trust."""
    contaminated = {**cfg, "baselines": {
        **cfg["baselines"],
        "gradient_boosting": {**cfg["baselines"]["gradient_boosting"],
                              "exclude_features": []}}}
    model = GradientBoosting(contaminated)
    assert model.features == FEATURES
    # Counted from the config's own list rather than hardcoded, so adding an
    # exclusion is a config decision and not also a test edit — the property
    # under test is "every excluded column is gone and nothing else is", not
    # how many there happen to be today.
    excluded = cfg["baselines"]["gradient_boosting"]["exclude_features"]
    kept = GradientBoosting(cfg).features
    assert set(kept) == set(FEATURES) - set(excluded)
    assert len(kept) == len(FEATURES) - len(excluded)
    # and it still trains and scores with the full set
    out = model.fit(frame).predict(frame, threshold=0.5)
    assert out["score"].between(0.0, 1.0).all()


def test_an_excluded_column_may_be_absent_from_the_frame(cfg, frame):
    """A column the model does not use must not be required to be present."""
    model = GradientBoosting(cfg).fit(frame)
    out = model.predict(frame.drop(columns=["days_since_last_8k"]),
                        threshold=0.5)
    assert len(out) == len(frame)
