"""Calibration — Brier, ECE, and the reliability curve.

Two tests carry this file. `test_brier_matches_hand_computed` and
`test_ece_matches_hand_computed` check values worked out on paper, so the
implementation is verified against arithmetic rather than against itself.

The other one worth reading is `test_useless_model_gets_a_great_brier_and_zero_skill`.
A model that detects nothing scores Brier 0.003 and ECE 0.000 — superb-looking
and perfectly calibrated, while being worthless. That is the accuracy trap in a
new costume, and the skill score is what exposes it.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.eval.contract import conform
from src.eval.metrics import (
    brier_score,
    calibration_summary,
    expected_calibration_error,
    reliability_curve,
)
from src.eval.synthetic import make_synthetic_predictions

TS = 1_725_148_800  # 2024-09-01 UTC


def rows(scores, labels, is_sched=None, items=None) -> pd.DataFrame:
    """One window per row. `labels` are 1 for positive, 0 for quiet."""
    n = len(scores)
    return conform(pd.DataFrame({
        "window_id": [f"w{i}" for i in range(n)],
        "ticker": ["AAA"] * n,
        "ts_utc": [TS] * n,
        "t0_utc": [TS if y else pd.NA for y in labels],
        "score": list(scores),
        "action": ["WAIT"] * n,
        "is_scheduled": [True if y else pd.NA for y in labels],
        "item_code": ["8.01" if y else pd.NA for y in labels],
    }))


# --- the Done-when: hand-computed values --------------------------------

HAND = rows([0.1, 0.1, 0.9, 0.9], [0, 1, 0, 1])


def test_brier_matches_hand_computed() -> None:
    """(0.01 + 0.81 + 0.81 + 0.01) / 4 = 0.41"""
    assert brier_score(HAND) == pytest.approx(0.41)


def test_ece_matches_hand_computed() -> None:
    """Two bins:
        [0, 0.5)  n=2  conf 0.1  actual 0.5  gap 0.4
        [0.5, 1]  n=2  conf 0.9  actual 0.5  gap 0.4
        (2/4)(0.4) + (2/4)(0.4) = 0.40
    """
    assert expected_calibration_error(HAND, n_bins=2) == pytest.approx(0.40)


def test_perfect_model_scores_zero() -> None:
    perfect = rows([0.0, 1.0, 0.0, 1.0], [0, 1, 0, 1])
    assert brier_score(perfect) == pytest.approx(0.0)
    assert expected_calibration_error(perfect, n_bins=2) == pytest.approx(0.0)


# --- the trap the skill score exists for --------------------------------


def useless(p: float, n_pos: int = 6, n_quiet: int = 1994) -> pd.DataFrame:
    """A model that ignores its input and always says the same thing,
    at a base rate close to the real 0.3%."""
    return rows([p] * (n_pos + n_quiet), [1] * n_pos + [0] * n_quiet)


def test_useless_model_gets_a_great_brier_and_zero_skill() -> None:
    """The whole reason `brier_skill_score` exists.

    Forecasting the base rate constantly detects nothing whatsoever, yet scores
    Brier ~0.003 and ECE ~0.000 — it is perfectly calibrated and useless. Only
    the skill score says so.
    """
    r = calibration_summary(useless(0.003))
    assert r.brier < 0.01                       # looks superb
    assert r.ece == pytest.approx(0.0, abs=1e-6)  # perfectly calibrated
    assert r.brier_skill_score == pytest.approx(0.0, abs=1e-3)   # and worthless


def test_confidently_wrong_model_has_negative_skill() -> None:
    r = calibration_summary(useless(0.5))
    assert r.brier_skill_score < 0


def test_skill_is_nan_when_every_label_is_the_same() -> None:
    """No positives means the baseline forecast is degenerate. nan, not a
    number that reads as meaningful."""
    r = calibration_summary(rows([0.2, 0.3], [0, 0]))
    assert math.isnan(r.brier_skill_score)


# --- refusing what calibration cannot measure ---------------------------


def test_unbounded_scores_are_refused() -> None:
    """A z-score baseline has no calibration. The error says so rather than
    squashing the values into a range and reporting a meaningless number."""
    df = make_synthetic_predictions(n_positive=5, n_quiet=20, n_tickers=4, seed=3)
    assert df["score"].min() < 0            # normal noise, definitely unbounded
    for fn in (brier_score, expected_calibration_error, calibration_summary):
        with pytest.raises(ValueError, match="calibration needs probabilities"):
            fn(df)


def test_error_names_the_observed_range() -> None:
    with pytest.raises(ValueError, match=r"scores range \[-1\.000, 2\.000\]"):
        brier_score(rows([-1.0, 2.0], [0, 1]))


# --- reliability curve --------------------------------------------------


def test_reliability_curve_columns_and_shape() -> None:
    curve = reliability_curve(HAND, n_bins=2)
    assert list(curve.columns) == [
        "bin", "bin_lower", "bin_upper", "count", "mean_predicted", "mean_actual",
    ]
    assert len(curve) == 2
    assert curve["count"].sum() == len(HAND)


def test_empty_bins_are_excluded() -> None:
    """A bin nobody landed in says nothing about calibration, and counting it
    as a zero gap would flatter ECE."""
    curve = reliability_curve(rows([0.05, 0.95], [0, 1]), n_bins=10)
    assert len(curve) == 2                      # not 10
    assert set(curve["bin"]) == {0, 9}


def test_boundary_probabilities_are_binned() -> None:
    """0.0 lands in the first bin and 1.0 in the last — the top bin is closed
    on the right so p=1.0 has a home."""
    curve = reliability_curve(rows([0.0, 1.0], [0, 1]), n_bins=10)
    assert set(curve["bin"]) == {0, 9}


def test_calibrated_model_sits_near_the_diagonal() -> None:
    """Predicted and actual should agree bin by bin when the data is generated
    to be calibrated."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 4000)
    y = (rng.uniform(0, 1, 4000) < p).astype(int)
    curve = reliability_curve(rows(p, y), n_bins=10)
    assert (curve["mean_actual"] - curve["mean_predicted"]).abs().max() < 0.05
    assert expected_calibration_error(rows(p, y)) < 0.03


def test_bin_count_comes_from_config() -> None:
    from src.utils.config import load_config

    r = calibration_summary(rows([0.1, 0.9], [0, 1]))
    assert r.n_bins == load_config()["eval"]["calibration_bins"]


def test_summary_reports_base_rate() -> None:
    r = calibration_summary(useless(0.003))
    assert r.base_rate == pytest.approx(6 / 2000)
    assert r.n_rows == 2000
