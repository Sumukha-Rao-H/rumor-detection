"""Precision at a fixed alert budget — the headline metric.

The test that matters most is `test_no_signal_gives_chance_level_precision`.
If a metric reports better-than-chance on data containing no signal, it is
flattering itself, and every model comparison built on it is worthless.
"""

from __future__ import annotations

import math
from dataclasses import fields

import pandas as pd
import pytest

from src.eval.contract import empty_frame
from src.eval.metrics import (
    BudgetResult,
    accuracy,
    precision_at_alert_budget,
    ticker_months,
    window_summary,
)
from src.eval.synthetic import make_synthetic_predictions

# A frame dense enough that the budget is a real constraint: with the sparse
# defaults the allowance exceeds the window count and every window alerts.
DENSE = dict(n_positive=40, n_quiet=1500, n_tickers=8, span_days=180)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return make_synthetic_predictions(**DENSE, signal_strength=2.0, seed=5)


def test_realised_alerts_match_the_budget(frame) -> None:
    """The task's Done-when."""
    r = precision_at_alert_budget(frame)
    assert not r.budget_exceeds_windows
    assert r.realised_alerts == r.budget_alerts


def test_accuracy_refuses(frame) -> None:
    """The task's other Done-when: no bare accuracy, ever."""
    with pytest.raises(NotImplementedError, match="base rate"):
        accuracy(frame)


def test_result_has_no_accuracy_field() -> None:
    """Not merely unimplemented — unrepresentable."""
    assert "accuracy" not in {f.name for f in fields(BudgetResult)}


def test_no_signal_gives_chance_level_precision() -> None:
    """THE honesty check.

    With signal_strength=0 the positive and quiet windows are drawn from the
    same distribution, so picking the highest-scoring windows cannot beat
    picking at random. Precision must land near the base rate.
    """
    df = make_synthetic_predictions(**DENSE, signal_strength=0.0, seed=5)
    r = precision_at_alert_budget(df)
    lift = r.precision / r.base_rate
    assert 0.0 <= lift <= 2.0, f"lift {lift:.2f} on data with no signal"


def test_precision_rises_with_signal() -> None:
    """A stronger footprint must score better. Guards against a metric that is
    insensitive to the thing it is supposed to measure."""
    precisions = [
        precision_at_alert_budget(
            make_synthetic_predictions(**DENSE, signal_strength=s, seed=5)
        ).precision
        for s in (0.0, 1.0, 2.0, 5.0)
    ]
    assert precisions == sorted(precisions), precisions


def test_threshold_is_the_kth_highest_peak(frame) -> None:
    """Threshold selection is rank-based and exact, not a search."""
    r = precision_at_alert_budget(frame)
    peaks = window_summary(frame)["peak_score"].sort_values(ascending=False)
    assert r.threshold == pytest.approx(peaks.iloc[r.budget_alerts - 1])


def test_alerts_are_counted_per_window_not_per_hour(frame) -> None:
    """An episode ends when it flags, so one window can spend at most one
    alert. Counting hours instead would blow the budget ~48x over."""
    r = precision_at_alert_budget(frame)
    assert r.realised_alerts <= r.n_windows
    assert r.n_windows < len(frame)


def test_budget_exceeding_windows_is_reported_not_hidden() -> None:
    """The sparse default frame: allowance 308 against 220 windows.

    Everything alerts and precision collapses to the base rate. That is what a
    too-generous budget genuinely means, so the flag is raised rather than the
    number quietly massaged.
    """
    df = make_synthetic_predictions(seed=1)  # sparse defaults
    r = precision_at_alert_budget(df)
    assert r.budget_exceeds_windows is True
    assert r.realised_alerts == r.n_windows
    assert r.precision == pytest.approx(r.base_rate)


def test_zero_budget_raises_no_alerts(frame) -> None:
    """Precision is nan, not 0 — there is nothing to be wrong about."""
    r = precision_at_alert_budget(frame, max_alerts=0)
    assert r.realised_alerts == 0
    assert math.isnan(r.precision)
    assert r.recall == 0.0


def test_frame_with_no_positives_gives_nan_recall() -> None:
    df = make_synthetic_predictions(n_positive=0, n_quiet=200, n_tickers=8,
                                    span_days=180, seed=2)
    r = precision_at_alert_budget(df, max_alerts=10)
    assert r.n_positive == 0
    assert math.isnan(r.recall)
    assert r.precision == 0.0


def test_max_alerts_override(frame) -> None:
    r = precision_at_alert_budget(frame, max_alerts=25)
    assert r.budget_alerts == 25
    assert r.realised_alerts == 25


def test_budget_rate_override_scales_the_allowance(frame) -> None:
    lo = precision_at_alert_budget(frame, budget_per_stock_per_month=1)
    hi = precision_at_alert_budget(frame, budget_per_stock_per_month=4)
    assert hi.budget_alerts == 4 * lo.budget_alerts
    assert hi.threshold <= lo.threshold   # a looser budget admits weaker scores


def test_recall_rises_as_the_budget_loosens(frame) -> None:
    tight = precision_at_alert_budget(frame, max_alerts=20)
    loose = precision_at_alert_budget(frame, max_alerts=200)
    assert loose.recall >= tight.recall


def test_ticker_months_counts_distinct_pairs() -> None:
    """Two tickers in one month is two ticker-months; one ticker across two
    months is also two."""
    base = 1_725_148_800  # 2024-09-01 UTC
    month = 31 * 24 * 3600
    df = pd.DataFrame({
        "window_id": ["a", "b", "c"],
        "ticker": ["AAA", "BBB", "AAA"],
        "ts_utc": [base, base, base + month],
        "t0_utc": [pd.NA] * 3,
        "score": [0.0] * 3,
        "action": ["WAIT"] * 3,
        "is_scheduled": [pd.NA] * 3,
        "item_code": [pd.NA] * 3,
    })
    assert ticker_months(df) == 3


def test_window_summary_collapses_hours(frame) -> None:
    s = window_summary(frame)
    assert len(s) == frame["window_id"].nunique()
    assert s["peak_score"].max() == frame["score"].max()


def test_invalid_frame_is_rejected(frame) -> None:
    """Contract validation runs before any arithmetic."""
    with pytest.raises(ValueError, match="empty"):
        precision_at_alert_budget(empty_frame())
    bad = frame.copy()
    bad.loc[bad.index[0], "action"] = "HOLD"
    with pytest.raises(ValueError, match="unknown action"):
        precision_at_alert_budget(bad)


def test_base_rate_is_reported_alongside_precision(frame) -> None:
    """Precision alone is unreadable; the pair is the point."""
    r = precision_at_alert_budget(frame)
    assert 0 < r.base_rate < 1
    assert r.precision > r.base_rate      # signal_strength=2.0 should beat chance
