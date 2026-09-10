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

import numpy as np

from src.eval.contract import WAIT, conform, empty_frame
from src.eval.metrics import (
    BudgetResult,
    accuracy,
    precision_at_alert_budget,
    ticker_months,
    window_summary,
)
from src.eval.synthetic import make_synthetic_predictions
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts

# A frame dense enough that the budget is a real constraint: with the sparse
# defaults the allowance exceeds the window count and every window alerts.
HOUR = 3600

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


def test_tie_group_at_cutoff_can_overshoot_budget() -> None:
    """Documented, intentional behavior: ties at the threshold are admitted as
    a whole group, so `realised_alerts` can exceed `budget_alerts` with no
    bound. This is not a bug (see the module docstring, `precision_at_alert_budget`),
    but it was previously untested, so a future change to tie-breaking (e.g.
    admitting only the first tied window by sort order) could silently change
    this behavior without any test noticing.

    5 windows, scores [3, 3, 3, 2, 1]; the two positives are the first window
    (score 3) and the 4th window (score 2). `max_alerts=2` asks for the top 2,
    but 3 windows tie for the top score, so all 3 are let through:

        budget_alerts   = 2
        realised_alerts = 3          (the whole 3-way tie at score 3)
        true_positives  = 1          (only the first tied window is positive)
        precision       = 1/3
    """
    df = pd.DataFrame({
        "window_id": ["w0", "w1", "w2", "w3", "w4"],
        "ticker": ["AAA"] * 5,
        "ts_utc": [1_725_148_800 + i * 3600 for i in range(5)],
        "t0_utc": [1_725_148_800, pd.NA, pd.NA, 1_725_148_800 + 3 * 3600, pd.NA],
        "score": [3.0, 3.0, 3.0, 2.0, 1.0],
        "action": ["WAIT"] * 5,
        "is_scheduled": [True, pd.NA, pd.NA, True, pd.NA],
        "item_code": ["8.01", pd.NA, pd.NA, "8.01", pd.NA],
    })
    r = precision_at_alert_budget(df, max_alerts=2)
    assert r.budget_alerts == 2
    assert r.realised_alerts == 3
    assert r.true_positives == 1
    assert r.precision == pytest.approx(1 / 3)
    assert r.budget_exceeds_windows is False   # this isn't the "budget > n_windows" case


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
    """One row per window, and a positive represented by its FINAL bar.

    The second assertion used to be `peak_score.max() == score.max()`, which
    held while a window scored as the maximum over all its hours. That IS the
    length asymmetry: it gave a 48-bar episode 48 chances against a quiet bar's
    one. A positive is now its decision point — the bar before t0 — so the
    frame's single highest score need not survive collapsing, and asserting
    that it does would re-pin the bug.
    """
    s = window_summary(frame)
    assert len(s) == frame["window_id"].nunique()

    pos = frame[frame["t0_utc"].notna()]
    final = (pos.sort_values("ts_utc").groupby("window_id").tail(1)
                .set_index("window_id")["score"])
    pd.testing.assert_series_equal(
        s.loc[final.index, "peak_score"], final, check_names=False)


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


def test_the_ceiling_is_reported_when_the_budget_exceeds_the_positives():
    """Issue 29, made structural instead of remembered.

    The budget is SPENT, not capped: the top `budget_alerts` windows are
    flagged, so once the budget exceeds the positives available, every alert
    beyond the last true positive is a false one by construction and precision
    cannot reach 1.0 however good the model is. The study's real shape —
    31,823 alerts against 6,737 positives — puts the ceiling at 21.2%, and a
    15% result read without it looks like failure rather than 71% of what was
    achievable.

    Built so the ceiling BITES: 2 positives, a budget of 10.
    """
    rows = []
    for i in range(10):
        rows.append({
            "window_id": f"w{i}", "ticker": "AAA", "ts_utc": 1_000 + i,
            "score": i / 10, "action": "WAIT",
            "t0_utc": 9_999 if i < 2 else None,
            "is_scheduled": True if i < 2 else None,
            "item_code": "8.01" if i < 2 else None,
        })
    frame = pd.DataFrame(rows)

    res = precision_at_alert_budget(frame, max_alerts=10)

    assert res.n_positive == 2
    assert res.budget_alerts == 10
    # 2 positives / 10 alerts — a flawless detector still scores only 20%.
    assert res.max_precision == pytest.approx(0.2)
    assert res.precision <= res.max_precision + 1e-9


def test_the_ceiling_is_one_when_positives_outnumber_the_budget():
    """The other direction: a budget tighter than the positives is not capped
    by this effect at all, and the ceiling must not spuriously drop below 1."""
    rows = []
    for i in range(10):
        rows.append({
            "window_id": f"w{i}", "ticker": "AAA", "ts_utc": 1_000 + i,
            "score": i / 10, "action": "WAIT",
            "t0_utc": 9_999, "is_scheduled": True, "item_code": "8.01",
        })
    res = precision_at_alert_budget(pd.DataFrame(rows), max_alerts=3)
    assert res.n_positive == 10
    assert res.budget_alerts == 3
    assert res.max_precision == pytest.approx(1.0)


def test_a_caller_supplied_max_alerts_reaches_the_reported_ceiling(frame) -> None:
    """`report_table(max_alerts=N)` used to size the threshold from N and the
    ceiling from something else entirely.

    N was passed to `precision_at_alert_budget` to pick the operating point and
    then dropped on the way into `evaluate`, whose own budget fell back to
    ticker-months x the config rate. At `max_alerts=400` the table advertised a
    ceiling of 1.0 beside a precision of 0.167 — the ceiling was being divided
    by an allowance thousands of alerts wide while precision was divided by the
    400 alerts actually issued.

    Phase 10 never passed `max_alerts` (`compare.comparison_table` does not),
    so nothing published moved; this pins the two together so they cannot
    disagree again.
    """
    from src.eval.report import report_table

    n = 400
    table = report_table(frame, max_alerts=n)
    row = table.query("slice == 'all'").iloc[0]

    assert row["n_alerts"] == n, "the operating point must spend exactly N"
    assert row["max_precision"] == pytest.approx(
        min(row["n_positive"], n) / n), (
        "the ceiling must be sized from the same N the threshold was")
    assert row["precision"] <= row["max_precision"] + 1e-12
    # `tie_spill_ratio` divides the alerts issued by the allowance, so it is
    # the column that still catches the dropped argument once the ceiling
    # shares precision's denominator: N alerts against an allowance of N is 1,
    # against the config-derived one it is not.
    assert row["tie_spill_ratio"] == pytest.approx(1.0), (
        "the row's allowance must be the N the caller asked for, not the one "
        "ticker-months x the config rate would have produced")


# --------------------------------------------------------------------------
# The null: a scorer with no information must score chance
#
# `test_no_signal_gives_chance_level_precision` above checks this, but only on
# `synthetic.py`'s frames, where BOTH classes are `horizon` rows long. The real
# evaluation frame is not that shape: `evalset.build_eval_frame` gives a
# positive 48 bars and a quiet window a SINGLE bar. A plain max-per-window then
# hands a positive 48 independent chances to cross the threshold and a quiet
# window one, at the same one-alert cost — so window LENGTH decides the
# ranking, not detection.
#
# Measured on the Phase 10 shape before the fix, pure random noise reached
# precision 0.0943 and 29.6x lift, beating every tuned detector in the final
# table. These tests build the asymmetric shape on purpose, because the
# symmetric fixtures above cannot see it.
# --------------------------------------------------------------------------

def _asymmetric_frame(seed: int, n_positive: int = 200,
                      n_quiet_per_ticker: int = 2000,
                      n_tickers: int = 12) -> pd.DataFrame:
    """The real frame's shape: 48-bar episodes, one-bar quiet windows."""
    horizon = load_config()["decision"]["horizon_hours"]
    rng = np.random.default_rng(seed)
    base = date_str_to_ts("2025-09-01")
    tickers = [f"TKR{i:02d}" for i in range(n_tickers)]
    blocks = []

    for i in range(n_positive):
        t0 = base + int(rng.integers(horizon, 20_000)) * HOUR
        ts = t0 - np.arange(horizon, 0, -1) * HOUR
        blocks.append(pd.DataFrame({
            "window_id": f"pos-{i:04d}", "ticker": tickers[i % n_tickers],
            "ts_utc": ts, "t0_utc": t0, "score": rng.normal(size=horizon),
            "action": WAIT, "is_scheduled": True, "item_code": "8.01"}))

    for tkr in tickers:
        ts = base + 40_000 * HOUR + np.arange(n_quiet_per_ticker) * HOUR
        blocks.append(pd.DataFrame({
            "window_id": [f"bar:{tkr}:{t}" for t in ts], "ticker": tkr,
            "ts_utc": ts, "t0_utc": pd.NA,
            "score": rng.normal(size=n_quiet_per_ticker),
            "action": WAIT, "is_scheduled": pd.NA, "item_code": pd.NA}))

    return conform(pd.concat(blocks, ignore_index=True))


def test_pure_noise_scores_chance_on_the_real_asymmetric_frame() -> None:
    """THE null. Averaged over seeds, a scorer that knows nothing must land on
    the base rate — on the frame shape the project actually evaluates, not only
    on the symmetric synthetic one.

    Before quiet windows were given the same span as an episode, this measured
    32.7x on exactly this frame.
    """
    lifts = []
    for seed in range(6):
        r = precision_at_alert_budget(_asymmetric_frame(seed))
        lifts.append(r.precision / r.base_rate)
    mean_lift = float(np.mean(lifts))
    assert 0.5 <= mean_lift <= 2.0, (
        f"pure noise scored {mean_lift:.2f}x on the asymmetric frame — window "
        f"length is deciding the ranking again")


def test_a_positive_is_represented_by_its_decision_point_not_its_best_hour() -> None:
    """Chosen by POSITION, never by score.

    Picking the episode's loudest hour is exactly the bug — it is what gave a
    positive 48 draws against a quiet bar's one. An episode whose best hour is
    early and whose decision point is quiet must score quiet.
    """
    horizon = load_config()["decision"]["horizon_hours"]
    base = date_str_to_ts("2025-09-01")
    t0 = base + 500 * HOUR
    loud_early = np.zeros(horizon)
    loud_early[0] = 99.0                       # the best hour, farthest from t0
    pos = pd.DataFrame({
        "window_id": "pos-0", "ticker": "AAA",
        "ts_utc": t0 - np.arange(horizon, 0, -1) * HOUR, "t0_utc": t0,
        "score": loud_early, "action": WAIT, "is_scheduled": True,
        "item_code": "8.01"})
    quiet = pd.DataFrame({
        "window_id": ["bar:AAA:1"], "ticker": ["AAA"],
        "ts_utc": [base + 900 * HOUR], "t0_utc": [pd.NA], "score": [1.0],
        "action": [WAIT], "is_scheduled": [pd.NA], "item_code": [pd.NA]})

    peaks = window_summary(conform(pd.concat([pos, quiet], ignore_index=True)))
    assert peaks.loc["pos-0", "peak_score"] == 0.0, (
        "the episode scored its loudest hour instead of its decision point")


def test_a_quiet_window_is_left_exactly_as_it_is() -> None:
    """Quiet windows are already one bar and collapsing must not touch them.

    Extending them to 48 instead was tried and is worse: quiet windows overlap,
    so a rolling maximum charges one sustained anomaly ~20 alerts while a
    48-bar episode still costs 1 — the original asymmetry, mirrored.
    """
    df = make_synthetic_predictions(**DENSE, signal_strength=2.0, seed=5)
    quiet_ids = df.loc[df["t0_utc"].isna(), "window_id"].unique()
    direct = (df[df["window_id"].isin(quiet_ids)]
              .groupby("window_id", sort=False)["score"].max())
    pd.testing.assert_series_equal(
        window_summary(df).loc[direct.index, "peak_score"], direct,
        check_names=False)
