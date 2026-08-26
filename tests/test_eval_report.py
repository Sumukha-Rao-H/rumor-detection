"""The report table — slices, action distribution, t0 variants.

Two tests carry this file:

`test_always_wait_is_flagged_degenerate` is the task's Done-when. The plan's
standing worry is a policy that learns to always wait and scores well by never
being wrong; one boolean column has to make that unmissable.

`test_scheduled_slice_keeps_quiet_windows` guards the trap. Slicing on
`is_scheduled` deletes every negative — they have no event type — and precision
becomes 1.0 for every model. `test_naive_slicing_would_give_precision_one`
demonstrates the wrong version so the reason the code looks odd is on record.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.eval.contract import FLAG, WAIT, actions_from_scores
from src.eval.metrics import precision_at_alert_budget
from src.eval.report import (
    COLUMNS,
    action_distribution,
    evaluate,
    report_table,
    slice_frames,
    t0_variant_gap,
)
from src.eval.synthetic import make_synthetic_predictions

DENSE = dict(n_positive=60, n_quiet=1500, n_tickers=8, span_days=180)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return make_synthetic_predictions(**DENSE, signal_strength=2.0, seed=5)


@pytest.fixture(scope="module")
def table(frame) -> pd.DataFrame:
    return report_table(frame)


# --- the Done-when -------------------------------------------------------


def test_always_wait_is_flagged_degenerate(frame) -> None:
    """A policy that never flags must be obvious at a glance, not something a
    reader infers from three nans."""
    t = report_table(frame, max_alerts=0)
    assert t["degenerate"].all()
    assert (t["n_alerts"] == 0).all()
    assert (t["pct_windows_alerted"] == 0.0).all()
    assert t["precision"].isna().all()


def test_a_working_model_is_not_flagged_degenerate(table) -> None:
    assert not table["degenerate"].any()


# --- the slicing trap ----------------------------------------------------


def test_scheduled_slice_keeps_quiet_windows(frame) -> None:
    """Every slice retains all the false-alarm opportunities."""
    slices = slice_frames(frame)
    n_quiet = int(frame["t0_utc"].isna().groupby(frame["window_id"]).first().sum())
    for name in ("scheduled", "unscheduled"):
        quiet_in_slice = slices[name].loc[slices[name]["t0_utc"].isna(), "window_id"].nunique()
        assert quiet_in_slice == n_quiet, name


def test_sliced_precision_is_not_trivially_one(table) -> None:
    for name in ("scheduled", "unscheduled"):
        p = table.loc[table["slice"] == name, "precision"].iloc[0]
        assert p < 1.0, f"{name} precision {p} — the negatives were dropped"


def test_naive_slicing_would_give_precision_one(frame) -> None:
    """The wrong implementation, demonstrated.

    Kept as a test so the reason `slice_frames` re-attaches quiet windows is on
    record rather than looking like an accident.
    """
    naive = frame[frame["is_scheduled"] == True]  # noqa: E712
    assert naive["t0_utc"].notna().all(), "naive slice contains no negatives"
    r = precision_at_alert_budget(naive, max_alerts=5)
    assert r.precision == 1.0


# --- one operating point -------------------------------------------------


def test_all_slices_share_one_threshold(table) -> None:
    """Per-slice thresholds would put each slice at its own operating point and
    quietly flatter the ones where the model is weak."""
    assert table["threshold"].nunique() == 1


def test_threshold_is_the_one_the_budget_picked(frame, table) -> None:
    expected = precision_at_alert_budget(frame).threshold
    assert table["threshold"].iloc[0] == pytest.approx(expected)


# --- action distribution -------------------------------------------------


def test_action_distribution_per_hour_and_per_window(frame) -> None:
    """Both levels are reported: the per-hour share is tiny by construction
    (one flag per 48-hour window), so the per-window rate is the readable one.
    """
    decided = actions_from_scores(frame, threshold=1.5)
    d = action_distribution(decided)
    assert d["n_wait_hours"] + d["n_flag_hours"] == len(decided)
    assert 0 < d["pct_hours_flagged"] < 0.05
    assert d["pct_windows_alerted"] > d["pct_hours_flagged"]


def test_action_distribution_on_an_all_wait_frame(frame) -> None:
    d = action_distribution(frame)
    assert d["n_flag_hours"] == 0
    assert d["pct_windows_alerted"] == 0.0


# --- t0 variants ---------------------------------------------------------


def test_both_t0_variants_appear() -> None:
    """The contract carries one t0 column, so the same model is evaluated once
    per variant and the results sit side by side."""
    filing = make_synthetic_predictions(**DENSE, signal_strength=2.0, seed=5)
    news = make_synthetic_predictions(**DENSE, signal_strength=1.5, seed=5)
    t = report_table({"filing": filing, "news_adjusted": news})
    assert set(t["t0_variant"]) == {"filing", "news_adjusted"}
    assert (t.groupby("t0_variant").size() > 1).all()


def test_t0_variant_gap_reports_a_difference() -> None:
    filing = make_synthetic_predictions(**DENSE, signal_strength=2.0, seed=5)
    news = make_synthetic_predictions(**DENSE, signal_strength=1.5, seed=5)
    t = report_table({"filing": filing, "news_adjusted": news})
    assert not math.isnan(t0_variant_gap(t))


def test_bare_dataframe_is_treated_as_one_variant(table) -> None:
    assert set(table["t0_variant"]) == {"default"}


# --- calibration in the table -------------------------------------------


def test_unbounded_scores_give_nan_calibration_not_an_error(table) -> None:
    """A report covering several models must not fail because one of them is a
    threshold rule. nan here means 'not applicable', not 'failed' —
    calibration_summary still refuses loudly when called directly.
    """
    assert table["brier"].isna().all()
    assert table["ece"].isna().all()


def test_calibration_appears_when_scores_are_probabilities(frame) -> None:
    probs = frame.copy()
    ranked = probs["score"].rank(pct=True)
    probs["score"] = ranked.astype("float64")
    t = report_table(probs)
    assert t["brier"].notna().all()
    assert t["ece"].notna().all()


# --- shape and edges -----------------------------------------------------


def test_table_columns_are_stable(table) -> None:
    """The report and dashboard both depend on this shape."""
    assert list(table.columns) == COLUMNS


def test_item_code_slices_are_present(table) -> None:
    item_rows = [s for s in table["slice"] if s.startswith("item ")]
    assert len(item_rows) >= 5


def test_slice_with_no_positives_still_emits_a_row(frame) -> None:
    """An item code with no events should be visibly absent, not silently
    missing from the table."""
    only_quiet = frame[frame["t0_utc"].isna()]
    row = evaluate(only_quiet, threshold=1.5)
    assert row["n_positive"] == 0
    assert math.isnan(row["recall"])


def test_evaluate_without_a_threshold_picks_its_own(frame) -> None:
    row = evaluate(frame)
    assert row["threshold"] == pytest.approx(precision_at_alert_budget(frame).threshold)


def test_split_by_is_read_from_config(frame) -> None:
    from src.utils.config import load_config

    splits = load_config()["eval"]["split_by"]
    names = set(slice_frames(frame))
    for s in splits:
        if s == "item_code":
            assert any(n.startswith("item ") for n in names)
        else:
            assert s in names
