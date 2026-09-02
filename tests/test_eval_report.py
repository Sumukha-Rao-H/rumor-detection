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


def test_scheduled_and_unscheduled_slices_are_not_swapped(frame) -> None:
    """`test_scheduled_slice_keeps_quiet_windows` is symmetric in `name` and
    would pass even if the two labels were swapped. This checks content, not
    just quiet-window counts: the "scheduled" slice's positives must actually
    be `is_scheduled == True` rows, and "unscheduled" must actually be
    `is_scheduled == False` rows.
    """
    slices = slice_frames(frame)
    for name, expected in (("scheduled", True), ("unscheduled", False)):
        positives = slices[name][slices[name]["t0_utc"].notna()]
        assert len(positives) > 0
        assert (positives["is_scheduled"] == expected).all(), name


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


def test_t0_variant_gap_sign_is_independent_of_dict_order() -> None:
    """Same two frames, only the `frames` dict's iteration order differs — the
    reported gap must have the SAME sign (and value) either way. Regression
    test for the order-dependent sign bug: `t0_variant_gap` used to read the
    two variants positionally off however `report_table` happened to append
    rows, so swapping the caller's dict order flipped the sign.
    """
    filing = make_synthetic_predictions(**DENSE, signal_strength=2.0, seed=5)
    news = make_synthetic_predictions(**DENSE, signal_strength=1.5, seed=5)

    forward = report_table({"filing": filing, "news_adjusted": news})
    backward = report_table({"news_adjusted": news, "filing": filing})

    gap_forward = t0_variant_gap(forward)
    gap_backward = t0_variant_gap(backward)

    assert not math.isnan(gap_forward)
    assert gap_forward == pytest.approx(gap_backward)


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


def test_split_by_without_scheduled_and_unscheduled_is_rejected(frame) -> None:
    """AGENTS.md rule 5's scheduled/unscheduled split is non-negotiable. If
    `split_by` (from config, or passed explicitly) drops either name,
    `slice_frames` must refuse loudly rather than silently emitting a
    pooled-only report.
    """
    with pytest.raises(ValueError, match="scheduled"):
        slice_frames(frame, split_by=["item_code"])
    with pytest.raises(ValueError, match="unscheduled"):
        slice_frames(frame, split_by=["scheduled", "item_code"])


def test_report_table_warns_on_a_fully_empty_slice(caplog) -> None:
    """A slice with zero rows at all (no matching positives, no quiet windows)
    must be visibly flagged, not silently dropped from the table."""
    only_scheduled = make_synthetic_predictions(
        n_positive=6, n_quiet=0, n_tickers=3, span_days=60,
        signal_strength=2.0, seed=11,
    ).copy()
    only_scheduled["is_scheduled"] = True

    with caplog.at_level("WARNING", logger="src.eval.report"):
        t = report_table(only_scheduled, max_alerts=3)

    assert "unscheduled" not in set(t["slice"])
    assert any("empty" in r.message for r in caplog.records)


def test_all_scheduled_positives_through_report_table() -> None:
    """All positives scheduled, run through `report_table`'s own per-slice
    loop (not just `evaluate` directly): the unscheduled slice must reduce to
    quiet-only (n_positive=0, recall nan), the scheduled slice must match the
    pooled row.
    """
    only_scheduled = make_synthetic_predictions(
        n_positive=6, n_quiet=200, n_tickers=4, span_days=60,
        signal_strength=2.0, seed=11,
    ).copy()
    only_scheduled["is_scheduled"] = only_scheduled["is_scheduled"].where(
        only_scheduled["t0_utc"].isna(), True
    )
    t = report_table(only_scheduled, max_alerts=3)

    all_row = t[t["slice"] == "all"].iloc[0]
    sched = t[t["slice"] == "scheduled"].iloc[0]
    unsched = t[t["slice"] == "unscheduled"].iloc[0]

    assert sched["n_positive"] == all_row["n_positive"]
    assert unsched["n_positive"] == 0
    assert math.isnan(unsched["recall"])


def test_all_unscheduled_positives_through_report_table() -> None:
    """Mirror of the all-scheduled case: all positives unscheduled."""
    only_unscheduled = make_synthetic_predictions(
        n_positive=6, n_quiet=200, n_tickers=4, span_days=60,
        signal_strength=2.0, seed=13,
    ).copy()
    only_unscheduled["is_scheduled"] = only_unscheduled["is_scheduled"].where(
        only_unscheduled["t0_utc"].isna(), False
    )
    t = report_table(only_unscheduled, max_alerts=3)

    all_row = t[t["slice"] == "all"].iloc[0]
    sched = t[t["slice"] == "scheduled"].iloc[0]
    unsched = t[t["slice"] == "unscheduled"].iloc[0]

    assert unsched["n_positive"] == all_row["n_positive"]
    assert sched["n_positive"] == 0
    assert math.isnan(sched["recall"])


def test_each_slice_is_validated_exactly_once(frame, monkeypatch) -> None:
    """Regression guard for P1-Xb.

    Validation used to run three times per slice — `evaluate`,
    `detection_delays` and `calibration_summary` each did their own. That was
    0.9 s of report_table's 1.1 s. The fix was a public/private split, and this
    test stops it drifting back: one validation per slice, plus a couple for
    the variant-level threshold.

    Deliberately NOT solved with a `validated` marker on `DataFrame.attrs`.
    Those attrs survive `sample()` too, which shuffles rows and breaks the
    "ts_utc ascends within a window" invariant — a guard that can silently lie
    is worse than a slow one.
    """
    import src.eval.contract as contract
    import src.eval.metrics as metrics
    import src.eval.report as report

    calls = []
    original = contract.validate_predictions

    def counted(df):
        calls.append(1)
        return original(df)

    for module in (contract, metrics, report):
        monkeypatch.setattr(module, "validate_predictions", counted, raising=False)

    table = report_table(frame)
    assert len(calls) <= len(table) + 3, (
        f"{len(calls)} validations for {len(table)} slices — "
        f"redundant validation has crept back"
    )


def test_public_metric_entry_points_still_validate(frame) -> None:
    """The private fast paths must not have loosened the public guards."""
    from src.eval.metrics import (
        brier_score,
        calibration_summary,
        detection_delay_summary,
        detection_delays,
        expected_calibration_error,
        reliability_curve,
    )

    bad = frame.copy()
    bad.loc[bad.index[0], "action"] = "HOLD"
    for fn in (detection_delays, detection_delay_summary, brier_score,
               reliability_curve, expected_calibration_error, calibration_summary):
        with pytest.raises(ValueError, match="unknown action"):
            fn(bad)


def test_a_multi_item_filing_reaches_every_component_item_slice():
    """An 8-K reports every item it covers, so `item_code` is a LIST.

    Matching that stored string atomically gave "2.02,8.01" its own private
    bucket and left it out of both `item 2.02` and `item 8.01`. 4,945 of
    16,842 real events carry more than one code, so roughly a third of the
    population was missing from the per-item breakdown.
    """
    from src.eval.report import slice_frames

    rows = []
    for wid, code, t0 in (("w-multi", "2.02,8.01", 1_000),
                          ("w-single", "2.02", 2_000),
                          ("w-quiet", None, None)):
        for i in range(3):
            rows.append({
                "window_id": wid, "ticker": "AAA", "ts_utc": 100 + i,
                "score": 0.1 * i, "action": "WAIT",
                "t0_utc": t0, "is_scheduled": None if t0 is None else True,
                "item_code": code,
            })
    df = pd.DataFrame(rows)

    out = slice_frames(df, split_by=["scheduled", "unscheduled", "item_code"])

    assert "item 2.02" in out and "item 8.01" in out
    assert "item 2.02,8.01" not in out, "the raw joined string became a bucket"

    def windows(name):
        f = out[name]
        return set(f[f["t0_utc"].notna()]["window_id"])

    assert windows("item 2.02") == {"w-multi", "w-single"}
    assert windows("item 8.01") == {"w-multi"}
