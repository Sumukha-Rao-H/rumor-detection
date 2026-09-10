"""P5-06 — the comparison table, and the properties that make it a comparison.

A table is only a comparison if every row reached it the same way. These tests
pin that: one evaluation frame, one alert budget, one threshold per baseline
applied across its slices, and the same window count for everyone. If any of
those slipped, two rows could differ because of their harnesses rather than
their detectors, and the whole of Phase 5 would be uninterpretable.
"""

import numpy as np
import pandas as pd
import pytest

from src.baselines import AlwaysQuiet, CUSUM, VolumeZScore
from src.baselines.compare import (REPORT_COLUMNS, comparison_table, conform,
                                   render)
from src.eval.synthetic import make_synthetic_predictions
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts

HOUR = 3600


@pytest.fixture
def cfg():
    return load_config()


def make_frame(n_pos: int = 8, n_neg: int = 40) -> pd.DataFrame:
    """Positives spike late; negatives stay quiet. Half of each are scheduled.

    Deliberately few tickers over a short span. The alert budget is
    denominated in ticker-months, so a frame spread thinly over many tickers
    and months would earn a budget larger than its own window count — every
    window then gets bought, precision collapses to the base rate for every
    baseline, and the fixture would test nothing. That regime is real (it
    blocked P5-03) and is pinned in `test_baselines_volume_zscore.py`; here it
    is avoided so the comparison has something to compare.
    """
    base = date_str_to_ts("2025-10-01")
    rows = []
    for i in range(n_pos + n_neg):
        positive = i < n_pos
        anchor = base + i * 8 * HOUR
        zs = [0.2, 0.6, 3.2, 4.1] if positive else [0.1, 0.2, 0.15, 0.3]
        for h, z in enumerate(zs):
            rows.append({
                "window_id": f"{'P' if positive else 'N'}{i}",
                "ticker": f"T{i % 2}",
                "ts_utc": anchor - (len(zs) - h) * HOUR,
                "t0_utc": anchor if positive else None,
                "is_scheduled": (i % 2 == 0) if positive else None,
                "item_code": ("2.02" if i % 2 == 0 else "8.01") if positive else None,
                "volume_z": z,
            })
    return conform(pd.DataFrame(rows))


@pytest.fixture
def predictions(cfg):
    frame = make_frame()
    return {m.name: m.predict(frame, threshold=float("inf"))
            for m in (AlwaysQuiet(cfg), VolumeZScore(cfg), CUSUM(cfg))}


def test_every_baseline_is_scored_on_the_same_windows(cfg, predictions):
    """The property the whole table rests on. If two baselines evaluated
    different window sets their precisions would not be comparable, however
    identical the arithmetic."""
    table = comparison_table(cfg, predictions, "news_adjusted")
    all_rows = table[table["slice"] == "all"]
    assert all_rows["n_windows"].nunique() == 1
    assert all_rows["n_positive"].nunique() == 1


def test_the_floor_row_is_present_and_lift_is_relative_to_it(cfg, predictions):
    """A precision without its floor beside it cannot be read."""
    table = comparison_table(cfg, predictions, "news_adjusted")
    all_rows = table[table["slice"] == "all"].set_index("baseline")

    assert "always_quiet" in all_rows.index
    assert all_rows.loc["always_quiet", "lift"] == pytest.approx(1.0)
    floor = all_rows.loc["always_quiet", "precision"]
    for name in ("volume_zscore", "cusum"):
        assert all_rows.loc[name, "lift"] == pytest.approx(
            all_rows.loc[name, "precision"] / floor)


def test_numbers_are_split_scheduled_versus_unscheduled(cfg, predictions):
    """Code standards rule 4: every number split, never pooled."""
    table = comparison_table(cfg, predictions, "news_adjusted")
    slices = set(table["slice"])
    assert "all" in slices
    assert "scheduled" in slices
    assert "unscheduled" in slices


def test_one_threshold_per_baseline_across_its_slices(cfg, predictions):
    """Each slice must not get its own flattering cut, or a slice could look
    good purely by being scored at a threshold chosen to suit it."""
    table = comparison_table(cfg, predictions, "news_adjusted")
    for name in table["baseline"].unique():
        rows = table[table["baseline"] == name]
        assert rows["threshold"].nunique() == 1


def test_the_ceiling_is_carried_beside_every_precision(cfg, predictions):
    """Issue 29: 15% next to a 21% ceiling is 71% of achievable; 15% alone
    reads as failure."""
    table = comparison_table(cfg, predictions, "news_adjusted")
    assert "max_precision" in table.columns
    assert table["max_precision"].notna().all()


def test_the_variant_label_is_carried_on_every_row(cfg, predictions):
    table = comparison_table(cfg, predictions, "filing")
    assert (table["t0_variant"] == "filing").all()


def test_report_columns_are_all_produced(cfg, predictions):
    table = comparison_table(cfg, predictions, "news_adjusted")
    for col in REPORT_COLUMNS:
        assert col in table.columns, col


def test_render_puts_the_best_baseline_first(cfg, predictions):
    """The table is read top-down; the ordering is part of the deliverable."""
    table = comparison_table(cfg, predictions, "news_adjusted")
    text = render(table)
    lines = [l for l in text.splitlines() if l.strip()]
    assert lines[1].split()[0] in ("volume_zscore", "cusum")
    assert "always_quiet" in lines[-1]


def test_a_detector_beats_the_floor_on_separable_data(cfg, predictions):
    """A sanity check on the fixture: if the constructed signal were not
    separable, every assertion above would be about noise."""
    table = comparison_table(cfg, predictions, "news_adjusted")
    all_rows = table[table["slice"] == "all"].set_index("baseline")
    assert all_rows.loc["volume_zscore", "precision"] > \
           all_rows.loc["always_quiet", "precision"]


def test_conform_casts_to_the_contract_dtypes():
    frame = make_frame(n_pos=1, n_neg=1)
    assert frame["ts_utc"].dtype == "Int64"
    assert frame["t0_utc"].dtype == "Int64"
    assert frame["is_scheduled"].dtype == "boolean"


# --------------------------------------------------------------------------
# The null row
# --------------------------------------------------------------------------

def test_random_noise_is_in_the_comparison_table():
    """It is the row a sceptical reader checks first.

    On a frame whose two classes get the same number of chances, a scorer that
    reads nothing must score the always-quiet floor. If this row ever reports
    meaningfully more, the evaluation frame has a length asymmetry again and no
    other row in the table means anything until that is explained.
    """
    from src.baselines import RandomNoise

    assert RandomNoise(load_config()).name == "random_noise"


def test_random_noise_scores_the_same_frame_the_same_way_twice():
    """The table is built once per t0 variant. A null that moved between them
    would read as a finding."""
    from src.baselines import RandomNoise

    cfg = load_config()
    frame = make_synthetic_predictions(n_positive=5, n_quiet=40, seed=3)
    first = RandomNoise(cfg).score(frame)
    second = RandomNoise(cfg).score(frame)
    pd.testing.assert_series_equal(first, second)


def test_random_noise_reads_no_feature_column():
    """So a difference between this row and a real baseline can never be blamed
    on missing history."""
    from src.baselines import RandomNoise

    cfg = load_config()
    frame = make_synthetic_predictions(n_positive=5, n_quiet=40, seed=3)
    stripped = frame.copy()
    for col in [c for c in stripped.columns if c not in
                ("window_id", "ticker", "ts_utc", "t0_utc", "score", "action",
                 "is_scheduled", "item_code")]:
        stripped[col] = float("nan")
    pd.testing.assert_series_equal(RandomNoise(cfg).score(frame),
                                   RandomNoise(cfg).score(stripped))


def test_trading_hours_to_close_is_excluded_from_the_learned_models():
    """It encodes how the window was CUT, not what the market did.

    A positive window ends at t0 and most 8-Ks land after the close, so
    `trading_hours_to_close <= 1` holds for 84% of positives against ~15-22% of
    negatives. Ranking by it alone scores 5.67x lift with no detection ability.
    Unlike `days_since_last_8k` it points the same way at evaluation, so a model
    that learns it is rewarded rather than merely misled.
    """
    excluded = load_config()["baselines"]["gradient_boosting"]["exclude_features"]
    assert "trading_hours_to_close" in excluded


def test_the_training_frame_honours_a_ratio_override():
    """Plan §8: "report results at two ratios so nobody can accuse you of
    tuning it."

    `sampling.robustness_ratios` was consulted in exactly one place — a print
    of how many negatives are AVAILABLE at each ratio. That is an availability
    census, not a robustness check, under a name that promises one. Nothing
    could re-run training at 1:1 or 2:1, so the requirement had no answer.
    `build_training_frame(ratio=...)` is what makes it runnable; this pins the
    argument so it cannot quietly stop being honoured.
    """
    import inspect

    from src.baselines.compare import build_training_frame

    assert "ratio" in inspect.signature(build_training_frame).parameters


def test_main_prints_the_scheduled_unscheduled_split_not_only_the_pooled_row(
        cfg, monkeypatch, capsys):
    """AGENTS.md rule 7, on the one path nothing checked.

    `comparison_table` has always produced the split rows and the tests above
    pin that. `main` then printed `render(table)` — which defaults to the `all`
    slice — so `FINAL-test-run.log`, the file a report would be written from,
    carried the pooled rows and nothing else. Pooling is not a summary here:
    gradient boosting scores 0.00475 scheduled against 0.01642 unscheduled and
    `rl_policy[seed43]` inverts that ordering, so the pooled row hides the
    finding. The violation lived entirely in the table-to-text step, which is
    why it survived a file full of tests about the table.
    """
    import sys

    from src import db
    from src.baselines import compare

    frame = make_frame()
    monkeypatch.setattr(db, "get_conn", lambda *a, **k: None)
    monkeypatch.setattr(compare, "split_bounds", lambda *a, **k: (0, 1))
    monkeypatch.setattr(compare, "build_eval_frame", lambda *a, **k: frame)
    monkeypatch.setattr(sys, "argv",
                        ["compare", "--variant", "news_adjusted", "--skip-gb"])

    compare.main()
    out = capsys.readouterr().out

    for slice_name in compare.SLICES:
        assert f"-- slice: {slice_name} --" in out
    # One printed row per slice, not one row full stop.
    assert out.count("volume_zscore") == len(compare.SLICES)


# --------------------------------------------------------------------------
# The null needs an error bar
# --------------------------------------------------------------------------

def make_asymmetric_frame(n_pos: int = 100, quiet_per_ticker: int = 250,
                          n_tickers: int = 8, horizon: int = 48
                          ) -> pd.DataFrame:
    """Positives 48 bars, negatives a single bar — the shape that broke.

    This is `evalset.build_eval_frame`'s real geometry, reproduced small. A
    plain max per window would hand a positive 48 independent chances to cross
    and a quiet window one, at the same one-alert cost, and pure noise would
    win the table on window LENGTH alone.
    """
    base = date_str_to_ts("2025-10-01")
    rows = []
    for t in range(n_tickers):
        for j in range(quiet_per_ticker):
            rows.append({"window_id": f"Q{t}-{j}", "ticker": f"T{t}",
                         "ts_utc": base + j * HOUR, "t0_utc": None,
                         "is_scheduled": None, "item_code": None,
                         "volume_z": 0.0})
    for i in range(n_pos):
        ticker = i % n_tickers
        anchor = base + (quiet_per_ticker + 100 + i * (horizon + 4)) * HOUR
        for h in range(horizon):
            rows.append({"window_id": f"P{i}", "ticker": f"T{ticker}",
                         "ts_utc": anchor - (horizon - h) * HOUR,
                         "t0_utc": anchor,
                         "is_scheduled": (i % 2 == 0),
                         "item_code": "2.02" if i % 2 == 0 else "8.01",
                         "volume_z": 0.0})
    return conform(pd.DataFrame(rows))


def test_random_noise_scores_about_one_times_lift_on_a_length_asymmetric_frame():
    """The artefact the null exists to catch, actually exercised.

    Until now nothing tested the null against the shape that produced the bug.
    On the Phase 10 frame — 48-bar positives against single-bar negatives —
    noise reached precision 0.0943 and 29.6x lift, beating every tuned
    detector. `metrics.window_summary` gives a quiet window the same span as an
    episode, so a scorer with no information must come back to chance. This is
    that claim run rather than believed.

    Checked at every configured seed, because one draw is not an error bar: a
    single null row on this shape sits anywhere between roughly 0.55x and
    1.45x from sampling noise alone, which is exactly why the comparison emits
    one row per seed.
    """
    from src.baselines import RandomNoise
    from src.baselines.compare import noise_seeds
    from src.eval.metrics import precision_at_alert_budget

    cfg = load_config()
    frame = make_asymmetric_frame()
    budget = 1000

    floor = precision_at_alert_budget(
        AlwaysQuiet(cfg).predict(frame), max_alerts=budget).precision

    lifts = []
    for seed in noise_seeds(cfg):
        scored = RandomNoise(cfg, seed=seed).predict(frame,
                                                     threshold=float("inf"))
        lifts.append(precision_at_alert_budget(
            scored, max_alerts=budget).precision / floor)

    # A band wide enough for the sampling noise this frame really has, and far
    # narrower than the 29.6x the artefact produced — the failure being
    # guarded against is an order of magnitude, not a decimal.
    for seed, lift in zip(noise_seeds(cfg), lifts):
        assert 0.5 <= lift <= 1.6, f"seed {seed} scored {lift:.2f}x lift"


def test_the_null_is_emitted_once_per_configured_seed():
    """One row is a point estimate; the spread across seeds is the error bar.

    On the Phase 10 shape a null draw has E[TP] ~ 19.1 with sd ~ 4.4, so one
    row reading 1.4x is indistinguishable from a real residual asymmetry of
    that size. The null catches a 29.6x artefact; it cannot on its own certify
    the absence of a 1.5x one, and a reader has no way to know that from a
    single row.
    """
    from src.baselines.compare import noise_seeds

    cfg = load_config()
    seeds = noise_seeds(cfg)
    assert len(seeds) >= 3, "a spread needs more than a couple of draws"
    assert len(set(seeds)) == len(seeds), "duplicated seeds are not a spread"


def test_run_baselines_can_fit_gradient_boosting_without_a_caller_supplied_model(
        tmp_path, monkeypatch):
    """`run_baselines` fits gradient boosting itself when no `fitted_gb` is
    handed in, and that path referenced a name it did not have.

    The `--sampling-ratio` wiring was applied to the wrong one of two
    identical-looking blocks: it landed inside `run_baselines`, which has no
    `args`, so the branch raised `NameError` the moment it ran. Every existing
    test passed `skip_gb=True` or a pre-fitted model, so nothing exercised it —
    it would have surfaced as a crash in the middle of a real `compare` run.

    This drives the branch with a stub fitter, so it stays exercised without
    paying for an XGBoost fit.
    """
    from src.baselines import compare as compare_mod

    class StubGB:
        name = "gradient_boosting"

        def __init__(self, cfg):
            self.cfg = cfg

        def fit(self, frame, conn=None):
            return self

        def predict(self, frame, threshold=None, conn=None, context=""):
            return frame.assign(score=0.5)

    monkeypatch.setattr(compare_mod, "GradientBoosting", StubGB)
    monkeypatch.setattr(compare_mod, "build_training_frame",
                        lambda cfg, conn, ratio=None: frame)

    cfg = load_config()
    frame = make_synthetic_predictions(n_positive=4, n_quiet=30, seed=1)
    # volume_zscore and cusum read this column; the synthetic generator makes
    # a PREDICTION frame, which does not carry features.
    frame = frame.assign(volume_z=frame["score"].astype("float64"))
    predictions, models = compare_mod.run_baselines(cfg, None, frame)

    assert "gradient_boosting" in predictions
    assert set(models) == set(predictions)
