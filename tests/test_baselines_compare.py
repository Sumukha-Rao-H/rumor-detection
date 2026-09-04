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
