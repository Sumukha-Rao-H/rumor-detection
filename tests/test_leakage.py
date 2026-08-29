"""Leakage tests — MANDATORY, and deliberately written before `features.py`.

One of the four restrictions this project claims over the prior work is
*forward, not retrospective*. A model that can see past the hour it is
deciding at makes that claim false, and the failure is silent: the scores come
out better, not broken.

`src/pipeline/features.py` does not exist yet. That is the point. A leakage
test written after the features it guards is a test shaped around code that
already passes it.

The detector
------------
Reading code for `center=True` only ever catches the leaks already on the list.
The general test is behavioural:

    1. compute the features
    2. change ONLY the values after time t
    3. recompute

If any feature value at or before `t` moved, it used the future. Nothing about
the implementation needs to be known, which is what lets this hold up against
Phase 4 code nobody has written.

What this proves and what it does not
-------------------------------------
It proves the ABSENCE OF LOOK-AHEAD. It is not a proof of correctness — a
feature that ignores its input entirely can never be caught, and would sail
through.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.config import load_config


# --------------------------------------------------------------------------
# the detector
# --------------------------------------------------------------------------

class LookaheadError(AssertionError):
    """A feature value changed when only later rows were altered."""


def _perturb_after(frame: pd.DataFrame, cut: int) -> pd.DataFrame:
    """Return a copy with everything strictly after `cut` made wildly different.

    Large and structural on purpose: multiplying by 1000 and adding noise makes
    any dependence on the future obvious, rather than something that could hide
    inside floating-point rounding.
    """
    out = frame.copy()
    mask = out.index > cut
    rng = np.random.default_rng(0)
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out.loc[mask, col] = (out.loc[mask, col] * 1000.0
                                  + rng.normal(0, 50, mask.sum()))
    return out


def find_lookahead(fn, frame: pd.DataFrame,
                   cuts: int = 5) -> list[tuple[int, str]]:
    """`(timestamp, column)` pairs whose value depends on data after that time.

    `fn` takes the input frame and returns a feature frame on the same index.
    Comparison is EXACT — any tolerance is a place for a small leak to hide —
    and NaN is treated as equal to NaN, because the warm-up rows of a correct
    trailing window are NaN by design.
    """
    base = fn(frame)
    index = list(frame.index)
    violations: list[tuple[int, str]] = []

    # Cut points spread across the middle; the first and last rows are useless
    # (nothing after, or nothing before).
    positions = np.linspace(len(index) // 4, len(index) - 2, cuts, dtype=int)
    for pos in sorted(set(int(p) for p in positions)):
        cut = index[pos]
        after = fn(_perturb_after(frame, cut))
        past = base.index <= cut
        for col in base.columns:
            a, b = base.loc[past, col], after.loc[past, col]
            differs = ~((a == b) | (a.isna() & b.isna()))
            for ts in a.index[differs]:
                violations.append((int(ts), col))
    return sorted(set(violations))


def assert_no_lookahead(fn, frame: pd.DataFrame) -> None:
    """Raise `LookaheadError` naming the first offending column and timestamp."""
    violations = find_lookahead(fn, frame)
    if violations:
        ts, col = violations[0]
        raise LookaheadError(
            f"feature '{col}' at ts={ts} changed when only rows AFTER ts={ts} "
            f"were altered — it is reading the future. "
            f"{len(violations)} violation(s) across "
            f"{len({c for _, c in violations})} column(s)."
        )


# --------------------------------------------------------------------------
# a small hourly series to run them against
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    """400 hourly bars, indexed by UTC epoch seconds — the project's time unit."""
    rng = np.random.default_rng(42)
    n = 400
    start = 1_725_148_800  # 2024-09-01 00:00 UTC, the study window start
    index = pd.Index([start + 3600 * i for i in range(n)], name="ts_utc")
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    volume = rng.lognormal(12, 0.4, n)
    return pd.DataFrame({"close": close, "volume": volume}, index=index)


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


# --------------------------------------------------------------------------
# correct feature builders
# --------------------------------------------------------------------------

def good_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Trailing windows only — the shape `code-standards.md` prescribes."""
    close, volume = frame["close"], frame["volume"]
    base = volume.shift(1).rolling(48, min_periods=12)
    return pd.DataFrame({
        "ret_1h": close.pct_change(1),
        "ret_4h": close.pct_change(4),
        "vol_z": (volume - base.mean()) / base.std(),
        "realised_vol": close.pct_change().rolling(24, min_periods=12).std(),
    }, index=frame.index)


def current_bar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Uses the bar AT time t, unshifted. Not look-ahead: that bar has closed."""
    volume = frame["volume"]
    base = volume.rolling(48, min_periods=12)
    return pd.DataFrame({"vol_z": (volume - base.mean()) / base.std()},
                        index=frame.index)


# --------------------------------------------------------------------------
# deliberately leaky builders — the proof the detector works
# --------------------------------------------------------------------------

def centered_window(frame: pd.DataFrame) -> pd.DataFrame:
    """`center=True` reaches forward. The canonical mistake."""
    volume = frame["volume"]
    base = volume.rolling(48, min_periods=12, center=True)
    return pd.DataFrame({"vol_z": (volume - base.mean()) / base.std()},
                        index=frame.index)


def full_series_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    """Whole-series mean and std leak every row into every other row."""
    volume = frame["volume"]
    return pd.DataFrame({"vol_z": (volume - volume.mean()) / volume.std()},
                        index=frame.index)


def backfilled(frame: pd.DataFrame) -> pd.DataFrame:
    """Leakage through imputation rather than a window."""
    close = frame["close"].copy()
    close.iloc[::10] = np.nan
    return pd.DataFrame({"close_filled": close.bfill()}, index=frame.index)


def next_hour_return(frame: pd.DataFrame) -> pd.DataFrame:
    """The blunt case: tomorrow's number on today's row."""
    return pd.DataFrame({"ret_next": frame["close"].shift(-1) / frame["close"] - 1},
                        index=frame.index)


# --------------------------------------------------------------------------
# correct code must not be flagged
# --------------------------------------------------------------------------

def test_a_trailing_feature_passes(bars):
    assert_no_lookahead(good_features, bars)


def test_using_the_current_bar_is_not_lookahead(bars):
    """At hour t the bar for t has closed and its data is public.

    `code-standards.md` separately requires the z-score baseline to shift by
    one, but that is baseline hygiene — keeping a spike out of its own normal —
    not look-ahead. Conflating the two would make this test fire on correct
    code.
    """
    assert_no_lookahead(current_bar_features, bars)


def test_nan_warmup_rows_do_not_count_as_a_change(bars):
    """The first rows of a trailing window are NaN by design, in both runs."""
    out = good_features(bars)
    assert out["vol_z"].isna().sum() > 0, "the fixture must exercise warm-up"
    assert find_lookahead(good_features, bars) == []


# --------------------------------------------------------------------------
# the Done-when: injected leakage fails loudly
# --------------------------------------------------------------------------

@pytest.mark.parametrize("builder, name", [
    (centered_window, "centered rolling window"),
    (full_series_zscore, "whole-series z-score"),
    (backfilled, "backward fill from later rows"),
    (next_hour_return, "next hour's return"),
])
def test_injected_leakage_is_caught(bars, builder, name):
    with pytest.raises(LookaheadError):
        assert_no_lookahead(builder, bars)


def test_the_violation_names_the_column_and_time(bars):
    """A failure nobody can act on is barely better than no test."""
    with pytest.raises(LookaheadError) as exc:
        assert_no_lookahead(full_series_zscore, bars)
    message = str(exc.value)
    assert "vol_z" in message
    assert "reading the future" in message
    assert "ts=" in message


def test_a_leak_of_one_row_is_still_caught(bars):
    """Not just the loud cases: a single future row must be enough."""
    def barely_leaky(frame: pd.DataFrame) -> pd.DataFrame:
        out = good_features(frame)
        out.iloc[100, out.columns.get_loc("ret_1h")] = frame["close"].iloc[-1]
        return out

    with pytest.raises(LookaheadError):
        assert_no_lookahead(barely_leaky, bars)


# --------------------------------------------------------------------------
# scalers: fit on train only
# --------------------------------------------------------------------------

def _split_points(cfg: dict, n: int) -> tuple[int, int]:
    train = int(n * cfg["split"]["train"])
    val = int(n * (cfg["split"]["train"] + cfg["split"]["val"]))
    return train, val


def test_scaler_fit_on_all_data_is_caught(bars, cfg):
    """The standard says fit on train ONLY. Fitting on everything leaks the
    test set's mean and variance into every training row."""
    train_end, _ = _split_points(cfg, len(bars))
    values = bars["volume"]

    all_fit = (values - values.mean()) / values.std()
    train_only = values.iloc[:train_end]
    correct = (values - train_only.mean()) / train_only.std()

    assert not np.allclose(all_fit.iloc[:train_end], correct.iloc[:train_end]), (
        "if these agreed the test would prove nothing about this dataset")


def test_scaler_fit_on_train_only_passes(bars, cfg):
    """Refitting after more future data arrives must not move a training row."""
    train_end, _ = _split_points(cfg, len(bars))
    values = bars["volume"]
    train = values.iloc[:train_end]

    scaled_now = (values.iloc[:train_end] - train.mean()) / train.std()
    extended = pd.concat([values, values * 1000])          # the future changes
    train_again = extended.iloc[:train_end]
    scaled_later = (extended.iloc[:train_end] - train_again.mean()) / train_again.std()

    pd.testing.assert_series_equal(scaled_now, scaled_later)


# --------------------------------------------------------------------------
# the split itself must not leak
# --------------------------------------------------------------------------

def test_the_split_is_temporal_not_random(cfg):
    assert cfg["split"]["method"] == "temporal", (
        "a random split puts later hours in train and earlier ones in test, "
        "which leaks the future through the split itself")


def test_test_split_is_strictly_after_train(bars, cfg):
    train_end, val_end = _split_points(cfg, len(bars))
    index = bars.index
    train, val, test = index[:train_end], index[train_end:val_end], index[val_end:]

    assert train.max() < val.min() < val.max() < test.min()
    assert len(train) + len(val) + len(test) == len(index)
    assert set(train) & set(test) == set()


def test_the_split_fractions_come_from_config(cfg):
    total = cfg["split"]["train"] + cfg["split"]["val"] + cfg["split"]["test"]
    assert abs(total - 1.0) < 1e-9, f"split fractions sum to {total}, not 1"
