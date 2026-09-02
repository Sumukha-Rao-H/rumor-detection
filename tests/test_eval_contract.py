"""The evaluation data contract.

Every baseline and the learned policy must emit this shape, so these tests
matter more than their size suggests: a contract nothing checks is a
convention, and conventions drift.

The important one is `test_row_scored_after_t0_is_rejected`. A row whose hour
falls after its window's t0 is a moment when the news was already public.
Scoring it is leakage, it would inflate every lead time in the report, and
nothing else in the pipeline would complain.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.eval.contract import (
    FLAG,
    NON_NULL,
    SCHEMA,
    WAIT,
    actions_from_scores,
    allowed_actions,
    conform,
    empty_frame,
    validate_predictions,
)
from src.eval.synthetic import make_synthetic_predictions

HOUR = 3600


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return make_synthetic_predictions(n_positive=5, n_quiet=10, seed=7)


def test_synthetic_frame_validates(frame) -> None:
    """The task's Done-when, stated literally."""
    out = validate_predictions(frame)
    assert len(out) == len(frame)
    assert list(out.columns) == list(SCHEMA)


def test_dtypes_match_the_contract(frame) -> None:
    for col, dtype in SCHEMA.items():
        assert str(frame[col].dtype) == dtype, col


def test_positive_and_quiet_windows_are_both_present(frame) -> None:
    """t0_utc is the label: present on positives, null on quiet windows."""
    assert frame["t0_utc"].notna().any()
    assert frame["t0_utc"].isna().any()


def test_quiet_windows_have_no_event_metadata(frame) -> None:
    quiet = frame[frame["t0_utc"].isna()]
    assert quiet["is_scheduled"].isna().all()
    assert quiet["item_code"].isna().all()


def test_all_wait_frame_is_valid(frame) -> None:
    """The always-quiet baseline must flow through the metrics, not error.

    It is the base-rate floor and the whole reason plain accuracy is banned, so
    it has to be representable.
    """
    assert (frame["action"] == WAIT).all()
    validate_predictions(frame)


def test_scores_may_be_unbounded(frame) -> None:
    """A volume z-score baseline emits values outside 0-1. Forcing a
    probability range here would exclude it."""
    df = frame.copy()
    df.loc[df.index[0], "score"] = 47.5
    df.loc[df.index[1], "score"] = -12.0
    validate_predictions(df)


@pytest.mark.parametrize("column", list(SCHEMA))
def test_missing_column_is_rejected(frame, column) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        conform(frame.drop(columns=[column]))


@pytest.mark.parametrize("column", NON_NULL)
def test_null_in_required_column_is_rejected(frame, column) -> None:
    df = frame.copy()
    df.loc[df.index[0], column] = pd.NA
    with pytest.raises(ValueError, match=f"{column!r} may not be null"):
        validate_predictions(df)


def test_unknown_action_is_rejected(frame) -> None:
    df = frame.copy()
    df.loc[df.index[0], "action"] = "HOLD"
    with pytest.raises(ValueError, match="unknown action"):
        validate_predictions(df)


def test_two_flags_in_one_window_are_rejected(frame) -> None:
    """An episode ends when it flags, so a second FLAG is a bug in the caller."""
    df = frame.copy()
    first = df["window_id"].iloc[0]
    idx = df.index[df["window_id"] == first][:2]
    df.loc[idx, "action"] = FLAG
    with pytest.raises(ValueError, match="at most one FLAG"):
        validate_predictions(df)


def test_one_flag_per_window_is_fine(frame) -> None:
    df = frame.copy()
    df.loc[df.index[0], "action"] = FLAG
    validate_predictions(df)


def test_row_scored_after_t0_is_rejected(frame) -> None:
    """THE leakage guard.

    An hour after t0 is a moment the market already knew. Scoring it would make
    the model look prescient about news that had already broken, and no other
    check in the pipeline would notice.
    """
    df = frame.copy()
    pos = df.index[df["t0_utc"].notna()][0]
    df.loc[pos, "ts_utc"] = df.loc[pos, "t0_utc"] + HOUR
    with pytest.raises(ValueError, match="scored AFTER their window's t0"):
        validate_predictions(df)


@pytest.mark.parametrize("column", ["t0_utc", "ticker", "is_scheduled", "item_code"])
def test_inconsistent_value_within_a_window_is_rejected(frame, column) -> None:
    """A window is one episode: t0, ticker and event metadata cannot change
    hour to hour. Corrupts exactly what `metrics.py` collapses with
    `.first()` per window, so this must be caught here, not there."""
    df = frame.copy()
    # a positive window has non-null t0_utc/is_scheduled/item_code to mutate
    pos_window = df.loc[df["t0_utc"].notna(), "window_id"].iloc[0]
    idx = df.index[df["window_id"] == pos_window]
    assert len(idx) >= 2

    if column == "t0_utc":
        df.loc[idx[0], column] = df.loc[idx[0], column] + HOUR
    elif column == "ticker":
        df.loc[idx[0], column] = "SOMEOTHERTICKER"
    elif column == "is_scheduled":
        df.loc[idx[0], column] = not bool(df.loc[idx[1], column])
    elif column == "item_code":
        df.loc[idx[0], column] = "9.99"

    with pytest.raises(ValueError, match=f"{column!r} must be constant"):
        validate_predictions(df)


def test_is_scheduled_null_on_a_positive_window_is_rejected(frame) -> None:
    """Event metadata missing on a window that has a t0 — the design doc calls
    mixed null/non-null `is_scheduled` within one window a bug."""
    df = frame.copy()
    pos_window = df.loc[df["t0_utc"].notna(), "window_id"].iloc[0]
    idx = df.index[df["window_id"] == pos_window]
    df.loc[idx, "is_scheduled"] = pd.NA
    with pytest.raises(ValueError, match="'is_scheduled' must be null exactly"):
        validate_predictions(df)


def test_item_code_present_on_a_quiet_window_is_rejected(frame) -> None:
    """Event metadata present on a window that has no t0 — a quiet window
    cannot also carry an item code."""
    df = frame.copy()
    quiet_window = df.loc[df["t0_utc"].isna(), "window_id"].iloc[0]
    idx = df.index[df["window_id"] == quiet_window]
    df.loc[idx, "item_code"] = "5.02"
    with pytest.raises(ValueError, match="'item_code' must be null exactly"):
        validate_predictions(df)


def test_infinite_score_is_rejected(frame) -> None:
    """+inf/-inf would poison Brier and calibration aggregates downstream;
    pd.isna() does not flag it, so it needs its own check."""
    df = frame.copy()
    df.loc[df.index[0], "score"] = float("inf")
    with pytest.raises(ValueError, match="must be finite"):
        validate_predictions(df)

    df = frame.copy()
    df.loc[df.index[0], "score"] = float("-inf")
    with pytest.raises(ValueError, match="must be finite"):
        validate_predictions(df)


def test_row_exactly_at_t0_is_allowed(frame) -> None:
    """t0 itself is the last decidable hour — the boundary is inclusive."""
    df = frame.copy()
    pos = df.index[df["t0_utc"].notna()][0]
    df.loc[pos, "ts_utc"] = df.loc[pos, "t0_utc"]
    validate_predictions(df.sort_values(["window_id", "ts_utc"]).reset_index(drop=True))


def test_duplicate_hour_is_rejected(frame) -> None:
    df = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    df = df.sort_values(["window_id", "ts_utc"]).reset_index(drop=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_predictions(df)


def test_unsorted_hours_are_rejected(frame) -> None:
    df = frame.copy()
    idx = df.index[df["window_id"] == df["window_id"].iloc[0]][:2]
    df.loc[idx[0], "ts_utc"], df.loc[idx[1], "ts_utc"] = (
        df.loc[idx[1], "ts_utc"], df.loc[idx[0], "ts_utc"],
    )
    with pytest.raises(ValueError, match="must ascend"):
        validate_predictions(df)


def test_empty_frame_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_predictions(empty_frame())


def test_actions_come_from_config() -> None:
    """Not hardcoded in the module."""
    from src.utils.config import load_config

    assert allowed_actions() == load_config()["decision"]["actions"]
    assert {WAIT, FLAG} <= set(allowed_actions())


def test_actions_from_scores_flags_first_crossing_only(frame) -> None:
    """A threshold baseline stops at the first crossing.

    Flagging every subsequent hour too would spend the alert budget many times
    over on a single window and make the baseline look far worse than it is.
    """
    out = actions_from_scores(frame, threshold=1.0)
    per_window = out[out["action"] == FLAG].groupby("window_id").size()
    assert (per_window <= 1).all()
    validate_predictions(out)


def test_actions_from_scores_flags_the_chronologically_first_crossing(frame) -> None:
    """The docstring promises the FIRST hour at or above threshold — first by
    `ts_utc`, not first by row position. A frame whose rows are not already
    time-sorted (built ticker-major, or reshaped) must still flag the
    earliest hour, not whichever crossing happens to appear first in the
    frame."""
    df = pd.DataFrame({
        "window_id": ["w1", "w1", "w1"],
        "ticker": ["TKR000"] * 3,
        "ts_utc": pd.array([200, 100, 300], dtype="Int64"),  # out of order
        "t0_utc": pd.array([pd.NA, pd.NA, pd.NA], dtype="Int64"),
        "score": [5.0, 5.0, 0.0],
        "action": pd.array([WAIT, WAIT, WAIT], dtype="string"),
        "is_scheduled": pd.array([pd.NA, pd.NA, pd.NA], dtype="boolean"),
        "item_code": pd.array([pd.NA, pd.NA, pd.NA], dtype="string"),
    })
    out = actions_from_scores(df, threshold=1.0)

    flagged = out.loc[out["action"] == FLAG]
    assert len(flagged) == 1
    assert flagged["ts_utc"].iloc[0] == 100, (
        "ts_utc=100 crosses first chronologically; flagging ts_utc=200 (which "
        "merely appears first in the frame) would be the row-order bug"
    )
    # row order is preserved — actions_from_scores must not reshuffle its input
    assert list(out["ts_utc"]) == [200, 100, 300]


def test_actions_from_scores_flags_nothing_when_threshold_unreachable(frame) -> None:
    out = actions_from_scores(frame, threshold=1e9)
    assert (out["action"] == WAIT).all()
    validate_predictions(out)


def test_synthetic_is_reproducible() -> None:
    a = make_synthetic_predictions(n_positive=3, n_quiet=3, seed=42)
    b = make_synthetic_predictions(n_positive=3, n_quiet=3, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_signal_strength_zero_makes_positives_indistinguishable() -> None:
    """Needed by P1-08: a metric must report chance-level performance on data
    with no signal, rather than something flattering."""
    df = make_synthetic_predictions(n_positive=60, n_quiet=60,
                                    signal_strength=0.0, seed=3)
    pos = df.loc[df["t0_utc"].notna(), "score"].mean()
    quiet = df.loc[df["t0_utc"].isna(), "score"].mean()
    assert abs(pos - quiet) < 0.15
