"""The evaluation data contract — the shape every model must produce.

One row per **(decision window, hour)**. Every baseline, and later the learned
policy, emits this and nothing else, so the metrics in `metrics.py` compare
like with like instead of needing a per-model adapter.

Columns
-------
======================  =========  ======  =================================
column                  type       null?   meaning
======================  =========  ======  =================================
window_id               string     no      one decision window (one episode)
ticker                  string     no
ts_utc                  Int64      no      the decision hour, UTC epoch secs
t0_utc                  Int64      YES     when the market learned.
                                           **Null means a quiet window**
score                   float64    no      higher = news more likely coming
action                  string     no      WAIT or FLAG (config.decision)
is_scheduled            boolean    YES     null on quiet windows
item_code               string     YES     null on quiet windows
======================  =========  ======  =================================

`t0_utc` IS the label. A window with a t0 is a positive (the hours before a
real 8-K); a window without one is a quiet negative. There is deliberately no
separate `label` column — it would be derivable from `t0_utc`, and a redundant
column is a column that can disagree with what it duplicates.

`window_id` groups the hours of one episode. It is needed for two things the
project cannot compute without it: the alert budget counts one alert per
*window*, not per hour, and detection delay is measured from the flag to that
window's t0. Under the stopping formulation an episode ends at FLAG, so there
is at most one FLAG per window and `validate_predictions` enforces it.

The two t0 variants
-------------------
`config.eval.t0_variants` is [filing, news_adjusted] and both get reported.
Rather than carry two columns, `t0_utc` means *the variant currently being
evaluated* and the whole evaluation is run twice. Reporting the gap is then a
comparison of two result tables, and every metric stays single-purpose.

Scores are not probabilities
----------------------------
A volume z-score baseline emits unbounded values. The contract does not force a
0-1 range; calibration (brier, ece) applies only to models that claim to output
probabilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import load_config

# column -> pandas dtype. Nullable extension types ("Int64", "boolean",
# "string") are used so a missing t0 is a real null rather than a NaN that
# would quietly coerce an integer column to float.
SCHEMA: dict[str, str] = {
    "window_id": "string",
    "ticker": "string",
    "ts_utc": "Int64",
    "t0_utc": "Int64",
    "score": "float64",
    "action": "string",
    "is_scheduled": "boolean",
    "item_code": "string",
}

#: Columns that may never be null. `t0_utc`, `is_scheduled` and `item_code` are
#: null exactly on quiet windows.
NON_NULL: tuple[str, ...] = ("window_id", "ticker", "ts_utc", "score", "action")

WAIT, FLAG = "WAIT", "FLAG"


def allowed_actions() -> list[str]:
    """The action vocabulary, from config — not hardcoded here."""
    return list(load_config()["decision"]["actions"])


def empty_frame() -> pd.DataFrame:
    """An empty frame with the right columns and dtypes, for accumulating rows."""
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in SCHEMA.items()})


def conform(df: pd.DataFrame) -> pd.DataFrame:
    """Cast a frame to the contract's dtypes, leaving values alone.

    Callers build frames from many sources; this puts them in one shape before
    validation so a legitimate frame is not rejected over an int64-vs-Int64
    mismatch.
    """
    missing = [c for c in SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"prediction frame is missing columns: {missing}")
    return df.astype(SCHEMA)[list(SCHEMA)]


#: Columns that must hold one value per window_id — a window is one episode,
#: so its t0, ticker and event metadata cannot change hour to hour.
CONSTANT_PER_WINDOW: tuple[str, ...] = ("t0_utc", "ticker", "is_scheduled", "item_code")


def validate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Check a prediction frame against the contract. Returns it, conformed.

    Raises ValueError naming the offending windows. Beyond dtypes this enforces
    six project rules, of which the third is the one that matters most:

    1. every `action` is in `config.decision.actions`
    2. at most one FLAG per window (an episode ends when it flags)
    3. **no scored hour may fall after its window's t0** — that hour is a
       moment when the news was already public, so scoring it is leakage and
       would inflate every lead time
    4. `ts_utc` is unique and sorted within each window
    5. `t0_utc`, `ticker`, `is_scheduled` and `item_code` are each constant
       within a window — a window is one episode, so `metrics.py` can safely
       collapse it to a single label and ticker with `.first()`
    6. `is_scheduled` and `item_code` are null exactly when `t0_utc` is null
       (event metadata only exists for positive windows)
    """
    if df.empty:
        raise ValueError("prediction frame is empty — nothing to evaluate")

    out = conform(df)

    for col in NON_NULL:
        if out[col].isna().any():
            bad = out.loc[out[col].isna(), "window_id"].unique()[:5]
            raise ValueError(f"{col!r} may not be null; first offending windows: {list(bad)}")

    non_finite = ~np.isfinite(out["score"])
    if non_finite.any():
        bad = out.loc[non_finite, "window_id"].unique()[:5]
        raise ValueError(
            f"'score' must be finite — +inf/-inf would poison every downstream "
            f"aggregate (Brier, calibration); first offending windows: {list(bad)}"
        )

    for col in CONSTANT_PER_WINDOW:
        nunique = out.groupby("window_id", sort=False)[col].nunique(dropna=False)
        inconsistent = nunique[nunique > 1]
        if len(inconsistent):
            raise ValueError(
                f"{col!r} must be constant within a window — a window is one "
                f"episode with one label; offending windows: "
                f"{list(inconsistent.index[:5])}"
            )

    mismatched = out["is_scheduled"].isna() != out["t0_utc"].isna()
    if mismatched.any():
        bad = out.loc[mismatched, "window_id"].unique()[:5]
        raise ValueError(
            f"'is_scheduled' must be null exactly on quiet windows (null "
            f"t0_utc); first offending windows: {list(bad)}"
        )

    mismatched = out["item_code"].isna() != out["t0_utc"].isna()
    if mismatched.any():
        bad = out.loc[mismatched, "window_id"].unique()[:5]
        raise ValueError(
            f"'item_code' must be null exactly on quiet windows (null "
            f"t0_utc); first offending windows: {list(bad)}"
        )

    allowed = set(allowed_actions())
    unknown = set(out["action"].dropna().unique()) - allowed
    if unknown:
        raise ValueError(f"unknown action(s) {sorted(unknown)}; allowed: {sorted(allowed)}")

    flags = out[out["action"] == FLAG].groupby("window_id").size()
    multi = flags[flags > 1]
    if len(multi):
        raise ValueError(
            f"an episode ends when it flags, so a window may hold at most one "
            f"{FLAG}; offending windows: {list(multi.index[:5])}"
        )

    positives = out["t0_utc"].notna()
    late = out.loc[positives & (out["ts_utc"] > out["t0_utc"])]
    if len(late):
        raise ValueError(
            f"{len(late)} row(s) are scored AFTER their window's t0 — the news "
            f"was already public at that hour, so including them is leakage. "
            f"First offending windows: {list(late['window_id'].unique()[:5])}"
        )

    dupes = out.duplicated(subset=["window_id", "ts_utc"])
    if dupes.any():
        raise ValueError(
            f"duplicate (window_id, ts_utc) rows would be counted twice by every "
            f"metric; first offending windows: "
            f"{list(out.loc[dupes, 'window_id'].unique()[:5])}"
        )

    # groupby().diff() rather than a Python loop over groups: 3 ms vs 131 ms on
    # 74k rows, and unlike a plain .diff() with a boundary mask it stays correct
    # when a window's rows are not contiguous in the frame.
    steps = out.groupby("window_id", sort=False)["ts_utc"].diff()
    descending = steps < 0
    if descending.any():
        bad = out.loc[descending, "window_id"].unique()[:5]
        raise ValueError(f"ts_utc must ascend within a window; offending: {list(bad)}")

    return out


def actions_from_scores(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Derive WAIT/FLAG from scores: flag the FIRST hour at or above `threshold`.

    Threshold-based baselines emit a score per hour and nothing else. Turning
    those into a stopping decision means taking the first crossing and stopping
    there — flagging every subsequent hour too would spend the alert budget
    many times over on one window and make the baseline look far worse than it
    is.

    "First" means first by `ts_utc`, not first by row position — a caller is
    not required to hand this a frame already sorted within each window, so
    the crossing is computed on a `ts_utc`-ordered view and the result is
    reassembled in the caller's original row order.

    The index is reset, deliberately. The crossing is reassembled by index
    label, and `Series.reindex` raises `ValueError: cannot reindex on an axis
    with duplicate labels` the moment two rows share one. That is not an exotic
    input: `validate_predictions` polices duplicate `(window_id, ts_utc)` pairs
    and says nothing at all about the pandas index, and `report.slice_frames`
    hands its `"all"` slice straight back to the caller with whatever index the
    caller built — so a frame concatenated without `ignore_index=True` upstream
    crashed here rather than being evaluated. Row ORDER is what the docstring
    above promises and `reset_index(drop=True)` preserves it exactly; only the
    labels change, and nothing downstream reads them (the metrics regroup by
    `window_id`).
    """
    out = df.reset_index(drop=True)
    chrono = out.sort_values(["window_id", "ts_utc"], kind="stable")
    crossed = chrono["score"] >= threshold
    first = crossed & ~crossed.groupby(chrono["window_id"]).cummax().groupby(
        chrono["window_id"]
    ).shift(1, fill_value=False)
    action = pd.Series(
        [FLAG if f else WAIT for f in first], index=chrono.index, dtype="string"
    )
    out["action"] = action.reindex(out.index)
    return out
