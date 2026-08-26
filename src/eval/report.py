"""The report table — P1-08 to P1-10 composed into what the report prints.

One row per (slice x t0 variant): precision and recall at the alert budget,
detection delay in trading hours, calibration, and the action distribution.

Two things here are easy to get wrong and both would produce flattering numbers,
so they are handled explicitly:

**Slicing must not drop the negatives.** Quiet windows have `is_scheduled = null`
because they are not events. Filtering a slice on `is_scheduled == True` deletes
every negative, leaving precision = TP/(TP+0) = 1.0 for every model, every time.
A slice therefore keeps ALL quiet windows and filters only the positives. A
false alarm is shared across slices rather than belonging to one.

**The threshold is chosen once, on the full frame.** If each slice picked its own
it would sit at its own operating point, the numbers would not be comparable,
and a slice where the model is weak would quietly get a looser threshold. One
live system has one alert budget.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from src.eval.contract import FLAG, WAIT, actions_from_scores, validate_predictions
from src.eval.metrics import (
    calibration_summary,
    detection_delay_summary,
    precision_at_alert_budget,
    window_summary,
)
from src.utils.config import load_config

#: Column order the report depends on. Pinned by a test.
COLUMNS = [
    "slice", "t0_variant", "n_windows", "n_positive", "base_rate",
    "threshold", "n_alerts", "precision", "recall",
    "median_lead_trading_h", "median_lead_wall_h", "n_missed",
    "n_wait_hours", "n_flag_hours", "pct_hours_flagged", "pct_windows_alerted",
    "brier", "brier_skill_score", "ece", "degenerate",
]


def action_distribution(df: pd.DataFrame) -> dict:
    """WAIT/FLAG counts, at both the hour and the window level.

    The per-hour share is reported because the plan asks for it, but it is
    misleading alone: with at most one FLAG per 48-hour window it never exceeds
    ~2% even for a busy model. `pct_windows_alerted` is the interpretable one.
    """
    n_hours = len(df)
    n_flag = int((df["action"] == FLAG).sum())
    alerted = df.loc[df["action"] == FLAG, "window_id"].nunique()
    n_windows = df["window_id"].nunique()
    return {
        "n_wait_hours": int((df["action"] == WAIT).sum()),
        "n_flag_hours": n_flag,
        "pct_hours_flagged": (n_flag / n_hours) if n_hours else float("nan"),
        "pct_windows_alerted": (alerted / n_windows) if n_windows else float("nan"),
    }


def slice_frames(df: pd.DataFrame, split_by: list[str] | None = None
                 ) -> dict[str, pd.DataFrame]:
    """Split into named slices, **keeping every quiet window in each one**.

    See the module docstring: filtering on `is_scheduled` or `item_code`
    directly would delete all the negatives and force precision to 1.0.
    """
    splits = split_by if split_by is not None else load_config()["eval"]["split_by"]
    quiet = df[df["t0_utc"].isna()]
    positive = df[df["t0_utc"].notna()]

    out: dict[str, pd.DataFrame] = {"all": df}

    def with_quiet(pos: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([pos, quiet], ignore_index=True)

    if "scheduled" in splits:
        out["scheduled"] = with_quiet(positive[positive["is_scheduled"] == True])  # noqa: E712
    if "unscheduled" in splits:
        out["unscheduled"] = with_quiet(positive[positive["is_scheduled"] == False])  # noqa: E712
    if "item_code" in splits:
        for code in sorted(positive["item_code"].dropna().unique()):
            out[f"item {code}"] = with_quiet(positive[positive["item_code"] == code])

    return out


def evaluate(df: pd.DataFrame, threshold: float | None = None,
             max_alerts: int | None = None) -> dict:
    """Every metric for one frame, at one operating point.

    `threshold` is passed in so that all slices share the operating point
    chosen on the full frame. Omit it only when evaluating a frame on its own.
    """
    frame = validate_predictions(df)

    if threshold is None:
        budget = precision_at_alert_budget(frame, max_alerts=max_alerts)
        threshold = budget.threshold
    decided = actions_from_scores(frame, threshold)

    windows = window_summary(decided)
    n_windows = len(windows)
    n_positive = int(windows["is_positive"].sum())
    alerted = windows[windows["peak_score"] >= threshold]
    tp = int(alerted["is_positive"].sum())

    delay = detection_delay_summary(decided)
    actions = action_distribution(decided)

    try:
        cal = calibration_summary(decided)
        brier, skill, ece = cal.brier, cal.brier_skill_score, cal.ece
    except ValueError:
        # Scores are not probabilities — a threshold baseline. nan means "not
        # applicable" here, not "failed"; calibration_summary still refuses
        # loudly when called directly.
        brier = skill = ece = float("nan")

    return {
        "n_windows": n_windows,
        "n_positive": n_positive,
        "base_rate": (n_positive / n_windows) if n_windows else float("nan"),
        "threshold": threshold,
        "n_alerts": len(alerted),
        "precision": (tp / len(alerted)) if len(alerted) else float("nan"),
        "recall": (tp / n_positive) if n_positive else float("nan"),
        "median_lead_trading_h": delay.median_trading_hours,
        "median_lead_wall_h": delay.median_wall_clock_hours,
        "n_missed": delay.n_missed,
        "brier": brier,
        "brier_skill_score": skill,
        "ece": ece,
        "degenerate": actions["n_flag_hours"] == 0,
        **actions,
    }


def report_table(frames: Mapping[str, pd.DataFrame] | pd.DataFrame,
                 max_alerts: int | None = None) -> pd.DataFrame:
    """The full table: one row per (slice x t0 variant).

    `frames` maps a t0 variant name to its prediction frame — the same model
    evaluated once per variant in `config.eval.t0_variants`, since the contract
    carries one t0 column. A bare DataFrame is treated as a single unnamed
    variant.

    An always-WAIT policy shows up as `degenerate=True` with zero alerts, which
    is the point of the column.
    """
    if isinstance(frames, pd.DataFrame):
        frames = {"default": frames}

    rows = []
    for variant, df in frames.items():
        frame = validate_predictions(df)
        # One operating point per variant, chosen on everything.
        threshold = precision_at_alert_budget(frame, max_alerts=max_alerts).threshold
        for name, sliced in slice_frames(frame).items():
            if sliced.empty:
                continue
            rows.append({"slice": name, "t0_variant": variant,
                         **evaluate(sliced, threshold=threshold)})

    return pd.DataFrame(rows, columns=COLUMNS)


def t0_variant_gap(table: pd.DataFrame, metric: str = "median_lead_trading_h",
                   slice_name: str = "all") -> float:
    """Difference in a metric between the two t0 variants.

    The plan requires both variants and the gap between them to be reported —
    that comparison is itself a small contribution, since it measures how much
    of the apparent warning was really time the market already knew.
    """
    row = table[table["slice"] == slice_name].set_index("t0_variant")[metric]
    if len(row) != 2:
        return float("nan")
    return float(row.iloc[0] - row.iloc[1])
