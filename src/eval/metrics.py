"""Evaluation metrics. Consumes the contract in `contract.py` and nothing else.

**Plain accuracy is not implemented here, deliberately.** The base rate is
about 0.3%, so a model that never alerts scores 99.7% and looks excellent while
being useless. `accuracy()` exists only to raise and say so, because the
temptation to reach for it is the single most likely way this project produces
a number that cannot be defended.

The headline metric is **precision at a fixed alert budget**, chosen before any
model existed (plan §9 phase 1) so it cannot be picked after seeing which
number flatters a result.

How the budget works
--------------------
`config.eval.alert_budget_per_stock_per_month` is 2. An analyst watching N
stocks for M months will tolerate roughly ``N * M * 2`` alerts in total; more
than that and the queue stops being triaged. So:

1. count the distinct ``(ticker, calendar month)`` pairs the frame covers
2. multiply by the per-stock-per-month rate to get the total alert allowance
3. take the highest-scoring windows until the allowance is spent
4. the score of the last one admitted is the threshold

An alert is **one FLAG per window**, not per hour — an episode ends when it
flags — so a window alerts if any of its hours crosses the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from src.eval.contract import FLAG, validate_predictions
from src.utils.config import load_config
from src.utils.timeutils import get_market_calendar, trading_hours_between


@dataclass(frozen=True)
class BudgetResult:
    """Precision and recall at a fixed alert budget.

    Note what is absent: there is no `accuracy` field, and there will not be
    one. See the module docstring.
    """

    threshold: float
    budget_alerts: int          # what the budget allowed
    realised_alerts: int        # what the threshold actually spends
    ticker_months: int
    n_windows: int
    n_positive: int
    true_positives: int
    precision: float
    recall: float
    base_rate: float            # reported so precision can be read in context
    budget_exceeds_windows: bool

    def as_dict(self) -> dict:
        return asdict(self)


def accuracy(*_args, **_kwargs) -> float:
    """Deliberately not implemented.

    The base rate here is ~0.3%. A policy that always says WAIT scores 99.7%
    accuracy while detecting nothing, so the number is not merely unhelpful —
    it actively rewards the degenerate answer. Every headline figure in this
    project is precision at a fixed alert budget, split scheduled vs
    unscheduled.
    """
    raise NotImplementedError(
        "accuracy is banned in this project: the base rate is ~0.3%, so "
        "always-WAIT scores 99.7% while detecting nothing. Use "
        "precision_at_alert_budget() instead — see AGENTS.md rule 7."
    )


def ticker_months(df: pd.DataFrame) -> int:
    """Distinct (ticker, calendar month) pairs the frame covers.

    This is the unit the alert budget is denominated in. Counted from the data
    rather than from the study window, so a partial or sampled evaluation set
    gets an allowance proportional to what it actually covers.
    """
    month = pd.to_datetime(df["ts_utc"].astype("int64"), unit="s").dt.to_period("M")
    return int(df.assign(_m=month).groupby(["ticker", "_m"], observed=True).ngroups)


def window_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse hourly rows to one row per window.

    A window alerts if ANY of its hours crosses the threshold, so the window's
    peak score is what decides. `is_positive` comes from t0 being present.
    """
    grouped = df.groupby("window_id", sort=False)
    return pd.DataFrame({
        "ticker": grouped["ticker"].first(),
        "peak_score": grouped["score"].max(),
        "is_positive": grouped["t0_utc"].first().notna(),
    })


def precision_at_alert_budget(
    df: pd.DataFrame,
    budget_per_stock_per_month: float | None = None,
    max_alerts: int | None = None,
) -> BudgetResult:
    """The headline metric.

    Picks the threshold that spends the alert budget, then reports precision
    and recall at it.

    `max_alerts` overrides the computed allowance — used for tuning curves and
    for tests that need an exact number rather than one derived from calendar
    coverage.
    """
    frame = validate_predictions(df)
    rate = (budget_per_stock_per_month
            if budget_per_stock_per_month is not None
            else load_config()["eval"]["alert_budget_per_stock_per_month"])

    windows = window_summary(frame)
    n_windows = len(windows)
    n_positive = int(windows["is_positive"].sum())
    months = ticker_months(frame)

    budget = int(max_alerts) if max_alerts is not None else int(round(months * rate))
    budget = max(0, budget)
    exceeds = budget >= n_windows

    # Highest peaks first; the allowance buys the top `budget` windows. Ties at
    # the boundary are admitted together, so realised can exceed budget — which
    # is reported rather than silently trimmed.
    ranked = windows.sort_values("peak_score", ascending=False, kind="mergesort")
    if budget <= 0:
        threshold = float("inf")
        alerted = ranked.iloc[:0]
    elif exceeds:
        threshold = float("-inf")
        alerted = ranked
    else:
        threshold = float(ranked["peak_score"].iloc[budget - 1])
        alerted = ranked[ranked["peak_score"] >= threshold]

    realised = len(alerted)
    tp = int(alerted["is_positive"].sum())

    return BudgetResult(
        threshold=threshold,
        budget_alerts=budget,
        realised_alerts=realised,
        ticker_months=months,
        n_windows=n_windows,
        n_positive=n_positive,
        true_positives=tp,
        precision=(tp / realised) if realised else float("nan"),
        recall=(tp / n_positive) if n_positive else float("nan"),
        base_rate=(n_positive / n_windows) if n_windows else float("nan"),
        budget_exceeds_windows=exceeds,
    )


@dataclass(frozen=True)
class DelayResult:
    """Distribution of advance warning on correctly flagged windows.

    All lead times are **trading hours**. `median_wall_clock_hours` is carried
    for contrast only — it quantifies how much the trading-hours correction
    matters — and is never the figure reported as lead time.
    """

    n_detections: int
    n_positive: int
    n_missed: int
    median_trading_hours: float
    p25_trading_hours: float
    p75_trading_hours: float
    min_trading_hours: float
    max_trading_hours: float
    median_wall_clock_hours: float   # contrast only — not the headline

    def as_dict(self) -> dict:
        return asdict(self)


def detection_delays(df: pd.DataFrame) -> pd.DataFrame:
    """One row per **correctly flagged** window: how much warning it bought.

    A correctly flagged window is a positive one (it has a t0) where the model
    raised a FLAG. Flags on quiet windows are false alarms and have no lead
    time; folding a zero in for them would understate the warning that real
    detections actually gave.

    Returns per-detection rows rather than a summary so that the
    scheduled/unscheduled split in P1-11 and the dashboard's histogram do not
    have to recompute anything.

    Columns: lead_trading_hours, lead_wall_hours, ticker, is_scheduled,
    item_code, flag_ts_utc, t0_utc.
    """
    frame = validate_predictions(df)
    hits = frame[(frame["action"] == FLAG) & frame["t0_utc"].notna()]

    if hits.empty:
        return pd.DataFrame(columns=[
            "lead_trading_hours", "lead_wall_hours", "ticker",
            "is_scheduled", "item_code", "flag_ts_utc", "t0_utc",
        ])

    cal = get_market_calendar()
    flag_ts = hits["ts_utc"].astype("int64").to_numpy()
    t0 = hits["t0_utc"].astype("int64").to_numpy()

    return pd.DataFrame({
        "lead_trading_hours": [
            trading_hours_between(a, b, cal) for a, b in zip(flag_ts, t0)
        ],
        "lead_wall_hours": (t0 - flag_ts) / 3600.0,
        "ticker": hits["ticker"].to_numpy(),
        "is_scheduled": hits["is_scheduled"].to_numpy(),
        "item_code": hits["item_code"].to_numpy(),
        "flag_ts_utc": flag_ts,
        "t0_utc": t0,
    }, index=pd.Index(hits["window_id"].to_numpy(), name="window_id"))


def detection_delay_summary(df: pd.DataFrame) -> DelayResult:
    """Summarise `detection_delays`, plus how many positives were missed.

    With no detections every statistic is `nan`, not `0.0`. Zero would read as
    "detected everything with no warning"; nan says there is nothing to
    measure. The always-quiet baseline lands here and must not error.
    """
    frame = validate_predictions(df)
    delays = detection_delays(frame)

    windows = window_summary(frame)
    n_positive = int(windows["is_positive"].sum())
    lead = delays["lead_trading_hours"]

    def q(p: float) -> float:
        return float(lead.quantile(p)) if len(lead) else float("nan")

    return DelayResult(
        n_detections=len(delays),
        n_positive=n_positive,
        n_missed=n_positive - len(delays),
        median_trading_hours=q(0.5),
        p25_trading_hours=q(0.25),
        p75_trading_hours=q(0.75),
        min_trading_hours=float(lead.min()) if len(lead) else float("nan"),
        max_trading_hours=float(lead.max()) if len(lead) else float("nan"),
        median_wall_clock_hours=(
            float(delays["lead_wall_hours"].median()) if len(delays) else float("nan")
        ),
    )
