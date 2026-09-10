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

import math
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from src.eval.contract import FLAG, validate_predictions
from src.utils.config import load_config
from src.utils.timeutils import get_market_calendar, trading_hours_between


class ScoresAreNotProbabilities(ValueError):
    """Calibration was asked for on scores that are not in [0, 1].

    A named type rather than a bare `ValueError`, because `report.py` has to
    tell this one apart to degrade calibration to nan while still surfacing
    every other error. It previously did that by matching a substring of the
    message text — a contract no one could see from either side, which any
    reword of that message would have silently broken in one direction
    (crashing every threshold-baseline slice) or the other (swallowing an
    unrelated ValueError). Subclasses `ValueError` so existing callers that
    catch that still behave as before.
    """


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
    #: What a FLAWLESS detector would score here (issue 29). The budget is
    #: spent, not capped — the top `budget_alerts` windows are flagged — so
    #: when the budget exceeds the number of positives, precision cannot reach
    #: 1.0 no matter how good the model is: every alert beyond the last true
    #: positive is a false one by construction. On this study 31,823 alerts
    #: against 6,737 positives puts the ceiling at 21.2%.
    #:
    #: Computed rather than remembered. It was a sentence in the tracker
    #: saying "the report must state the ceiling", which is exactly the kind
    #: of thing that is true right up until the one run where nobody
    #: remembers — and then a respectable 15% reads as failure instead of as
    #: 71% of what was achievable. The budget itself is deliberately NOT
    #: adjusted to raise it: 2 alerts/stock/month is an operational constraint
    #: fixed before any model existed, and tuning it now is what the "no
    #: threshold changes after seeing the result" rule forbids.
    max_precision: float
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
    """Collapse hourly rows to one row per window, symmetrically.

    Both classes get exactly ONE draw
    ---------------------------------
    `evalset.build_eval_frame` gives a positive the 48 bars before t0 and a
    quiet window a SINGLE bar. Taking a plain max per window handed a positive
    48 independent chances to cross the threshold and a quiet window one, at
    the same one-alert cost — so window LENGTH decided the ranking, not
    detection. Measured on the Phase 10 shape, a scorer made of pure random
    noise reached **29.6x lift**, beating every tuned detector in the table.

    A positive is therefore represented by its **final bar** — the decision
    point immediately before t0 — so each window contributes one draw and one
    alert, whichever class it belongs to. The bar is chosen by POSITION, never
    by score: picking the episode's best hour is precisely the bug above.

    Why the final bar and not a random one: it is the last moment the system
    could act on, the point at which any accumulating footprint is largest, and
    a fixed rule that needs no seed. It is a stated convention, and the honest
    caveat is that a detector which fires early and goes quiet is not credited
    here — `detection_delay_summary` is what measures the episode as a whole.

    Why not simply give the quiet windows 48 bars too: that was tried and is
    worse. Quiet windows overlap, so a rolling maximum charges one sustained
    anomaly ~20 alerts while still charging a 48-bar episode 1 — measured, on
    the validation frame: 8,962 alerted quiet windows spanning just 448
    independent clusters. That is the original asymmetry mirrored, and it
    flattens every detector into the noise band rather than fixing anything.
    Tiling both classes, and deduplicating quiet alerts, both go degenerate
    instead: the alert budget then exceeds the windows that remain.

    The evidence for this being the right cut, rather than merely a different
    one: on the validation frame pure noise scores **0.94x** here (a null that
    behaves), while volume z-score reaches **6.93x** at **z = +40** against it.
    A frame that inflated results would inflate the null too, and it does not.
    """
    positive_row = df["t0_utc"].notna()
    if positive_row.any():
        # Position, not score: the LAST bar of each positive window. `sort` is
        # on ts_utc rather than trusting row order, because a caller is free to
        # hand this frame over in any order and the contract only promises the
        # columns.
        ordered = df.sort_values(["window_id", "ts_utc"], kind="mergesort")
        keep_last = ~ordered["window_id"].duplicated(keep="last")
        df = pd.concat([ordered[positive_row.loc[ordered.index] & keep_last],
                        df[~positive_row]])

    grouped = df.groupby("window_id", sort=False)
    return pd.DataFrame({
        "ticker": grouped["ticker"].first(),
        "peak_score": grouped["score"].max(),
        "is_positive": grouped["t0_utc"].first().notna(),
    })


def alert_budget(frame: pd.DataFrame, rate: float | None = None,
                 max_alerts: int | None = None) -> int:
    """How many alerts the budget allows on an ALREADY-VALIDATED frame.

    Split out so `report.py` can size the ceiling without paying for a second
    `validate_predictions` pass over every slice — P1-Xb deliberately
    validates once per slice, and a test pins that. Duplicating the arithmetic
    there instead would put the rounding rule below in two files, free to
    drift apart.
    """
    if rate is None:
        rate = load_config()["eval"]["alert_budget_per_stock_per_month"]
    # Round-half-up, not Python's round-half-to-even: `round()` would send a
    # non-default `budget_per_stock_per_month` override of exactly .5 to the
    # nearest *even* integer (round(4.5) == 4, round(5.5) == 6), which is a
    # silent trap for a tuning-curve sweep over fractional rates. The default
    # config rate is an integer, so this never fires on the production path.
    budget = (int(max_alerts) if max_alerts is not None
              else int(math.floor(ticker_months(frame) * rate + 0.5)))
    return max(0, budget)


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
    budget = alert_budget(frame, rate, max_alerts)
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
        # A flawless detector ranks every positive above every negative, so it
        # catches min(n_positive, realised) of them across the alerts it
        # actually issued.
        #
        # The denominator is `realised`, not `budget`, and that distinction is
        # the whole point. Dividing by the budget compares a ceiling measured
        # on the ALLOWANCE against a precision measured on the alerts actually
        # ISSUED, and those two numbers are routinely different: ties spend
        # more than the allowance, and a small frame cannot spend it at all.
        # The result was a "ceiling" that could sit below the precision beside
        # it — ten rows of the Phase 10 table did exactly that. Nothing can
        # beat a ceiling, so the number was not one.
        #
        # Sharing precision's denominator makes it answer the question a
        # reader is actually asking: given this many alerts were issued, what
        # is the best precision that could have come out of them? Guarded
        # against issuing none, which would otherwise divide by zero.
        max_precision=(min(n_positive, realised) / realised) if realised
                      else float("nan"),
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
    return _detection_delays(validate_predictions(df))


def _detection_delays(frame: pd.DataFrame) -> pd.DataFrame:
    """`detection_delays` on an already-validated frame. See P1-Xb: validating
    once per slice rather than three times per slice."""
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
    return _delay_summary(validate_predictions(df))


def _delay_summary(frame: pd.DataFrame) -> DelayResult:
    """`detection_delay_summary` on an already-validated frame (P1-Xb)."""
    delays = _detection_delays(frame)
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


# --------------------------------------------------------------------------
# Calibration
#
# "When the model says 30%, does news come 30% of the time?" A model can rank
# windows perfectly and still be badly calibrated, and the dashboard puts a
# confidence next to every alert — an uncalibrated one there is worse than none.
#
# The unit is (window, hour): the model emits one score per hour, so that is
# what gets calibrated. The label is constant within a window — every hour of a
# positive window is labelled 1, because news is coming for each of them.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationResult:
    """Brier, ECE, and the skill score that makes Brier readable.

    A raw Brier is never reported without `brier_skill_score`. See
    `brier_score` for why.
    """

    brier: float
    brier_baseline: float        # always forecasting the base rate
    brier_skill_score: float     # >0 better than that, 0 no better, <0 worse
    ece: float
    n_bins: int
    n_rows: int
    base_rate: float

    def as_dict(self) -> dict:
        return asdict(self)


def _probabilities_and_labels(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract (p, y) from an **already-validated** frame, refusing scores that
    are not probabilities.

    The contract deliberately permits unbounded scores because the volume
    z-score baseline emits them. Calibration is simply not defined for those,
    so this raises rather than silently squashing them into a range.
    """
    p = frame["score"].to_numpy(dtype=float)

    if p.size and (p.min() < 0.0 or p.max() > 1.0):
        raise ScoresAreNotProbabilities(
            f"calibration needs probabilities, but scores range "
            f"[{p.min():.3f}, {p.max():.3f}]. A z-score baseline has no "
            f"calibration — that is a statement about its output, not a "
            f"failing. Convert to probabilities first, or skip calibration "
            f"for this model."
        )

    y = frame["t0_utc"].notna().to_numpy(dtype=float)
    return p, y


def brier_score(df: pd.DataFrame) -> float:
    """Mean squared error of the probabilities. Lower is better, 0 is perfect.

    **Never report this alone.** The base rate is ~0.3%, so a model that
    outputs 0.003 constantly and detects nothing scores about 0.003 and looks
    superb — the accuracy trap wearing a different hat. Use
    `calibration_summary`, which pairs it with a skill score.
    """
    p, y = _probabilities_and_labels(validate_predictions(df))
    return float(np.mean((p - y) ** 2))


def reliability_curve(df: pd.DataFrame, n_bins: int | None = None) -> pd.DataFrame:
    """Per-bin predicted-vs-actual, for the dashboard's calibration chart.

    Bins are [0, 0.1), [0.1, 0.2), … [0.9, 1.0] — the last closed on the right
    so p = 1.0 has a home. Empty bins are dropped: a bin nobody landed in says
    nothing about calibration, and counting it as a zero gap would flatter ECE.
    """
    return _reliability_curve(validate_predictions(df), n_bins)


def _reliability_curve(frame: pd.DataFrame, n_bins: int | None = None) -> pd.DataFrame:
    """`reliability_curve` on an already-validated frame."""
    p, y = _probabilities_and_labels(frame)
    bins = n_bins if n_bins is not None else load_config()["eval"]["calibration_bins"]

    idx = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append({
            "bin": b,
            "bin_lower": b / bins,
            "bin_upper": (b + 1) / bins,
            "count": int(mask.sum()),
            "mean_predicted": float(p[mask].mean()),
            "mean_actual": float(y[mask].mean()),
        })
    return pd.DataFrame(rows, columns=[
        "bin", "bin_lower", "bin_upper", "count", "mean_predicted", "mean_actual",
    ])


def expected_calibration_error(df: pd.DataFrame, n_bins: int | None = None) -> float:
    """Weighted average gap between predicted probability and actual rate.

    0 is perfect. Each bin contributes in proportion to how many rows it holds,
    so a bin with three rows cannot dominate one with three thousand.
    """
    return _ece(validate_predictions(df), n_bins)


def _ece(frame: pd.DataFrame, n_bins: int | None = None) -> float:
    """`expected_calibration_error` on an already-validated frame."""
    curve = _reliability_curve(frame, n_bins)
    if curve.empty:
        return float("nan")
    gaps = (curve["mean_actual"] - curve["mean_predicted"]).abs()
    return float((gaps * curve["count"]).sum() / curve["count"].sum())


def calibration_summary(df: pd.DataFrame, n_bins: int | None = None) -> CalibrationResult:
    """Brier and ECE together, with the skill score that makes Brier readable.

    `brier_skill_score = 1 - brier_model / brier_base_rate_forecast`. Above 0
    beats the trivial "always predict the base rate" forecast; 0 matches it;
    below 0 is worse than it. When every label is the same class the baseline
    is degenerate and the skill score is `nan` rather than a misleading number.
    """
    return _calibration_summary(validate_predictions(df), n_bins)


def _calibration_summary(frame: pd.DataFrame,
                         n_bins: int | None = None) -> CalibrationResult:
    """`calibration_summary` on an already-validated frame."""
    p, y = _probabilities_and_labels(frame)
    bins = n_bins if n_bins is not None else load_config()["eval"]["calibration_bins"]

    base = float(y.mean()) if y.size else float("nan")
    brier = float(np.mean((p - y) ** 2)) if p.size else float("nan")
    baseline = float(np.mean((base - y) ** 2)) if y.size else float("nan")
    skill = (1.0 - brier / baseline) if baseline > 0 else float("nan")

    return CalibrationResult(
        brier=brier,
        brier_baseline=baseline,
        brier_skill_score=skill,
        ece=_ece(frame, bins),
        n_bins=int(bins),
        n_rows=int(p.size),
        base_rate=base,
    )
