"""Quickest change detection — has the evidence been piling up?

The textbook method for the problem this project actually poses: watch a
stream and decide as early as possible that its behaviour has shifted, without
alarming too often on noise.

    S(t) = max(0, S(t-1) + x(t) - k)        alarm when S >= h, then reset

Where volume z-score asks *is this hour unusual*, CUSUM asks *has the evidence
been accumulating*. One 3-sigma hour and three consecutive 1.5-sigma hours
look very different to the first and identical to the second. That is the
hypothesis this baseline exists to test, which is why it scores the same
`volume_z` feature P5-03 does — the comparison then isolates accumulation
rather than confounding it with a different input.

`k` is the slack: evidence below it decays the statistic, so ordinary noise
cannot drift into an alarm. `h` is the decision boundary.

Why the statistic is continuous per ticker, not per window
----------------------------------------------------------
This is the design decision of the module, and it follows from P5-03's
evaluation frame, where a positive is a 48-bar episode and a negative is a
single bar.

Accumulate per *window* and CUSUM gets 48 bars of evidence on every positive
and exactly one on every negative. It would score superbly, and the result
would be an artefact of how the labels were laid out rather than a fact about
the market.

So the statistic runs continuously over each ticker's bars in time order, and
every decision point reads off the value standing at its own hour. A negative
bar carries the accumulation from its own preceding hours exactly as a
positive bar does. This is also what live operation looks like: the monitor
runs continuously and does not know which hours will turn out to precede an
announcement.

The cold start is real and is not engineered around: `S` begins at 0 on the
first bar of whatever frame it is given, so on a validation frame it does not
carry state across the split boundary. That is a small pessimism — a change
already underway at the boundary starts from zero — and back-filling it is
exactly the kind of convenience that becomes leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.baselines.always_quiet import AlwaysQuiet
from src.baselines.base import Baseline
from src.eval.metrics import detection_delay_summary, precision_at_alert_budget

FEATURE = "volume_z"


def cusum_statistic(values: np.ndarray, k: float, h: float,
                    reset_after_alarm: bool = True) -> np.ndarray:
    """The one-sided upper CUSUM over one ordered series.

    Returns the statistic AT each position — the alarm value is emitted before
    any reset, so the score reflects the evidence that triggered it.

    NaN carries the state forward unchanged and emits NaN. No evidence is not
    evidence of normality: treating a missing bar as a zero observation would
    decay `S` by `k` and quietly assert the stock was calm. Emitting NaN hands
    the window to `Baseline`'s unscoreable machinery instead, so the decision
    is made in one place for every baseline.
    """
    out = np.full(len(values), np.nan, dtype="float64")
    s = 0.0
    for i, x in enumerate(values):
        if not np.isfinite(x):
            continue                      # state preserved, no claim made
        s = max(0.0, s + float(x) - k)
        out[i] = s
        if reset_after_alarm and s >= h:
            s = 0.0
    return out


@dataclass(frozen=True)
class CusumOperatingPoint:
    """The tuned (k, h), what they scored, and what they beat."""

    drift: float
    threshold: float
    action_threshold: float
    precision: float
    max_precision: float
    recall: float
    base_rate: float
    floor_precision: float
    median_trading_hours: float
    n_windows: int
    n_positive: int
    grid_size: int

    @property
    def beats_floor(self) -> bool:
        return self.precision > self.floor_precision


class CUSUM(Baseline):
    """Accumulates `volume_z` evidence per ticker and scores the running total."""

    name = "cusum"

    def __init__(self, cfg: dict | None = None, drift: float | None = None,
                 threshold: float | None = None) -> None:
        super().__init__(cfg)
        c = self.cfg.get("baselines", {}).get("cusum", {})
        self.drift = c.get("drift", 0.5) if drift is None else drift
        self.threshold_h = (c.get("threshold", 5.0)
                            if threshold is None else threshold)
        self.reset_after_alarm = bool(c.get("reset_after_alarm", True))

    def score(self, frame: pd.DataFrame) -> pd.Series:
        if FEATURE not in frame.columns:
            raise ValueError(
                f"{self.name}: frame has no {FEATURE!r} column — this baseline "
                f"accumulates P4-07's feature and cannot run without it")

        # Sort internally rather than trusting the caller: the statistic is
        # order-dependent, so a frame handed over grouped by window instead of
        # by time would silently produce a different number.
        chrono = frame.sort_values(["ticker", "ts_utc"], kind="stable")
        out = pd.Series(np.nan, index=frame.index, dtype="float64")
        for _, block in chrono.groupby("ticker", sort=False):
            out.loc[block.index] = cusum_statistic(
                block[FEATURE].to_numpy(dtype="float64"),
                self.drift, self.threshold_h, self.reset_after_alarm)
        return out


def tune_cusum(cfg: dict, frame: pd.DataFrame, conn=None,
               drift_grid: list[float] | None = None,
               threshold_grid: list[float] | None = None) -> CusumOperatingPoint:
    """Choose `k` and `h` by a recorded 2-D sweep on the given frame.

    **The procedure, written down so nobody can say it was hand-picked:**

    1. Evaluate every `(k, h)` pair in the configured grids. Both matter here,
       which is a real difference from P5-03 rather than an inconsistency:
       `reset_after_alarm` couples the boundary back into the estimator, so
       crossing `h` zeroes `S` and changes every later value the metric ranks
       on. For volume z-score the threshold provably could not move precision;
       here it can, and a test pins each fact.
    2. Rank pairs by precision at the alert budget. Ties break on **longer
       median lead time**, then on **smaller k** and **smaller h**, so the
       result is deterministic rather than dependent on grid order.
    3. The WAIT/FLAG cut is then the budget-implied threshold for the winning
       pair — the same rule every other baseline uses, so all four spend the
       same allowance and P5-06 compares like with like. Letting `h` double as
       the action threshold would hand CUSUM a different budget from the rest.
    4. Always-quiet is run on the identical frame and carried alongside,
       because a precision figure without its floor cannot be read.

    `conn` is passed through to `predict()`, so the sealed test split is
    refused here as everywhere.
    """
    c = cfg["baselines"]["cusum"]
    drift_grid = drift_grid or c["drift_grid"]
    threshold_grid = threshold_grid or c["threshold_grid"]

    floor = precision_at_alert_budget(
        AlwaysQuiet(cfg).predict(frame, conn=conn, context="cusum tune floor"))

    best = None
    evaluated = 0
    for k in drift_grid:
        for h in threshold_grid:
            # h >= k, ALWAYS. Not a preference — with `reset_after_alarm` the
            # boundary is also a cap on how much evidence the statistic may
            # carry, so a boundary below the slack means S can never hold more
            # than h and the recursion saw-tooths instead of accumulating. At
            # the pair this sweep previously chose (k=1.5, h=1.0), six
            # consecutive 2-sigma bars peaked at 1.0 while a single 4.5-sigma
            # bar reached 3.0 — the exact inversion of the hypothesis this
            # baseline exists to test, and it made CUSUM a shifted per-bar
            # z-score wearing a change-detector's name.
            #
            # config.yaml stated this rule ("h is bounded below by k") beside a
            # grid that violated it, and nothing enforced it. Skipping the
            # pairs here rather than editing the grids keeps both grids
            # readable and keeps the constraint true for every k.
            if h < k:
                continue
            evaluated += 1
            model = CUSUM(cfg, drift=k, threshold=h)
            scored = model.predict(frame, threshold=float("inf"), conn=conn,
                                   context=f"cusum tune k={k} h={h}")
            budget = precision_at_alert_budget(scored)

            at_cut = model.predict(frame, threshold=budget.threshold, conn=conn,
                                   context=f"cusum tune k={k} h={h} at cut")
            delay = detection_delay_summary(at_cut)
            lead = (delay.median_trading_hours if delay.n_detections
                    else float("-inf"))

            key = (budget.precision, lead, -float(k), -float(h))
            if best is None or key > best[0]:
                best = (key, k, h, budget, lead)

    if best is None:
        raise SystemExit(
            f"no (k, h) pair satisfies h >= k across drift_grid={drift_grid} "
            f"and threshold_grid={threshold_grid}. A CUSUM whose boundary sits "
            f"below its slack cannot accumulate, so there is nothing here "
            f"worth tuning — widen threshold_grid upward.")
    _, k, h, budget, lead = best
    return CusumOperatingPoint(
        drift=float(k),
        threshold=float(h),
        action_threshold=float(budget.threshold),
        precision=float(budget.precision),
        max_precision=float(budget.max_precision),
        recall=float(budget.recall),
        base_rate=float(budget.base_rate),
        floor_precision=float(floor.precision),
        median_trading_hours=float(lead),
        n_windows=int(budget.n_windows),
        n_positive=int(budget.n_positive),
        # The pairs actually EVALUATED, not the cartesian product:
        # the h < k half is skipped, so reporting the product would
        # overstate the search by roughly a third.
        grid_size=evaluated,
    )
