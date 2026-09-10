"""Alert when volume is unusual for this stock. The one most likely to win.

The backlog flags this baseline as the probable winner, and that warning is
the reason it is built carefully rather than as a foil. If a 2.5-sigma volume
spike beats the learned policy in Phase 6, that is the project's finding and
it gets reported as one — a strawman here would make the whole comparison
worthless.

What is tuned, and what is not
------------------------------
It is easy to tune the wrong thing here, so the distinction is stated rather
than assumed.

`precision_at_alert_budget` **derives its own cut**: it ranks windows by peak
score and buys the top `budget`. The headline precision therefore does not
depend on this baseline's `threshold` at all — sweeping threshold against
precision would draw a flat line and pick an arbitrary winner from noise.

What the threshold does control is the **action** column: whether a window
flags, and at which hour. That drives detection delay, lead time and the
action distribution. So the two knobs are settled differently:

* **threshold** — set to the budget-implied cut the metric itself returns on
  the tuning frame, so the operating point spends exactly the allowance it is
  given. Derived and reported, never guessed.
* **`min_wait_hours`** — genuinely swept, but **not on precision**, and the
  distinction is the same one made three paragraphs up. `min_wait_hours` only
  rewrites the `action` column, and `precision_at_alert_budget` never reads
  `action` — it ranks windows by `peak_score`. So precision is *invariant*
  across the grid by construction, and a test pins that rather than leaving it
  to be believed. What the sweep actually decides is lead time and the action
  distribution, and `tune`'s key `(precision, lead, -wait)` therefore settles
  it entirely on the lead-time tie-break: 0 wins because it flags at the
  earliest crossing. The sweep is honest and stays; the reason it used to give
  was not.

Tuning uses the sampled 3:1 frame, whose config comment reads "training only".
The headline comparison at the true base rate (0.284%, every in-universe bar)
is P5-06's job and a different population.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.baselines.always_quiet import AlwaysQuiet
from src.baselines.base import Baseline
from src.eval.metrics import detection_delay_summary, precision_at_alert_budget

FEATURE = "volume_z"


@dataclass(frozen=True)
class OperatingPoint:
    """The tuned settings, plus what they scored and what they beat.

    Carries the floor alongside the result on purpose: a precision figure that
    is not stated against the base rate cannot be read, and the backlog's
    warning about this baseline winning is only meaningful next to what it won
    against.
    """

    threshold: float
    min_wait_hours: float
    precision: float
    max_precision: float
    recall: float
    base_rate: float
    floor_precision: float
    median_trading_hours: float
    n_windows: int
    n_positive: int
    grid: tuple[float, ...]

    @property
    def beats_floor(self) -> bool:
        return self.precision > self.floor_precision


class VolumeZScore(Baseline):
    """Scores each hour by that hour's volume z-score.

    No transformation: the feature P4-07 built already answers "how unusual is
    this hour's volume for this stock", and re-scaling it here would put a
    second, untested definition of the same idea in the codebase.
    """

    name = "volume_zscore"

    def __init__(self, cfg: dict | None = None,
                 min_wait_hours: float | None = None) -> None:
        super().__init__(cfg)
        vz = self.cfg.get("baselines", {}).get("volume_zscore", {})
        self._min_wait = (vz.get("min_wait_hours", 0)
                          if min_wait_hours is None else min_wait_hours)

    def _common(self) -> dict:
        """Override the shared `min_wait_hours` with this baseline's own.

        `Baseline._apply_min_wait` reads `baselines.common`; the sweep needs to
        vary the value per candidate without mutating global config, so the
        merged view is produced here.
        """
        common = dict(super()._common())
        common["min_wait_hours"] = self._min_wait
        return common

    def score(self, frame: pd.DataFrame) -> pd.Series:
        if FEATURE not in frame.columns:
            raise ValueError(
                f"{self.name}: frame has no {FEATURE!r} column — this baseline "
                f"scores P4-07's feature and cannot run without it"
            )
        return frame[FEATURE]


def tune(cfg: dict, frame: pd.DataFrame, conn=None,
         grid: list[float] | None = None) -> OperatingPoint:
    """Choose `min_wait_hours` and the threshold on the given frame.

    **The procedure, written down so nobody can say it was hand-picked:**

    1. For each candidate `min_wait_hours` in the configured grid, score the
       frame and compute `precision_at_alert_budget`. Precision is rank-based,
       so neither the candidate's threshold nor its wait affects it — the
       metric ranks on `peak_score` and `min_wait_hours` only rewrites
       `action`. Every candidate therefore returns the *same* precision, and
       that is a property of the metric rather than a finding about the data.
    2. Rank candidates by precision. Because step 1 ties them all, the winner
       is in practice decided by the next key: **longer median lead time** —
       between two equally precise operating points the earlier warning is the
       more useful one, and lead time is the project's second headline. A
       remaining tie breaks on the smaller `min_wait_hours`, so the result is
       deterministic rather than dependent on dict order. Precision is kept as
       the first key anyway, so that a future change which *does* move it
       cannot be silently outvoted by lead time.
    3. The threshold is then the budget-implied cut the metric returns for the
       winning candidate, so the reported operating point spends exactly its
       allowance.
    4. Always-quiet is run on the identical frame and its precision carried
       alongside, because "0.31" means nothing without the floor beside it.

    `conn` is passed through to `predict()`, so the sealed test split is
    refused here exactly as it is everywhere else.
    """
    grid = grid or cfg["baselines"]["volume_zscore"]["min_wait_hours_grid"]
    floor = precision_at_alert_budget(
        AlwaysQuiet(cfg).predict(frame, conn=conn, context="tune floor"))

    best = None
    for wait in grid:
        model = VolumeZScore(cfg, min_wait_hours=wait)
        # A finite placeholder threshold: precision is rank-based and does not
        # depend on it, and the reported threshold is replaced below by the
        # budget-implied cut. Using -inf here would flag every window's first
        # legal hour, which is what the min-wait sweep is meant to measure.
        scored = model.predict(frame, threshold=float("inf"), conn=conn,
                               context=f"tune wait={wait}")
        budget = precision_at_alert_budget(scored)

        # Lead time needs actual flags, so re-derive actions at the cut the
        # budget implies rather than at the placeholder.
        at_cut = model.predict(frame, threshold=budget.threshold, conn=conn,
                               context=f"tune wait={wait} at cut")
        delay = detection_delay_summary(at_cut)
        lead = delay.median_trading_hours if delay.n_detections else float("-inf")

        key = (budget.precision, lead, -float(wait))
        if best is None or key > best[0]:
            best = (key, wait, budget, lead)

    _, wait, budget, lead = best
    return OperatingPoint(
        threshold=float(budget.threshold),
        min_wait_hours=float(wait),
        precision=float(budget.precision),
        max_precision=float(budget.max_precision),
        recall=float(budget.recall),
        base_rate=float(budget.base_rate),
        floor_precision=float(floor.precision),
        median_trading_hours=float(lead),
        n_windows=int(budget.n_windows),
        n_positive=int(budget.n_positive),
        grid=tuple(float(g) for g in grid),
    )
