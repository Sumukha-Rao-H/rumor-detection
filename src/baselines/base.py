"""The one shape every baseline is written against.

A baseline's job is to look at the hours before a possible announcement and
say, each hour, whether to WAIT or to FLAG. Only the *scoring* differs between
them: always-quiet never alarms, volume z-score watches one feature, CUSUM
accumulates evidence, gradient boosting learns a function. Everything between
a score and a number in the comparison table is identical — and is written
once, here.

That is not tidiness. The shared part is the part that is easy to get wrong in
a way that produces a better-looking number rather than an error:

* an episode ends when it flags, so a window holds **at most one FLAG**;
* a window a baseline cannot score must **stay in the denominator**, or
  precision rises without anything being detected;
* the **sealed test set** must stay untouched until Phase 10;
* the output must satisfy the P1-07 contract **before** any metric sees it.

Written four times, those drift apart and `P5-06`'s table compares detectors
that measured different things. Written here, a subclass cannot skip them: it
supplies `score()` and nothing else.

Unscoreable windows
-------------------
`sampling.py` defers one decision to this module by name: issue 32 found 234
positive windows with `volume_z` NaN throughout, because fewer than
`features.min_baseline_bars` bars precede them. No volume-based detector can
score those windows.

They are **ranked last, never dropped**. Dropping them would flatter the
metric — `precision_at_alert_budget` divides by a window count, and removing
hard windows raises precision without detecting anything — and it would leave
volume z-score scored on a smaller window set than always-quiet, so the two
rows of the comparison table would not be about the same task.

The floor is derived from the frame (`min(finite scores) - 1`) rather than a
configured constant, because the contract deliberately does not bound scores:
any fixed sentinel could sit above a real score on some future baseline. It is
finite because the contract rejects ±inf, a rule that exists so one infinity
cannot poison Brier and calibration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from src.eval import contract
from src.pipeline import split
from src.utils.config import load_config
from src.utils.timeutils import get_market_calendar, trading_hours_between

#: Columns the feature matrix carries that are also contract columns. Copied
#: through untouched so a baseline cannot accidentally relabel a window.
_CARRIED = ("window_id", "ticker", "ts_utc", "t0_utc", "is_scheduled", "item_code")


def unscoreable_floor(scores: pd.Series) -> float:
    """A finite score strictly below every real score in `scores`.

    Used to rank a window a baseline could not score without removing it from
    the evaluation. Returns 0.0 when nothing is finite — a baseline whose
    scores are all NaN is degenerate, but it should still produce a valid
    frame that the metrics report as catching nothing, rather than a NaN that
    propagates into every aggregate.
    """
    finite = scores[np.isfinite(scores)]
    if finite.empty:
        return 0.0
    return float(finite.min()) - 1.0


class Baseline(ABC):
    """A detector that scores each hour of a window and stops at the first FLAG.

    Subclasses implement `score()`. `fit()` is a no-op unless overridden —
    always-quiet has nothing to learn, gradient boosting does.

    `name` is used in the comparison table and in error messages; it defaults
    to the class name.
    """

    name: str = ""

    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = cfg if cfg is not None else load_config()
        if not self.name:
            self.name = type(self).__name__

    # -- what a subclass provides ---------------------------------------
    @abstractmethod
    def score(self, frame: pd.DataFrame) -> pd.Series:
        """Score every row of `frame`; higher means news is more likely coming.

        Must return a Series aligned to `frame.index`. NaN is allowed and
        means "this hour cannot be scored" — the base class decides what
        happens to it, so that decision is the same for every baseline.

        Scores need not be probabilities and need not be bounded; the contract
        says so, and only a baseline that claims to emit probabilities is
        measured by Brier or ECE.
        """

    def fit(self, train: pd.DataFrame) -> "Baseline":
        """Learn from the training split. Default: nothing to learn.

        Returns self so `Baseline().fit(train).predict(val)` reads in one line.
        """
        return self

    # -- what every baseline gets for free -------------------------------
    def _common(self) -> dict:
        return self.cfg.get("baselines", {}).get("common", {})

    def _apply_unscoreable(self, frame: pd.DataFrame, scores: pd.Series) -> pd.Series:
        """Floor windows this baseline could not score at all.

        A window is unscoreable only when **none** of its hours got a finite
        score. A single NaN hour inside an otherwise scoreable window is
        floored on its own and the window keeps its other hours — losing a
        whole window over one missing bar would throw away information the
        detector does have.
        """
        policy = self._common().get("unscoreable", "rank_last")
        finite = pd.Series(np.isfinite(scores), index=scores.index)
        if finite.all():
            return scores

        any_finite = finite.groupby(frame["window_id"], sort=False).transform("any")
        dead_windows = frame.loc[~any_finite, "window_id"].unique()

        if policy == "error" and len(dead_windows):
            raise ValueError(
                f"{self.name}: {len(dead_windows)} window(s) have no finite "
                f"score and 'baselines.common.unscoreable' is 'error'; first: "
                f"{list(dead_windows[:5])}"
            )
        if policy not in ("rank_last", "error"):
            raise ValueError(
                f"unknown baselines.common.unscoreable={policy!r}; "
                f"expected 'rank_last' or 'error'"
            )

        # One floor for the whole frame, so a dead window ranks below every
        # real score everywhere — not merely below its own window's scores.
        return scores.fillna(unscoreable_floor(scores)).replace(
            [np.inf, -np.inf], unscoreable_floor(scores)
        )

    def _apply_min_wait(self, frame: pd.DataFrame, actions: pd.Series) -> pd.Series:
        """Forbid a FLAG in the opening `min_wait_hours` **trading** hours.

        Measured with the market-hours helpers, never by subtracting two
        timestamps: most windows straddle an overnight or a weekend, so
        wall-clock hours would silently let a flag through (or block one) at
        the wrong point. Rule 3 of the code standards.

        A window whose first legal hour never arrives simply never flags.
        """
        hours = float(self._common().get("min_wait_hours", 0) or 0)
        if hours <= 0:
            return actions

        # The calendar OBJECT, not the code string: `trading_hours_between`
        # takes an ExchangeCalendar and would fail on a bare "XNYS". Resolved
        # once here rather than per row — `get_market_calendar` is cached, but
        # the lookup still is not free at one call per flagged hour.
        calendar = get_market_calendar(self.cfg["market"]["calendar"])
        out = actions.copy()
        starts = frame.groupby("window_id", sort=False)["ts_utc"].transform("min")
        flagged = out == contract.FLAG
        for idx in out.index[flagged]:
            elapsed = trading_hours_between(
                int(starts.loc[idx]), int(frame.loc[idx, "ts_utc"]), calendar
            )
            if elapsed < hours:
                out.loc[idx] = contract.WAIT
        return out

    def predict(
        self,
        frame: pd.DataFrame,
        threshold: float,
        conn=None,
        context: str = "",
    ) -> pd.DataFrame:
        """Score `frame` and return a validated contract frame.

        `threshold` decides WAIT vs FLAG: the first hour at or above it flags,
        and the episode ends there. It is explicit rather than derived inside
        this call because P5-03 and P5-04 tune it on validation and P5-06
        re-derives it from the alert budget; a threshold that appeared by
        magic would be a threshold nobody could report.

        Pass `conn` to have the sealed test split enforced. It is optional
        because a synthetic frame has no database behind it, but on the real
        path it should always be supplied — `assert_not_test` fails closed, so
        the check is only skipped when there is genuinely nothing to check
        against.
        """
        if frame.empty:
            raise ValueError(
                f"{self.name}: prediction frame is empty — nothing to evaluate"
            )
        missing = [c for c in _CARRIED if c not in frame.columns]
        if missing:
            raise ValueError(f"{self.name}: feature frame is missing {missing}")

        if conn is not None:
            split.assert_not_test(
                self.cfg, conn, frame["ts_utc"], context or f"{self.name}.predict"
            )

        scores = pd.Series(self.score(frame), index=frame.index, dtype="float64")
        scores = self._apply_unscoreable(frame, scores)

        out = frame.loc[:, list(_CARRIED)].copy()
        out["score"] = scores
        out = contract.actions_from_scores(out, threshold)
        out["action"] = self._apply_min_wait(frame, out["action"])

        # Emit in chronological order within each window. `actions_from_scores`
        # deliberately works regardless of the caller's row order, but the
        # contract requires ascending ts_utc, so an unsorted feature frame
        # would score correctly and then fail validation. Sorting here means a
        # baseline never has to think about it.
        out = out.sort_values(["window_id", "ts_utc"], kind="stable").reset_index(drop=True)
        return contract.validate_predictions(out)
