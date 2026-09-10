"""The detector that never alarms — the floor under every other number.

At a base rate near 0.3%, a policy that always says WAIT is right 99.7% of the
time and detects nothing. That is the trap this project's headline metric
exists to avoid, and this module is the trap made concrete: it is run through
exactly the same evaluation path as every real baseline, so the comparison
table carries the do-nothing number in the same column as the rest.

It has two faces, and both are true at once.

As a **stopping policy** it never FLAGs. Zero alerts, zero detections. This is
the 99.7% figure in its literal form — and the reason the project reports
precision at a fixed alert budget instead of accuracy.

As a **ranker** it emits one constant score, so it cannot order windows at
all. Every window ties, `precision_at_alert_budget` admits the whole tied
block, and precision comes out **equal to the base rate**. That equality is
the point: base-rate precision is what "no information" looks like on this
metric, and it is the line P5-03 through P5-05 have to clear to have found
anything.

Accuracy is not computed here, despite being what the module is about.
`metrics.accuracy()` raises on purpose; the trap is stated — base rate,
recall, zero flags — **never reported as a result**. A number that rewards the
degenerate answer does not become a finding just because the degenerate answer
is under test.

"Never calculated" would be too strong, and was: `sampling.eval_decision_points`
does compute what accuracy would say, and `sampling.print_report` prints it,
labelled so it cannot be quoted as a metric. That is the same rhetorical move
this module makes by existing — showing the reader the 99.7% and why it is
worthless is the argument for the headline metric. What must never happen is
the figure appearing in a results table as though it measured something.
"""

from __future__ import annotations

import pandas as pd

from src.baselines.base import Baseline


class AlwaysQuiet(Baseline):
    """Scores every hour identically and never flags.

    The constant's *value* is arbitrary by construction — a constant ranker is
    uninformative at any level — so it lives in config only to keep a literal
    out of `src/`. A test pins that changing it changes no reported number.
    """

    name = "always_quiet"

    def __init__(self, cfg: dict | None = None) -> None:
        super().__init__(cfg)
        self.constant = float(
            self.cfg.get("baselines", {}).get("always_quiet", {}).get("score", 0.0)
        )

    def score(self, frame: pd.DataFrame) -> pd.Series:
        """One value everywhere. Reads no feature, so nothing is unscoreable."""
        return pd.Series(self.constant, index=frame.index, dtype="float64")

    def predict(
        self,
        frame: pd.DataFrame,
        threshold: float = float("inf"),
        conn=None,
        context: str = "",
    ) -> pd.DataFrame:
        """Never flags, whatever threshold the caller passes.

        The guarantee has to be structural. A constant score of 0.0 with a
        threshold of -1 — easily produced by a tuning sweep that does not know
        which baseline it is driving — would flag the first hour of every
        window, and the floor would quietly become an alarm-on-everything
        detector while still calling itself always-quiet.

        So the caller's threshold is ignored and `+inf` is used instead.
        Contract scores are finite, so no crossing is possible. Everything else
        still runs through `Baseline.predict`: the seal check, the
        chronological ordering, the contract validation. The floor must be
        produced by the same machinery as the numbers it is compared against,
        or it proves nothing about them.
        """
        return super().predict(
            frame,
            threshold=float("inf"),
            conn=conn,
            context=context or f"{self.name}.predict",
        )
