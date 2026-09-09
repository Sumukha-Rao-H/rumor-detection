"""A scorer that knows nothing — the null the whole table is measured against.

`always_quiet` is the floor for a detector that never alarms. This is the floor
for one that alarms at random, and the two are not the same number whenever the
evaluation frame is not perfectly symmetric.

Why this exists
---------------
It was added after a review found that the evaluation frame gave a positive
episode 48 bars and a quiet window a single bar, while a window scores as the
maximum over its rows. A positive therefore got 48 independent chances to cross
the threshold and a quiet window got one, at the same one-alert cost — so
window LENGTH decided the ranking rather than detection. Measured on the
Phase 10 shape, pure noise reached precision 0.0943 and **29.6x lift**, beating
every tuned detector in the final table.

`metrics.window_summary` now gives a quiet window the same span as an episode,
and noise scores 1.0x again. But "the metric is fixed" is a claim, and a claim
that only lives in a commit message is one nobody can check. Running the null
as an ordinary baseline puts it in the same column as everything else, every
time the table is built, so the artefact cannot come back unnoticed — and a
reader who does not trust the fix does not have to take it on faith.

If this row ever reports meaningfully more than the always-quiet floor, the
evaluation frame has developed an asymmetry again and **no other row in the
table means anything** until that is explained.

The seed comes from config so the row is reproducible. Its value should not
matter — a null that moves with its seed is a null with too few windows to say
anything — and the spread across seeds is itself worth a glance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.baselines.base import Baseline


class RandomNoise(Baseline):
    """Draws one uniform score per hour, from a seeded generator.

    Reads no feature at all, so nothing is ever unscoreable and the row is
    never affected by missing history — which matters, because it means a
    difference between this row and a real baseline cannot be blamed on
    coverage.
    """

    name = "random_noise"

    def __init__(self, cfg: dict | None = None) -> None:
        super().__init__(cfg)
        self.seed = int(
            self.cfg.get("baselines", {}).get("random_noise", {}).get("seed", 0)
        )

    def score(self, frame: pd.DataFrame) -> pd.Series:
        """Uniform noise, seeded per call.

        Seeded per call rather than per instance so that scoring the same frame
        twice gives the same answer — the comparison table is rebuilt more than
        once per run (once per t0 variant), and a row that moved between them
        would look like a finding.
        """
        rng = np.random.default_rng(self.seed)
        return pd.Series(rng.random(len(frame)), index=frame.index,
                         dtype="float64")
