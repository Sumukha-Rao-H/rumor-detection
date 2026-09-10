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

One seed is not enough
----------------------
"Its value should not matter" is a claim about sampling noise, and on the
Phase 10 shape it is not quite true: a null draw there has E[TP] ~ 19.1 with
sd ~ 4.4, which is a lift anywhere from about 0.55x to 1.45x at two standard
deviations from chance alone. A single row reading 1.4x is therefore
indistinguishable from a real residual asymmetry of that size. The null as
built catches a 29.6x artefact; it cannot on its own certify the absence of a
1.5x one.

So `compare.run_baselines` emits this baseline once per seed in
`baselines.random_noise.seeds`, as separate rows named `random_noise[42]` and
so on — the same convention the per-seed policy rows already use. The spread
across those rows IS the error bar, and reading it is how a reader tells noise
from a finding. It reads no feature, so each extra row costs one score vector.
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

    def __init__(self, cfg: dict | None = None,
                 seed: int | None = None) -> None:
        super().__init__(cfg)
        noise = self.cfg.get("baselines", {}).get("random_noise", {})
        self.seed = int(noise.get("seed", 0) if seed is None else seed)
        if seed is not None:
            # An explicitly seeded draw is one point in the spread, not "the"
            # null, so it says which point it is. The unseeded construction
            # keeps the bare name, because that is what every existing caller
            # and the config's single `seed` entry mean by it.
            self.name = f"random_noise[{self.seed}]"

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
