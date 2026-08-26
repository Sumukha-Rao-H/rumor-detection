"""Fake prediction frames, for building metrics before any model exists.

The plan requires the evaluation code to be written **before** the first model
(plan §9 phase 1), so that the headline metric cannot be chosen after seeing
which one flatters a result. That is only possible if there is something to run
the metrics on — hence this module.

It lives in `src/eval/` rather than `tests/` because it is a working tool for
that sequencing, not a test fixture. Tests use it too.

Nothing here touches the database, the network, or real prices. Every frame is
generated from a seed and is reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.contract import FLAG, WAIT, conform
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts

HOUR = 3600


def make_synthetic_predictions(
    n_positive: int = 20,
    n_quiet: int = 200,
    horizon_hours: int | None = None,
    signal_strength: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """A prediction frame shaped exactly like a real one.

    Positive windows end at a t0; quiet windows have none. Scores are noise
    plus, on positive windows, a ramp that grows as t0 approaches — a stand-in
    for a footprint building before the announcement.

    `signal_strength=0` makes positives and negatives statistically identical,
    which is the useful case for checking that a metric reports chance-level
    performance rather than something flattering.

    Actions are all WAIT. Callers that need decisions derive them with
    `contract.actions_from_scores`, the same way a threshold baseline will.
    """
    cfg = load_config()
    horizon = horizon_hours or cfg["decision"]["horizon_hours"]
    rng = np.random.default_rng(seed)

    base = date_str_to_ts(cfg["study_window"]["start"])
    tickers = [f"TKR{i:03d}" for i in range(max(1, (n_positive + n_quiet) // 8))]
    items = cfg["items"]["unscheduled_focus"] + cfg["items"]["scheduled"]
    scheduled_set = set(cfg["items"]["scheduled"])

    rows: list[pd.DataFrame] = []

    for i in range(n_positive):
        t0 = base + int(rng.integers(horizon, 24 * 300)) * HOUR
        hours = np.arange(horizon, 0, -1)          # hours remaining until t0
        ts = t0 - hours * HOUR
        ramp = signal_strength * (1.0 - hours / horizon)   # 0 far out, ->1 at t0
        item = str(rng.choice(items))
        rows.append(pd.DataFrame({
            "window_id": f"pos-{i:04d}",
            "ticker": str(rng.choice(tickers)),
            "ts_utc": ts,
            "t0_utc": t0,
            "score": rng.normal(0.0, 1.0, horizon) + ramp,
            "action": WAIT,
            "is_scheduled": item in scheduled_set,
            "item_code": item,
        }))

    for i in range(n_quiet):
        start = base + int(rng.integers(0, 24 * 300)) * HOUR
        rows.append(pd.DataFrame({
            "window_id": f"quiet-{i:04d}",
            "ticker": str(rng.choice(tickers)),
            "ts_utc": start + np.arange(horizon) * HOUR,
            "t0_utc": pd.NA,
            "score": rng.normal(0.0, 1.0, horizon),
            "action": WAIT,
            "is_scheduled": pd.NA,
            "item_code": pd.NA,
        }))

    return conform(pd.concat(rows, ignore_index=True))
