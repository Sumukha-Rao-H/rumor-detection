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
    n_tickers: int | None = None,
    span_days: int = 300,
    seed: int = 0,
) -> pd.DataFrame:
    """A prediction frame shaped exactly like a real one.

    Positive windows end at a t0; quiet windows have none. Scores are noise
    plus, on positive windows, a ramp that grows as t0 approaches — a stand-in
    for a footprint building before the announcement.

    `signal_strength=0` makes positives and negatives statistically identical,
    which is the useful case for checking that a metric reports chance-level
    performance rather than something flattering.

    `n_tickers` and `span_days` control DENSITY — how many windows fall in one
    ticker-month. That matters because the alert budget is defined per stock
    per month: spread a few windows over many tickers and months and the budget
    exceeds the number of windows, making any budgeted metric vacuous. Defaults
    are deliberately sparse; pass a small `n_tickers` for a realistic monitoring
    load.

    Actions are all WAIT. Callers that need decisions derive them with
    `contract.actions_from_scores`, the same way a threshold baseline will.
    """
    cfg = load_config()
    horizon = horizon_hours or cfg["decision"]["horizon_hours"]
    rng = np.random.default_rng(seed)

    base = date_str_to_ts(cfg["study_window"]["start"])
    n_tickers = n_tickers or max(1, (n_positive + n_quiet) // 8)
    tickers = np.array([f"TKR{i:03d}" for i in range(n_tickers)])
    items = np.array(cfg["items"]["unscheduled_focus"] + cfg["items"]["scheduled"])
    scheduled_set = set(cfg["items"]["scheduled"])

    # Every column is built once across all windows and flattened row-major, so
    # each window's hours stay contiguous and ascending. Building one small
    # frame per window and concatenating them cost ~0.8 s at 1,500 windows.
    # [horizon .. 1] — stops one hour before t0, deliberately: this mirrors
    # src/pipeline/features.py, whose window is STRICTLY before t0
    # (test_stricter_than_the_evaluation_contract), so `ts_utc == t0_utc`
    # never appears in a real feature matrix either. contract.py's leakage
    # check allows that boundary (`ts_utc <= t0_utc`) for defensiveness, but
    # nothing production emits reaches it, so this generator does not
    # manufacture it — doing so would make "shaped exactly like a real one"
    # false. The boundary is exercised by a hand-mutated row in
    # test_row_exactly_at_t0_is_allowed instead.
    hours_to_t0 = np.arange(horizon, 0, -1)          # [horizon .. 1]
    ramp = signal_strength * (1.0 - hours_to_t0 / horizon)

    blocks: list[dict] = []

    if n_positive:
        t0s = base + rng.integers(horizon, 24 * span_days, size=n_positive) * HOUR
        item_per_window = rng.choice(items, size=n_positive)
        blocks.append({
            "window_id": np.repeat([f"pos-{i:04d}" for i in range(n_positive)], horizon),
            "ticker": np.repeat(rng.choice(tickers, size=n_positive), horizon),
            "ts_utc": (t0s[:, None] - hours_to_t0[None, :] * HOUR).ravel(),
            "t0_utc": np.repeat(t0s, horizon),
            "score": (rng.normal(0.0, 1.0, (n_positive, horizon)) + ramp).ravel(),
            "is_scheduled": np.repeat(
                np.isin(item_per_window, list(scheduled_set)), horizon),
            "item_code": np.repeat(item_per_window, horizon),
        })

    if n_quiet:
        starts = base + rng.integers(0, 24 * span_days, size=n_quiet) * HOUR
        blocks.append({
            "window_id": np.repeat([f"quiet-{i:04d}" for i in range(n_quiet)], horizon),
            "ticker": np.repeat(rng.choice(tickers, size=n_quiet), horizon),
            "ts_utc": (starts[:, None] + np.arange(horizon)[None, :] * HOUR).ravel(),
            "t0_utc": np.full(n_quiet * horizon, pd.NA, dtype=object),
            "score": rng.normal(0.0, 1.0, (n_quiet, horizon)).ravel(),
            "is_scheduled": np.full(n_quiet * horizon, pd.NA, dtype=object),
            "item_code": np.full(n_quiet * horizon, pd.NA, dtype=object),
        })

    if not blocks:
        from src.eval.contract import empty_frame
        return empty_frame()

    data = {
        key: np.concatenate([b[key] for b in blocks])
        for key in blocks[0]
    }
    data["action"] = np.full(len(data["window_id"]), WAIT)
    return conform(pd.DataFrame(data))
