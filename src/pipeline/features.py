"""Feature builders — everything the model is allowed to see.

One rule governs this module, and breaking it does not raise an error, it
produces a number that looks good and is wrong:

    every value at time t uses only data timestamped <= t,
    through backward-looking windows only.

`tests/test_leakage.py` was written in P2-08 against this module before any of
it existed, precisely so the check could not be shaped around the
implementation. It perturbs rows after `t`, recomputes, and flags any earlier
value that moved.

**Frame contract.** Each builder takes ONE ticker's bars: a DataFrame indexed by
`ts_utc` (UTC epoch seconds, ascending) with at least `close` and `volume`, and
returns a DataFrame on the same index. One ticker at a time is not a
convenience — a frame holding several would let a lag reach across a company
boundary and compute Apple's return from Microsoft's price, and no test of
"does it read the future" would catch that, because the timestamps would be
perfectly ordered.

**Horizons are in BARS, not wall-clock hours.** Bars exist only while the market
is open, so a positional lag is automatically a trading-time lag — which is what
this project asks for everywhere else. A wall-clock lag would reach across a
weekend and compare Monday's price with Friday's while calling it 24 hours. The
imprecision worth naming: a session yields 7 bars in 6.5 hours (six full hours
plus the 15:30 half-hour stub), so 24 bars is about 22.3 trading hours.
"""

from __future__ import annotations

import pandas as pd

from src.utils.config import load_config


def returns(frame: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Trailing returns over each configured horizon.

        ret_{h}h(t) = close(t) / close(t - h bars) - 1

    Both prices are at or before `t`, so the value is knowable at `t`.

    NaN where the horizon reaches before the series starts, or where a bar is
    missing. Deliberately not filled: a zero would be a fabricated observation
    of "no move", and forward-filling would invent a price that never traded.
    `pct_change` no longer fills by default on pandas 3, so this is the library
    behaviour rather than something layered on top.
    """
    cfg = cfg or load_config()
    close = frame["close"]
    return pd.DataFrame(
        {f"ret_{h}h": close.pct_change(h)
         for h in cfg["features"]["return_horizons_h"]},
        index=frame.index,
    )
