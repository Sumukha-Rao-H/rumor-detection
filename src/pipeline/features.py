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


def volume_zscore(frame: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """How unusual this hour's volume is against the stock's own recent normal.

    The project's central feature. If the detector works at all, it works
    because of this number.

        base = volume.shift(1).rolling(window, min_periods=min_baseline_bars)
        z    = (volume - base.mean()) / base.std()

    **`.shift(1)` is the point.** Without it the current bar sits inside its own
    baseline, so a genuine spike inflates the mean and standard deviation it is
    measured against and damps the very signal we are hunting. A 10x volume
    hour would score lower than it should, and nothing would look wrong.

    **The leakage detector cannot catch that.** It perturbs rows AFTER t;
    including the current bar uses data AT t, which the no-future rule permits.
    So the property has its own test in `test_features_volume.py` rather than
    relying on `assert_no_lookahead`.

    Zero variance yields NaN, never infinity. If every baseline bar has the same
    volume the standard deviation is 0, and any differing current bar would
    divide to +/-inf — which propagates silently through every downstream
    statistic and blows up any model that meets it. The frozen snapshot holds
    34,709 zero-volume bars, so a flat baseline is real, not hypothetical.
    Undefined is the honest value.
    """
    cfg = cfg or load_config()
    fcfg = cfg["features"]
    volume = frame["volume"]

    base = volume.shift(1).rolling(fcfg["volume_zscore_window_h"],
                                   min_periods=fcfg["min_baseline_bars"])
    std = base.std()
    z = (volume - base.mean()) / std
    return pd.DataFrame({"volume_z": z.where(std > 0)}, index=frame.index)


def realised_volatility(frame: pd.DataFrame,
                        cfg: dict | None = None) -> pd.DataFrame:
    """How much this stock has been moving lately: the sd of one-bar returns.

    Context for the footprint — a 3% move means something different in a calm
    stock than in a jumpy one.

    **No `.shift(1)` here, and that is deliberate.** P4-07 made the shift the
    whole task, so its absence needs a reason. The z-score is a *comparison*:
    it asks how far THIS bar sits from its baseline, so the bar must not be in
    the baseline or it drags normal toward itself. Volatility is a
    *description*: it asks how much the stock has been moving, and the move
    realised at `t` is part of the honest answer. It is knowable at `t`, from
    `close(t)` and `close(t-1)`, both at or before `t`. Excluding it would not
    be safer, only staler — reporting the previous bar's volatility while
    labelling it the current one.

    `min_periods` equals the window, not `min_baseline_bars`: volatility "over
    120 bars" should mean 120 bars. Reusing `min_baseline_bars` would also set a
    trap, since it happens to equal the window today — shrinking the window
    would silently make the feature NaN forever.

    Not annualised. Annualising multiplies by a constant, which changes nothing
    a model can use, and the constant needs a bars-per-year figure the
    7-bars-per-6.5-hour session makes awkward to state honestly.
    """
    cfg = cfg or load_config()
    window = cfg["features"]["volatility_window_h"]
    one_bar = frame["close"].pct_change(1)
    return pd.DataFrame(
        {"volatility": one_bar.rolling(window, min_periods=window).std()},
        index=frame.index,
    )
