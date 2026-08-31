"""P4-08 — realised volatility, and why it does NOT shift the way P4-07 did.

The z-score is a comparison, so the bar must leave its own baseline. Volatility
is a description, so the move realised at `t` belongs in it. Both are
backward-looking; only one is a self-comparison. The last test here documents
that contrast, because "the previous feature shifted and this one does not"
looks like an inconsistency until someone states the reason.
"""

import numpy as np
import pandas as pd
import pytest

from src.pipeline.features import realised_volatility
from src.utils.config import load_config
from tests.test_leakage import assert_no_lookahead


HOUR = 3600
START = 1_756_684_800


@pytest.fixture
def cfg():
    return load_config()


def bars(closes):
    idx = pd.Index([START + HOUR * i for i in range(len(closes))],
                   name="ts_utc")
    return pd.DataFrame({"close": [float(c) for c in closes],
                         "volume": [1e6] * len(closes)}, index=idx)


def walk(n, sd, seed=11, start=100.0):
    rng = np.random.default_rng(seed)
    return start * np.exp(np.cumsum(rng.normal(0, sd, n)))


# --------------------------------------------------------------------------
# the arithmetic
# --------------------------------------------------------------------------

def test_volatility_is_the_sd_of_one_bar_returns(cfg):
    w = cfg["features"]["volatility_window_h"]
    closes = walk(w + 5, 0.01)
    got = realised_volatility(bars(closes), cfg)["volatility"].iloc[-1]

    rets = pd.Series(closes).pct_change(1)
    expected = rets.iloc[-w:].std()
    assert got == pytest.approx(expected)


def test_a_flat_series_has_zero_volatility(cfg):
    """No denominator here, unlike the z-score — zero is a real answer."""
    w = cfg["features"]["volatility_window_h"]
    v = realised_volatility(bars([100.0] * (w + 5)), cfg)["volatility"]
    assert v.iloc[-1] == pytest.approx(0.0)


def test_a_jumpy_series_scores_higher_than_a_calm_one(cfg):
    """The feature has to discriminate, or it is decoration."""
    w = cfg["features"]["volatility_window_h"] + 5
    calm = realised_volatility(bars(walk(w, 0.002)), cfg)["volatility"].iloc[-1]
    jumpy = realised_volatility(bars(walk(w, 0.02)), cfg)["volatility"].iloc[-1]
    assert jumpy > calm * 5


# --------------------------------------------------------------------------
# the window
# --------------------------------------------------------------------------

def test_nan_until_the_window_is_full(cfg):
    """`min_periods` is the window: 120 bars means 120, not what happens to exist."""
    w = cfg["features"]["volatility_window_h"]
    v = realised_volatility(bars(walk(w + 3, 0.01)), cfg)["volatility"]
    assert v.iloc[:w].isna().all()
    assert v.iloc[w:].notna().all()


def test_the_first_non_nan_is_at_index_window(cfg):
    """The off-by-one, pinned: the first bar has no return at all, so a full
    window of returns is only available at index `window`, not `window - 1`."""
    w = cfg["features"]["volatility_window_h"]
    v = realised_volatility(bars(walk(w + 3, 0.01)), cfg)["volatility"]
    assert v.index.get_loc(v.first_valid_index()) == w


def test_a_jump_leaves_the_window_after_exactly_window_bars(cfg):
    """Rolling decay, made deliberate rather than assumed."""
    w = cfg["features"]["volatility_window_h"]
    closes = [100.0] * 5 + [150.0] * (w + 5)          # one jump, then flat
    v = realised_volatility(bars(closes), cfg)["volatility"]
    jump_idx = 5
    assert v.iloc[jump_idx + w - 1] > 0               # jump still inside
    assert v.iloc[jump_idx + w] == pytest.approx(0.0)  # it has aged out


def test_window_comes_from_config(cfg):
    small = {**cfg, "features": {**cfg["features"], "volatility_window_h": 10}}
    v = realised_volatility(bars(walk(20, 0.01)), small)["volatility"]
    assert v.iloc[:10].isna().all() and v.iloc[10:].notna().all()


# --------------------------------------------------------------------------
# contracts
# --------------------------------------------------------------------------

def test_volatility_passes_the_leakage_detector(cfg):
    assert_no_lookahead(lambda f: realised_volatility(f, cfg),
                        bars(walk(400, 0.01)))


def test_including_the_current_return_is_deliberate(cfg):
    """Documents the contrast with P4-07's `.shift(1)`.

    Volatility describes movement up to and including now. A shifted version
    would report the previous bar's answer under the current bar's label —
    staler, not safer. The two differ, and this pins which one we compute.
    """
    w = cfg["features"]["volatility_window_h"]
    closes = list(walk(w + 1, 0.005)) + [1000.0]      # a violent final bar
    frame = bars(closes)

    ours = realised_volatility(frame, cfg)["volatility"].iloc[-1]
    shifted = (frame["close"].pct_change(1).shift(1)
               .rolling(w, min_periods=w).std().iloc[-1])
    assert ours > shifted        # the final move is counted, as intended
