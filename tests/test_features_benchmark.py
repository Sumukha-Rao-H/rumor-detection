"""P4-09 — stripping the market out, so what is left is company-specific.

The test that carries the feature's meaning is the first one: a stock that rose
exactly as much as the market must score zero. Everything else guards the
mechanics, of which alignment-before-differencing is the one most easily got
backwards.
"""

import numpy as np
import pandas as pd
import pytest

from src.pipeline.features import benchmark_relative
from src.utils.config import load_config
from tests.test_leakage import assert_no_lookahead


HOUR = 3600
START = 1_756_684_800


@pytest.fixture
def cfg():
    return load_config()


def bars(closes, start=START, step=HOUR):
    idx = pd.Index([start + step * i for i in range(len(closes))],
                   name="ts_utc")
    return pd.DataFrame({"close": [float(c) for c in closes],
                         "volume": [1e6] * len(closes)}, index=idx)


def walk(n, sd=0.01, seed=5, start=100.0):
    rng = np.random.default_rng(seed)
    return start * np.exp(np.cumsum(rng.normal(0, sd, n)))


# --------------------------------------------------------------------------
# what the feature means
# --------------------------------------------------------------------------

def test_a_stock_moving_with_the_market_scores_zero(cfg):
    """THE point: up 10% on a day the market rose 10% is not company news."""
    stock = bars([100, 110, 121])
    bench = bars([50, 55, 60.5])          # identical percentage moves
    rel = benchmark_relative(stock, bench, cfg)
    assert rel["ret_rel_1h"].iloc[1] == pytest.approx(0.0, abs=1e-12)
    assert rel["ret_rel_1h"].iloc[2] == pytest.approx(0.0, abs=1e-12)


def test_outperformance_is_positive(cfg):
    rel = benchmark_relative(bars([100, 110]), bars([100, 105]), cfg)
    assert rel["ret_rel_1h"].iloc[1] == pytest.approx(0.05)


def test_underperformance_is_negative(cfg):
    """Flat while the market rallied is a real negative signal."""
    rel = benchmark_relative(bars([100, 100]), bars([100, 104]), cfg)
    assert rel["ret_rel_1h"].iloc[1] == pytest.approx(-0.04)


def test_the_benchmark_against_itself_is_zero(cfg):
    """SPY sits in `bars` and could be passed by accident."""
    spy = bars(walk(30))
    rel = benchmark_relative(spy, spy, cfg)
    assert rel["ret_rel_1h"].dropna().abs().max() == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# alignment — the mechanic most easily inverted
# --------------------------------------------------------------------------

def test_alignment_happens_before_differencing(cfg):
    """The benchmark's "h bars ago" must be h bars in the STOCK's index.

    Here the benchmark carries an extra timestamp the stock does not have. If
    the return were computed on the benchmark's own index and reindexed
    afterwards, the 1-bar move compared against would be the wrong one.
    """
    stock = bars([100.0, 110.0], start=START, step=2 * HOUR)   # t0, t0+2h
    bench = pd.DataFrame(
        {"close": [100.0, 999.0, 100.0], "volume": [1e6] * 3},
        index=pd.Index([START, START + HOUR, START + 2 * HOUR], name="ts_utc"))

    rel = benchmark_relative(stock, bench, cfg)
    # On the stock's index the benchmark went 100 -> 100, i.e. 0%.
    # The spurious 999 bar is not on the stock's index and must not be used.
    assert rel["ret_rel_1h"].iloc[1] == pytest.approx(0.10)


def test_a_missing_benchmark_bar_gives_nan_not_a_stale_price(cfg):
    """Forward-filling would compare this hour's move to an older market move."""
    stock = bars([100.0, 110.0, 121.0])
    bench = pd.DataFrame({"close": [100.0], "volume": [1e6]},
                         index=pd.Index([START], name="ts_utc"))
    rel = benchmark_relative(stock, bench, cfg)
    assert rel["ret_rel_1h"].iloc[1:].isna().all()


def test_extra_benchmark_timestamps_are_ignored(cfg):
    stock = bars(walk(20))
    bench = bars(walk(40, seed=9))
    rel = benchmark_relative(stock, bench, cfg)
    assert rel.index.equals(stock.index) and len(rel) == 20


# --------------------------------------------------------------------------
# contracts
# --------------------------------------------------------------------------

def test_one_column_per_horizon(cfg):
    rel = benchmark_relative(bars(walk(200)), bars(walk(200, seed=8)), cfg)
    assert list(rel.columns) == [f"ret_rel_{h}h"
                                 for h in cfg["features"]["return_horizons_h"]]


def test_disabled_returns_an_empty_frame_on_the_same_index(cfg):
    """So a caller can concatenate unconditionally."""
    off = {**cfg, "features": {**cfg["features"],
                               "include_benchmark_relative": False}}
    stock = bars(walk(20))
    rel = benchmark_relative(stock, bars(walk(20, seed=8)), off)
    assert rel.empty and rel.index.equals(stock.index)


def test_benchmark_relative_passes_the_leakage_detector(cfg):
    bench = bars(walk(400, seed=8))
    assert_no_lookahead(lambda f: benchmark_relative(f, bench, cfg),
                        bars(walk(400)))
