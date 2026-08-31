"""P4-06 — the first feature, and the first real test of the leakage detector.

`tests/test_leakage.py` was written in P2-08 against a `features.py` that did
not exist, so the check could not be shaped around the implementation. This is
where that bet pays off: the same detector now runs against real code.

The tests that matter are the two at the bottom. The arithmetic ones would fail
loudly; a leak would not.
"""

import numpy as np
import pandas as pd
import pytest

from src.pipeline.features import returns
from src.utils.config import load_config
from tests.test_leakage import LookaheadError, assert_no_lookahead, find_lookahead


HOUR = 3600
START = 1_756_684_800   # 2025-09-01 00:00 UTC, the study window start


@pytest.fixture
def cfg():
    return load_config()


def bars(closes, volumes=None, start=START, step=HOUR):
    idx = pd.Index([start + step * i for i in range(len(closes))], name="ts_utc")
    return pd.DataFrame(
        {"close": [float(c) for c in closes],
         "volume": volumes if volumes is not None else [1e6] * len(closes)},
        index=idx)


def random_bars(n=400, seed=7):
    rng = np.random.default_rng(seed)
    return bars(100 + np.cumsum(rng.normal(0, 0.5, n)),
                volumes=rng.lognormal(12, 0.4, n))


# --------------------------------------------------------------------------
# the arithmetic
# --------------------------------------------------------------------------

def test_returns_are_computed_against_the_lagged_close(cfg):
    f = returns(bars([100, 110, 121, 133.1]), cfg)
    assert f["ret_1h"].tolist()[1:] == pytest.approx([0.10, 0.10, 0.10])


def test_a_longer_horizon_spans_more_bars(cfg):
    f = returns(bars([100, 101, 102, 103, 104]), cfg)
    assert f["ret_4h"].iloc[4] == pytest.approx(0.04)
    assert f["ret_1h"].iloc[4] == pytest.approx(104 / 103 - 1)


def test_one_column_per_configured_horizon(cfg):
    f = returns(random_bars(), cfg)
    assert list(f.columns) == [f"ret_{h}h"
                               for h in cfg["features"]["return_horizons_h"]]


def test_horizons_come_from_config_not_the_code(cfg):
    custom = {**cfg, "features": {**cfg["features"], "return_horizons_h": [2, 7]}}
    assert list(returns(random_bars(), custom).columns) == ["ret_2h", "ret_7h"]


def test_a_fall_is_negative(cfg):
    assert returns(bars([100, 90]), cfg)["ret_1h"].iloc[1] == pytest.approx(-0.10)


# --------------------------------------------------------------------------
# what is NOT filled in
# --------------------------------------------------------------------------

def test_the_first_h_rows_are_nan(cfg):
    """Undefined, not zero. Zero would fabricate an observation of "no move"."""
    f = returns(bars(list(range(100, 110))), cfg)
    assert f["ret_4h"].iloc[:4].isna().all()
    assert not f["ret_4h"].iloc[4:].isna().any()


def test_a_missing_bar_is_not_bridged(cfg):
    """Forward-filling would invent a price that never traded."""
    f = returns(bars([100, np.nan, 102, 103]), cfg)
    assert f["ret_1h"].iloc[1:3].isna().all()
    assert f["ret_1h"].iloc[3] == pytest.approx(103 / 102 - 1)


def test_a_frame_shorter_than_the_horizon_is_all_nan(cfg):
    f = returns(bars([100, 101, 102]), cfg)
    assert f["ret_120h"].isna().all()


def test_index_and_length_are_preserved(cfg):
    frame = random_bars(50)
    f = returns(frame, cfg)
    assert len(f) == len(frame)
    assert f.index.equals(frame.index)
    assert f.index.name == "ts_utc"


def test_an_overnight_gap_is_one_step_not_many(cfg):
    """Bars exist only when the market is open, so a positional lag IS a
    trading-time lag. A wall-clock lag would compare Monday with Friday and
    call it an hour."""
    idx = pd.Index([START, START + HOUR, START + 20 * HOUR], name="ts_utc")
    frame = pd.DataFrame({"close": [100.0, 101.0, 110.0],
                          "volume": [1e6] * 3}, index=idx)
    assert returns(frame, cfg)["ret_1h"].iloc[2] == pytest.approx(110 / 101 - 1)


# --------------------------------------------------------------------------
# leakage — the tests that matter
# --------------------------------------------------------------------------

def test_returns_pass_the_leakage_detector(cfg):
    """P2-08's detector, written before this module existed, against real code."""
    assert_no_lookahead(lambda f: returns(f, cfg), random_bars())


def test_perturbing_the_future_changes_nothing_earlier(cfg):
    """The property directly, without the helper: alter only rows after a
    cut point, recompute, and every earlier value must be identical."""
    frame = random_bars(200)
    cut = 120
    before = returns(frame, cfg).iloc[:cut]

    tampered = frame.copy()
    tampered.iloc[cut:, tampered.columns.get_loc("close")] *= 3.0
    after = returns(tampered, cfg).iloc[:cut]

    pd.testing.assert_frame_equal(before, after)


def test_the_detector_still_catches_a_centred_window(cfg):
    """Guards the guard: if the detector had gone blind, the test above would
    pass for the wrong reason."""
    def leaky(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"ret_centred": frame["close"].rolling(5, center=True).mean()},
            index=frame.index)

    assert find_lookahead(leaky, random_bars())
    with pytest.raises(LookaheadError, match="reading the future"):
        assert_no_lookahead(leaky, random_bars())
