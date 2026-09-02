"""P4-07 — the volume z-score, and the bug no leakage test can find.

The Done-when is that no z-score is computed from a window containing its own
bar. That property is NOT protected by `assert_no_lookahead`: the detector
perturbs rows AFTER t, while including the current bar uses data AT t, which
the no-future rule allows. So it is tested directly here — and one test proves
the detector really is blind to it, so nobody later deletes these on the
assumption that leakage coverage already handles it.
"""

import numpy as np
import pandas as pd
import pytest

from src.pipeline.features import volume_zscore
from src.utils.config import load_config
from tests.test_leakage import assert_no_lookahead, find_lookahead


HOUR = 3600
START = 1_756_684_800


@pytest.fixture
def cfg():
    return load_config()


def bars(volumes, closes=None):
    idx = pd.Index([START + HOUR * i for i in range(len(volumes))],
                   name="ts_utc")
    return pd.DataFrame(
        {"close": closes if closes is not None else [100.0] * len(volumes),
         "volume": [float(v) for v in volumes]}, index=idx)


def noisy(n, seed=3, scale=1e6):
    rng = np.random.default_rng(seed)
    return rng.lognormal(np.log(scale), 0.25, n)


def leaky_zscore(frame, cfg):
    """The same feature WITHOUT the shift — the bug this task prevents."""
    fcfg = cfg["features"]
    v = frame["volume"]
    base = v.rolling(fcfg["volume_zscore_window_h"],
                     min_periods=fcfg["min_baseline_bars"])
    std = base.std()
    return pd.DataFrame({"volume_z": ((v - base.mean()) / std).where(std > 0)},
                        index=frame.index)


# --------------------------------------------------------------------------
# the Done-when
# --------------------------------------------------------------------------

def test_a_spike_is_not_in_its_own_baseline(cfg):
    """THE acceptance criterion.

    The baseline behind the spike is ordinary noise, so a 10x hour must score
    far out. If the spike were inside its own window it would inflate the mean
    and standard deviation it is compared against.

    Threshold is 20, not 10: on this exact fixture the leaked version (spike
    left inside its own baseline) scores ~11.0 — a threshold of 10 would not
    have caught a deleted `.shift(1)` on its own. 20 sits strictly between the
    leaked ~11.0 and the correct ~31.8, so this test fails by itself if the
    shift regresses, without relying on the companion test below.
    """
    n = cfg["features"]["min_baseline_bars"] + 20
    v = list(noisy(n)) + [1e7]
    z = volume_zscore(bars(v), cfg)["volume_z"]
    assert z.iloc[-1] > 20


def test_including_the_current_bar_would_damp_the_spike(cfg):
    """Proves the shift is load-bearing, not decorative.

    The leaky version puts the spike inside its own baseline, raising the mean
    and standard deviation it is measured against and pulling the score down.

    Measured on this fixture: **z = 31.8 correct, 11.0 leaked** — the spike
    scores about a THIRD of its true magnitude. Not a rounding difference; a
    detector trained on the leaked version would see a much flatter world and
    would need a far lower threshold to fire at all.
    """
    n = cfg["features"]["min_baseline_bars"] + 20
    frame = bars(list(noisy(n)) + [1e7])
    correct = volume_zscore(frame, cfg)["volume_z"].iloc[-1]
    damped = leaky_zscore(frame, cfg)["volume_z"].iloc[-1]
    assert damped < correct
    assert correct / damped > 2          # measured ~2.9x, not a rounding artefact


def test_the_leakage_detector_cannot_see_this_bug(cfg):
    """Why the two tests above must exist.

    The leaky builder reads no future data — it reads the CURRENT bar — so the
    detector passes it. Recorded so nobody deletes those tests believing
    leakage coverage already protects this.
    """
    frame = bars(noisy(400))
    assert find_lookahead(lambda f: leaky_zscore(f, cfg), frame) == []


# --------------------------------------------------------------------------
# the guards
# --------------------------------------------------------------------------

def test_z_is_nan_before_min_baseline_bars(cfg):
    m = cfg["features"]["min_baseline_bars"]
    z = volume_zscore(bars(noisy(m + 5)), cfg)["volume_z"]
    assert z.iloc[:m].isna().all()
    assert z.iloc[m:].notna().any()


def test_a_flat_baseline_gives_nan_not_infinity(cfg):
    """34,709 zero-volume bars exist, so a flat baseline is real.

    An infinity would propagate silently through every downstream statistic.
    """
    n = cfg["features"]["min_baseline_bars"] + 20
    z = volume_zscore(bars([1e6] * n + [1e7]), cfg)["volume_z"]
    assert z.iloc[-1] != z.iloc[-1]              # NaN
    assert not np.isinf(z.to_numpy(dtype=float)).any()


def test_a_zero_volume_bar_scores_strongly_negative(cfg):
    """A legitimate observation, not missing data: no trading is unusual."""
    n = cfg["features"]["min_baseline_bars"] + 20
    z = volume_zscore(bars(list(noisy(n)) + [0.0]), cfg)["volume_z"]
    assert z.iloc[-1] < -2


def test_a_typical_bar_scores_near_zero(cfg):
    """The scale is sane — an ordinary hour is not an anomaly."""
    z = volume_zscore(bars(noisy(600)), cfg)["volume_z"]
    assert abs(z.dropna().median()) < 1.0


def test_window_and_minimum_come_from_config(cfg):
    small = {**cfg, "features": {**cfg["features"],
                                 "volume_zscore_window_h": 10,
                                 "min_baseline_bars": 5}}
    z = volume_zscore(bars(noisy(30)), small)["volume_z"]
    assert z.iloc[:5].isna().all() and z.iloc[5:].notna().all()


# --------------------------------------------------------------------------
# contracts
# --------------------------------------------------------------------------

def test_volume_zscore_passes_the_leakage_detector(cfg):
    assert_no_lookahead(lambda f: volume_zscore(f, cfg), bars(noisy(400)))


def test_index_and_length_are_preserved(cfg):
    frame = bars(noisy(200))
    out = volume_zscore(frame, cfg)
    assert len(out) == len(frame) and out.index.equals(frame.index)
    assert list(out.columns) == ["volume_z"]
