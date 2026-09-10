"""P5-04 — accumulation, and the structural bias it had to avoid.

The test that matters most here is
`test_state_is_continuous_per_ticker_not_per_window`. The evaluation frame
gives a positive 48 bars and a negative one bar, so a per-window CUSUM would
score brilliantly for reasons that have nothing to do with the market. The
statistic runs per ticker instead, and that property is pinned rather than
trusted.
"""

import numpy as np
import pandas as pd
import pytest

from src import db
from src.baselines import CUSUM, AlwaysQuiet, VolumeZScore, cusum_statistic, tune_cusum
from src.eval import contract
from src.eval.metrics import precision_at_alert_budget
from src.pipeline.split import boundaries, seal
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts

HOUR = 3600


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def conn(tmp_path):
    return db.get_conn(tmp_path / "cusum.db")


def make_frame(specs) -> pd.DataFrame:
    """specs: list of (window_id, ticker, positive?, [z per hour])."""
    base = date_str_to_ts("2025-10-01")
    rows, offset = [], {}
    for wid, ticker, positive, scores in specs:
        i = offset.setdefault(ticker, 0)
        anchor = base + (i + len(scores) + 1) * HOUR
        offset[ticker] = i + len(scores) + 1
        for h, z in enumerate(scores):
            rows.append({
                "window_id": wid, "ticker": ticker,
                "ts_utc": anchor - (len(scores) - h) * HOUR,
                "t0_utc": anchor if positive else None,
                "is_scheduled": True if positive else None,
                "item_code": "8.01" if positive else None,
                "volume_z": z,
            })
    return pd.DataFrame(rows).astype({
        "window_id": "string", "ticker": "string", "ts_utc": "Int64",
        "t0_utc": "Int64", "is_scheduled": "boolean", "item_code": "string"})


# --------------------------------------------------------------------------
# The statistic
# --------------------------------------------------------------------------
def test_statistic_matches_the_textbook_recursion():
    got = cusum_statistic(np.array([1.0, 1.0, 1.0]), k=0.5, h=100.0)
    assert got == pytest.approx([0.5, 1.0, 1.5])


def test_statistic_never_goes_negative():
    """The max(0, ...) floor: a long calm stretch parks at zero rather than
    running up a credit a later spike could spend."""
    got = cusum_statistic(np.array([-5.0] * 10), k=0.5, h=100.0)
    assert (got == 0.0).all()


def test_slack_decays_noise():
    """What k exists for: evidence below the slack never accumulates."""
    got = cusum_statistic(np.array([0.2] * 20), k=0.5, h=100.0)
    assert (got == 0.0).all()


def test_accumulation_beats_a_single_spike():
    """CUSUM's whole hypothesis, made concrete: sustained moderate evidence
    should outscore one louder hour."""
    sustained = cusum_statistic(np.array([1.5, 1.5, 1.5]), k=0.5, h=100.0)
    spike = cusum_statistic(np.array([0.0, 0.0, 3.0]), k=0.5, h=100.0)
    assert sustained.max() > spike.max()


def test_reset_after_alarm_zeroes_the_statistic():
    """The alarm value itself is emitted, then the state clears."""
    got = cusum_statistic(np.array([3.0, 3.0, 3.0]), k=0.0, h=5.0)
    assert got == pytest.approx([3.0, 6.0, 3.0])


def test_reset_can_be_disabled():
    got = cusum_statistic(np.array([3.0, 3.0, 3.0]), k=0.0, h=5.0,
                          reset_after_alarm=False)
    assert got == pytest.approx([3.0, 6.0, 9.0])


def test_nan_carries_state_but_emits_no_score():
    """No evidence is not evidence of normality. Treating a missing bar as a
    zero observation would decay S by k and assert the stock was calm."""
    got = cusum_statistic(np.array([1.0, np.nan, 1.0]), k=0.5, h=100.0)
    assert got[0] == pytest.approx(0.5)
    assert np.isnan(got[1])
    assert got[2] == pytest.approx(1.0)          # resumed from 0.5, not 0.0


def test_no_lookahead():
    """Tamper with the future, demand the past does not move."""
    base = np.array([1.0, 1.2, 0.8, 1.1, 0.9])
    tampered = base.copy()
    tampered[3:] *= 50
    a = cusum_statistic(base, k=0.5, h=100.0)
    b = cusum_statistic(tampered, k=0.5, h=100.0)
    assert a[:3] == pytest.approx(b[:3])


# --------------------------------------------------------------------------
# The anti-bias property
# --------------------------------------------------------------------------
def test_state_is_continuous_per_ticker_not_per_window(cfg):
    """The decision this module exists to get right.

    A single-bar negative window preceded by three strong hours must inherit
    that accumulation. If the statistic restarted per window it would read
    only its own bar, positives would get 48 bars of runway against a
    negative's one, and CUSUM would score brilliantly for reasons that have
    nothing to do with the market.
    """
    f = make_frame([
        ("W_prior", "AAA", False, [2.0, 2.0, 2.0]),
        ("W_single", "AAA", False, [0.1]),
    ])
    out = CUSUM(cfg, drift=0.5, threshold=100.0).predict(f, threshold=1e9)
    single = out[out["window_id"] == "W_single"]["score"].iloc[0]

    # Per-window it would be max(0, 0.1 - 0.5) = 0. Continuous, it inherits.
    assert single > 0.5


def test_state_does_not_leak_across_tickers(cfg):
    """Continuity is per ticker. B's statistic must not see A's spike."""
    f = make_frame([
        ("A1", "AAA", False, [9.0, 9.0, 9.0]),
        ("B1", "BBB", False, [0.1]),
    ])
    out = CUSUM(cfg, drift=0.5, threshold=1e9).predict(f, threshold=1e9)
    assert out[out["window_id"] == "B1"]["score"].iloc[0] == 0.0


def test_row_order_does_not_change_the_result(cfg):
    """The statistic is order-dependent, so the module sorts internally rather
    than trusting whatever order the caller assembled."""
    f = make_frame([("W1", "AAA", True, [1.0, 2.0, 3.0, 1.0])])
    a = CUSUM(cfg).predict(f, threshold=1e9).set_index(["window_id", "ts_utc"])
    b = (CUSUM(cfg).predict(f.iloc[::-1].reset_index(drop=True), threshold=1e9)
         .set_index(["window_id", "ts_utc"]))
    pd.testing.assert_series_equal(a["score"], b.loc[a.index, "score"])


# --------------------------------------------------------------------------
# The honest contrast with P5-03
# --------------------------------------------------------------------------
def test_the_threshold_moves_the_score_here(cfg):
    """Not an inconsistency with P5-03 but a consequence of the reset: crossing
    h zeroes S, so h is part of the estimator, not merely an alarm rule."""
    f = make_frame([("W1", "AAA", True, [3.0, 3.0, 3.0, 3.0])])
    low = CUSUM(cfg, drift=0.0, threshold=3.0).predict(f, threshold=1e9)
    high = CUSUM(cfg, drift=0.0, threshold=1e9).predict(f, threshold=1e9)
    assert not np.allclose(low["score"], high["score"])


def test_the_threshold_cannot_move_volume_zscore(cfg):
    """The other half of the contrast, so the difference is documented as a
    property of the two detectors rather than an accident of this test file."""
    f = make_frame([("W1", "AAA", True, [1.0, 3.0]),
                    ("N1", "BBB", False, [0.1, 0.2])])
    a = precision_at_alert_budget(
        VolumeZScore(cfg).predict(f, threshold=0.5), max_alerts=1)
    b = precision_at_alert_budget(
        VolumeZScore(cfg).predict(f, threshold=99.0), max_alerts=1)
    assert a.precision == b.precision


# --------------------------------------------------------------------------
# Tuning
# --------------------------------------------------------------------------
@pytest.fixture
def tuning_frame():
    """Positives accumulate; negatives stay under the slack."""
    specs = []
    for i in range(4):
        specs.append((f"P{i}", f"T{i}", True, [1.4, 1.5, 1.6, 1.7]))
    for i in range(12):
        specs.append((f"N{i}", f"Q{i}", False, [0.1, 0.2, 0.1, 0.2]))
    return make_frame(specs)


def test_tune_sweeps_the_full_grid_and_returns_a_grid_point(cfg, tuning_frame):
    op = tune_cusum(cfg, tuning_frame, drift_grid=[0.25, 0.5],
                    threshold_grid=[3.0, 5.0])
    assert op.grid_size == 4
    assert op.drift in (0.25, 0.5)
    assert op.threshold in (3.0, 5.0)


def test_tune_is_deterministic(cfg, tuning_frame):
    a = tune_cusum(cfg, tuning_frame, drift_grid=[0.25, 0.5],
                   threshold_grid=[3.0, 5.0])
    b = tune_cusum(cfg, tuning_frame, drift_grid=[0.25, 0.5],
                   threshold_grid=[3.0, 5.0])
    assert a == b


def test_tune_reports_the_floor_alongside(cfg, tuning_frame):
    op = tune_cusum(cfg, tuning_frame, drift_grid=[0.5], threshold_grid=[5.0])
    floor = precision_at_alert_budget(AlwaysQuiet(cfg).predict(tuning_frame))
    assert op.floor_precision == pytest.approx(floor.precision)


def test_tune_uses_the_configured_grids_by_default(cfg, tuning_frame):
    """The defaults come from config, not from grids hardcoded in the sweep.

    `grid_size` counts the pairs actually EVALUATED, which is the legal half:
    pairs with h < k are skipped because a boundary below the slack caps the
    statistic and stops it accumulating. Reporting the cartesian product would
    overstate the search by about a third.
    """
    op = tune_cusum(cfg, tuning_frame)
    c = cfg["baselines"]["cusum"]
    legal = sum(1 for k in c["drift_grid"] for h in c["threshold_grid"] if h >= k)
    assert op.grid_size == legal
    assert legal < len(c["drift_grid"]) * len(c["threshold_grid"])


def test_tune_never_returns_a_boundary_below_its_slack(cfg, tuning_frame):
    """The property the whole baseline rests on.

    With `reset_after_alarm`, h is not only the alarm rule — it caps how much
    evidence S may carry. At h < k the recursion saw-tooths and a single loud
    bar outranks sustained elevation, which is the inverse of the hypothesis
    CUSUM is here to test. The sweep once chose exactly that (k=1.5, h=1.0).
    """
    op = tune_cusum(cfg, tuning_frame)
    assert op.threshold >= op.drift


def test_the_configured_boundary_is_not_below_its_own_slack(cfg):
    """The defect that shipped: `drift: 1.5` with `threshold: 1.0`.

    With `reset_after_alarm`, h is not only the alarm rule — the statistic
    resets on crossing it, so h also CAPS how much evidence S can hold. Below
    the slack, S can never exceed h at all and the recursion saw-tooths.

    Stated honestly, because a stronger claim would be wrong: h >= k does NOT
    make sustained elevation outrank every single spike. The reset caps a
    sustained run at ~h while a lone bar of x >> k alarms immediately with the
    uncapped value x - k, so a large enough spike always wins on peak score.
    That is a property of reset-based CUSUM, not a bug, and it is worth knowing
    when reading the scores. What h >= k buys is that the statistic can
    accumulate across bars at all before it alarms — which at h < k it cannot.
    """
    c = cfg["baselines"]["cusum"]
    assert c["threshold"] >= c["drift"], (
        f"config ships drift={c['drift']}, threshold={c['threshold']}: a "
        f"boundary below the slack caps S below one bar's own increment, so "
        f"the statistic cannot accumulate and this stops being CUSUM")


def test_a_boundary_below_the_slack_cannot_accumulate(cfg):
    """Why the rule above exists, demonstrated rather than asserted."""
    import numpy as np

    reset = cfg["baselines"]["cusum"]["reset_after_alarm"]
    bad = cusum_statistic(np.full(6, 2.0), k=1.5, h=1.0, reset_after_alarm=reset)
    good = cusum_statistic(np.full(6, 2.0), k=1.5, h=3.0, reset_after_alarm=reset)
    assert bad.max() <= 1.0, "S is capped below what one bar contributes"
    assert good.max() > bad.max(), (
        "a legal boundary lets six consecutive 2-sigma bars build past what an "
        "illegal one allows")


def test_tune_never_touches_the_sealed_test_split(cfg, conn):
    seal(cfg, conn)
    _, val_end = boundaries(cfg)
    f = make_frame([("P0", "AAA", True, [1.0, 3.0])])
    f["ts_utc"] = pd.array([val_end + i * HOUR for i in range(len(f))],
                           dtype="Int64")
    f["t0_utc"] = pd.array([val_end + 99 * HOUR] * len(f), dtype="Int64")

    with pytest.raises(SystemExit, match="SEALED TEST SET"):
        tune_cusum(cfg, f, conn=conn, drift_grid=[0.5], threshold_grid=[5.0])


# --------------------------------------------------------------------------
# Inherited behaviour
# --------------------------------------------------------------------------
def test_output_passes_the_contract(cfg, tuning_frame):
    out = CUSUM(cfg).predict(tuning_frame, threshold=2.0)
    pd.testing.assert_frame_equal(out, contract.validate_predictions(out))


def test_a_frame_without_the_feature_is_refused(cfg, tuning_frame):
    with pytest.raises(ValueError, match="volume_z"):
        CUSUM(cfg).predict(tuning_frame.drop(columns=["volume_z"]),
                           threshold=2.0)


def test_an_all_nan_window_is_ranked_last_not_dropped(cfg, tuning_frame):
    dead = make_frame([("DEAD", "ZZZ", False, [np.nan] * 4)])
    mixed = pd.concat([tuning_frame, dead], ignore_index=True)
    out = CUSUM(cfg).predict(mixed, threshold=2.0)
    assert out["window_id"].nunique() == mixed["window_id"].nunique()
    assert (out[out["window_id"] == "DEAD"]["action"] == contract.WAIT).all()


def test_name_is_stable_for_the_report(cfg):
    assert CUSUM(cfg).name == "cusum"
