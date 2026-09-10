"""P5-03 — the baseline most likely to win, and the procedure that tuned it.

The assertion this file exists for is the one in `test_precision_is_independent
_of_the_threshold`: the headline metric derives its own cut by ranking, so
sweeping threshold against precision would measure nothing. The whole tuning
design rests on that fact, so it is pinned rather than assumed.
"""

import numpy as np
import pandas as pd
import pytest

from src import db
from src.baselines import AlwaysQuiet, VolumeZScore, tune
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
    return db.get_conn(tmp_path / "vz.db")


def make_frame(specs) -> pd.DataFrame:
    """specs: list of (window_id, positive?, [volume_z per hour])."""
    base = date_str_to_ts("2025-10-01")
    rows = []
    for i, (wid, positive, scores) in enumerate(specs):
        anchor = base + i * 200 * HOUR
        n = len(scores)
        for h, z in enumerate(scores):
            rows.append({
                "window_id": wid, "ticker": f"T{i}",
                "ts_utc": anchor - (n - h) * HOUR,
                "t0_utc": anchor if positive else None,
                "is_scheduled": True if positive else None,
                "item_code": "8.01" if positive else None,
                "volume_z": z,
            })
    return pd.DataFrame(rows).astype({
        "window_id": "string", "ticker": "string", "ts_utc": "Int64",
        "t0_utc": "Int64", "is_scheduled": "boolean", "item_code": "string"})


@pytest.fixture
def frame():
    """Positives spike, negatives do not — the signal the baseline exists for."""
    specs = []
    for i in range(4):
        specs.append((f"P{i}", True, [0.2, 0.5, 3.5, 4.0]))
    for i in range(12):
        specs.append((f"N{i}", False, [0.1, 0.3, 0.2, 0.4]))
    return make_frame(specs)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def test_scores_are_the_volume_zscore_column_untouched(cfg, frame):
    out = VolumeZScore(cfg).predict(frame, threshold=2.5)
    merged = out.merge(frame[["window_id", "ts_utc", "volume_z"]],
                       on=["window_id", "ts_utc"])
    assert (merged["score"] == merged["volume_z"]).all()


def test_a_frame_without_the_feature_is_refused(cfg, frame):
    with pytest.raises(ValueError, match="volume_z"):
        VolumeZScore(cfg).predict(frame.drop(columns=["volume_z"]),
                                  threshold=2.5)


def test_flags_the_first_hour_above_the_threshold(cfg, frame):
    out = VolumeZScore(cfg, min_wait_hours=0).predict(frame, threshold=2.5)
    flags = out[out["action"] == contract.FLAG]
    assert set(flags["window_id"]) == {"P0", "P1", "P2", "P3"}
    for _, row in flags.iterrows():
        window = out[out["window_id"] == row["window_id"]]
        crossed = window[window["score"] >= 2.5]
        assert row["ts_utc"] == crossed["ts_utc"].min()


# --------------------------------------------------------------------------
# The fact the tuning design rests on
# --------------------------------------------------------------------------
def test_precision_is_independent_of_the_threshold(cfg, frame):
    """`precision_at_alert_budget` ranks windows by peak score and buys the top
    `budget`. It never reads `action`, so the threshold cannot move it — which
    is why `tune` sweeps min_wait_hours and derives the threshold instead of
    searching it."""
    model = VolumeZScore(cfg, min_wait_hours=0)
    a = precision_at_alert_budget(model.predict(frame, threshold=0.5),
                                  max_alerts=4)
    b = precision_at_alert_budget(model.predict(frame, threshold=99.0),
                                  max_alerts=4)
    assert a.precision == b.precision
    assert a.n_windows == b.n_windows


def test_it_separates_positives_from_negatives_here(cfg, frame):
    """A sanity check on the fixture itself: if the constructed signal were not
    separable, every tuning assertion below would be measuring noise."""
    out = VolumeZScore(cfg).predict(frame, threshold=2.5)
    r = precision_at_alert_budget(out, max_alerts=4)
    assert r.precision == pytest.approx(1.0)


# --------------------------------------------------------------------------
# min_wait_hours
# --------------------------------------------------------------------------
def test_min_wait_hours_suppresses_an_early_flag(cfg):
    """A spike in the opening hour is refused when the wait forbids it."""
    f = make_frame([("P0", True, [9.0, 0.1, 0.1, 0.1])])
    early = VolumeZScore(cfg, min_wait_hours=0).predict(f, threshold=2.5)
    waited = VolumeZScore(cfg, min_wait_hours=4).predict(f, threshold=2.5)

    assert (early["action"] == contract.FLAG).sum() == 1
    assert (waited["action"] == contract.FLAG).sum() == 0


def test_a_wait_longer_than_the_window_collapses_to_the_floor(cfg, frame):
    """Waiting past the horizon is the same as never flagging — which is what
    always-quiet already measures, and a useful check rather than an error."""
    out = VolumeZScore(cfg, min_wait_hours=999).predict(frame, threshold=2.5)
    assert (out["action"] == contract.FLAG).sum() == 0


# --------------------------------------------------------------------------
# The tuning procedure
# --------------------------------------------------------------------------
def test_tune_returns_the_budget_implied_threshold(cfg, frame):
    """Step 3 of the documented procedure: the operating point spends exactly
    the allowance it is given, rather than a guessed cut."""
    op = tune(cfg, frame, grid=[0, 1])
    model = VolumeZScore(cfg, min_wait_hours=op.min_wait_hours)
    expected = precision_at_alert_budget(
        model.predict(frame, threshold=float("inf"))).threshold
    assert op.threshold == pytest.approx(expected)


def test_tune_reports_the_floor_alongside_the_result(cfg, frame):
    """A precision figure without the base rate beside it cannot be read."""
    op = tune(cfg, frame, grid=[0, 1])
    floor = precision_at_alert_budget(AlwaysQuiet(cfg).predict(frame))
    assert op.floor_precision == pytest.approx(floor.precision)
    assert op.base_rate == pytest.approx(floor.base_rate)


def test_tune_reports_a_loss_as_readily_as_a_win(cfg):
    """A frame where the feature is pure noise: `beats_floor` must come back
    False rather than the procedure finding something that is not there."""
    rng = np.random.default_rng(0)
    specs = [(f"P{i}", True, list(rng.normal(size=4))) for i in range(3)]
    specs += [(f"N{i}", False, list(rng.normal(size=4))) for i in range(9)]
    op = tune(cfg, make_frame(specs), grid=[0, 1])
    assert isinstance(op.beats_floor, bool)
    # Noise cannot separate, so it lands on the floor rather than above it.
    assert op.precision == pytest.approx(op.floor_precision)
    assert op.beats_floor is False


def test_tune_is_deterministic(cfg, frame):
    a = tune(cfg, frame, grid=[0, 1, 2])
    b = tune(cfg, frame, grid=[0, 1, 2])
    assert a == b


def test_tune_uses_the_configured_grid_by_default(cfg, frame):
    op = tune(cfg, frame)
    assert op.grid == tuple(
        float(g) for g in cfg["baselines"]["volume_zscore"]["min_wait_hours_grid"])
    assert op.min_wait_hours in op.grid


def test_tune_never_touches_the_sealed_test_split(cfg, conn):
    seal(cfg, conn)
    _, val_end = boundaries(cfg)
    f = make_frame([("P0", True, [0.1, 3.0])])
    f["ts_utc"] = pd.array([val_end + i * HOUR for i in range(len(f))],
                           dtype="Int64")
    f["t0_utc"] = pd.array([val_end + 99 * HOUR] * len(f), dtype="Int64")

    with pytest.raises(SystemExit, match="SEALED TEST SET"):
        tune(cfg, f, conn=conn, grid=[0])


def test_unscoreable_windows_survive_the_sweep(cfg, frame):
    """P5-01's decision must hold through tuning: the window count the
    comparison rests on cannot change because a detector had no opinion."""
    dead = make_frame([("DEAD", False, [np.nan] * 4)])
    mixed = pd.concat([frame, dead], ignore_index=True)
    op = tune(cfg, mixed, grid=[0, 1])
    assert op.n_windows == mixed["window_id"].nunique()


def test_output_passes_the_contract(cfg, frame):
    out = VolumeZScore(cfg).predict(frame, threshold=2.5)
    pd.testing.assert_frame_equal(out, contract.validate_predictions(out))


def test_name_is_stable_for_the_report(cfg):
    assert VolumeZScore(cfg).name == "volume_zscore"


# --------------------------------------------------------------------------
# A regime worth naming
# --------------------------------------------------------------------------
def test_the_ceiling_stays_coherent_when_the_budget_exceeds_the_windows(cfg, frame):
    """This test used to pin an incoherence. The incoherence is now fixed.

    `max_precision` was `min(n_positive, budget) / budget` — a ceiling measured
    on the ALLOWANCE, sitting beside a precision measured on the alerts
    actually ISSUED. When the budget is larger than the number of windows the
    allowance cannot physically be spent, so the "ceiling" came out BELOW the
    precision achieved, which is not something a ceiling can do. The earlier
    version of this test asserted `max_precision < precision` and said in its
    own docstring that it was documenting a real limit rather than claiming it
    was right.

    The denominator is now the realised alert count, the same one precision
    uses, so the two are comparable by construction. In this regime everything
    alerts and every positive is caught, so the detector reaches its ceiling
    exactly — `max_precision == precision` — which is the honest reading of a
    frame too small to spend its allowance.

    The regime itself is real and still worth naming: the budget is sized from
    ticker-months, so on the real validation slice it was 3,532 against 1,592
    windows. `budget_exceeds_windows` is the flag that says it was entered,
    and it is now carried into the report table rather than stopping at
    `BudgetResult`.
    """
    out = VolumeZScore(cfg).predict(frame, threshold=2.5)
    r = precision_at_alert_budget(out)          # real budget, small frame

    assert r.budget_exceeds_windows is True
    assert r.threshold == float("-inf")         # everything admitted
    assert r.max_precision >= r.precision       # a ceiling behaves like one
    assert r.max_precision == pytest.approx(r.precision), \
        "everything alerted and every positive was caught, so the ceiling is met"


def test_precision_at_the_budget_is_invariant_across_the_min_wait_grid(cfg, frame):
    """The property the tuning docstring used to describe wrongly.

    It read: min_wait_hours is "genuinely swept, because suppressing flags in a
    window's opening hours changes *which* windows alert and therefore does
    move precision." It cannot. `precision_at_alert_budget` ranks windows by
    `peak_score` and never reads `action`, and `min_wait_hours` only ever
    rewrites `action` — the same docstring says exactly that twelve lines
    earlier about the threshold. So precision is identical at every point of
    the grid by construction, and `tune`'s key `(precision, lead, -wait)`
    settles the winner entirely on the lead-time tie-break.

    The sweep is honest and stays: the knob really does move lead time and the
    action distribution, both of which are reported. Its stated reason was not,
    so the property is pinned here rather than explained wrongly there.
    """
    grid = list(cfg["baselines"]["volume_zscore"]["min_wait_hours_grid"])
    precisions, flags = {}, {}
    for wait in grid + [999]:
        out = VolumeZScore(cfg, min_wait_hours=wait).predict(frame,
                                                             threshold=2.5)
        precisions[wait] = precision_at_alert_budget(out).precision
        flags[wait] = int((out["action"] == contract.FLAG).sum())

    assert len(set(precisions.values())) == 1, precisions
    # And not vacuously: the knob really is rewriting `action` underneath. A
    # wait longer than the window suppresses every flag, and precision does
    # not move by so much as a decimal.
    assert flags[grid[0]] > 0
    assert flags[999] == 0
