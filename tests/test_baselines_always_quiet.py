"""P5-02 — the do-nothing floor, and the trap it exists to expose.

The interesting assertions here are not that a constant is constant. They are
that always-quiet produces **base-rate precision** through the ordinary
evaluation path, that it cannot be talked into flagging by a badly chosen
threshold, and that the configured constant is genuinely inert — if changing
it moved a number, the floor would be a tunable knob rather than a fact about
the data.
"""

import numpy as np
import pandas as pd
import pytest

from src import db
from src.baselines import AlwaysQuiet
from src.eval import contract
from src.eval.metrics import accuracy, precision_at_alert_budget
from src.pipeline.split import boundaries, seal
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts

HOUR = 3600


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def conn(tmp_path):
    return db.get_conn(tmp_path / "always_quiet.db")


def make_frame(n_windows: int = 10, n_positive: int = 3,
               hours: int = 4, feature=0.0) -> pd.DataFrame:
    """`n_positive` positive windows, the rest quiet, hours strictly before t0."""
    base = date_str_to_ts("2025-10-01")
    rows = []
    for w in range(n_windows):
        anchor = base + w * 100 * HOUR
        positive = w < n_positive
        for h in range(hours):
            rows.append({
                "window_id": f"W{w}",
                "ticker": f"T{w}",
                "ts_utc": anchor - (hours - h) * HOUR,
                "t0_utc": anchor if positive else None,
                "is_scheduled": True if positive else None,
                "item_code": "8.01" if positive else None,
                "volume_z": feature,
            })
    return pd.DataFrame(rows).astype({
        "window_id": "string", "ticker": "string", "ts_utc": "Int64",
        "t0_utc": "Int64", "is_scheduled": "boolean", "item_code": "string"})


# --------------------------------------------------------------------------
# It never flags
# --------------------------------------------------------------------------
def test_never_flags(cfg):
    out = AlwaysQuiet(cfg).predict(make_frame())
    assert (out["action"] == contract.WAIT).all()
    assert (out["action"] == contract.FLAG).sum() == 0


def test_never_flags_even_when_handed_a_threshold_below_its_score(cfg):
    """The guarantee has to be structural. A tuning sweep that does not know
    which baseline it is driving must not turn the floor into an
    alarm-on-everything detector."""
    out = AlwaysQuiet(cfg).predict(make_frame(), threshold=-1e9)
    assert (out["action"] == contract.FLAG).sum() == 0


def test_never_flags_at_its_own_score_exactly(cfg):
    """The boundary case: threshold == the constant. `actions_from_scores`
    flags on `>=`, so this is the one value that would slip through if the
    override were not there."""
    constant = AlwaysQuiet(cfg).constant
    out = AlwaysQuiet(cfg).predict(make_frame(), threshold=constant)
    assert (out["action"] == contract.FLAG).sum() == 0


# --------------------------------------------------------------------------
# The deliverable: base-rate precision
# --------------------------------------------------------------------------
def test_precision_at_budget_equals_the_base_rate(cfg):
    """What "no information" looks like on the headline metric, and the line
    every other baseline has to clear."""
    out = AlwaysQuiet(cfg).predict(make_frame(n_windows=10, n_positive=3))
    r = precision_at_alert_budget(out, max_alerts=2)

    assert r.base_rate == pytest.approx(0.3)
    assert r.precision == pytest.approx(r.base_rate)


def test_it_flags_nothing_yet_still_reports_base_rate_precision(cfg):
    """Both faces at once, and both true: zero alerts as a stopping policy,
    base-rate precision as a ranker. The report carries both."""
    out = AlwaysQuiet(cfg).predict(make_frame(n_windows=10, n_positive=3))
    r = precision_at_alert_budget(out, max_alerts=2)

    assert (out["action"] == contract.FLAG).sum() == 0
    assert r.true_positives == 3
    assert r.precision == pytest.approx(r.base_rate)


def test_the_overspend_is_visible_not_hidden(cfg):
    """A detector that cannot rank cannot spend a budget sensibly. Ties are
    admitted together, so realised exceeds budget — reported, not trimmed."""
    out = AlwaysQuiet(cfg).predict(make_frame(n_windows=10, n_positive=3))
    r = precision_at_alert_budget(out, max_alerts=2)

    assert r.realised_alerts > r.budget_alerts
    assert r.realised_alerts == r.n_windows


# --------------------------------------------------------------------------
# The constant is inert
# --------------------------------------------------------------------------
def test_score_is_the_configured_constant(cfg):
    out = AlwaysQuiet(cfg).predict(make_frame())
    assert (out["score"] == cfg["baselines"]["always_quiet"]["score"]).all()


def test_changing_the_configured_score_changes_no_reported_number(cfg):
    """If the floor moved with this knob it would be a tunable, not a fact
    about the data. Two very different constants, identical numbers."""
    frame = make_frame(n_windows=10, n_positive=3)
    other = {**cfg, "baselines": {**cfg["baselines"],
                                  "always_quiet": {"score": 7.5}}}

    a = precision_at_alert_budget(AlwaysQuiet(cfg).predict(frame), max_alerts=2)
    b = precision_at_alert_budget(AlwaysQuiet(other).predict(frame), max_alerts=2)

    assert a.precision == pytest.approx(b.precision)
    assert a.base_rate == pytest.approx(b.base_rate)
    assert a.recall == pytest.approx(b.recall)
    assert a.true_positives == b.true_positives


# --------------------------------------------------------------------------
# Inherited behaviour still holds
# --------------------------------------------------------------------------
def test_output_passes_the_contract_unchanged(cfg):
    out = AlwaysQuiet(cfg).predict(make_frame())
    pd.testing.assert_frame_equal(out, contract.validate_predictions(out))


def test_nothing_is_ever_unscoreable(cfg):
    """The one baseline that reads no feature. Even under the strict policy,
    and even on an all-NaN frame, it has an opinion about every window."""
    strict = {**cfg, "baselines": {**cfg["baselines"],
                                   "always_quiet": {"score": 0.0},
                                   "common": {"unscoreable": "error"}}}
    frame = make_frame(feature=np.nan)
    out = AlwaysQuiet(strict).predict(frame)
    assert len(out) == len(frame)
    assert np.isfinite(out["score"]).all()


def test_the_seal_is_enforced_through_the_inherited_path(cfg, conn):
    seal(cfg, conn)
    _, val_end = boundaries(cfg)
    frame = make_frame(n_windows=1, n_positive=1)
    frame["ts_utc"] = pd.array([val_end + i * HOUR for i in range(len(frame))],
                               dtype="Int64")
    frame["t0_utc"] = pd.array([val_end + 99 * HOUR] * len(frame), dtype="Int64")

    with pytest.raises(SystemExit, match="SEALED TEST SET"):
        AlwaysQuiet(cfg).predict(frame, conn=conn)


def test_empty_frame_raises(cfg):
    with pytest.raises(ValueError, match="empty"):
        AlwaysQuiet(cfg).predict(make_frame().iloc[:0])


def test_name_is_stable_for_the_report(cfg):
    """P5-06 keys its table by this; it must not drift to the class name."""
    assert AlwaysQuiet(cfg).name == "always_quiet"


# --------------------------------------------------------------------------
# The trap is stated, never computed
# --------------------------------------------------------------------------
def test_accuracy_is_still_refused(cfg):
    """This module is *about* the 99.7% figure and still does not compute it.
    A number that rewards the degenerate answer does not get calculated just
    because the degenerate answer is the one under test."""
    out = AlwaysQuiet(cfg).predict(make_frame())
    with pytest.raises(NotImplementedError, match="accuracy is banned"):
        accuracy(out)
