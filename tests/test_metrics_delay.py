"""Detection delay — how much advance warning a correct flag actually bought.

The test that carries the task is `test_overnight_flag_gives_zero_trading_hours`:
a flag one wall-clock hour before t0, raised overnight, is 0.0 trading hours of
warning. No trading happened in that hour, so no warning was available. Reporting
1.0 there would be the exact error the whole trading-hours rule exists to prevent.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.eval.contract import FLAG, WAIT, actions_from_scores, conform
from src.eval.metrics import (
    detection_delay_summary,
    detection_delays,
    precision_at_alert_budget,
)
from src.eval.synthetic import make_synthetic_predictions
from src.utils.timeutils import get_market_calendar, trading_hours_between

HOUR = 3600


def ts(iso: str) -> int:
    return int(datetime.strptime(iso, "%Y-%m-%d %H:%M")
               .replace(tzinfo=timezone.utc).timestamp())


def window(window_id: str, hours: list[int], t0: int | None,
           flag_at: int | None = None, ticker: str = "AAA",
           is_scheduled=None, item_code=None) -> pd.DataFrame:
    """One decision window, hand-built so the timestamps are exact.

    `is_scheduled`/`item_code` must be null exactly on quiet windows (the
    contract enforces this — event metadata only exists when there is an
    event). So the default here follows `t0`, not a blanket `pd.NA`: a
    positive window without an explicit override still gets non-null
    placeholder metadata rather than failing validation.
    """
    default_meta = t0 is not None
    return pd.DataFrame({
        "window_id": window_id,
        "ticker": ticker,
        "ts_utc": hours,
        "t0_utc": pd.NA if t0 is None else t0,
        "score": 0.0,
        "action": [FLAG if h == flag_at else WAIT for h in hours],
        "is_scheduled": ((False if default_meta else pd.NA)
                         if is_scheduled is None else is_scheduled),
        "item_code": (("8.01" if default_meta else pd.NA)
                      if item_code is None else item_code),
    })


def frame(*windows: pd.DataFrame) -> pd.DataFrame:
    return conform(pd.concat(windows, ignore_index=True))


@pytest.fixture(scope="module")
def cal():
    return get_market_calendar()


# --- the Done-when -------------------------------------------------------


def test_overnight_flag_gives_zero_trading_hours(cal) -> None:
    """A wall-clock hour of 'warning' containing no trading is no warning.

    Flag at 02:00 UTC, t0 at 03:00 UTC — the middle of the night in New York.
    Wall clock says 1.0 hours. The market was shut, so the answer is 0.0.
    """
    df = frame(window("w", [ts("2024-11-22 02:00"), ts("2024-11-22 03:00")],
                      t0=ts("2024-11-22 03:00"), flag_at=ts("2024-11-22 02:00")))
    d = detection_delays(df)
    assert d["lead_trading_hours"].iloc[0] == 0.0
    assert d["lead_wall_hours"].iloc[0] == 1.0


def test_weekend_gap_matches_hand_computed_trading_hours(cal) -> None:
    """Independently hand-computed, not just re-derived from the same helper.

    This is exactly the module docstring's own example: flag at 20:00 UTC on
    an ordinary Friday, t0 at 14:00 UTC the following Monday (no holiday in
    between). Hand computation using known NYSE regular-session hours in
    November (EST, UTC-5, so the session runs 14:30-21:00 UTC, no early
    close that week):

        Friday   20:00 -> 21:00 UTC (close)   = 1.0 trading hour
        Monday   14:00 UTC is *before* the 14:30 UTC open, so 0 trading
                 hours have elapsed by t0
        Saturday, Sunday                       = market shut, 0 hours
        -----------------------------------------------------------
        total                                  = 1.0 trading hour

    Wall clock, by hand: Friday 20:00 -> Monday 20:00 is exactly 3 * 24 = 72
    hours; Monday 14:00 is 6 hours earlier, so 72 - 6 = 66 wall-clock hours.

    2024-11-22 is a Friday and 2024-11-25 the following Monday, an ordinary
    trading week (Thanksgiving that year is Thursday 2024-11-28, the week
    after) — confirmed via `cal.is_session` for both dates before relying on
    them here.
    """
    assert cal.is_session("2024-11-22") and cal.is_session("2024-11-25")

    flag_at = ts("2024-11-22 20:00")
    t0 = ts("2024-11-25 14:00")
    df = frame(window("w", [flag_at, t0], t0=t0, flag_at=flag_at))

    d = detection_delays(df)
    assert d["lead_wall_hours"].iloc[0] == pytest.approx(66.0)
    assert d["lead_trading_hours"].iloc[0] == pytest.approx(1.0)


def test_lead_time_matches_trading_hours_between(cal) -> None:
    """Row for row, the values come from P1-05 and nowhere else."""
    df = make_synthetic_predictions(n_positive=25, n_quiet=200, n_tickers=6,
                                    span_days=120, signal_strength=2.0, seed=11)
    flagged = actions_from_scores(df, threshold=1.5)
    d = detection_delays(flagged)
    assert len(d) > 0
    for _, row in d.iterrows():
        assert row["lead_trading_hours"] == trading_hours_between(
            int(row["flag_ts_utc"]), int(row["t0_utc"]), cal
        )


# --- what counts as a detection -----------------------------------------


def test_only_correctly_flagged_windows_appear() -> None:
    """A flag on a quiet window is a false alarm, not a fast detection.

    Including it as a zero would drag the median down and understate the
    warning that real detections gave.
    """
    df = frame(
        window("pos", [ts("2024-11-22 15:00"), ts("2024-11-22 17:00")],
               t0=ts("2024-11-22 17:00"), flag_at=ts("2024-11-22 15:00")),
        window("quiet", [ts("2024-11-21 15:00"), ts("2024-11-21 17:00")],
               t0=None, flag_at=ts("2024-11-21 15:00")),
    )
    d = detection_delays(df)
    assert list(d.index) == ["pos"]


def test_unflagged_positive_is_a_miss_not_a_zero() -> None:
    df = frame(
        window("hit", [ts("2024-11-22 15:00"), ts("2024-11-22 17:00")],
               t0=ts("2024-11-22 17:00"), flag_at=ts("2024-11-22 15:00")),
        window("miss", [ts("2024-11-21 15:00"), ts("2024-11-21 17:00")],
               t0=ts("2024-11-21 17:00")),
    )
    s = detection_delay_summary(df)
    assert s.n_positive == 2 and s.n_detections == 1 and s.n_missed == 1


def test_flag_at_t0_is_zero_not_missing() -> None:
    """Detected, with no warning at all. A real outcome, not an absence."""
    df = frame(window("w", [ts("2024-11-22 15:00")],
                      t0=ts("2024-11-22 15:00"), flag_at=ts("2024-11-22 15:00")))
    d = detection_delays(df)
    assert len(d) == 1 and d["lead_trading_hours"].iloc[0] == 0.0


# --- degenerate frames must survive -------------------------------------


def test_all_wait_frame_gives_empty_result() -> None:
    """The always-quiet baseline is required by the plan and must not error."""
    df = make_synthetic_predictions(n_positive=5, n_quiet=20, n_tickers=4, seed=2)
    assert (df["action"] == WAIT).all()
    assert detection_delays(df).empty


def test_median_is_nan_not_zero_when_nothing_detected() -> None:
    """nan means 'nothing to measure'. 0.0 would read as 'detected everything
    with no warning' — a different and much worse claim."""
    df = make_synthetic_predictions(n_positive=5, n_quiet=20, n_tickers=4, seed=2)
    s = detection_delay_summary(df)
    assert s.n_detections == 0
    assert math.isnan(s.median_trading_hours)
    assert s.n_missed == s.n_positive


def test_flags_only_on_quiet_windows_detect_nothing() -> None:
    df = frame(
        window("q1", [ts("2024-11-21 15:00")], t0=None, flag_at=ts("2024-11-21 15:00")),
        window("p1", [ts("2024-11-22 15:00")], t0=ts("2024-11-22 15:00")),
    )
    s = detection_delay_summary(df)
    assert s.n_detections == 0 and math.isnan(s.median_trading_hours)


# --- sanity and invariants ----------------------------------------------


def test_earlier_flags_give_longer_lead_times() -> None:
    t0 = ts("2024-11-22 20:00")
    hours = [ts("2024-11-22 15:00"), ts("2024-11-22 18:00"), t0]
    early = detection_delays(frame(window("w", hours, t0, flag_at=hours[0])))
    late = detection_delays(frame(window("w", hours, t0, flag_at=hours[1])))
    assert early["lead_trading_hours"].iloc[0] > late["lead_trading_hours"].iloc[0]


def test_wall_clock_never_below_trading_hours() -> None:
    """Invariant: the market cannot have been open longer than time passed."""
    df = make_synthetic_predictions(n_positive=30, n_quiet=300, n_tickers=6,
                                    span_days=120, signal_strength=2.0, seed=9)
    d = detection_delays(actions_from_scores(df, threshold=1.0))
    assert (d["lead_wall_hours"] >= d["lead_trading_hours"]).all()


def test_metadata_is_carried_for_splitting() -> None:
    """P1-11 splits by scheduled/unscheduled and item code; it should not have
    to go back to the prediction frame to do it."""
    df = frame(window("w", [ts("2024-11-22 15:00"), ts("2024-11-22 17:00")],
                      t0=ts("2024-11-22 17:00"), flag_at=ts("2024-11-22 15:00"),
                      is_scheduled=True, item_code="2.02"))
    d = detection_delays(df)
    assert d["is_scheduled"].iloc[0] is True or bool(d["is_scheduled"].iloc[0]) is True
    assert d["item_code"].iloc[0] == "2.02"


def test_summary_quartiles_are_ordered() -> None:
    df = make_synthetic_predictions(n_positive=40, n_quiet=1500, n_tickers=8,
                                    span_days=180, signal_strength=2.0, seed=5)
    s = detection_delay_summary(actions_from_scores(df, threshold=1.0))
    assert s.min_trading_hours <= s.p25_trading_hours <= s.median_trading_hours
    assert s.median_trading_hours <= s.p75_trading_hours <= s.max_trading_hours


def test_pipeline_from_budget_threshold_to_delay() -> None:
    """The real sequence: budget picks a threshold, threshold picks flags,
    flags give lead times.

    These two now count DIFFERENT things, on purpose, and the inequality below
    is the honest relationship rather than a weakened equality.

    `true_positives` comes from the budget, which scores a positive at its
    DECISION POINT — the bar before t0 — so that an episode and a quiet bar
    each get one draw and cost one alert. Without that symmetry a 48-bar
    episode had 48 chances to cross against a quiet bar's one, and pure noise
    scored 29.6x lift.

    `n_detections` comes from the episode: it counts a positive as detected if
    the detector flagged at ANY hour of its window, which is what lead time has
    to mean — a detector that fires 20 hours early and goes quiet has still
    given 20 hours of warning.

    So episode detections are a superset of decision-point detections, and the
    two are reported as separate quantities rather than one number pretending
    to be both. On this fixture it is 36 against 7.
    """
    df = make_synthetic_predictions(n_positive=40, n_quiet=1500, n_tickers=8,
                                    span_days=180, signal_strength=2.0, seed=5)
    r = precision_at_alert_budget(df)
    s = detection_delay_summary(actions_from_scores(df, r.threshold))
    assert s.n_detections >= r.true_positives, (
        "an episode flagged at its decision point is flagged at some hour of "
        "it, so episode detections cannot be the smaller count")
    assert s.median_trading_hours <= s.median_wall_clock_hours


def test_invalid_frame_is_rejected() -> None:
    df = make_synthetic_predictions(n_positive=3, n_quiet=5, n_tickers=2, seed=1)
    bad = df.copy()
    bad.loc[bad.index[0], "action"] = "HOLD"
    with pytest.raises(ValueError, match="unknown action"):
        detection_delays(bad)
