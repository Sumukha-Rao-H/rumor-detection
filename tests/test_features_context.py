"""P4-10 — context signals, and the leak the backlog warned about.

The Done-when is that `days_since_last_8k` counts only filings already accepted
at time `t`. Two tests carry it: the boundary (a filing at exactly `t` counts,
one a second later does not), and the demonstration that the event's OWN filing
drops the feature to zero at t0 — which is why the decision window has to be
strictly before t0.

As in P4-07, the leakage detector cannot see either failure, because reading a
filing AT `t` is not reading the future. One test pins that blindness.
"""

import numpy as np
import pandas as pd
import pytest

from src.pipeline.features import context_signals
from src.utils.config import load_config
from src.utils.timeutils import iso_utc_to_ts
from tests.test_leakage import find_lookahead, assert_no_lookahead


HOUR = 3600
DAY = 86400
# A Wednesday inside a normal trading week, mid-session (15:00 UTC = 11:00 ET).
MID = iso_utc_to_ts("2026-03-04T15:00:00Z")


@pytest.fixture
def cfg():
    return load_config()


def bars(stamps):
    idx = pd.Index(list(stamps), name="ts_utc")
    return pd.DataFrame({"close": [100.0] * len(idx),
                         "volume": [1e6] * len(idx)}, index=idx)


# --------------------------------------------------------------------------
# the Done-when
# --------------------------------------------------------------------------

def test_only_filings_accepted_by_t_are_counted(cfg):
    """THE acceptance criterion."""
    filings = np.array([MID - 10 * DAY, MID + 5 * DAY])   # one past, one future
    out = context_signals(bars([MID]), filings, cfg=cfg)
    assert out["days_since_last_8k"].iloc[0] == pytest.approx(10.0)


def test_a_filing_at_exactly_t_counts(cfg):
    """It is public at `t`, so the model may know it."""
    out = context_signals(bars([MID]), np.array([MID]), cfg=cfg)
    assert out["days_since_last_8k"].iloc[0] == pytest.approx(0.0)


def test_a_filing_one_second_later_does_not(cfg):
    """The exclusive half of the boundary."""
    out = context_signals(bars([MID]), np.array([MID + 1]), cfg=cfg)
    assert np.isnan(out["days_since_last_8k"].iloc[0])


def test_days_since_is_nan_before_any_filing(cfg):
    """Absence is not zero, and not a huge number either."""
    out = context_signals(bars([MID]), np.array([], dtype=np.int64), cfg=cfg)
    assert np.isnan(out["days_since_last_8k"].iloc[0])


def test_the_most_recent_prior_filing_wins(cfg):
    filings = np.array([MID - 30 * DAY, MID - 3 * DAY, MID - 90 * DAY])
    filings.sort()
    out = context_signals(bars([MID]), filings, cfg=cfg)
    assert out["days_since_last_8k"].iloc[0] == pytest.approx(3.0)


# --------------------------------------------------------------------------
# the trap beyond the obvious one
# --------------------------------------------------------------------------

def test_the_event_itself_makes_the_feature_zero_at_t0(cfg):
    """The constraint P4-11 must enforce, demonstrated rather than described.

    The event being predicted IS an 8-K. At t0 its own filing is accepted, so
    `days_since_last_8k` is 0 — the feature does not merely leak, it announces
    the event. The decision window must be STRICTLY before t0.
    """
    t0 = MID
    window = [t0 - 3 * HOUR, t0 - 2 * HOUR, t0 - HOUR, t0]
    filings = np.array([t0 - 40 * DAY, t0])
    out = context_signals(bars(window), filings, cfg=cfg)["days_since_last_8k"]

    assert out.iloc[:3].min() > 39          # before t0: stale, uninformative
    assert out.iloc[3] == pytest.approx(0.0)  # AT t0: the answer, handed over


def test_the_leakage_detector_cannot_see_that(cfg):
    """Why the test above must exist.

    The detector perturbs rows after `t`; this reads a filing AT `t`. Same
    blind spot P4-07 found for the volume z-score.
    """
    stamps = [MID + i * HOUR for i in range(40)]
    filings = np.array(stamps)              # a filing on every single bar
    assert find_lookahead(
        lambda f: context_signals(f, filings, cfg=cfg), bars(stamps)) == []


# --------------------------------------------------------------------------
# earnings
# --------------------------------------------------------------------------

def test_earnings_uses_the_last_not_the_next(cfg):
    """"Days until the next earnings" would mean reading a future filing."""
    earnings = np.array([MID - 45 * DAY, MID + 45 * DAY])
    out = context_signals(bars([MID]), np.array([], dtype=np.int64),
                          earnings_times=earnings, cfg=cfg)
    assert out["days_since_last_earnings"].iloc[0] == pytest.approx(45.0)


def test_no_earnings_filing_gives_nan(cfg):
    """8 universe members file nothing at all."""
    out = context_signals(bars([MID]), np.array([], dtype=np.int64), cfg=cfg)
    assert np.isnan(out["days_since_last_earnings"].iloc[0])


# --------------------------------------------------------------------------
# the two units
# --------------------------------------------------------------------------

def test_days_since_is_calendar_days_not_trading_days(cfg):
    """News ages over a weekend, even though no trading happens.

    Friday 15:00 UTC to the following Monday 15:00 UTC is 3 calendar days and
    only about 1 trading day.
    """
    friday = iso_utc_to_ts("2026-03-06T15:00:00Z")
    monday = iso_utc_to_ts("2026-03-09T15:00:00Z")
    out = context_signals(bars([monday]), np.array([friday]), cfg=cfg)
    assert out["days_since_last_8k"].iloc[0] == pytest.approx(3.0)


def test_hours_to_close_is_trading_hours(cfg):
    """15:00 UTC on 4 March 2026 is 10:00 ET, six hours before the 16:00 close.

    Note the DST detail, which caught this test on the first run: US clocks
    move on the second Sunday of March, so 4 March is still EST (UTC-5) and
    15:00 UTC is 10:00 ET, not 11:00. Issue 6 recorded the same trap in P1-06 —
    every test there used a winter date and a fixed-offset assumption would
    have passed while two-thirds of the study window came out wrong.
    """
    out = context_signals(bars([MID]), cfg=cfg)
    assert out["trading_hours_to_close"].iloc[0] == pytest.approx(6.0, abs=0.05)


def test_hours_to_close_is_correct_on_the_other_side_of_dst(cfg):
    """The same clock time in summer, when ET is UTC-4 rather than UTC-5.

    15:00 UTC is 11:00 EDT, so five hours remain. If the calendar were being
    treated as a fixed offset, one of these two tests would fail.
    """
    summer = iso_utc_to_ts("2026-06-10T15:00:00Z")
    out = context_signals(bars([summer]), cfg=cfg)
    assert out["trading_hours_to_close"].iloc[0] == pytest.approx(5.0, abs=0.05)


def test_hours_to_close_before_the_open_is_a_full_session(cfg):
    """A pre-open bar has the whole next session ahead of it."""
    pre_open = iso_utc_to_ts("2026-03-04T10:00:00Z")   # 05:00 ET
    out = context_signals(bars([pre_open]), cfg=cfg)
    assert out["trading_hours_to_close"].iloc[0] == pytest.approx(6.5, abs=0.05)


# --------------------------------------------------------------------------
# contracts
# --------------------------------------------------------------------------

def test_disabled_flags_omit_their_columns(cfg):
    off = {**cfg, "features": {**cfg["features"],
                               "include_days_since_last_8k": False,
                               "include_earnings_proximity": False}}
    out = context_signals(bars([MID]), np.array([MID - DAY]), cfg=off)
    assert list(out.columns) == ["trading_hours_to_close"]


def test_context_signals_pass_the_leakage_detector(cfg):
    stamps = [MID + i * HOUR for i in range(60)]
    filings = np.array([MID - 5 * DAY])
    assert_no_lookahead(lambda f: context_signals(f, filings, cfg=cfg),
                        bars(stamps))
