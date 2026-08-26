"""Market-hours helpers — is_market_open, next_market_*, trading_hours_between.

The dates here are chosen deliberately, not at random:

  2024-11-27  Wednesday, an ordinary full session
  2024-11-28  Thanksgiving — a weekday the exchange is shut
  2024-11-29  the day after — an early close at 13:00 ET, not 16:00
  2024-11-30  Saturday
  2024-12-25  Christmas, which fell on a Wednesday in 2024

Between them they cover every way a timestamp can fail to be a trading moment.
The half-day is the important one: a naive "weekday minus holidays" check gets
it wrong and overstates any lead time crossing it by three hours.

P1-06 extends this file to the helpers built on top of this one.
"""

from __future__ import annotations

import pytest

from src.utils.timeutils import (
    date_str_to_ts,
    dt_to_ts,
    get_market_calendar,
    is_market_open,
    next_market_close,
    next_market_open,
    trading_hours_between,
)
from datetime import datetime, timezone


def ts(iso: str) -> int:
    """'2024-11-27 15:00' (UTC) -> epoch seconds."""
    return dt_to_ts(datetime.strptime(iso, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc))


@pytest.fixture(scope="module")
def cal():
    return get_market_calendar()


def test_open_during_normal_session(cal) -> None:
    """Wednesday mid-session. 15:00 UTC = 10:00 ET."""
    assert is_market_open(ts("2024-11-27 15:00"), cal) is True


def test_closed_overnight(cal) -> None:
    """03:00 UTC on a weekday is 22:00 ET the evening before."""
    assert is_market_open(ts("2024-11-27 03:00"), cal) is False


def test_closed_on_weekend(cal) -> None:
    assert is_market_open(ts("2024-11-30 15:00"), cal) is False


def test_closed_on_holiday(cal) -> None:
    """Christmas 2024 was a Wednesday — a weekday check alone would say open."""
    assert is_market_open(ts("2024-12-25 15:00"), cal) is False
    assert is_market_open(ts("2024-11-28 15:00"), cal) is False  # Thanksgiving


def test_half_day_closes_early(cal) -> None:
    """2024-11-29 closed at 13:00 ET = 18:00 UTC, not 16:00 ET = 21:00 UTC.

    17:59 UTC is inside the session; 18:00 UTC is not. A calendar unaware of
    early closes would report both as open.
    """
    assert is_market_open(ts("2024-11-29 17:59"), cal) is True
    assert is_market_open(ts("2024-11-29 18:00"), cal) is False
    assert is_market_open(ts("2024-11-29 19:00"), cal) is False


def test_session_boundaries_are_left_closed(cal) -> None:
    """[open, close) — the opening bell counts, the closing bell does not.

    Pinned by a test because P1-05 measures durations against this convention;
    if it ever flipped, an event landing exactly on the close would change
    which session it belongs to.
    """
    assert is_market_open(ts("2024-11-27 14:29"), cal) is False  # 09:29 ET
    assert is_market_open(ts("2024-11-27 14:30"), cal) is True   # 09:30 ET, open
    assert is_market_open(ts("2024-11-27 20:59"), cal) is True   # 15:59 ET
    assert is_market_open(ts("2024-11-27 21:00"), cal) is False  # 16:00 ET, closed


def test_out_of_range_raises_with_bounds_named(cal) -> None:
    """Never False for an unknown date.

    Silently answering "closed" would let the live monitor run past the end of
    the calendar and stop alerting without anyone noticing.
    """
    with pytest.raises(ValueError, match=r"outside the XNYS calendar"):
        is_market_open(ts("1990-01-03 15:00"), cal)
    with pytest.raises(ValueError, match=r"covers \d{4}-\d{2}-\d{2} to"):
        is_market_open(ts("2099-01-04 15:00"), cal)


def test_calendar_is_cached() -> None:
    assert get_market_calendar() is get_market_calendar()


def test_calendar_code_comes_from_config() -> None:
    """The default calendar is whatever config says, not a hardcoded string."""
    from src.utils.config import load_config

    assert get_market_calendar().name == load_config()["market"]["calendar"]


def test_study_window_start_is_a_trading_moment() -> None:
    """Sanity link to the real config: the window starts on a Sunday in 2024,
    so its UTC midnight is not a trading moment. Guards against anyone reading
    date_str_to_ts as if it produced a market time."""
    from src.utils.config import load_config

    start = date_str_to_ts(load_config()["study_window"]["start"])
    assert is_market_open(start) is False


# --------------------------------------------------------------------------
# P1-04 — next_market_close / next_market_open
#
# Thanksgiving week is the whole test in miniature: Wednesday is normal,
# Thursday is shut, Friday closes early. Anything that walks forward naively
# gets at least one of those wrong.
# --------------------------------------------------------------------------


def test_next_close_mid_session_is_the_current_session(cal) -> None:
    """From inside a session, the next close is that session's own.

    This is what the 'trading hours to close' feature needs — from 10:00 ET the
    relevant close is 16:00 ET the same day, not tomorrow's.
    """
    assert next_market_close(ts("2024-11-27 15:00"), cal) == ts("2024-11-27 21:00")


def test_next_close_after_close_skips_to_the_next_session(cal) -> None:
    """The task's Done-when, stated literally.

    16:30 ET Wednesday. The next close is NOT that day's (already passed) and
    NOT Thursday's (Thanksgiving) — it is Friday's.
    """
    assert next_market_close(ts("2024-11-27 21:30"), cal) == ts("2024-11-29 18:00")


def test_next_close_respects_the_half_day(cal) -> None:
    """Friday 2024-11-29 closes 13:00 ET = 18:00 UTC, not 16:00 ET = 21:00 UTC.

    A hand-rolled 'next weekday at 21:00 UTC' would be three hours late here,
    and every lead time measured across that Friday would inherit the error.
    """
    close = next_market_close(ts("2024-11-29 15:00"), cal)
    assert close == ts("2024-11-29 18:00")
    assert close != ts("2024-11-29 21:00")


def test_next_open_from_weekend_is_monday(cal) -> None:
    assert next_market_open(ts("2024-11-30 12:00"), cal) == ts("2024-12-02 14:30")


def test_next_open_from_holiday_is_the_next_session(cal) -> None:
    assert next_market_open(ts("2024-11-28 15:00"), cal) == ts("2024-11-29 14:30")


def test_next_open_lands_on_an_open_minute(cal) -> None:
    """Whatever comes back must itself be a trading moment.

    Ties the two halves of this module together: if next_market_open ever
    returned a holiday or an out-of-hours instant, is_market_open would say so.
    """
    for start in ["2024-11-28 15:00", "2024-11-30 12:00", "2024-11-27 21:30"]:
        assert is_market_open(next_market_open(ts(start), cal), cal) is True


def test_returns_epoch_seconds_as_int(cal) -> None:
    """Not a pandas Timestamp — everything in this project is an epoch int."""
    assert isinstance(next_market_close(ts("2024-11-27 15:00"), cal), int)
    assert isinstance(next_market_open(ts("2024-11-27 15:00"), cal), int)


def test_next_helpers_raise_out_of_range(cal) -> None:
    """Same rule as is_market_open: never invent a plausible answer."""
    with pytest.raises(ValueError, match=r"outside the XNYS calendar"):
        next_market_close(ts("2099-01-04 15:00"), cal)
    with pytest.raises(ValueError, match=r"outside the XNYS calendar"):
        next_market_open(ts("1990-01-03 15:00"), cal)


# --------------------------------------------------------------------------
# P1-05 — trading_hours_between
#
# The function every lead-time number in the report is computed with. Wall
# clock is not a substitute, and the weekend test below is the proof: the same
# two moments are 66 hours apart on a clock and 1.0 hours apart in the market.
# --------------------------------------------------------------------------


def test_within_one_session(cal) -> None:
    assert trading_hours_between(ts("2024-11-22 14:30"), ts("2024-11-22 15:30"), cal) == 1.0


def test_full_session_is_six_and_a_half_hours(cal) -> None:
    """09:30-16:00 ET. The closing bell is excluded, so this is 6.5 exactly."""
    assert trading_hours_between(ts("2024-11-22 14:30"), ts("2024-11-22 21:00"), cal) == 6.5


def test_half_day_session_is_three_and_a_half(cal) -> None:
    """2024-11-29 closes at 13:00 ET. Counting it as a normal session would
    inflate any lead time crossing that Friday by three hours."""
    assert trading_hours_between(ts("2024-11-29 14:30"), ts("2024-11-29 18:00"), cal) == 3.5


def test_overnight_weekend_gap_is_the_whole_point(cal) -> None:
    """The task's Done-when, and the reason this function exists.

    Friday 20:00 UTC (15:00 ET, market open) to Monday 14:00 UTC (09:00 ET,
    market not yet open). Sixty-six hours pass on a clock. Only the last hour
    of Friday's session was tradeable.
    """
    a, b = ts("2024-11-22 20:00"), ts("2024-11-25 14:00")
    wall_clock = (b - a) / 3600
    assert wall_clock == 66.0
    assert trading_hours_between(a, b, cal) == 1.0


def test_skips_a_weekday_holiday(cal) -> None:
    """Wednesday close to Friday open spans Thanksgiving, which contributes
    nothing. Only Friday's half session before 15:30 ET counts."""
    hours = trading_hours_between(ts("2024-11-27 21:00"), ts("2024-11-29 15:30"), cal)
    assert hours == 1.0  # 14:30-15:30 UTC on the Friday


def test_span_entirely_outside_market_hours_is_zero(cal) -> None:
    assert trading_hours_between(ts("2024-11-27 02:00"), ts("2024-11-27 03:00"), cal) == 0.0
    assert trading_hours_between(ts("2024-11-30 10:00"), ts("2024-12-01 10:00"), cal) == 0.0


def test_end_is_exclusive(cal) -> None:
    """[start, end) — one minute apart is one minute, not two.

    Matches the [open, close) convention pinned in P1-03. Without this,
    consecutive spans would each claim the minute they share and a sum of
    hourly steps would drift upward.
    """
    one_minute = trading_hours_between(ts("2024-11-22 14:30"), ts("2024-11-22 14:31"), cal)
    assert one_minute == pytest.approx(1 / 60)


def test_consecutive_spans_sum_to_the_whole(cal) -> None:
    """Tiling property: splitting a session anywhere must not change the total."""
    whole = trading_hours_between(ts("2024-11-22 14:30"), ts("2024-11-22 21:00"), cal)
    first = trading_hours_between(ts("2024-11-22 14:30"), ts("2024-11-22 17:00"), cal)
    second = trading_hours_between(ts("2024-11-22 17:00"), ts("2024-11-22 21:00"), cal)
    assert first + second == whole


def test_same_timestamp_is_zero(cal) -> None:
    assert trading_hours_between(ts("2024-11-22 15:00"), ts("2024-11-22 15:00"), cal) == 0.0


def test_reversed_arguments_raise(cal) -> None:
    """A flag after the news is a bug, not a negative lead time. Argument order
    is easy to get wrong and this is the function where that corrupts the
    headline number."""
    with pytest.raises(ValueError, match="end precedes start"):
        trading_hours_between(ts("2024-11-22 15:00"), ts("2024-11-22 14:00"), cal)


def test_out_of_range_raises_naming_the_endpoint(cal) -> None:
    with pytest.raises(ValueError, match=r"outside the XNYS calendar"):
        trading_hours_between(ts("1990-01-03 15:00"), ts("2024-11-22 15:00"), cal)
    with pytest.raises(ValueError, match=r"outside the XNYS calendar"):
        trading_hours_between(ts("2024-11-22 15:00"), ts("2099-01-04 15:00"), cal)
