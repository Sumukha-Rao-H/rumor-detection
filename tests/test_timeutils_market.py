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

All of those are WINTER dates. The P1-06 section at the bottom covers summer,
because the session moves an hour in UTC across a daylight-saving change and a
suite that only ever tests November would never notice.
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


# --------------------------------------------------------------------------
# P1-06 — daylight saving, and invariants
#
# Everything above uses November/December dates. The exchange runs on New York
# local time, so the session moves an hour in UTC across the year:
#
#   2025-01-15   14:30-21:00 UTC   (09:30 ET, EST = UTC-5)
#   2025-07-15   13:30-20:00 UTC   (09:30 ET, EDT = UTC-4)
#
# If anything in the stack assumed a fixed 14:30 UTC open, every test above
# would still pass and two-thirds of the study window would be silently wrong.
# --------------------------------------------------------------------------


def test_summer_session_opens_an_hour_earlier_in_utc(cal) -> None:
    """13:30 UTC in July is 09:30 ET — open. The same clock time in winter is
    pre-market."""
    assert is_market_open(ts("2025-07-15 13:30"), cal) is True
    assert is_market_open(ts("2025-01-15 13:30"), cal) is False


def test_winter_close_time_is_already_shut_in_summer(cal) -> None:
    """The same UTC clock time falls in different places in the session.

    20:00 UTC is 15:00 EST in winter — mid-session, open.
    20:00 UTC is 16:00 EDT in summer — exactly the close, so shut under the
    [open, close) convention.
    """
    assert is_market_open(ts("2025-01-15 20:00"), cal) is True   # 15:00 ET
    assert is_market_open(ts("2025-07-15 20:00"), cal) is False  # 16:00 EDT


def test_summer_session_is_still_six_and_a_half_hours(cal) -> None:
    """The session's LENGTH does not change with DST, only its UTC placement."""
    assert trading_hours_between(ts("2025-07-15 13:30"), ts("2025-07-15 20:00"), cal) == 6.5
    assert trading_hours_between(ts("2025-01-15 14:30"), ts("2025-01-15 21:00"), cal) == 6.5


def test_span_across_the_dst_transition(cal) -> None:
    """US clocks moved forward on Sunday 2025-03-09.

    Friday's session opened 14:30 UTC; Monday's opened 13:30 UTC. A span from
    an hour before Friday's close to an hour after Monday's open is 2.0 trading
    hours, and naive arithmetic on UTC offsets would get it wrong.
    """
    a, b = ts("2025-03-07 20:00"), ts("2025-03-10 14:30")
    assert (b - a) / 3600 == 66.5                              # wall clock
    assert trading_hours_between(a, b, cal) == 2.0             # 1h Fri + 1h Mon


def test_trading_hours_never_exceed_wall_clock(cal) -> None:
    """An invariant, not an example: the market cannot be open for longer than
    the time that actually passed. Holds for every span, so it catches whole
    classes of arithmetic error rather than the cases someone thought of."""
    spans = [
        ("2024-11-27 15:00", "2024-11-27 16:00"),
        ("2024-11-22 20:00", "2024-11-25 14:00"),
        ("2025-03-07 20:00", "2025-03-10 14:30"),
        ("2025-07-15 13:30", "2025-07-15 20:00"),
        ("2024-12-24 18:00", "2024-12-26 15:00"),
    ]
    for a, b in spans:
        wall = (ts(b) - ts(a)) / 3600
        assert trading_hours_between(ts(a), ts(b), cal) <= wall, f"{a} -> {b}"


def test_hours_to_close_agrees_with_next_market_close(cal) -> None:
    """Cross-check between P1-04 and P1-05.

    Measuring from mid-session to that session's close must give the hours
    remaining. If either helper drifts, the two stop agreeing.
    """
    for moment, expected in [("2025-01-15 15:00", 6.0), ("2025-07-15 14:30", 5.5)]:
        t0 = ts(moment)
        assert trading_hours_between(t0, next_market_close(t0, cal), cal) == expected


def test_study_window_edges_are_usable(cal) -> None:
    """Every other test uses dates I picked. This one uses the dates the
    project actually runs on, so a window moved past the calendar's coverage
    fails here rather than mid-collection."""
    from src.utils.config import load_config

    window = load_config()["study_window"]
    start, end = date_str_to_ts(window["start"]), date_str_to_ts(window["end"])
    assert trading_hours_between(start, end, cal) > 0
    is_market_open(start, cal)  # must not raise
    is_market_open(end, cal)
