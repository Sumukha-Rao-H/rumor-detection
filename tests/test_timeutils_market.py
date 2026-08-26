"""Market-hours helpers — is_market_open.

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
