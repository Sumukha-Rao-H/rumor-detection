"""Dependency guard for the market calendar.

`exchange_calendars` is the only pinned package in requirements.txt, because it
ships holiday tables and early-close rules as *data*. These tests do not check
the helper functions in `src.utils.timeutils` — that is P1-06. They check the
one thing those helpers cannot: that the calendar the config asks for exists,
is the right calendar, and actually spans the study window.

No network, no database.
"""

from __future__ import annotations

import datetime as dt

import exchange_calendars as xc
import pytest

from src.utils.config import load_config

SESSIONS_PER_YEAR = 252  # NYSE, approximately


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def calendar(cfg: dict):
    return xc.get_calendar(cfg["market"]["calendar"])


def _window_dates(cfg: dict) -> tuple[dt.date, dt.date]:
    window = cfg["study_window"]
    return (
        dt.date.fromisoformat(window["start"]),
        dt.date.fromisoformat(window["end"]),
    )


def test_configured_calendar_loads(cfg: dict, calendar) -> None:
    """The code in config/config.yaml resolves to a real calendar."""
    assert calendar.name == cfg["market"]["calendar"]


def test_calendar_timezone_is_new_york(calendar) -> None:
    """Sessions must be defined in exchange local time.

    A UTC-based calendar would put session boundaries in the wrong place and
    every 'was the market open at time t?' answer would drift by hours across
    daylight-saving changes.
    """
    assert str(calendar.tz) == "America/New_York"


def test_calendar_covers_study_window(cfg: dict, calendar) -> None:
    """The calendar's session range must span the whole study window.

    exchange_calendars builds a bounded range. If a future version narrows it,
    or the window moves, this fails here rather than as a wall of NaNs in
    phase 4.
    """
    start, end = _window_dates(cfg)
    assert calendar.first_session.date() <= start, (
        f"calendar starts {calendar.first_session.date()}, after the study "
        f"window start {start}"
    )
    assert calendar.last_session.date() >= end, (
        f"calendar ends {calendar.last_session.date()}, before the study "
        f"window end {end}"
    )


def test_window_contains_plausible_session_count(cfg: dict, calendar) -> None:
    """Guards against an empty or degenerate calendar.

    A calendar that loads but yields no sessions would make every trading-hour
    count silently zero.
    """
    start, end = _window_dates(cfg)
    sessions = calendar.sessions_in_range(str(start), str(end))
    years = (end - start).days / 365.25
    expected = SESSIONS_PER_YEAR * years
    assert 0.9 * expected <= len(sessions) <= 1.1 * expected, (
        f"{len(sessions)} sessions in {years:.2f} years; expected "
        f"~{expected:.0f}"
    )


def test_known_holiday_is_not_a_session(calendar) -> None:
    """Christmas Day 2024 fell on a Wednesday and the exchange was closed.

    A weekday holiday is the cheapest proof that the holiday table is loaded
    at all, rather than the calendar just excluding weekends.
    """
    assert not calendar.is_session("2024-12-25")
    assert calendar.is_session("2024-12-24")  # the half-day before it
