"""UTC time and market-hours helpers.

Rule (plan §0.3): all timestamps are stored as UTC epoch seconds; conversion
to anything human-readable happens at display time only. Naive datetimes are
banned — every helper here is timezone-aware.

The market-hours half of this module exists because lead time in this project
is measured in TRADING hours, never wall-clock hours. Most 8-Ks are accepted
after the close, so "the six hours before the filing" is usually the previous
trading session. Answering that correctly needs a real exchange calendar:
holidays move, and the exchange closes early three times a year.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache

import exchange_calendars as xc
import numpy as np
import pandas as pd

GDELT_FMT = "%Y%m%d%H%M%S"  # e.g. 20250101000000, always UTC


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_ts() -> int:
    return int(utc_now().timestamp())


def ts_to_dt(ts: int | float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def dt_to_ts(dt: datetime) -> int:
    if dt.tzinfo is None:
        raise ValueError(f"Naive datetime not allowed: {dt!r}")
    return int(dt.timestamp())


def date_str_to_ts(s: str) -> int:
    """'YYYY-MM-DD' (interpreted as UTC midnight) -> epoch seconds."""
    return dt_to_ts(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc))


def iso_utc_to_ts(s: str) -> int:
    """ISO-8601 UTC (EDGAR's `acceptanceDateTime`) -> epoch seconds.

    EDGAR sends `2026-07-30T20:30:28.000Z`. The trailing `Z` means UTC and is
    the whole point: read as local time, every acceptance time in the study
    shifts by four or five hours, and by a *different* amount either side of a
    daylight-saving change. `fromisoformat` accepts the offset form, so `Z` is
    normalised first.
    """
    text = s.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        raise ValueError(
            f"Timestamp without a timezone: {s!r} — refusing to guess UTC"
        )
    return int(dt.timestamp())


def ts_to_gdelt(ts: int | float) -> str:
    return ts_to_dt(ts).strftime(GDELT_FMT)


def gdelt_to_ts(s: str) -> int:
    """GDELT 'seendate' like '20250101123000' or '20250101T123000Z'."""
    cleaned = s.replace("T", "").replace("Z", "").strip()
    return dt_to_ts(datetime.strptime(cleaned, GDELT_FMT).replace(tzinfo=timezone.utc))


def ts_to_iso(ts: int | float) -> str:
    return ts_to_dt(ts).strftime("%Y-%m-%d %H:%M:%SZ")


# --------------------------------------------------------------------------
# Market hours
#
# Sessions are LEFT-CLOSED, RIGHT-OPEN — [open, close). The opening minute
# counts as open; the closing minute does not. Fixed here so that every
# function measuring time agrees about an event landing exactly on the bell.
# --------------------------------------------------------------------------


@lru_cache(maxsize=4)
def get_market_calendar(code: str | None = None) -> xc.ExchangeCalendar:
    """The exchange calendar, cached.

    `code` defaults to `market.calendar` in config.yaml. Config is imported
    lazily so that importing this module does not require a config file to
    exist, and so that reading config is never a side effect of an import.

    Cached because building a calendar is not free, and the pipeline asks it
    one question per hour per ticker.
    """
    if code is None:
        from src.utils.config import load_config

        code = load_config()["market"]["calendar"]
    return xc.get_calendar(code)


@lru_cache(maxsize=4)
def _session_bounds(calendar_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Every session's open and close for one calendar, as epoch seconds.

    Keyed on the calendar's name rather than the object because
    `xc.get_calendar` hands back the same instance for a given name, and a
    string is the obvious cache key. Cached because `trading_hours_between`
    is asked one question per hour per ticker across a 322k-row feature
    matrix, and rebuilding two ~5,000-element arrays on every one of those
    calls would dominate the pipeline's runtime.

    Session opens and closes are always minute-aligned, so whole seconds hold
    them exactly; the fractional part of a real timestamp is carried by the
    other side of the arithmetic.

    Refuses a calendar with a lunch break. `trading_hours_between` reads a
    session as one unbroken interval `[open, close)`, which is true of XNYS
    but not of XTKS or XHKG — those shut for lunch, and counting the break as
    tradeable would inflate every lead time crossing it by an hour or more.
    Changing `market.calendar` to one of those must stop here rather than
    quietly produce bigger numbers.
    """
    cal = xc.get_calendar(calendar_name)
    if not cal.break_starts.isna().all():
        raise ValueError(
            f"{cal.name} has a lunch break, and trading_hours_between treats "
            f"each session as one unbroken [open, close) interval — it would "
            f"count the break as tradeable. Subtract the break interval here "
            f"before using this calendar for market.calendar."
        )
    opens = cal.opens.to_numpy(dtype="datetime64[s]").astype("int64")
    closes = cal.closes.to_numpy(dtype="datetime64[s]").astype("int64")
    return opens, closes


def _out_of_range(cal: xc.ExchangeCalendar, minute: pd.Timestamp) -> ValueError:
    """The error every market-hours helper raises when asked about a date the
    calendar does not cover. The library's own message omits the bounds, which
    is the one thing the reader needs."""
    return ValueError(
        f"{minute:%Y-%m-%d %H:%M:%S}Z is outside the {cal.name} calendar, "
        f"which covers {cal.first_session.date()} to {cal.last_session.date()}. "
        f"Upgrade exchange-calendars (currently pinned) if the study window "
        f"has moved past it."
    )


def is_market_open(ts: int | float,
                   calendar: xc.ExchangeCalendar | None = None) -> bool:
    """Was the exchange open at this exact UTC epoch second?

    Handles weekends, holidays, and early closes — the day after Thanksgiving
    closes at 13:00 ET, not 16:00, and getting that wrong overstates a lead
    time by three hours.

    Raises if `ts` falls outside the calendar's coverage. Returning False there
    would be worse than useless: the live monitor would conclude the market is
    permanently shut and quietly stop alerting.
    """
    cal = calendar or get_market_calendar()
    minute = pd.Timestamp(ts, unit="s", tz="UTC")
    try:
        return bool(cal.is_open_on_minute(minute))
    except ValueError as exc:  # MinuteOutOfBounds subclasses ValueError
        raise _out_of_range(cal, minute) from exc


def next_market_close(ts: int | float,
                      calendar: xc.ExchangeCalendar | None = None) -> int:
    """Epoch second of the next market close strictly after `ts`.

    Strictly after, matching the module's `[open, close)` convention: asked at
    the closing bell itself, that session is already over, so the answer is the
    NEXT session's close.

    From inside a session this is that session's own close — which is what the
    "trading hours to close" feature wants. From after the close it is the next
    trading session's, skipping weekends and holidays, and it respects early
    closes: after Wednesday's close in Thanksgiving week this returns Friday
    18:00 UTC (13:00 ET), not 21:00 UTC.
    """
    cal = calendar or get_market_calendar()
    minute = pd.Timestamp(ts, unit="s", tz="UTC")
    try:
        return int(cal.next_close(minute).timestamp())
    except ValueError as exc:
        raise _out_of_range(cal, minute) from exc


def next_market_open(ts: int | float,
                     calendar: xc.ExchangeCalendar | None = None) -> int:
    """Epoch second of the next market open strictly after `ts`.

    Used by the live monitor to decide when to next wake up rather than polling
    through a closed market.
    """
    cal = calendar or get_market_calendar()
    minute = pd.Timestamp(ts, unit="s", tz="UTC")
    try:
        return int(cal.next_open(minute).timestamp())
    except ValueError as exc:
        raise _out_of_range(cal, minute) from exc


def trading_hours_between(start_ts: int | float, end_ts: int | float,
                          calendar: xc.ExchangeCalendar | None = None) -> float:
    """Hours the market was OPEN between two UTC epoch seconds.

    The function every lead-time number in the report is computed with. Wall
    clock is not a substitute: a flag at 20:00 UTC Friday and an 8-K accepted
    14:00 UTC Monday is 66 wall-clock hours but **1.0 trading hours**, because
    everything between the Friday close and the Monday open is time nobody
    could trade in.

    Half-open `[start, end)`, matching the session convention in this module —
    the closing bell is not counted, so consecutive spans tile without
    double-counting the minute they share.

    Raises if `end` precedes `start`: that means the flag came after the news,
    which is an argument-order mistake or a meaningless lead time, and either
    should stop rather than land silently in a results table.
    """
    cal = calendar or get_market_calendar()
    a = pd.Timestamp(start_ts, unit="s", tz="UTC")
    b = pd.Timestamp(end_ts, unit="s", tz="UTC")

    # Before anything else: a missing timestamp. Without this the comparisons
    # below all fall through on NaT, the bounds check fails, and the error
    # formatter then dies with "NaTType does not support strftime" — loud, but
    # naming the wrong problem to whoever has to debug it.
    if a is pd.NaT or b is pd.NaT:
        missing = "start_ts" if a is pd.NaT else "end_ts"
        raise ValueError(
            f"trading_hours_between got a missing timestamp for {missing} "
            f"({start_ts!r}, {end_ts!r}). A lead time cannot be measured from "
            f"or to an unknown instant — find why it is NaN rather than "
            f"treating it as zero."
        )

    if b < a:
        raise ValueError(
            f"end precedes start: {b:%Y-%m-%d %H:%M:%S}Z < {a:%Y-%m-%d %H:%M:%S}Z. "
            f"Lead time is measured forward — check the argument order."
        )
    if b == a:
        return 0.0

    for t in (a, b):
        if not (cal.first_minute <= t <= cal.last_minute):
            raise _out_of_range(cal, t)

    # The answer is the total overlap between [start, end) and the sessions
    # themselves — summed directly from session opens and closes rather than by
    # counting the library's trading MINUTES.
    #
    # That distinction is the point of this implementation. `is_trading_minute`
    # and `minutes_in_range` answer under `exchange_calendars`' minute-grid
    # convention, set by the calendar's `side` (this project's calendars are
    # "left", so a session contributes 390 minutes, not 391). That convention is
    # a library default we do not pin, and a flip would move every lead-time
    # number in the report by a minute per session crossed — silently, with
    # every test still passing. Counting from opens and closes depends on no
    # such convention.
    #
    # It is also exact below a minute, which the grid is not: real timestamps
    # here (EDGAR acceptanceDateTime, news article times) carry seconds, and an
    # earlier version that handed sub-minute values to `minutes_in_range` got
    # them floored — sometimes to zero hours for a span that was open
    # throughout. Here the endpoints enter the arithmetic as they are.
    #
    # Holidays, weekends, early closes and DST are all already baked into
    # `opens`/`closes` in UTC, so none of them needs a special case. A session
    # is read as one unbroken interval, which `_session_bounds` refuses to let
    # a lunch-break calendar violate.
    opens, closes = _session_bounds(cal.name)
    overlap = np.clip(np.minimum(closes, float(end_ts))
                      - np.maximum(opens, float(start_ts)), 0.0, None)
    return float(overlap.sum()) / 3600.0
