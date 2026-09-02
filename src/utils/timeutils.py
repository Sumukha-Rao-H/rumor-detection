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
    """Epoch second of the next market close at or after `ts`.

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

    # `exchange_calendars` only knows whole trading minutes — it has no notion
    # of a fraction of a minute. Real timestamps in this project (EDGAR
    # acceptanceDateTime, news article times) carry seconds, so `a` and `b`
    # are not, in general, minute-aligned, and the sub-minute remainder at
    # each end has to be handled by hand rather than handed to the library.
    #
    # A previous version asked `minutes_in_range(a, b - 1 minute)` for a
    # half-open span. That is only a correct way to exclude `b`'s minute when
    # `a` and `b` are themselves exactly on minute boundaries: internally,
    # `exchange_calendars` FLOORS any sub-minute timestamp to its containing
    # minute before comparing (see `calendar_helpers.parse_timestamp`, which
    # floors because this calendar's `side` is "left"). That floors `b - 1
    # minute` right back down whenever `b` itself isn't aligned, which can
    # even push it before `a` and silently return 0 minutes for a span that
    # was open the whole time — or, the other direction, floors `a` and `b`
    # to their own minutes and then counts each of those minutes as whole,
    # overcounting a sub-minute span that merely touches two different
    # minutes. Neither direction is a rounding error; both are wrong answers.
    #
    # The correct decomposition treats `a`'s minute and `b`'s minute as
    # special and sums three pieces:
    #   1. the open seconds remaining in `a`'s own minute (from `a` to the
    #      start of the next minute), counted only if `a`'s minute is itself
    #      a trading minute;
    #   2. every whole trading minute strictly between `a`'s minute and `b`'s
    #      minute — this is exactly what `minutes_in_range` is for, since
    #      both endpoints here ARE minute-aligned;
    #   3. the open seconds already elapsed in `b`'s own minute (from the
    #      start of that minute to `b`), counted only if `b`'s minute is
    #      itself a trading minute.
    # When `a` and `b` fall in the same minute, only that one (possibly
    # partial) minute matters. When `a` and `b` are both exactly minute-
    # aligned (the case every previous test exercised), this reduces to
    # exactly the old behaviour: piece 1 contributes a's minute in full,
    # piece 3 contributes nothing from b's minute, and piece 2 is the whole
    # minutes strictly in between — the same count as before.
    one_minute = pd.Timedelta(minutes=1)
    a_minute = a.floor("min")
    b_minute = b.floor("min")

    if a_minute == b_minute:
        seconds = (b - a).total_seconds() if cal.is_trading_minute(a_minute) else 0.0
        return seconds / 3600.0

    seconds = 0.0
    if cal.is_trading_minute(a_minute):
        seconds += (a_minute + one_minute - a).total_seconds()
    if cal.is_trading_minute(b_minute):
        seconds += (b - b_minute).total_seconds()

    between_start = a_minute + one_minute
    between_end = b_minute - one_minute
    if between_start <= between_end:
        seconds += len(cal.minutes_in_range(between_start, between_end)) * 60.0

    return seconds / 3600.0
