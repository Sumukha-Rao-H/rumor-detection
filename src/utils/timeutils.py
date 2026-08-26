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
