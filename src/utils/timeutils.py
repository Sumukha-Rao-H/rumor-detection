"""UTC time helpers.

Rule (plan §0.3): all timestamps are stored as UTC epoch seconds; conversion
to anything human-readable happens at display time only. Naive datetimes are
banned — every helper here is timezone-aware.
"""

from __future__ import annotations

from datetime import datetime, timezone

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
