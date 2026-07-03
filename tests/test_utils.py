"""Tests for UTC time helpers, rate limiter, and config loading."""

import time
from datetime import datetime

import pytest

from src.utils import timeutils
from src.utils.config import load_config
from src.utils.ratelimit import RateLimiter


def test_gdelt_roundtrip():
    ts = timeutils.date_str_to_ts("2025-06-15")
    assert timeutils.ts_to_gdelt(ts) == "20250615000000"
    assert timeutils.gdelt_to_ts("20250615000000") == ts
    assert timeutils.gdelt_to_ts("20250615T000000Z") == ts  # artlist variant


def test_naive_datetime_rejected():
    with pytest.raises(ValueError):
        timeutils.dt_to_ts(datetime(2025, 1, 1))


def test_rate_limiter_enforces_interval():
    rl = RateLimiter(0.05)
    start = time.monotonic()
    rl.wait()
    rl.wait()
    rl.wait()
    assert time.monotonic() - start >= 0.09  # 2 enforced gaps


def test_config_loads_and_resolves_paths():
    cfg = load_config()
    assert cfg["reward"]["T_max"] == 48
    assert cfg["paths"]["db"].endswith("data/db/rumor.db")
    assert "wallstreetbets" in cfg["subreddits"]
    assert cfg["reddit"]["live_min_interval_s"] >= 7
