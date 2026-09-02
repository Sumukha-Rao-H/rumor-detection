"""Tests for UTC time helpers, rate limiter, and config loading."""

import time
from datetime import datetime

import pytest

from src.utils import timeutils
from src.utils.config import load_config
from src.utils.ratelimit import Backoff, RateLimiter


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


def test_backoff_grows_exponentially(monkeypatch):
    """60, 120, 240, ... per the docstring — checked against the actual
    sleep durations requested, not just internal state."""
    slept = []
    monkeypatch.setattr("src.utils.ratelimit.time.sleep", slept.append)
    backoff = Backoff(base_s=60.0, cap_s=1e9)  # cap far out of reach here
    for _ in range(5):
        backoff.sleep()
    assert slept == [60.0, 120.0, 240.0, 480.0, 960.0]


def test_backoff_is_capped(monkeypatch):
    """However many failures pile up, the sleep never exceeds the cap."""
    slept = []
    monkeypatch.setattr("src.utils.ratelimit.time.sleep", slept.append)
    backoff = Backoff(base_s=60.0, cap_s=200.0)
    for _ in range(6):
        backoff.sleep()
    assert slept == [60.0, 120.0, 200.0, 200.0, 200.0, 200.0]
    assert max(slept) == 200.0


def test_backoff_reset_zeroes_failures(monkeypatch):
    """`reset()` (called on a successful request) must restart the sequence
    from the base delay, not continue climbing."""
    slept = []
    monkeypatch.setattr("src.utils.ratelimit.time.sleep", slept.append)
    backoff = Backoff(base_s=60.0, cap_s=1e9)
    backoff.sleep()
    backoff.sleep()
    assert backoff.failures == 2
    backoff.reset()
    assert backoff.failures == 0
    backoff.sleep()
    assert slept[-1] == 60.0  # back to the base delay, not 240


def test_backoff_cap_defaults_from_config(monkeypatch):
    """Project rule: no hardcoded rate limits. The default cap must come from
    config.yaml's `ratelimit.backoff_cap_s`, not a bare Python default."""
    monkeypatch.setattr("src.utils.ratelimit.time.sleep", lambda s: None)
    cfg_cap = load_config()["ratelimit"]["backoff_cap_s"]
    assert Backoff().cap_s == cfg_cap


def test_backoff_cap_still_overridable_explicitly(monkeypatch):
    """A caller (or a test) can still bypass config and set its own cap."""
    monkeypatch.setattr("src.utils.ratelimit.time.sleep", lambda s: None)
    assert Backoff(cap_s=42.0).cap_s == 42.0


def test_config_loads_and_resolves_paths():
    cfg = load_config()
    assert cfg["paths"]["db"].endswith("data/db/footprints.db")
    assert cfg["decision"]["horizon_hours"] > 0
    # SEC caps traffic at 10 req/s; staying under it is a terms-of-service rule.
    assert cfg["edgar"]["max_requests_per_s"] <= 10
    # 9.01 is an attachment marker, not an event type — it must be excluded or
    # it dominates the label distribution.
    assert "9.01" in cfg["items"]["exclude"]
    # The headline metric is precision at a fixed alert budget, never accuracy.
    assert cfg["eval"]["alert_budget_per_stock_per_month"] > 0
    # No hardcoded rate limits (rule 7): the Backoff ceiling comes from here.
    assert cfg["ratelimit"]["backoff_cap_s"] > 0
