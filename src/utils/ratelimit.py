"""Rate limiting + exponential backoff for every external API in the project.

The plan's hard rules (§5.2): conservative pacing, exponential backoff on
429/403, never hammer. All collectors go through RateLimiter — no bare
time.sleep() pacing anywhere else.
"""

from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)


class RateLimiter:
    """Enforces a minimum interval between calls (a 1-token token bucket)."""

    def __init__(self, min_interval_s: float):
        self.min_interval_s = float(min_interval_s)
        self._last_call = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        remaining = self.min_interval_s - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_call = time.monotonic()


class Backoff:
    """Exponential backoff: 60s, 120s, 240s, ... capped; reset on success."""

    def __init__(self, base_s: float = 60.0, cap_s: float | None = None):
        """`cap_s` defaults to `ratelimit.backoff_cap_s` in config.yaml — never
        a hardcoded ceiling — so callers that just do `Backoff(base_s=...)`
        (every current one) still get a config-driven cap rather than a bare
        Python default. Config is imported lazily, matching
        `timeutils.get_market_calendar`, so importing this module never
        requires a config file to exist and reading config is never a side
        effect of an import. Pass `cap_s` explicitly to override it (e.g. in
        tests) without touching config.
        """
        self.base_s = base_s
        if cap_s is None:
            from src.utils.config import load_config

            cap_s = load_config()["ratelimit"]["backoff_cap_s"]
        self.cap_s = cap_s
        self.failures = 0

    def sleep(self, reason: str = "") -> None:
        delay = min(self.base_s * (2 ** self.failures), self.cap_s)
        self.failures += 1
        log.warning("Backing off %.0fs (attempt %d) %s", delay, self.failures, reason)
        time.sleep(delay)

    def reset(self) -> None:
        self.failures = 0
