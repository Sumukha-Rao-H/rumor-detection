"""Rotating pool of API keys for one provider.

Free tiers are metered per key, not per user, so a dozen teammates' keys are a
dozen independent quotas. This module turns them into one logical credential:
requests go to whichever key is ready soonest, and a key that hits its daily
limit steps out until that provider's quota resets instead of stalling the run.

Three failure modes, deliberately treated differently — conflating them is how
a pool destroys itself:

  per-minute   the key is fine, we were just fast. Back off and retry the *same*
               key; rotating here would burn all twelve within a minute.
  per-day      the key is spent until the provider's local midnight. Cool it
               until then and move on.
  invalid      revoked, mistyped or never enabled. Retiring it for the session
               matters most with borrowed keys, where a couple are usually dead.

State lives in a JSON file (`paths.llm_state`) keyed by a fingerprint of the
key, never the key itself, so an interrupted run does not re-discover the same
exhausted keys tomorrow morning. Labels are env var names, safe to log.

Usage:
  python -m src.utils.keypool          # show every key and its state
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.utils.ratelimit import RateLimiter
from src.utils.timeutils import next_midnight_ts, ts_to_iso, utc_now_ts

log = logging.getLogger(__name__)

# Providers phrase the same two limits a dozen ways: Gemini puts
# "GenerateRequestsPerDayPerProjectPerModel" in error.details[].quotaId, Groq
# says "rate limit reached ... on requests per day (RPD)".
PER_DAY_RE = re.compile(r"per[\s_-]?day|\bRPD\b|\bTPD\b|daily", re.IGNORECASE)
PER_MINUTE_RE = re.compile(r"per[\s_-]?minute|\bRPM\b|\bTPM\b", re.IGNORECASE)

DAY = "day"
MINUTE = "minute"
UNKNOWN = "unknown"

FOREVER = 4_102_444_800  # 2100-01-01, i.e. "not coming back this session"


def classify_quota(text: str) -> str:
    """Decide from a 429 body whether a daily or per-minute limit was hit."""
    # Order matters: a daily-quota body often names the per-minute metric too.
    if PER_DAY_RE.search(text or ""):
        return DAY
    if PER_MINUTE_RE.search(text or ""):
        return MINUTE
    return UNKNOWN


# The closing quote is optional: an error stored by an older, truncating
# version of this code can end mid-message, and that is exactly the text most
# in need of tidying.
MESSAGE_RE = re.compile(r'"message"\s*:\s*"([^"]*)')
STATUS_RE = re.compile(r"HTTP (\d+):?\s*")


def short_reason(reason: str) -> str:
    """Reduce a provider error to one readable line.

    Both vendors answer with pretty-printed JSON, so the raw string is a dozen
    lines of braces around one useful sentence. Keeping the status code and
    that sentence is what makes `keypool` output scannable. Idempotent, so
    re-shortening an already-shortened reason is a no-op.
    """
    reason = " ".join(reason.split())
    status = STATUS_RE.match(reason)
    prefix = f"HTTP {status.group(1)}: " if status else ""
    body = reason[status.end():] if status else reason
    message = MESSAGE_RE.search(body)
    return (prefix + (message.group(1) if message else body).strip())[:110]


def fingerprint(value: str) -> str:
    """Short, non-reversible id for a secret — safe to write to disk and logs."""
    return hashlib.sha1(value.encode()).hexdigest()[:12]


def discover_keys(env_name: str) -> list[tuple[str, str]]:
    """Find every key for `env_name` in the environment, as (label, value).

    Three accepted spellings, so a pool can grow without editing config:
    `GEMINI_API_KEY` (one key), `GEMINI_API_KEYS` (comma or whitespace
    separated) and `GEMINI_API_KEY_1..N` (one teammate per line in .env).
    Duplicates are dropped — two people pasting the same key is one quota.
    """
    found: list[tuple[str, str]] = []

    single = os.getenv(env_name, "").strip()
    if single:
        found.append((env_name, single))

    for i, part in enumerate(re.split(r"[,\s]+", os.getenv(f"{env_name}S", "")), 1):
        if part.strip():
            found.append((f"{env_name}S#{i}", part.strip()))

    numbered = re.compile(rf"^{re.escape(env_name)}_(\d+)$")
    indexed = []
    for name, value in os.environ.items():
        match = numbered.match(name)
        if match and value.strip():
            indexed.append((int(match.group(1)), name, value.strip()))
    found.extend((name, value) for _, name, value in sorted(indexed))

    seen: dict[str, str] = {}
    for label, value in found:
        seen.setdefault(value, label)
    return [(label, value) for value, label in seen.items()]


@dataclass
class ApiKey:
    label: str
    value: str = field(repr=False)
    fingerprint: str
    limiter: RateLimiter
    cooling_until: int = 0
    reason: str = ""

    def is_available(self, now: int) -> bool:
        return now >= self.cooling_until

    def status(self, now: int) -> str:
        if self.is_available(now):
            return "ready"
        if self.cooling_until >= FOREVER:
            return f"disabled ({self.reason})"
        return f"cooling until {ts_to_iso(self.cooling_until)} ({self.reason})"


class NoKeysAvailable(RuntimeError):
    """Every key for this provider is cooling down or disabled."""


class KeyPool:
    """The keys for one provider, with per-key pacing and cooldown state."""

    def __init__(self, provider: str, cfg: dict, state_path: str | Path | None = None):
        pcfg = cfg["llm"][provider]
        self.provider = provider
        self.env_name = pcfg["key_env"]
        self.reset_tz = pcfg.get("quota_reset_tz", "UTC")
        self.cooldown_s = int(cfg["llm"].get("key_cooldown_s", 900))
        self.state_path = Path(state_path or cfg["paths"]["llm_state"])

        min_interval = float(pcfg.get("min_interval_s", cfg["llm"]["min_interval_s"]))
        pairs = discover_keys(self.env_name)
        if not pairs:
            raise RuntimeError(
                f"No API keys for {provider}: set {self.env_name}, "
                f"{self.env_name}S (comma-separated) or {self.env_name}_1.. in .env"
            )
        self.keys = [
            ApiKey(label=label, value=value, fingerprint=fingerprint(value),
                   limiter=RateLimiter(min_interval))
            for label, value in pairs
        ]
        self._load_state()
        log.info("%s: %d keys (%d ready)", provider, len(self.keys),
                 sum(k.is_available(utc_now_ts()) for k in self.keys))

    def __len__(self) -> int:
        return len(self.keys)

    # ------------------------------------------------------------------ state

    def _load_state(self) -> None:
        """Re-apply cooldowns recorded by an earlier run; expired ones lapse."""
        try:
            blob = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        entries = blob.get(self.provider, {})
        now = utc_now_ts()
        for key in self.keys:
            entry = entries.get(key.fingerprint)
            if entry and entry.get("cooling_until", 0) > now:
                key.cooling_until = int(entry["cooling_until"])
                # Shorten on read too, so state written before short_reason()
                # existed displays cleanly without waiting to be re-cooled.
                key.reason = short_reason(entry.get("reason", ""))

    def _save_state(self) -> None:
        try:
            blob = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            blob = {}
        # Only cooling keys are worth recording; a ready key is the default.
        blob[self.provider] = {
            key.fingerprint: {"label": key.label,
                              "cooling_until": key.cooling_until,
                              "reason": key.reason}
            for key in self.keys if key.cooling_until
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(blob, indent=2), encoding="utf-8")

    # ----------------------------------------------------------------- public

    def available(self, now: int | None = None) -> list[ApiKey]:
        now = utc_now_ts() if now is None else now
        return [key for key in self.keys if key.is_available(now)]

    def acquire(self) -> ApiKey:
        """The usable key that will be ready soonest, after waiting for it.

        Picking by readiness rather than by position spreads a batch across
        every teammate's quota instead of draining the first one, and makes the
        pool's throughput the sum of its keys' rates rather than one key's.
        """
        ready = self.available()
        if not ready:
            raise NoKeysAvailable(
                f"all {len(self.keys)} {self.provider} keys unavailable: "
                + self.describe()
            )
        key = min(ready, key=lambda k: (k.limiter.time_until_ready(),
                                        self.keys.index(k)))
        key.limiter.wait()
        return key

    def cool(self, key: ApiKey, scope: str, reason: str = "") -> None:
        """Take a key out of rotation for as long as its failure implies."""
        now = utc_now_ts()
        if scope == DAY:
            key.cooling_until = next_midnight_ts(self.reset_tz, now)
        elif scope == "disabled":
            key.cooling_until = FOREVER
        else:
            key.cooling_until = now + self.cooldown_s
        key.reason = short_reason(reason)
        self._save_state()
        log.warning("%s %s out of rotation: %s", self.provider, key.label,
                    key.status(now))

    def describe(self) -> str:
        """A one-line summary for logs and exceptions.

        Deliberately not per-key: this string lands in every "no keys left"
        error, and spelling out twelve keys' states buried the two facts that
        matter — how many are left and when the next one returns. Run
        `python -m src.utils.keypool` for the full table.
        """
        now = utc_now_ts()
        disabled = [k for k in self.keys if k.cooling_until >= FOREVER]
        cooling = [k for k in self.keys
                   if not k.is_available(now) and k.cooling_until < FOREVER]
        parts = [f"{len(self.available(now))}/{len(self.keys)} ready"]
        if cooling:
            soonest = min(k.cooling_until for k in cooling)
            parts.append(f"{len(cooling)} cooling until {ts_to_iso(soonest)}")
        if disabled:
            parts.append(f"{len(disabled)} disabled "
                         f"({', '.join(k.label for k in disabled)})")
        return ", ".join(parts)

    def reset(self) -> None:
        """Put every key back in rotation — for after a key is re-issued."""
        for key in self.keys:
            key.cooling_until, key.reason = 0, ""
        self._save_state()


def main() -> None:
    """Print the state of every configured pool without calling any API."""
    import argparse

    from src.utils.config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", action="append",
                        help="limit to this provider (repeatable)")
    parser.add_argument("--reset", action="store_true",
                        help="clear all cooldowns — use after replacing a key, "
                             "or to re-test one the provider had refused")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    cfg = load_config()
    providers = args.provider or [
        name for name in (cfg["llm"]["provider"], cfg["llm"].get("fallback")) if name
    ]
    now = utc_now_ts()
    for provider in providers:
        pool = KeyPool(provider, cfg)
        if args.reset:
            pool.reset()
        ready = len(pool.available(now))
        print(f"\n{provider}: {ready}/{len(pool)} ready "
              f"(resets at midnight {pool.reset_tz})")
        for key in pool.keys:
            print(f"  {key.label:<22} {key.fingerprint}  {key.status(now)}")


if __name__ == "__main__":
    main()
