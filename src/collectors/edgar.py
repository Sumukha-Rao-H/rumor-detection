"""EDGAR collector — 8-K filings, item codes, and exact acceptance times.

This is the answer key. Every label in the study traces back to one row this
module wrote, so the transport layer is deliberately paranoid: SEC's terms of
service are a descriptive User-Agent with a contact address and no more than
10 requests per second, and the 2026 failure mode everyone hits is an HTTP 200
carrying redirect HTML that parses to nothing while the pipeline reports
success.

`EdgarClient` is cache-first. Every raw response is written under
`paths.edgar_raw` before anything parses it, and a URL already on disk is
served without touching the network — so a repeat run of the full universe
makes zero requests, and a body that turns out not to be JSON is dropped rather
than cached forever.

Parsing, the universe build, and the filings upsert are P2-02 … P2-04.

Usage (P2-05 adds the CLI):
  from src.collectors.edgar import EdgarClient
  client = EdgarClient(load_config())
  data = client.get_json(client.submissions_url("0000320193"))
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

import requests

from src.utils.ratelimit import Backoff, RateLimiter

log = logging.getLogger(__name__)

#: Status codes SEC uses to say "slow down" or "not right now". Anything else
#: (404, 403) is a fact about the URL, not a transient condition, so retrying
#: it just burns the request budget.
RETRY_STATUS = (429, 502, 503, 504)


class EdgarRequestError(RuntimeError):
    """A URL that could not be fetched or did not come back as JSON."""


class EdgarClient:
    """Cache-first HTTP for EDGAR: one session, one limiter, one cache root.

    Held together in an object because Phase 2 makes ~1,500 requests to one
    host and every call site needs the same four things. The limiter must
    outlive a single call or back-to-back requests would not pace against each
    other.
    """

    def __init__(self, cfg: dict, session: requests.Session | None = None):
        self.cfg = cfg
        ecfg = cfg["edgar"]
        self.session = session or requests.Session()
        self.session.headers.update({
            # SEC's entire terms of service: say who you are and how to reach
            # you. Requests without this are blocked.
            "User-Agent": cfg["http"]["user_agent"],
            "Accept-Encoding": "gzip, deflate",
        })
        self.cache_root = Path(cfg["paths"]["edgar_raw"])
        self.limiter = RateLimiter(ecfg["min_interval_s"])
        self.max_retries = int(ecfg["max_retries"])
        self._backoff_base_s = float(ecfg["backoff_base_s"])

    # -- URLs ---------------------------------------------------------------

    def submissions_url(self, cik: str) -> str:
        """Submissions JSON for a zero-padded 10-digit CIK."""
        return f"{self.cfg['edgar']['submissions_base']}/CIK{cik}.json"

    def company_tickers_url(self) -> str:
        """The ticker -> CIK map used to build the universe (P2-02)."""
        return self.cfg["universe"]["company_tickers_url"]

    # -- Cache --------------------------------------------------------------

    def cache_path(self, url: str) -> Path:
        """Where a URL's raw body lives on disk.

        The path mirrors the URL rather than hashing it, so the cache can be
        read, audited, and pruned by hand. The host is included because the
        phase talks to both www.sec.gov and data.sec.gov.
        """
        parsed = urlparse(url)
        rel = parsed.path.lstrip("/") or "index"
        if parsed.query:
            # No EDGAR endpoint used here takes a query string, but one arriving
            # later must not silently overwrite the query-less entry.
            rel = f"{rel}__{parsed.query.replace('&', '_').replace('=', '-')}"
        return self.cache_root / parsed.netloc / rel

    def _write_cache(self, path: Path, body: bytes) -> None:
        """Atomically, so an interrupted run never leaves a truncated cache hit."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(body)
        os.replace(tmp, path)

    # -- Fetch --------------------------------------------------------------

    def get_bytes(self, url: str, force: bool = False) -> bytes:
        """Raw response body for `url`, from the cache when it is already there.

        Set `force` to re-fetch and overwrite. Raises `EdgarRequestError` after
        `edgar.max_retries` failed attempts.
        """
        path = self.cache_path(url)
        if path.exists() and not force:
            # Deliberately before the limiter: a fully cached re-run of the
            # universe should be instant, not 1,500 x 0.125 s of sleeping.
            log.debug("cache hit %s", url)
            return path.read_bytes()

        backoff = Backoff(base_s=self._backoff_base_s)
        for attempt in range(1, self.max_retries + 1):
            self.limiter.wait()  # before every attempt, retries included
            try:
                resp = self.session.get(url, timeout=30)
            except requests.RequestException as exc:
                log.warning("EDGAR request failed (%d/%d) for %s: %s",
                            attempt, self.max_retries, url, exc)
                backoff.sleep(f"request error for {url}")
                continue

            if resp.status_code in RETRY_STATUS:
                log.warning("HTTP %d from EDGAR (%d/%d) for %s",
                            resp.status_code, attempt, self.max_retries, url)
                backoff.sleep(f"HTTP {resp.status_code} from {url}")
                continue

            if resp.status_code != 200:
                # 404 and 403 are facts about the URL. Retrying wastes budget.
                raise EdgarRequestError(
                    f"HTTP {resp.status_code} from EDGAR for {url} — not retried"
                )

            self._write_cache(path, resp.content)
            return resp.content

        raise EdgarRequestError(
            f"giving up on {url} after {self.max_retries} attempts"
        )

    def get_json(self, url: str, force: bool = False) -> dict | list:
        """Parsed JSON for `url`.

        The raw body is cached before parsing, per the collector contract. If it
        does not parse — the classic 200-carrying-redirect-HTML — the cache
        entry is removed, because a poisoned cache would serve that HTML to
        every later run without ever making a request again.
        """
        body = self.get_bytes(url, force=force)
        try:
            return json.loads(body)
        except (ValueError, UnicodeDecodeError) as exc:
            path = self.cache_path(url)
            path.unlink(missing_ok=True)
            raise EdgarRequestError(
                f"EDGAR returned non-JSON for {url} "
                f"(first 120 bytes: {body[:120]!r}) — cache entry discarded"
            ) from exc
