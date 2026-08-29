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

On top of that sits the universe build (P2-02): SEC's ticker->CIK map,
filtered to the configured exchanges and collapsed to one row per company.
Per-company submissions and the 8-K parse are P2-03 and P2-04.

Usage:
  python -m src.collectors.edgar --build-universe
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

import requests

from src import db
from src.utils.config import load_config
from src.utils.ratelimit import Backoff, RateLimiter
from src.utils.timeutils import date_str_to_ts

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


# --------------------------------------------------------------------------
# P2-02 — the universe
# --------------------------------------------------------------------------

def pick_primary_ticker(tickers: list[str]) -> str:
    """The one listing that represents a company, out of all its listings.

    A CIK routinely carries several tickers — share classes, preferred series,
    structured notes. On the real file, 895 of 6,054 companies do. `companies`
    is keyed by CIK, so taking whichever arrives last would leave JPMorgan
    labelled `VYLD` (a structured note): the filings would still be right,
    while the prices and news joined to them would be a different instrument.
    Nothing would error.

    A hyphen in this file marks a preferred series or a share class
    (`JPM-PC`, `ORCL-PD`), so prefer a plain ticker; among equals keep SEC's
    own order, which runs from most to least prominent. Companies whose only
    listings are hyphenated — genuine dual-class commons like `BRK-B`, or a
    preferred-only filer — keep the first of those rather than being dropped.
    """
    plain = [t for t in tickers if "-" not in t]
    return (plain or tickers)[0]


def company_rows(cfg: dict, payload: dict) -> list[dict]:
    """`company_tickers_exchange.json` -> one `companies` row per CIK.

    The universe is dated at the START of the study window, never today: a map
    built from today has already dropped every company that was acquired or
    delisted, which is exactly the dramatic events this study is about.
    """
    fields = [f.lower() for f in payload["fields"]]
    idx = {name: fields.index(name) for name in ("cik", "name", "ticker", "exchange")}
    keep = set(cfg["universe"]["exchanges"])
    as_of = date_str_to_ts(cfg["study_window"]["start"])

    # Insertion-ordered, so "first in the file" survives to pick_primary_ticker.
    seen: dict[str, dict] = {}
    for record in payload["data"]:
        exchange = record[idx["exchange"]]
        ticker = record[idx["ticker"]]
        if exchange not in keep or not ticker:
            continue
        cik = str(record[idx["cik"]]).zfill(10)  # string, ten digits, always
        entry = seen.setdefault(cik, {
            "cik": cik,
            "name": record[idx["name"]],
            "exchange": exchange,
            "universe_as_of": as_of,
            "in_universe": None,   # the Phase 3 liquidity filter decides
            "_tickers": [],
        })
        entry["_tickers"].append(ticker)

    rows = []
    for entry in seen.values():
        tickers = entry.pop("_tickers")
        rows.append({**entry, "ticker": pick_primary_ticker(tickers)})
    return rows


def build_universe(cfg: dict, conn, client: EdgarClient | None = None,
                   force: bool = False) -> int:
    """Fetch the ticker map and upsert it into `companies`. Returns rows written."""
    client = client or EdgarClient(cfg)
    payload = client.get_json(client.company_tickers_url(), force=force)
    rows = company_rows(cfg, payload)

    if not rows and cfg["logging"]["fail_on_zero_records"]:
        raise RuntimeError(
            f"universe: parsed ZERO companies from "
            f"{client.company_tickers_url()} for exchanges "
            f"{cfg['universe']['exchanges']} — refusing to report success"
        )

    new = db.upsert_companies(conn, rows)
    total = conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
    log.info("universe: %d companies parsed, %d new, %d rows in companies",
             len(rows), new, total)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-universe", action="store_true",
                        help="fetch the SEC ticker map into `companies`")
    parser.add_argument("--force", action="store_true",
                        help="re-fetch instead of serving from the raw cache")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config()
    if not args.build_universe:
        parser.error("nothing to do — pass --build-universe")

    conn = db.get_conn(cfg["paths"]["db"])
    build_universe(cfg, conn, force=args.force)


if __name__ == "__main__":
    main()
