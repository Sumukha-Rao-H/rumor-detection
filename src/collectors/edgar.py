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
filtered to the configured exchanges and collapsed to one row per company; and
the per-company submissions fetch (P2-03), which follows SEC's older-filings
pages so a heavy filer's window is not silently truncated; and the 8-K parse
(P2-04), which is where item codes stay strings and acceptance times stay UTC.

Usage:
  python -m src.collectors.edgar --build-universe
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests

from src import db
from src.utils.config import load_config, require_sec_user_agent
from src.utils.ratelimit import Backoff, RateLimiter
from src.utils.timeutils import (
    date_str_to_ts, iso_utc_to_ts, ts_to_dt, utc_now_ts,
)

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
        if session is None:
            # About to talk to the SEC for real, so the contact address has to
            # be a real one. Checked here rather than in each caller because
            # this is the one chokepoint every live request passes through.
            # An injected session means a test or a replay, which never
            # reaches the SEC and so needs no address.
            require_sec_user_agent(cfg)
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

    def submissions_page_url(self, page_name: str) -> str:
        """An older-filings page sits beside the main submissions file."""
        return f"{self.cfg['edgar']['submissions_base']}/{page_name}"

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
    if not isinstance(payload, dict) or "fields" not in payload:
        raise EdgarRequestError(
            f"company_tickers payload missing 'fields' — expected a dict "
            f"with 'fields' and 'data', got "
            f"{sorted(payload.keys()) if isinstance(payload, dict) else type(payload).__name__}"
        )
    fields = [f.lower() for f in payload["fields"]]
    try:
        idx = {name: fields.index(name) for name in ("cik", "name", "ticker", "exchange")}
    except ValueError as exc:
        raise EdgarRequestError(
            f"company_tickers payload's 'fields' is missing one of "
            f"('cik', 'name', 'ticker', 'exchange') — got {fields}"
        ) from exc
    if "data" not in payload:
        raise EdgarRequestError(
            f"company_tickers payload missing 'data' — expected a dict with "
            f"'fields' and 'data', got top-level keys {sorted(payload.keys())}"
        )
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


def check_rate_limit_config(ecfg: dict) -> None:
    """Make `edgar.max_requests_per_s` load-bearing instead of decorative.

    Only `RateLimiter(ecfg["min_interval_s"])` is ever read on the request
    path — `max_requests_per_s` was pure documentation, so editing it alone
    (e.g. bumping it to a still-SEC-legal 10) silently changed nothing. That
    is fine for tests, which deliberately zero `min_interval_s` for speed and
    have no real request to pace, but it is exactly the kind of silent no-op
    rule 7 exists to prevent for a real run — so the pair is validated here,
    at CLI startup, rather than inside `EdgarClient` where every test
    constructs one. Wiring it in this way (validating `min_interval_s`
    against it) is the smaller, safer change over deriving one value from the
    other outright, which would also have to decide which one wins.
    """
    if "max_requests_per_s" not in ecfg:
        return
    max_rps = float(ecfg["max_requests_per_s"])
    if max_rps <= 0:
        raise ValueError(f"edgar.max_requests_per_s must be > 0, got {max_rps}")
    implied_min_interval_s = 1.0 / max_rps
    min_interval_s = float(ecfg["min_interval_s"])
    if min_interval_s < implied_min_interval_s - 1e-9:
        raise ValueError(
            f"config/config.yaml: edgar.min_interval_s ({min_interval_s}) "
            f"paces faster than edgar.max_requests_per_s ({max_rps}) allows "
            f"(needs >= {implied_min_interval_s:.6f}s) — the two keys "
            f"disagree about the request rate"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-universe", action="store_true",
                        help="fetch the SEC ticker map into `companies`")
    parser.add_argument("--universe", action="store_true",
                        help="collect filings for every company in `companies`")
    parser.add_argument("--tickers",
                        help="comma-separated subset, e.g. TSLA,AAPL")
    parser.add_argument("--resume", action="store_true",
                        help="skip companies that already have filings stored")
    parser.add_argument("--link-predecessors", action="store_true",
                        help="find reorganised companies and add their "
                             "predecessor CIKs to `companies`")
    parser.add_argument("--dry-run", action="store_true",
                        help="with --link-predecessors: report, write nothing")
    parser.add_argument("--report", action="store_true",
                        help="print the sanity report on what has been collected")
    parser.add_argument("--force", action="store_true",
                        help="re-fetch instead of serving from the raw cache "
                             "(applies to --build-universe and to "
                             "--universe/--tickers alike)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config()
    check_rate_limit_config(cfg["edgar"])
    if not (args.build_universe or args.universe or args.tickers
            or args.report or args.link_predecessors):
        parser.error("nothing to do — pass --build-universe, --universe, "
                     "--tickers, --link-predecessors or --report")

    conn = db.get_conn(cfg["paths"]["db"])
    if args.report:
        print_filings_report(filings_report(cfg, conn))
        return
    if args.link_predecessors:
        result = link_predecessors(cfg, conn, dry_run=args.dry_run)
        print(f"\ncandidates {result['candidates']}  fetch_failed "
              f"{result.get('fetch_failed', 0)}  successors "
              f"{result['successors']}  linked {len(result['linked'])}  "
              f"unresolved {len(result['unresolved'])}")
        for row in result["linked"]:
            print(f"  linked   {row['ticker']:<8} {row['successor']} <- "
                  f"{row['predecessor']}  {row['name']}")
        for row in result["unresolved"]:
            print(f"  UNRESOLVED {row['ticker']:<6} {row['successor']} "
                  f"{row['name']}")
            for reason in row["reasons"][:3]:
                print(f"      {reason}")
        return
    if args.build_universe:
        build_universe(cfg, conn, force=args.force)
    if args.universe or args.tickers:
        tickers = ([t.strip().upper() for t in args.tickers.split(",") if t.strip()]
                   if args.tickers else None)
        collect_many(cfg, conn, tickers=tickers, resume=args.resume, force=args.force)



# --------------------------------------------------------------------------
# P2-03 — per-company submissions, including the older-filings pages
# --------------------------------------------------------------------------

def pages_to_fetch(files_block: list[dict], since_ts: int,
                   until_ts: int) -> list[str]:
    """Which older-filings pages overlap the study window.

    `filings.recent` is not a company's whole history: SEC keeps the most
    recent 1,000 filings or one year there, whichever is larger, and the rest
    in these pages. JPMorgan files enough that one year fills 25,937 records,
    so its `recent` block starts 2025-08-29 while the window opens 2024-09-01.
    Reading `recent` alone would drop eleven months for exactly the companies
    that file the most — with no error, just a company that appears to have had
    no events.

    Only overlapping pages are fetched: for JPMorgan that is 11 of 69. A page
    with a missing bound is fetched rather than guessed at, because a wrong
    skip is invisible in the output.
    """
    wanted = []
    for page in files_block or []:
        name = page.get("name")
        if not name:
            continue
        frm, to = page.get("filingFrom"), page.get("filingTo")
        if not frm or not to:
            wanted.append(name)   # fail safe: never narrow the window by guess
            continue
        if date_str_to_ts(to) >= since_ts and date_str_to_ts(frm) <= until_ts:
            wanted.append(name)
    return wanted


def records_from_block(block: dict) -> list[dict]:
    """SEC's parallel arrays -> one dict per filing.

    `recent` and each page store a list per field rather than a list of
    records. Zipping ragged arrays would silently truncate to the shortest and
    misalign every field after the gap, so the lengths are checked instead.
    """
    if not block:
        return []
    fields = list(block.keys())
    # An older-filings page is the bare block; the main file wraps it under
    # `filings.recent`. Mixing the two up otherwise surfaces as a KeyError
    # deep in the zip, which says nothing about what went wrong.
    if not all(isinstance(block[f], list) for f in fields):
        raise EdgarRequestError(
            f"expected a block of parallel arrays, got fields "
            f"{fields[:5]} whose values are not lists"
        )
    lengths = {len(block[f]) for f in fields}
    if len(lengths) > 1:
        raise EdgarRequestError(
            f"submissions block has ragged arrays: "
            f"{ {f: len(block[f]) for f in fields} } — zipping would misalign fields"
        )
    n = lengths.pop() if lengths else 0
    return [{f: block[f][i] for f in fields} for i in range(n)]


def fetch_company_filings(cfg: dict, client: EdgarClient, cik: str,
                          since_ts: int | None = None,
                          until_ts: int | None = None,
                          force: bool = False) -> list[dict]:
    """Every filing record for one CIK that could fall inside the window.

    Returns raw records — all form types, SEC's own field names and string
    values. The 8-K filter and the type conversions are P2-04, because page
    selection depends on all filings' dates: a page holding one 8-K among
    2,000 Form 4s must still be fetched.

    `force` bypasses the raw cache for this company's submissions file and
    every older-filings page it reads, so a stuck or corrupted cache entry can
    be deliberately refreshed instead of requiring someone to delete it from
    disk by hand.
    """
    since_ts = since_ts if since_ts is not None else date_str_to_ts(
        cfg["study_window"]["start"])
    until_ts = until_ts if until_ts is not None else date_str_to_ts(
        cfg["study_window"]["end"])

    payload = client.get_json(client.submissions_url(cik), force=force)
    filings = payload.get("filings", {})
    records = records_from_block(filings.get("recent", {}))

    for name in pages_to_fetch(filings.get("files", []), since_ts, until_ts):
        page = client.get_json(client.submissions_page_url(name), force=force)
        records.extend(records_from_block(page))

    # Consecutive pages share an edge date, so the same filing can arrive
    # twice. The upsert would absorb it, but a duplicated count reported as
    # fact would not be caught anywhere.
    seen: dict[str, dict] = {}
    for record in records:
        acc = record.get("accessionNumber")
        if acc and acc not in seen:
            seen[acc] = record
    return list(seen.values())


# --------------------------------------------------------------------------
# P2-04 — 8-K rows into `filings`
# --------------------------------------------------------------------------

def normalise_items(raw) -> str:
    """SEC's `items` field -> a clean comma-separated string of codes.

    Item codes are STRINGS and must stay strings. Through a float, `"1.01"`
    becomes `1.01` and still looks right — but `"1.10"` becomes `1.1`, and so
    does `"1.1"`, merging two different item codes into one with no error.
    The codes are the event taxonomy the whole study splits on, so a numeric
    one is raised on rather than quietly coerced.

    Whitespace is stripped because `"2.02, 9.01"` and `"2.02,9.01"` must not be
    two different values to the Phase 4 item filter.
    """
    if raw is None or raw == "":
        return ""                      # normal for some 8-Ks; "" not NULL so
                                       # a LIKE filter still behaves
    if not isinstance(raw, str):
        raise EdgarRequestError(
            f"item codes must be strings, got {type(raw).__name__} {raw!r} — "
            f"as a number 1.10 and 1.1 are the same value and two distinct "
            f"item codes would silently merge"
        )
    return ",".join(part.strip() for part in raw.split(",") if part.strip())


def filing_rows(cfg: dict, records: list[dict], cik: str,
                ticker: str | None) -> list[dict]:
    """Raw submission records -> `filings` rows, for the configured forms only.

    Forms are matched exactly against `edgar.forms`, never by prefix:
    `startswith("8-K")` would also swallow `8-K12B`, a different form.

    Blank dates become NULL rather than 0. A zero would read as 1 January 1970
    and become the oldest "event" in the study, which nothing downstream would
    flag as odd.

    A record that fails to parse (a malformed date, a missing required field)
    is skipped and counted rather than raised: SEC's older pages are less
    clean than `recent`, and one bad row must not throw away every good row
    already parsed for this company's whole history — that would make a
    company with 200 real 8-Ks indistinguishable from one that filed nothing.
    """
    keep = set(cfg["edgar"]["forms"])
    fetched = utc_now_ts()
    rows = []
    skipped = 0
    for record in records:
        if record.get("form") not in keep:
            continue
        try:
            acceptance = record.get("acceptanceDateTime") or None
            filing_date = record.get("filingDate") or None
            report_date = record.get("reportDate") or None
            rows.append({
                "accession_no": record["accessionNumber"],
                "cik": cik,
                "ticker": ticker,
                "form": record["form"],
                "items": normalise_items(record.get("items")),
                # The `Z` on acceptanceDateTime means UTC. Misread as local
                # time, every t0 in the study moves by four or five hours —
                # and by a different amount either side of a daylight-saving
                # change.
                "acceptance_utc": iso_utc_to_ts(acceptance) if acceptance else None,
                "filing_date_utc": date_str_to_ts(filing_date) if filing_date else None,
                "report_date_utc": date_str_to_ts(report_date) if report_date else None,
                "primary_doc": record.get("primaryDocument") or None,
                "fetched_utc": fetched,
            })
        except (ValueError, KeyError, EdgarRequestError) as exc:
            skipped += 1
            log.warning(
                "skipping malformed %s record for %s (%s), accession=%r: %s: %s",
                record.get("form"), ticker or "?", cik,
                record.get("accessionNumber", "?"), type(exc).__name__, exc,
            )
    if skipped:
        log.warning("%s (%s): skipped %d malformed record(s), kept %d",
                     ticker or "?", cik, skipped, len(rows))
    return rows


def collect_company(cfg: dict, conn, client: EdgarClient, cik: str,
                    ticker: str | None, force: bool = False) -> tuple[int, int]:
    """Fetch one company's submissions and store its 8-K rows.

    Returns `(records fetched, new rows)`. The record count is what the
    run-level guard watches: a company with no 8-Ks is ordinary, but a company
    with no records at all means the endpoint gave us nothing.
    """
    records = fetch_company_filings(cfg, client, cik, force=force)
    rows = filing_rows(cfg, records, cik, ticker)
    new = db.upsert_filings(conn, rows)
    log.info("%s (%s): %d records fetched, %d %s rows, %d new",
             ticker or "?", cik, len(records), len(rows),
             "/".join(cfg["edgar"]["forms"]), new)
    return len(records), new


# --------------------------------------------------------------------------
# P2-05 — running it over many companies, and refusing to lie about it
# --------------------------------------------------------------------------

#: Namespace for this collector's rows in `fetch_state`.
FETCH_SOURCE = "edgar"


def collect_many(cfg: dict, conn, client: EdgarClient | None = None,
                 tickers: list[str] | None = None,
                 resume: bool = False, force: bool = False) -> int:
    """Collect filings for many companies. Returns new rows written.

    Per-company failures are logged and the run continues — one 404 must not
    cost the other 6,053 companies. The guard is run-level for the same reason
    P1-15 made the news guard run-level: zero 8-Ks for ONE company is ordinary
    (plenty of small companies file none in two years), while zero records
    across EVERY company means the endpoint is broken.

    Every company's outcome is written to `fetch_state` and committed as it
    happens, so `resume=True` continues an interrupted run instead of
    restarting. `KeyboardInterrupt` is deliberately not caught here — `except
    Exception` does not cover it — so Ctrl-C stops the run with everything
    collected so far already committed.
    """
    client = client or EdgarClient(cfg)
    companies = db.companies_for_collection(conn, tickers)
    if not companies:
        if tickers:
            total = conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
            raise SystemExit(
                f"no company in `companies` matches --tickers {tickers} "
                f"(table holds {total} companies) — check for a typo, or "
                f"that the ticker is actually in `edgar.exchanges`. "
                f"`--build-universe` will not fix a wrong ticker."
            )
        raise SystemExit(
            "companies table is empty — run "
            "`python -m src.collectors.edgar --build-universe` first."
        )

    skip = db.completed_keys(conn, FETCH_SOURCE) if resume else set()
    if skip:
        log.info("resume: skipping %d companies already collected",
                 sum(1 for c in companies if c["cik"] in skip))

    total_records = total_new = failed = attempted = 0
    for company in companies:
        cik, ticker = company["cik"], company["ticker"]
        if cik in skip:
            continue
        attempted += 1
        try:
            records, new = collect_company(cfg, conn, client, cik, ticker, force=force)
        except Exception as exc:
            failed += 1
            log.exception("failed to collect %s (%s) — continuing", ticker, cik)
            db.set_fetch_state(conn, FETCH_SOURCE, cik, "failed",
                               error=f"{type(exc).__name__}: {exc}"[:500])
            continue
        # After the upsert, never before: a crash between the two re-fetches one
        # company, which is free. The reverse order would mark a company done
        # whose rows never landed.
        db.set_fetch_state(conn, FETCH_SOURCE, cik, "ok",
                           records=records, rows_written=new)
        total_records += records
        total_new += new

    n_filings = conn.execute("SELECT COUNT(*) FROM filings").fetchone()[0]
    log.info("Done. %d companies attempted (%d failed); %d records parsed, "
             "%d new rows; filings table now holds %d.",
             attempted, failed, total_records, total_new, n_filings)

    # The silent-failure guard. A 200 carrying redirect HTML already raises in
    # get_json; this catches the other shape of the same failure — every
    # response valid JSON, and nothing in any of them. `total_records` (every
    # form fetched) catches a dead endpoint; `total_new` (actual 8-K rows
    # written) catches the narrower case where EDGAR answers with real data
    # for every company but none of it matches `edgar.forms` — a config typo
    # or a schema change would otherwise leave `filings` frozen forever with
    # every run exiting 0.
    if cfg["logging"]["fail_on_zero_records"]:
        if attempted and failed == attempted:
            raise SystemExit(
                f"EVERY one of {attempted} companies failed — EDGAR is not "
                f"answering. Do not treat this run as successful."
            )
        if attempted and total_records == 0:
            raise SystemExit(
                f"ZERO records parsed across {attempted} companies — a 200 "
                f"response carrying nothing usable. Do not treat this run as "
                f"successful."
            )
        if attempted and total_records > 0 and total_new == 0:
            raise SystemExit(
                f"{total_records} records parsed across {attempted} "
                f"companies but ZERO matched edgar.forms {cfg['edgar']['forms']} "
                f"— check that against EDGAR's current schema. Do not treat "
                f"this run as successful."
            )
    elif attempted and total_records == 0:
        log.error("ZERO records parsed across %d companies "
                  "(logging.fail_on_zero_records is off)", attempted)
    elif attempted and total_new == 0:
        log.error("%d records parsed across %d companies but ZERO matched "
                  "edgar.forms %s (logging.fail_on_zero_records is off)",
                  total_records, attempted, cfg["edgar"]["forms"])
    return total_new


# --------------------------------------------------------------------------
# P2-10 — the sanity report
# --------------------------------------------------------------------------

def acceptance_hour_histogram(conn, start_ts: int, end_ts: int) -> list[dict]:
    """Filings per hour-of-day of acceptance, UTC, with the New York equivalent.

    This is the evidence the whole t0 correction rests on. The plan asserts
    8-Ks cluster after the US close while the press release that moved the
    market went out earlier; if this table is flat, the project's headline
    contribution has no basis.

    New York is computed from a real timestamp in each hour rather than a fixed
    -4 offset, or half the year is wrong by an hour (UI-context rule 7: every
    timestamp displays with its timezone).
    """
    rows = conn.execute(
        """SELECT CAST(strftime('%H', acceptance_utc, 'unixepoch') AS INTEGER) AS hour,
                  COUNT(*) AS n,
                  MIN(acceptance_utc) AS sample_ts
           FROM filings
           WHERE acceptance_utc IS NOT NULL
             AND acceptance_utc BETWEEN ? AND ?
           GROUP BY hour ORDER BY hour""",
        (start_ts, end_ts),
    ).fetchall()
    total = sum(r["n"] for r in rows) or 1
    return [{
        "hour_utc": r["hour"],
        "n": r["n"],
        "pct": 100.0 * r["n"] / total,
        "new_york": new_york_label(r["hour"]),
    } for r in rows]


def new_york_label(hour_utc: int) -> str:
    """'16:00 EDT / 15:00 EST' for a UTC hour.

    Both are shown because the US moves its clocks and the study window spans
    the change: the same UTC hour is 16:00 in July and 15:00 in January.
    Printing one would be wrong for half the data — and UI-context rule 7 makes
    a bare hour a bug.
    """
    ny = ZoneInfo("America/New_York")
    labels = []
    for month in (7, 1):                      # a summer and a winter reference
        ref = datetime(2026, month, 15, hour_utc, 0, tzinfo=timezone.utc)
        local = ref.astimezone(ny)
        labels.append(f"{local:%H:%M %Z}")
    return " / ".join(labels)


def item_code_counts(cfg: dict, conn, start_ts: int, end_ts: int) -> list[dict]:
    """One row per ITEM CODE, not per combination.

    `items` holds '2.02,9.01', so a naive GROUP BY counts combinations and
    hides how often each event type actually occurs. Scheduled codes are
    marked: UI-context rule 4 forbids pooling scheduled with unscheduled, and
    an unmarked table invites exactly that.
    """
    scheduled = set(cfg["items"]["scheduled"])
    excluded = set(cfg["items"]["exclude"])
    counts: dict[str, int] = {}
    for (items,) in conn.execute(
        """SELECT items FROM filings
           WHERE acceptance_utc BETWEEN ? AND ? AND items IS NOT NULL""",
        (start_ts, end_ts),
    ):
        for code in items.split(","):
            code = code.strip()
            if code:
                counts[code] = counts.get(code, 0) + 1
    return [
        {"item": code, "n": n,
         "scheduled": code in scheduled,
         "excluded": code in excluded}
        for code, n in sorted(counts.items(), key=lambda kv: -kv[1])
    ]


def filings_per_company(conn, start_ts: int, end_ts: int) -> dict:
    counts = [r[0] for r in conn.execute(
        """SELECT COUNT(*) FROM filings
           WHERE acceptance_utc BETWEEN ? AND ?
           GROUP BY cik""", (start_ts, end_ts))]
    if not counts:
        return {"companies": 0}
    counts.sort()
    return {
        "companies": len(counts),
        "min": counts[0],
        "median": counts[len(counts) // 2],
        "max": counts[-1],
        "mean": round(sum(counts) / len(counts), 1),
    }


def ciks_with_no_history_before_the_window(conn, start_ts: int) -> list[dict]:
    """Companies whose EARLIEST filing of any date is after the window opened.

    The detector for issue 18. SEC's ticker map points a ticker at the CIK that
    holds it TODAY, so a company that reorganised mid-window has its earlier
    filings under a predecessor CIK that carries no ticker and is never
    fetched. ExxonMobil is the known case: `0002115436` (ExxonMobil Holdings
    Corp) has nothing before 2026-07-07, while `0000034088` holds the history.

    A first draft flagged companies merely quiet for 90 days, which caught 502
    companies — mostly ordinary firms that file a few times a year. Asking
    instead whether the CIK existed AT ALL before the window is far sharper:
    it cannot miss a reorganisation, and its false positives are genuine new
    registrants (IPOs, SPACs) rather than every quiet company.

    Still a pointer, not a verdict. P2-11 decides the rule.
    """
    return [
        {"cik": r["cik"], "ticker": r["ticker"],
         "first_filing": ts_to_dt(r["first_ts"]).date().isoformat(),
         "n": r["n"]}
        for r in conn.execute(
            """SELECT cik, ticker, MIN(acceptance_utc) AS first_ts, COUNT(*) AS n
               FROM filings
               GROUP BY cik
               HAVING first_ts > ?
               ORDER BY n DESC""", (start_ts,))
    ]


def filings_report(cfg: dict, conn) -> dict:
    """Every figure in the sanity report, as data. Printing is separate."""
    start_ts = date_str_to_ts(cfg["study_window"]["start"])
    end_ts = date_str_to_ts(cfg["study_window"]["end"])
    total = conn.execute("SELECT COUNT(*) FROM filings").fetchone()[0]
    in_window = conn.execute(
        "SELECT COUNT(*) FROM filings WHERE acceptance_utc BETWEEN ? AND ?",
        (start_ts, end_ts)).fetchone()[0]
    return {
        "window": (cfg["study_window"]["start"], cfg["study_window"]["end"]),
        "total_rows": total,
        "in_window": in_window,
        # `filings` deliberately keeps 8-Ks from outside the window (P2-04),
        # so showing only one of these two figures would mislead either way.
        "outside_window": total - in_window,
        "no_acceptance_time": conn.execute(
            "SELECT COUNT(*) FROM filings WHERE acceptance_utc IS NULL"
        ).fetchone()[0],
        "hours": acceptance_hour_histogram(conn, start_ts, end_ts),
        "items": item_code_counts(cfg, conn, start_ts, end_ts),
        "per_company": filings_per_company(conn, start_ts, end_ts),
        "no_prior_history": ciks_with_no_history_before_the_window(
            conn, start_ts),
    }


def print_filings_report(report: dict) -> None:
    """Human-readable rendering. Wording follows UI-context: no implication of
    intent, scheduled never pooled with unscheduled, timezones always named."""
    if report["total_rows"] == 0:
        print("filings table is EMPTY — run "
              "`python -m src.collectors.edgar --universe` first.")
        return

    w0, w1 = report["window"]
    print(f"\nFILINGS SANITY REPORT   study window {w0} .. {w1}")
    print(f"  rows in table {report['total_rows']:>9,}")
    print(f"  in window     {report['in_window']:>9,}")
    print(f"  outside       {report['outside_window']:>9,}   (kept on purpose; "
          f"Phase 4 filters)")
    print(f"  no acceptance {report['no_acceptance_time']:>9,}")

    print("\n  ACCEPTANCE HOUR (UTC)  -- the evidence t0 is not the filing time")
    for row in sorted(report["hours"], key=lambda r: -r["n"])[:8]:
        bar = "#" * int(row["pct"] / 2)
        print(f"    {row['hour_utc']:02d}:00 UTC = {row['new_york']:>9}  "
              f"{row['n']:>7,}  {row['pct']:5.1f}%  {bar}")

    print("\n  ITEM CODES  (scheduled marked -- never pooled with unscheduled)")
    for row in report["items"][:10]:
        tag = "scheduled" if row["scheduled"] else ""
        tag = "excluded" if row["excluded"] else tag
        print(f"    {row['item']:>6}  {row['n']:>7,}   {tag}")

    p = report["per_company"]
    print(f"\n  FILINGS PER COMPANY  companies {p['companies']:,}  "
          f"min {p.get('min')}  median {p.get('median')}  "
          f"mean {p.get('mean')}  max {p.get('max')}")

    new_ciks = report["no_prior_history"]
    print(f"\n  CIKs WITH NO FILING HISTORY BEFORE THE WINDOW: "
          f"{len(new_ciks):,} companies")
    print("    a pointer for P2-11: a company that reorganised mid-window has")
    print("    its earlier filings under a predecessor CIK that carries no")
    print("    ticker and is never fetched. New registrants look the same.")
    for row in new_ciks[:5]:
        print(f"    {row['ticker'] or '?':<8} {row['cik']}  "
              f"first {row['first_filing']}  ({row['n']} filings)")
    print()


# --------------------------------------------------------------------------
# P2-11 — predecessor CIKs for companies that reorganised
# --------------------------------------------------------------------------

#: Words that say nothing about which company this is. Dropped before matching,
#: so `ExxonMobil Holdings Corp` and `EXXON MOBIL CORP` reduce to the same stem.
_LEGAL_SUFFIX = re.compile(
    r"\b(corp|corporation|inc|incorporated|llc|ltd|limited|co|company|plc|"
    r"holdings?|group|the|new|sa|nv|ag|lp|trust)\b", re.I)

#: Below this length a stem matches half the register ("bp", "ge"), so a prefix
#: match on it would be worse than no match at all.
MIN_STEM_PREFIX = 5


def name_stem(name: str) -> str:
    """A company name reduced to the part that identifies it."""
    return re.sub(r"[^a-z0-9]", "", _LEGAL_SUFFIX.sub("", (name or "").lower()))


def _submission_records(client: EdgarClient, submissions: dict) -> list[dict]:
    """Every filing record in an already-fetched submissions payload.

    Includes `filings.files` pages, not just `recent` — the same reason
    `fetch_company_filings` (P2-03) pages at all: `recent` covers only the
    most recent 1,000 filings or one year, and for an actively-traded company
    a marker filing (8-K12B) or a predecessor's prior 8-Ks can have rolled off
    it. Unlike `fetch_company_filings`, every page is fetched rather than only
    the ones overlapping the study window: these checks need the CIK's whole
    history, since a reorganisation or a predecessor's last 8-K can land
    outside it.

    A page that cannot be fetched is logged and skipped rather than allowed to
    propagate: an unguarded `get_json` here crashed the whole
    `--link-predecessors` run on a single transient 503 for one candidate CIK,
    unlike every sibling fetch in this module. Skipping fails in the SAFE
    direction — a missing page can only hide a marker, so `is_successor`
    answers False and the link is merely missed, never wrongly attributed,
    which is the trade this module explicitly chooses ("a wrong link silently
    attributes another company's 8-Ks to this ticker, which is worse than the
    gap it is meant to close").
    """
    filings = submissions.get("filings", {})
    records = records_from_block(filings.get("recent", {}))
    for page in filings.get("files", []) or []:
        name = page.get("name")
        if not name:
            continue
        try:
            block = client.get_json(client.submissions_page_url(name))
        except EdgarRequestError as exc:
            log.warning("submissions page %s unavailable (%s) — skipping it; "
                        "this CIK's filing history is incomplete for this "
                        "check, so a marker on that page cannot be seen",
                        name, exc)
            continue
        records.extend(records_from_block(block))
    return records


def is_successor(cfg: dict, submissions: dict,
                 client: EdgarClient | None = None) -> bool:
    """Did this CIK file a form declaring it continues another company?

    Form 8-K12B is "registration of securities of successor issuers". It is the
    only unambiguous marker available without reading filing text: of the 423
    CIKs with no history before the window, exactly 11 filed one.

    Pass `client` to also check `filings.files` pages, not just `recent` — for
    a heavy filer, the marker can have rolled off `recent` since it was filed.
    Omitting `client` checks `recent` only (the historical, pre-pagination
    behaviour), which is exact whenever the marker is still within `recent`.
    """
    if client is not None:
        records = _submission_records(client, submissions)
        forms = {r.get("form") for r in records}
    else:
        forms = set(submissions.get("filings", {}).get("recent", {}).get("form", []))
    return bool(forms & set(cfg["edgar"]["successor_forms"]))


def load_cik_lookup(cfg: dict, client: EdgarClient) -> dict[str, list[tuple[str, str]]]:
    """SEC's full name->CIK register, keyed by name stem.

    40 MB, fetched once and cached like every other response. It is the only
    place a predecessor can be found by name, because a CIK with no ticker is
    absent from `company_tickers_exchange.json` by construction — which is the
    whole reason it was never collected.
    """
    raw = client.get_bytes(cfg["edgar"]["cik_lookup_url"])
    lookup: dict[str, list[tuple[str, str]]] = {}
    for line in raw.decode("latin-1").splitlines():
        line = line.strip().rstrip(":")
        if not line or ":" not in line:
            continue
        name, _, cik = line.rpartition(":")
        if not cik.isdigit():
            continue
        lookup.setdefault(name_stem(name), []).append((name, cik.zfill(10)))
    return lookup


def propose_predecessors(name: str, lookup: dict, exclude_cik: str
                         ) -> list[tuple[str, str]]:
    """Candidate (name, cik) pairs for a successor's predecessor.

    Proposes only. An exact stem match first; failing that, the longest stem in
    the register that this name starts with. Names are weak evidence — five of
    eleven real cases match exactly and two of those return two candidates — so
    everything here is checked by `verify_predecessor` before it is believed.
    """
    stem = name_stem(name)
    if not stem:
        return []
    exact = [(n, c) for n, c in lookup.get(stem, []) if c != exclude_cik]
    if exact:
        return exact
    best: list[tuple[str, str]] = []
    best_len = 0
    for other, entries in lookup.items():
        if (len(other) >= MIN_STEM_PREFIX and other != stem
                and stem.startswith(other) and len(other) > best_len):
            candidates = [(n, c) for n, c in entries if c != exclude_cik]
            if candidates:
                best, best_len = candidates, len(other)
    return best


def verify_predecessor(cfg: dict, conn, client: EdgarClient, candidate_cik: str,
                       successor_sic: str | None, window_start: int
                       ) -> tuple[bool, str]:
    """Is this candidate really the predecessor? Returns (ok, reason).

    A wrong link silently attributes another company's 8-Ks to this ticker,
    which is worse than the gap it is meant to close. So every condition has to
    hold, and a candidate that cannot be confirmed is reported rather than
    guessed at.
    """
    if conn.execute("SELECT 1 FROM companies WHERE cik = ? AND successor_cik IS NULL",
                    (candidate_cik,)).fetchone():
        return False, "already has a ticker of its own"
    try:
        submissions = client.get_json(client.submissions_url(candidate_cik))
    except EdgarRequestError as exc:
        return False, f"submissions unavailable ({exc})"
    # NOT checked: `submissions["tickers"]`. A predecessor's own record keeps
    # listing the ticker after a reorganisation — both Columbia Financial CIKs
    # claim CLBK, and both Uranium Royalty CIKs claim UROY. The authority on
    # who holds a ticker TODAY is `company_tickers_exchange.json`, which is
    # what `companies` was built from and what the check above uses. Trusting
    # the submissions field here rejected two real predecessors.
    if successor_sic and submissions.get("sic") != successor_sic:
        return False, (f"SIC {submissions.get('sic')} != successor's "
                       f"{successor_sic}")
    records = _submission_records(client, submissions)
    keep = set(cfg["edgar"]["forms"])
    prior = [r for r in records
             if r.get("form") in keep and r.get("acceptanceDateTime")
             and iso_utc_to_ts(r["acceptanceDateTime"]) < window_start]
    if not prior:
        return False, "no 8-K filings before the window — a stub, not a predecessor"
    return True, f"{len(prior)} 8-K(s) before the window"


def link_predecessors(cfg: dict, conn, client: EdgarClient | None = None,
                      dry_run: bool = False) -> dict:
    """Find reorganised companies and add their predecessor CIKs to `companies`.

    A resolved predecessor is stored carrying the SUCCESSOR'S TICKER, with
    `successor_cik` naming what it feeds, so the existing collector picks it up
    unchanged and its filings land under the right ticker.
    """
    client = client or EdgarClient(cfg)
    window_start = date_str_to_ts(cfg["study_window"]["start"])

    candidates = ciks_with_no_history_before_the_window(conn, window_start)
    successors, unresolved, linked = [], [], []
    fetch_failed = 0
    for row in candidates:
        try:
            submissions = client.get_json(client.submissions_url(row["cik"]))
        except EdgarRequestError as exc:
            # A bare `continue` here made a total EDGAR outage during this run
            # look identical to "no reorganised companies" — same "successors
            # 0" line, exit 0, no signal anywhere. Count and log it instead,
            # consistent with how `collect_many` reports per-company failures.
            fetch_failed += 1
            log.warning("could not fetch submissions for candidate CIK %s "
                        "(%s): %s", row["cik"], row["ticker"], exc)
            continue
        if is_successor(cfg, submissions, client=client):
            successors.append((row, submissions))

    log.info("%d candidate CIK(s) with no prior history; %d fetch failure(s); "
             "%d filed %s", len(candidates), fetch_failed, len(successors),
             "/".join(cfg["edgar"]["successor_forms"]))
    if candidates and fetch_failed == len(candidates):
        # Every single candidate fetch failed — EDGAR is not answering, not
        # "nothing to link this run". Same class of guard as `collect_many`'s
        # "EVERY one of N companies failed".
        raise SystemExit(
            f"EVERY one of {len(candidates)} candidate CIK fetch(es) failed "
            f"— EDGAR is not answering. Do not treat this run's "
            f"'successors 0' as a real result."
        )
    if not successors:
        return {"candidates": len(candidates), "successors": 0,
                "fetch_failed": fetch_failed, "linked": [], "unresolved": []}

    lookup = load_cik_lookup(cfg, client)
    for row, submissions in successors:
        name = submissions.get("name", "")
        proposals = propose_predecessors(name, lookup, row["cik"])
        accepted = []
        reasons = []
        for cand_name, cand_cik in proposals[:20]:
            ok, why = verify_predecessor(cfg, conn, client, cand_cik,
                                         submissions.get("sic"), window_start)
            reasons.append(f"{cand_name} ({cand_cik}): {why}")
            if ok:
                accepted.append((cand_name, cand_cik))
        if len(accepted) == 1:
            cand_name, cand_cik = accepted[0]
            linked.append({"ticker": row["ticker"], "successor": row["cik"],
                           "predecessor": cand_cik, "name": cand_name})
            if not dry_run:
                db.upsert_companies(conn, [{
                    "cik": cand_cik, "ticker": row["ticker"], "name": cand_name,
                    "exchange": None, "sic": submissions.get("sic"),
                    "in_universe": None, "adv_usd": None, "last_price": None,
                    "universe_as_of": window_start,
                    "successor_cik": row["cik"],
                }])
            log.info("linked %s: %s <- %s (%s)", row["ticker"], row["cik"],
                     cand_cik, cand_name)
        else:
            unresolved.append({"ticker": row["ticker"], "successor": row["cik"],
                               "name": name, "accepted": len(accepted),
                               "reasons": reasons})
            log.warning("UNRESOLVED %s (%s) %r: %d candidate(s) survived "
                        "verification — reported, not guessed",
                        row["ticker"], row["cik"], name, len(accepted))
    return {"candidates": len(candidates), "successors": len(successors),
            "fetch_failed": fetch_failed, "linked": linked, "unresolved": unresolved}


if __name__ == "__main__":
    main()
