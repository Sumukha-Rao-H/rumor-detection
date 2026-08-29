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
from pathlib import Path
from urllib.parse import urlparse

import requests

from src import db
from src.utils.config import load_config
from src.utils.ratelimit import Backoff, RateLimiter
from src.utils.timeutils import date_str_to_ts, iso_utc_to_ts, utc_now_ts

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
    parser.add_argument("--universe", action="store_true",
                        help="collect filings for every company in `companies`")
    parser.add_argument("--tickers",
                        help="comma-separated subset, e.g. TSLA,AAPL")
    parser.add_argument("--resume", action="store_true",
                        help="skip companies that already have filings stored")
    parser.add_argument("--force", action="store_true",
                        help="re-fetch instead of serving from the raw cache")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config()
    if not (args.build_universe or args.universe or args.tickers):
        parser.error("nothing to do — pass --build-universe, --universe or --tickers")

    conn = db.get_conn(cfg["paths"]["db"])
    if args.build_universe:
        build_universe(cfg, conn, force=args.force)
    if args.universe or args.tickers:
        tickers = ([t.strip().upper() for t in args.tickers.split(",") if t.strip()]
                   if args.tickers else None)
        collect_many(cfg, conn, tickers=tickers, resume=args.resume)



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
                          until_ts: int | None = None) -> list[dict]:
    """Every filing record for one CIK that could fall inside the window.

    Returns raw records — all form types, SEC's own field names and string
    values. The 8-K filter and the type conversions are P2-04, because page
    selection depends on all filings' dates: a page holding one 8-K among
    2,000 Form 4s must still be fetched.
    """
    since_ts = since_ts if since_ts is not None else date_str_to_ts(
        cfg["study_window"]["start"])
    until_ts = until_ts if until_ts is not None else date_str_to_ts(
        cfg["study_window"]["end"])

    payload = client.get_json(client.submissions_url(cik))
    filings = payload.get("filings", {})
    records = records_from_block(filings.get("recent", {}))

    for name in pages_to_fetch(filings.get("files", []), since_ts, until_ts):
        page = client.get_json(client.submissions_page_url(name))
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
    """
    keep = set(cfg["edgar"]["forms"])
    fetched = utc_now_ts()
    rows = []
    for record in records:
        if record.get("form") not in keep:
            continue
        acceptance = record.get("acceptanceDateTime") or None
        filing_date = record.get("filingDate") or None
        report_date = record.get("reportDate") or None
        rows.append({
            "accession_no": record["accessionNumber"],
            "cik": cik,
            "ticker": ticker,
            "form": record["form"],
            "items": normalise_items(record.get("items")),
            # The `Z` on acceptanceDateTime means UTC. Misread as local time,
            # every t0 in the study moves by four or five hours — and by a
            # different amount either side of a daylight-saving change.
            "acceptance_utc": iso_utc_to_ts(acceptance) if acceptance else None,
            "filing_date_utc": date_str_to_ts(filing_date) if filing_date else None,
            "report_date_utc": date_str_to_ts(report_date) if report_date else None,
            "primary_doc": record.get("primaryDocument") or None,
            "fetched_utc": fetched,
        })
    return rows


def collect_company(cfg: dict, conn, client: EdgarClient, cik: str,
                    ticker: str | None) -> tuple[int, int]:
    """Fetch one company's submissions and store its 8-K rows.

    Returns `(records fetched, new rows)`. The record count is what the
    run-level guard watches: a company with no 8-Ks is ordinary, but a company
    with no records at all means the endpoint gave us nothing.
    """
    records = fetch_company_filings(cfg, client, cik)
    rows = filing_rows(cfg, records, cik, ticker)
    new = db.upsert_filings(conn, rows)
    log.info("%s (%s): %d records fetched, %d %s rows, %d new",
             ticker or "?", cik, len(records), len(rows),
             "/".join(cfg["edgar"]["forms"]), new)
    return len(records), new


# --------------------------------------------------------------------------
# P2-05 — running it over many companies, and refusing to lie about it
# --------------------------------------------------------------------------

def collect_many(cfg: dict, conn, client: EdgarClient | None = None,
                 tickers: list[str] | None = None,
                 resume: bool = False) -> int:
    """Collect filings for many companies. Returns new rows written.

    Per-company failures are logged and the run continues — one 404 must not
    cost the other 6,053 companies. The guard is run-level for the same reason
    P1-15 made the news guard run-level: zero 8-Ks for ONE company is ordinary
    (plenty of small companies file none in two years), while zero records
    across EVERY company means the endpoint is broken.
    """
    client = client or EdgarClient(cfg)
    companies = db.companies_for_collection(conn, tickers)
    if not companies:
        raise SystemExit(
            "companies table is empty — run "
            "`python -m src.collectors.edgar --build-universe` first."
        )

    skip = db.ciks_with_filings(conn) if resume else set()
    if skip:
        log.info("resume: skipping %d companies that already have filings",
                 sum(1 for c in companies if c["cik"] in skip))

    total_records = total_new = failed = attempted = 0
    for company in companies:
        if company["cik"] in skip:
            continue
        attempted += 1
        try:
            records, new = collect_company(cfg, conn, client,
                                           company["cik"], company["ticker"])
        except Exception:
            failed += 1
            log.exception("failed to collect %s (%s) — continuing",
                          company["ticker"], company["cik"])
            continue
        total_records += records
        total_new += new

    n_filings = conn.execute("SELECT COUNT(*) FROM filings").fetchone()[0]
    log.info("Done. %d companies attempted (%d failed); %d records parsed, "
             "%d new rows; filings table now holds %d.",
             attempted, failed, total_records, total_new, n_filings)

    # The silent-failure guard. A 200 carrying redirect HTML already raises in
    # get_json; this catches the other shape of the same failure — every
    # response valid JSON, and nothing in any of them.
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
    elif attempted and total_records == 0:
        log.error("ZERO records parsed across %d companies "
                  "(logging.fail_on_zero_records is off)", attempted)
    return total_new


if __name__ == "__main__":
    main()
