"""SEC EDGAR full-text search — the confirmation source (plan §6.4).

Added 2026-08-05 after the first labeling pass produced 9 genuine TRUE labels
from 1,170 events. The cause was not the labeling rules but retrieval: the free
news APIs return mostly aggregators, and GDELT — the only one with real domains
across the whole window — is throttled to roughly one answered query in three.

EDGAR fixes the shortage at its root. For the claims this project cares about,
a company's own filing *is* the confirmation, and a better one than a headline
about it: an 8-K Item 2.01 is the acquisition completing, not a report that it
might. It is keyless, covers 2001 onward, and SEC permits ~10 requests a second
with a declared User-Agent, so all 1,010 event windows cost minutes rather than
the days GDELT wants.

What a filing gives us and what it does not:

  text     full-text search returns metadata, not prose. The readable part is
           the form type plus its 8-K item codes, which map almost one-to-one
           onto our claim types (1.03 bankruptcy, 2.01 acquisition completed,
           3.01 delisting notice). Those become the headline the labeler reads.
  time     `file_date` is a date with no clock. t_official drives the reward's
           early-commit bonus, so the timestamp is placed at
           `news.edgar_filing_hour_utc` — late in the US business day — which
           errs towards claiming we learned *later* than we did. Erring early
           would invent foresight; erring late only forfeits bonus.

Rows land in the same `news` table under api='edgar' with source_domain
'sec.gov', which is already tier 1 in the credibility whitelist, so the labeler
picks them up with no changes.

Usage:
  python -m src.collectors.edgar --ticker AAPL --start 2025-01-01 --end 2025-03-31
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import requests

from src import db
from src.utils.config import load_config
from src.utils.ratelimit import Backoff
from src.utils.timeutils import date_str_to_ts, ts_to_dt, utc_now_ts

log = logging.getLogger(__name__)

SEC_DOMAIN = "sec.gov"

# SEC's 8-K item taxonomy. A regulatory constant, not a tuning knob — the codes
# are what make a filing readable as a claim ("2.01" means an acquisition
# actually closed), so the labeler needs them spelled out.
ITEM_DESCRIPTIONS = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "1.05": "Material Cybersecurity Incident",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition",
    "2.03": "Creation of a Direct Financial Obligation",
    "2.04": "Triggering Events That Accelerate a Financial Obligation",
    "2.05": "Costs Associated with Exit or Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": "Notice of Delisting or Failure to Satisfy a Listing Rule",
    "3.02": "Unregistered Sales of Equity Securities",
    "3.03": "Material Modification to Rights of Security Holders",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financial Statements",
    "5.01": "Changes in Control of Registrant",
    "5.02": "Departure or Election of Directors or Officers",
    "5.03": "Amendments to Articles of Incorporation or Bylaws",
    "5.07": "Submission of Matters to a Vote of Security Holders",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}


class EdgarUnavailable(RuntimeError):
    """The query never completed — distinct from completing with no filings.

    Same hazard as the news collectors: §6.4 reads silence as FALSE, so a
    failed request must never be recorded as "this company filed nothing".
    """


def load_cik_map(cfg: dict, session: requests.Session,
                 refresh: bool = False) -> dict[str, str]:
    """Ticker -> zero-padded CIK, cached on disk.

    SEC publishes the whole mapping as one small file, so this is a single
    request rather than a lookup per ticker.
    """
    path = Path(cfg["paths"]["symbol_dir"]) / "sec_company_tickers.json"
    if refresh or not path.exists():
        resp = session.get(cfg["news"]["edgar_tickers_url"], timeout=60)
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(resp.json()))
    data = json.loads(path.read_text())
    return {row["ticker"].upper(): str(row["cik_str"]).zfill(10)
            for row in data.values()}


def fetch_filings(cfg: dict, session: requests.Session, cik: str,
                  start_ts: int, end_ts: int) -> list[dict]:
    """Full-text search hits for one company in one window."""
    ncfg = cfg["news"]
    params = {
        "ciks": cik,
        "forms": ",".join(ncfg["edgar_forms"]),
        "startdt": ts_to_dt(start_ts).strftime("%Y-%m-%d"),
        "enddt": ts_to_dt(end_ts).strftime("%Y-%m-%d"),
    }
    backoff = Backoff(base_s=2)
    for _ in range(4):
        try:
            resp = session.get(ncfg["edgar_base"], params=params, timeout=30)
            if resp.status_code >= 500 or resp.status_code == 429:
                # EDGAR returns a bare 500 intermittently for queries that
                # succeed on retry, so a 5xx here is noise, not an answer.
                backoff.sleep(f"HTTP {resp.status_code} from EDGAR")
                continue
            resp.raise_for_status()
            return resp.json().get("hits", {}).get("hits", [])
        except requests.RequestException as exc:
            backoff.sleep(f"EDGAR request failed: {exc}")
        except ValueError as exc:
            backoff.sleep(f"EDGAR non-JSON response: {exc}")
    raise EdgarUnavailable(f"EDGAR gave up after repeated failures for CIK {cik}")


def describe(hit: dict) -> str:
    """A filing rendered as a headline the labeler can reason about."""
    source = hit.get("_source", {})
    form = source.get("root_form") or source.get("file_type") or "filing"
    items = [i for i in (source.get("items") or []) if i in ITEM_DESCRIPTIONS]
    names = source.get("display_names") or []
    who = names[0].split("  (")[0].strip() if names else ""
    what = "; ".join(f"Item {i}: {ITEM_DESCRIPTIONS[i]}" for i in items)
    return " — ".join(part for part in (f"SEC {form}", what, who) if part)


def filing_url(hit: dict) -> str:
    """Canonical EDGAR archive URL, which doubles as the news table's key."""
    raw = hit.get("_id") or ""
    accession, _, document = raw.partition(":")
    cik = (hit.get("_source", {}).get("ciks") or ["0"])[0]
    return (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
            f"{accession.replace('-', '')}/{document or accession}")


def filings_to_rows(hits: list[dict], ticker: str, cfg: dict) -> list[tuple]:
    """EDGAR hits -> news rows (url, ticker, title, domain, ts, api)."""
    hour = int(cfg["news"]["edgar_filing_hour_utc"])
    rows = []
    for hit in hits:
        file_date = hit.get("_source", {}).get("file_date")
        if not file_date:
            continue
        try:
            seen_utc = date_str_to_ts(file_date) + hour * 3600
        except ValueError:
            continue
        rows.append((filing_url(hit), ticker, describe(hit), SEC_DOMAIN,
                     seen_utc, "edgar"))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", help="YYYY-MM-DD (default: now)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    session = requests.Session()
    session.headers["User-Agent"] = cfg["news"]["edgar_user_agent"]

    ticker = args.ticker.upper()
    cik = load_cik_map(cfg, session).get(ticker)
    if not cik:
        raise SystemExit(f"{ticker} has no CIK in SEC's ticker file")
    hits = fetch_filings(cfg, session, cik,
                         date_str_to_ts(args.start),
                         date_str_to_ts(args.end) if args.end else utc_now_ts())
    rows = filings_to_rows(hits, ticker, cfg)
    new = db.upsert_news(conn, rows)
    log.info("EDGAR %s (CIK %s): %d filings, %d new", ticker, cik, len(hits), new)
    for row in rows:
        log.info("  %s  %s", ts_to_dt(row[4]).strftime("%Y-%m-%d"), row[2])


if __name__ == "__main__":
    main()
