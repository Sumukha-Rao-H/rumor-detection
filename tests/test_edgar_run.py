"""Run-level behaviour: the zero-record guard, resume, per-company failures.

The guard exists because of one specific 2026 failure mode — an HTTP 200
carrying redirect or rate-limit HTML. The process exits 0, a scheduler reports
success, and nothing is written for days.

It is run-level, not per-company, for the same reason P1-15 made the news guard
run-level: zero 8-Ks for ONE company is ordinary (plenty of small companies
file none in two years), while zero records across EVERY company means the
endpoint is broken.
"""

import copy
import json

import pytest

from src import db
from src.collectors.edgar import (
    EdgarClient, EdgarRequestError, collect_many, page_selection_floor_ts,
    pages_to_fetch,
)
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts, ts_to_dt


APPLE_8K = {
    "accessionNumber": "0000320193-26-000018", "form": "8-K",
    "items": "2.02,9.01", "acceptanceDateTime": "2026-07-30T20:30:28.000Z",
    "filingDate": "2026-07-30", "reportDate": "2026-07-30",
    "primaryDocument": "aapl-20260730.htm",
}


def submissions(*records) -> dict:
    """Records -> a submissions payload in SEC's parallel-array shape."""
    fields = sorted({f for r in records for f in r})
    return {"filings": {
        "recent": {f: [r.get(f, "") for r in records] for f in fields},
        "files": [],
    }}


class FakeClient:
    """Serves a payload (or raises) per CIK, and counts the calls."""

    def __init__(self, by_cik):
        self.by_cik = by_cik
        self.calls: list[str] = []

    def submissions_url(self, cik) -> str:
        return f"https://data.sec.gov/submissions/CIK{cik}.json"

    def submissions_page_url(self, name) -> str:
        return f"https://data.sec.gov/submissions/{name}"

    def get_json(self, url, force=False):
        cik = url.rsplit("CIK", 1)[-1].removesuffix(".json")
        self.calls.append(cik)
        result = self.by_cik[cik]
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


@pytest.fixture
def conn(tmp_path):
    c = db.get_conn(tmp_path / "test.db")
    db.upsert_companies(c, [
        {"cik": "0000320193", "ticker": "AAPL", "name": "Apple", "exchange": "Nasdaq"},
        {"cik": "0000789019", "ticker": "MSFT", "name": "Microsoft", "exchange": "Nasdaq"},
        {"cik": "0001318605", "ticker": "TSLA", "name": "Tesla", "exchange": "Nasdaq"},
    ])
    yield c
    c.close()


def flag_off(cfg) -> dict:
    c = copy.deepcopy(cfg)
    c["logging"]["fail_on_zero_records"] = False
    return c


# -- the Done-when ---------------------------------------------------------

def test_redirect_html_raises_instead_of_reporting_success(cfg, tmp_path):
    """A 200 carrying SEC's rate-threshold page must not pass as success."""
    html = (b"<html><head><title>SEC.gov | Request Rate Threshold Exceeded"
            b"</title></head><body>...</body></html>")

    class HtmlSession:
        headers: dict = {}

        def get(self, url, timeout=None, allow_redirects=None):
            class R:
                status_code = 200
                content = html
            return R()

    c = copy.deepcopy(cfg)
    c["paths"]["edgar_raw"] = str(tmp_path / "raw")
    c["edgar"]["min_interval_s"] = 0.0
    client = EdgarClient(c, session=HtmlSession())

    with pytest.raises(EdgarRequestError, match="non-JSON"):
        client.get_json("https://data.sec.gov/submissions/CIK0000320193.json")
    assert not client.cache_path(
        "https://data.sec.gov/submissions/CIK0000320193.json").exists(), (
        "caching the HTML would serve it to every later run without a request")


def test_a_run_of_only_redirect_html_fails_loudly(cfg, conn):
    """Same failure seen from the run: every company errors, so the run raises."""
    err = EdgarRequestError("EDGAR returned non-JSON for ... — cache entry discarded")
    client = FakeClient({"0000320193": err, "0000789019": err, "0001318605": err})
    with pytest.raises(SystemExit, match="EVERY one of 3"):
        collect_many(cfg, conn, client=client)


# -- the guard proper ------------------------------------------------------

def test_zero_records_across_the_whole_run_raises(cfg, conn):
    """Valid JSON from every company, and nothing in any of it."""
    empty = {"filings": {"recent": {}, "files": []}}
    client = FakeClient({c: empty for c in
                         ("0000320193", "0000789019", "0001318605")})
    with pytest.raises(SystemExit, match="ZERO records"):
        collect_many(cfg, conn, client=client)


def test_a_company_with_no_8ks_is_not_a_failure(cfg, conn):
    """Small companies file none in two years. That is data, not an outage."""
    form4 = {"accessionNumber": "x-1", "form": "4", "items": "",
             "acceptanceDateTime": "2026-01-02T20:00:00.000Z",
             "filingDate": "2026-01-02", "reportDate": "2026-01-02",
             "primaryDocument": "f.xml"}
    client = FakeClient({
        "0000320193": submissions(APPLE_8K),
        "0000789019": submissions(form4),
        "0001318605": submissions(form4),
    })
    assert collect_many(cfg, conn, client=client) == 1
    assert conn.execute("SELECT COUNT(*) FROM filings").fetchone()[0] == 1


def test_one_failing_company_does_not_abort_the_run(cfg, conn):
    client = FakeClient({
        "0000320193": submissions(APPLE_8K),
        "0000789019": EdgarRequestError("HTTP 404 from EDGAR for ... — not retried"),
        "0001318605": submissions({**APPLE_8K, "accessionNumber": "t-1"}),
    })
    assert collect_many(cfg, conn, client=client) == 2, (
        "one 404 must not cost the other companies")
    assert len(client.calls) == 3


def test_the_guard_respects_the_config_flag(cfg, conn):
    empty = {"filings": {"recent": {}, "files": []}}
    client = FakeClient({c: empty for c in
                         ("0000320193", "0000789019", "0001318605")})
    assert collect_many(flag_off(cfg), conn, client=client) == 0


# -- selection and resume --------------------------------------------------

def test_tickers_selection_collects_only_those(cfg, conn):
    client = FakeClient({"0001318605": submissions(APPLE_8K)})
    collect_many(cfg, conn, client=client, tickers=["TSLA"])
    assert client.calls == ["0001318605"]


def test_resume_skips_companies_already_collected(cfg, conn):
    payloads = {
        "0000320193": submissions(APPLE_8K),
        "0000789019": submissions({**APPLE_8K, "accessionNumber": "m-1"}),
        "0001318605": submissions({**APPLE_8K, "accessionNumber": "t-1"}),
    }
    first = FakeClient(payloads)
    collect_many(cfg, conn, client=first)
    assert len(first.calls) == 3

    second = FakeClient(payloads)
    collect_many(cfg, conn, client=second, resume=True)
    assert second.calls == [], "a completed run has nothing left to resume"


def test_resume_on_a_fresh_db_collects_everything(cfg, conn):
    client = FakeClient({c: submissions({**APPLE_8K, "accessionNumber": c})
                         for c in ("0000320193", "0000789019", "0001318605")})
    collect_many(cfg, conn, client=client, resume=True)
    assert len(client.calls) == 3


def test_resume_does_not_refetch_a_company_with_no_8ks(cfg, conn):
    """This was P2-05's known limitation, closed by P2-06.

    From `filings` alone, "fetched and files no 8-Ks" was indistinguishable
    from "never tried", so a quiet company was re-fetched on every resume.
    `fetch_state` records the outcome, so 'ok' with rows_written = 0 now means
    "do not come back".
    """
    form4 = {"accessionNumber": "x-1", "form": "4", "items": "",
             "acceptanceDateTime": "2026-01-02T20:00:00.000Z",
             "filingDate": "2026-01-02", "reportDate": "2026-01-02",
             "primaryDocument": "f.xml"}
    payloads = {"0000320193": submissions(APPLE_8K),
                "0000789019": submissions(form4),
                "0001318605": submissions(form4)}
    collect_many(cfg, conn, client=FakeClient(payloads))
    second = FakeClient(payloads)
    collect_many(cfg, conn, client=second, resume=True)
    assert second.calls == []


def test_universe_with_an_empty_companies_table_says_what_to_run(cfg, tmp_path):
    empty_db = db.get_conn(tmp_path / "empty.db")
    with pytest.raises(SystemExit, match="--build-universe"):
        collect_many(cfg, empty_db, client=FakeClient({}))
    empty_db.close()


def test_a_ticker_typo_is_not_reported_as_an_empty_table(cfg, conn):
    """`companies` holds 3 rows here; a typo'd ticker matching none of them
    used to raise the exact same 'table is empty' message as a genuinely
    empty table, sending the user toward `--build-universe` instead of toward
    their typo."""
    with pytest.raises(SystemExit) as exc:
        collect_many(cfg, conn, client=FakeClient({}), tickers=["ZZZZZINVALID"])
    msg = str(exc.value)
    assert "table is empty" not in msg, (
        "table has rows — the error must not claim otherwise")
    assert "ZZZZZINVALID" in msg
    assert "3 companies" in msg


# -- the total_new guard ----------------------------------------------------

def test_real_data_for_every_company_but_none_of_the_configured_forms_raises(cfg, conn):
    """A config typo in `edgar.forms` (wrong case, a schema change) can leave
    every company answering with real, non-empty data while zero of it
    matches — `total_records` alone would never catch this."""
    form4 = {"accessionNumber": "x-1", "form": "4", "items": "",
             "acceptanceDateTime": "2026-01-02T20:00:00.000Z",
             "filingDate": "2026-01-02", "reportDate": "2026-01-02",
             "primaryDocument": "f.xml"}
    client = FakeClient({c: submissions(form4) for c in
                         ("0000320193", "0000789019", "0001318605")})
    with pytest.raises(SystemExit, match="ZERO matched"):
        collect_many(cfg, conn, client=client)


def test_a_clean_rerun_of_a_complete_table_is_not_a_failure(cfg, conn):
    """The gap that let the guard read the wrong counter for so long.

    Every other guard test starts from an empty `filings` table, where "rows
    parsed" and "rows new" happen to be the same number. On the second run of
    an already-complete table they diverge: every 8-K is parsed again and the
    upsert reports none of them as new. The guard used to watch the new count,
    so a finished collection raised "ZERO matched edgar.forms" — factually
    false, since every record matched — every single time it was re-run.
    """
    payloads = {"0000320193": submissions(APPLE_8K),
                "0000789019": submissions({**APPLE_8K, "accessionNumber": "m-1"}),
                "0001318605": submissions({**APPLE_8K, "accessionNumber": "t-1"})}
    assert collect_many(cfg, conn, client=FakeClient(payloads)) == 3

    assert collect_many(cfg, conn, client=FakeClient(payloads)) == 0, (
        "nothing is new the second time — that is a complete table, not a "
        "broken endpoint")
    assert conn.execute("SELECT COUNT(*) FROM filings").fetchone()[0] == 3


def test_a_resumed_rerun_of_a_complete_table_is_also_silent(cfg, conn):
    """`--resume` masked the bug rather than fixing it: skipped companies never
    reach `attempted += 1`, so the guard was never armed. It must stay silent
    for the right reason now, and still collect nothing new."""
    payloads = {"0000320193": submissions(APPLE_8K),
                "0000789019": submissions({**APPLE_8K, "accessionNumber": "m-1"}),
                "0001318605": submissions({**APPLE_8K, "accessionNumber": "t-1"})}
    collect_many(cfg, conn, client=FakeClient(payloads))
    second = FakeClient(payloads)
    assert collect_many(cfg, conn, client=second, resume=True) == 0
    assert second.calls == []


def test_the_total_new_guard_respects_the_config_flag(cfg, conn):
    form4 = {"accessionNumber": "x-1", "form": "4", "items": "",
             "acceptanceDateTime": "2026-01-02T20:00:00.000Z",
             "filingDate": "2026-01-02", "reportDate": "2026-01-02",
             "primaryDocument": "f.xml"}
    client = FakeClient({c: submissions(form4) for c in
                         ("0000320193", "0000789019", "0001318605")})
    assert collect_many(flag_off(cfg), conn, client=client) == 0


# -- --force ----------------------------------------------------------------

def test_force_reaches_the_filings_collection_path(cfg, conn):
    """`--force` was wired only to `--build-universe`; a stuck or corrupted
    per-company submissions cache had no CLI-level fix. `force` must reach
    `client.get_json` on the filings path too."""
    seen_force = []

    class ForceCapturingClient(FakeClient):
        def get_json(self, url, force=False):
            seen_force.append(force)
            return super().get_json(url, force=force)

    client = ForceCapturingClient({"0000320193": submissions(APPLE_8K)})
    collect_many(cfg, conn, client=client, tickers=["AAPL"], force=True)
    assert seen_force and all(seen_force), (
        "force=True on collect_many must reach every client.get_json call")


# -- the page-selection floor (what the universe filter needs) --------------

def test_the_page_floor_reaches_back_past_the_window_for_the_universe(cfg):
    """`filings` feeds the study AND `universe.require_prior_8k`, which asks
    for an 8-K in the year BEFORE the window. Fetching only the window leaves
    that year unfetched for heavy filers and drops them from the universe."""
    start = date_str_to_ts(cfg["study_window"]["start"])
    floor = page_selection_floor_ts(cfg)
    assert floor == start - cfg["universe"]["prior_8k_lookback_days"] * 86400
    assert floor < start


def test_a_page_inside_the_prior_8k_lookback_is_selected(cfg):
    """The JPMorgan case, at `pages_to_fetch` level: a page that ends inside
    [start - prior_8k_lookback_days, start) holds exactly the prior 8-K the
    universe filter looks for, and used to be skipped as "before the window"."""
    start = date_str_to_ts(cfg["study_window"]["start"])
    end = date_str_to_ts(cfg["study_window"]["end"])
    floor = page_selection_floor_ts(cfg)
    prior = [{"name": "prior.json",
              "filingFrom": ts_to_dt(floor + 10 * 86400).strftime("%Y-%m-%d"),
              "filingTo": ts_to_dt(start - 10 * 86400).strftime("%Y-%m-%d")}]

    assert pages_to_fetch(prior, start, end) == [], (
        "keyed on the study window alone, this page is invisible")
    assert pages_to_fetch(prior, floor, end) == ["prior.json"]


def test_collect_many_actually_passes_the_widened_floor(cfg, conn):
    """The floor is only worth anything if the collection path uses it — the
    obvious "tidy-up" is to hand `fetch_company_filings` the study window."""
    seen = {}

    class PageCapturingClient(FakeClient):
        def get_json(self, url, force=False):
            if url.endswith("prior.json"):
                seen["fetched"] = True
                return {"form": ["8-K"],
                        "accessionNumber": ["prior-1"],
                        "items": ["1.01"],
                        "acceptanceDateTime": ["2025-01-02T20:00:00.000Z"],
                        "filingDate": ["2025-01-02"],
                        "reportDate": ["2025-01-02"],
                        "primaryDocument": ["p.htm"]}
            return super().get_json(url, force=force)

    start = date_str_to_ts(cfg["study_window"]["start"])
    floor = page_selection_floor_ts(cfg)
    payload = submissions(APPLE_8K)
    payload["filings"]["files"] = [{
        "name": "prior.json",
        "filingFrom": ts_to_dt(floor + 10 * 86400).strftime("%Y-%m-%d"),
        "filingTo": ts_to_dt(start - 10 * 86400).strftime("%Y-%m-%d")}]

    collect_many(cfg, conn, client=PageCapturingClient({"0000320193": payload}),
                 tickers=["AAPL"])
    assert seen.get("fetched"), (
        "the page holding the prior 8-K was never fetched — the universe "
        "filter will see no prior 8-K and drop this company")
    assert conn.execute(
        "SELECT COUNT(*) FROM filings WHERE accession_no = 'prior-1'"
    ).fetchone()[0] == 1


# -- the module has to actually run ----------------------------------------

def test_the_main_guard_is_the_last_thing_in_every_collector():
    """Caught for real during P2-05, before commit.

    `collect_many` was appended after `if __name__ == "__main__": main()`, so
    the module imported fine, every unit test passed — and running
    `python -m src.collectors.edgar --tickers AAPL` died with
    `NameError: name 'collect_many' is not defined`, because main() ran at the
    guard while the function below it did not exist yet.

    Nothing that imports the module can see this; only executing it can.
    """
    from pathlib import Path

    for name in ("edgar", "market", "news"):
        path = Path("src/collectors") / f"{name}.py"
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        assert lines[-2:] == ['if __name__ == "__main__":', "    main()"], (
            f"{path} must end with the __main__ guard — anything defined "
            f"below it does not exist when main() runs")
