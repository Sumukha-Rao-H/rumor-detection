"""Predecessor-CIK linking — detection, proposal, and every rejection reason.

SEC's ticker map points a ticker at the CIK that holds it TODAY. A company
that reorganised mid-window has its filings split across two CIKs, and the
older one carries no ticker — so it is never in `companies`, never fetched,
and its 8-Ks are simply absent. ExxonMobil lost 15 of its 17 in-window
filings that way.

A wrong link is worse than the gap: it silently attributes another company's
8-Ks to this ticker. So the name rule only proposes, and every candidate has to
survive verification or be reported unresolved.
"""

import copy

import pytest

from src import db
from src.collectors import edgar
from src.collectors.edgar import (
    MAX_PREDECESSOR_CANDIDATES, EdgarRequestError, is_successor,
    link_predecessors, name_stem, propose_predecessors, verify_predecessor,
)
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts, iso_utc_to_ts


WINDOW_START = date_str_to_ts("2025-09-01")


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


@pytest.fixture
def conn(tmp_path):
    c = db.get_conn(tmp_path / "p.db")
    yield c
    c.close()


def submissions(forms, *, name="X", sic="2911", tickers=None,
                acceptances=None) -> dict:
    forms = list(forms)
    acceptances = acceptances or ["2020-01-02T20:00:00Z"] * len(forms)
    return {
        "name": name, "sic": sic, "tickers": tickers or [],
        "filings": {"recent": {
            "form": forms,
            "acceptanceDateTime": acceptances,
            "accessionNumber": [f"a-{i}" for i in range(len(forms))],
        }, "files": []},
    }


class FakeClient:
    def __init__(self, by_cik, pages=None):
        self.by_cik = by_cik
        #: page name -> bare parallel-array block, for pagination tests.
        self.pages = pages or {}

    def submissions_url(self, cik) -> str:
        return f"https://data.sec.gov/submissions/CIK{cik}.json"

    def submissions_page_url(self, name) -> str:
        return f"https://data.sec.gov/submissions/{name}"

    def get_json(self, url, force=False):
        name = url.rsplit("/", 1)[-1]
        if name in self.pages:
            return self.pages[name]
        cik = name.removeprefix("CIK").removesuffix(".json")
        result = self.by_cik[cik]
        if isinstance(result, Exception):
            raise result
        return result


# -- detection -------------------------------------------------------------

def test_a_cik_filing_8k12b_is_a_successor(cfg):
    """8-K12B is "registration of securities of successor issuers" — a company
    stating in a filing that it continues another."""
    assert is_successor(cfg, submissions(["8-K", "8-K12B", "4"]))


def test_an_ordinary_new_registrant_is_not(cfg):
    """412 of the 423 candidates are ordinary new listings and must be left alone."""
    assert not is_successor(cfg, submissions(["8-K", "10-Q", "4", "S-1"]))


def test_8k12g3_also_counts(cfg):
    assert is_successor(cfg, submissions(["8-K12G3"]))


def test_a_cik_with_no_filings_is_not_a_successor(cfg):
    assert not is_successor(cfg, {"filings": {"recent": {}}})


# -- name stems ------------------------------------------------------------

def test_name_stem_drops_legal_suffixes():
    assert name_stem("ExxonMobil Holdings Corp") == "exxonmobil"
    assert name_stem("Pinnacle Financial Partners, Inc.") == "pinnaclefinancialpartners"


def test_stem_matches_across_spacing_and_case():
    """`EXXON MOBIL CORP` and `ExxonMobil Holdings Corp` are the same company."""
    assert name_stem("EXXON MOBIL CORP") == name_stem("ExxonMobil Holdings Corp")


def test_an_exact_stem_match_is_proposed():
    lookup = {"exxonmobil": [("EXXON MOBIL CORP", "0000034088")]}
    assert propose_predecessors("ExxonMobil Holdings Corp", lookup, "0002115436") \
        == [("EXXON MOBIL CORP", "0000034088")]


def test_the_successor_never_proposes_itself():
    lookup = {"exxonmobil": [("ExxonMobil Holdings Corp", "0002115436")]}
    assert propose_predecessors("ExxonMobil Holdings", lookup, "0002115436") == []


def test_a_longest_prefix_is_proposed_when_no_exact_match():
    """`Paramount Skydance Corp` has no exact match; `Paramount` is the lead."""
    lookup = {"paramount": [("Paramount Global", "0000813828")],
              "para": [("PARA CO", "0000000001")]}
    assert propose_predecessors("Paramount Skydance Corp", lookup, "x") \
        == [("Paramount Global", "0000813828")]


def test_a_very_short_stem_is_not_used_as_a_prefix():
    """A two-letter stem matches half the register; a match on it is noise."""
    lookup = {"bp": [("BP CO", "0000000002")]}
    assert propose_predecessors("BPX Energy Holdings", lookup, "x") == []


# -- verification: every rejection reason ----------------------------------

def test_a_real_predecessor_is_accepted(cfg, conn):
    client = FakeClient({"0000034088": submissions(
        ["8-K", "8-K"], sic="2911",
        acceptances=["2024-02-02T20:00:00Z", "2025-02-02T20:00:00Z"])})
    ok, why = verify_predecessor(cfg, conn, client, "0000034088", "2911",
                                 WINDOW_START)
    assert ok and "2 8-K" in why


def test_a_candidate_with_no_prior_history_is_rejected(cfg, conn):
    """A same-named stub is not a predecessor."""
    client = FakeClient({"0001226649": submissions(
        ["8-K"], sic="2911", acceptances=["2026-02-02T20:00:00Z"])})
    ok, why = verify_predecessor(cfg, conn, client, "0001226649", "2911",
                                 WINDOW_START)
    assert not ok and "stub" in why


def test_a_candidate_with_a_different_sic_is_rejected(cfg, conn):
    """Paramount Group (REIT, 6798) is not Paramount Skydance (broadcast, 4833)."""
    client = FakeClient({"0001605607": submissions(
        ["8-K"], sic="6798", acceptances=["2020-02-02T20:00:00Z"])})
    ok, why = verify_predecessor(cfg, conn, client, "0001605607", "4833",
                                 WINDOW_START)
    assert not ok and "SIC" in why


def test_a_successor_with_no_sic_cannot_verify_anything(cfg, conn):
    """The SIC test used to be skipped when the SUCCESSOR had no `sic` of its
    own, which quietly reduced verification to "has no ticker" + "filed an 8-K
    before the window" — conditions thousands of CIKs satisfy. 480 of the 6,135
    cached submissions payloads (7.8%) carry an empty or absent `sic`, so this
    is a real path. Unverifiable has to be reported, not assumed."""
    client = FakeClient({"0001605607": submissions(
        ["8-K"], sic="6798", acceptances=["2020-02-02T20:00:00Z"])})
    for successor_sic in (None, ""):
        ok, why = verify_predecessor(cfg, conn, client, "0001605607",
                                     successor_sic, WINDOW_START)
        assert not ok
        assert "no SIC" in why


def test_a_candidate_that_already_has_its_own_ticker_is_rejected(cfg, conn):
    """If it were in `companies` it would already be collected."""
    db.upsert_companies(conn, [{"cik": "0000320193", "ticker": "AAPL",
                                "name": "Apple", "exchange": "Nasdaq"}])
    ok, why = verify_predecessor(cfg, conn, FakeClient({}), "0000320193",
                                 "3571", WINDOW_START)
    assert not ok and "ticker of its own" in why


def test_the_submissions_ticker_field_is_not_used_to_reject(cfg, conn):
    """A predecessor's own record keeps listing the ticker after a
    reorganisation — both Columbia Financial CIKs claim CLBK, and both Uranium
    Royalty CIKs claim UROY. Trusting that field rejected two real
    predecessors; `companies` is the authority on who holds a ticker today.
    """
    client = FakeClient({"0001723596": submissions(
        ["8-K"], sic="6035", tickers=["CLBK"],
        acceptances=["2020-02-02T20:00:00Z"])})
    ok, _ = verify_predecessor(cfg, conn, client, "0001723596", "6035",
                               WINDOW_START)
    assert ok


def test_an_unreachable_candidate_is_rejected_not_assumed(cfg, conn):
    client = FakeClient({"0000000009": EdgarRequestError("HTTP 404")})
    ok, why = verify_predecessor(cfg, conn, client, "0000000009", "2911",
                                 WINDOW_START)
    assert not ok and "unavailable" in why


def test_only_the_configured_forms_count_as_prior_history(cfg, conn):
    """A predecessor with a decade of Form 4s but no 8-K is not what we need."""
    client = FakeClient({"0000000010": submissions(
        ["4", "4"], sic="2911",
        acceptances=["2019-01-02T20:00:00Z", "2020-01-02T20:00:00Z"])})
    ok, why = verify_predecessor(cfg, conn, client, "0000000010", "2911",
                                 WINDOW_START)
    assert not ok and "stub" in why


# -- storage ---------------------------------------------------------------

def test_a_linked_predecessor_carries_the_successor_ticker(conn):
    """That is what makes the existing collector store its filings under XOM."""
    db.upsert_companies(conn, [
        {"cik": "0002115436", "ticker": "XOM", "name": "ExxonMobil Holdings",
         "exchange": "NYSE"},
        {"cik": "0000034088", "ticker": "XOM", "name": "EXXON MOBIL CORP",
         "exchange": None, "successor_cik": "0002115436"},
    ])
    rows = conn.execute(
        "SELECT cik, successor_cik FROM companies WHERE ticker = 'XOM' "
        "ORDER BY cik").fetchall()
    assert [tuple(r) for r in rows] == [
        ("0000034088", "0002115436"), ("0002115436", None)]


def test_predecessor_rows_are_excluded_from_company_counts(conn):
    """A predecessor carries its successor's ticker, so an unfiltered count
    double-counts every reorganised company."""
    db.upsert_companies(conn, [
        {"cik": "0002115436", "ticker": "XOM", "name": "New", "exchange": "NYSE"},
        {"cik": "0000034088", "ticker": "XOM", "name": "Old", "exchange": None,
         "successor_cik": "0002115436"},
    ])
    assert conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0] == 2
    assert db.real_company_count(conn) == 1


def test_linking_is_idempotent(conn):
    row = {"cik": "0000034088", "ticker": "XOM", "name": "Old",
           "exchange": None, "successor_cik": "0002115436"}
    db.upsert_companies(conn, [row])
    db.upsert_companies(conn, [row])
    assert conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0] == 1


def test_a_predecessor_is_still_collected_like_any_company(conn):
    """No change to P2-03..P2-07 — that is the point of storing it this way."""
    db.upsert_companies(conn, [
        {"cik": "0002115436", "ticker": "XOM", "name": "New", "exchange": "NYSE"},
        {"cik": "0000034088", "ticker": "XOM", "name": "Old", "exchange": None,
         "successor_cik": "0002115436"},
    ])
    ciks = {r["cik"] for r in db.companies_for_collection(conn, ["XOM"])}
    assert ciks == {"0002115436", "0000034088"}


# -- pagination: `recent` alone is not a CIK's whole history ---------------

def test_is_successor_checks_files_pages_when_given_a_client(cfg):
    """An actively-traded successor can file enough Form 4s that its own
    8-K12B rolls off `recent`. Checking `recent` alone then misses it forever,
    with no error — a company's whole predecessor link silently never
    happens."""
    subs = submissions(["10-Q", "4"])
    subs["filings"]["files"] = [{"name": "old-page.json"}]
    pages = {"old-page.json": {
        "form": ["8-K12B"], "acceptanceDateTime": ["2020-01-02T20:00:00Z"],
        "accessionNumber": ["old-1"],
    }}
    client = FakeClient({}, pages=pages)
    assert not is_successor(cfg, subs), (
        "without a client, only `recent` is checked — the historical behaviour")
    assert is_successor(cfg, subs, client=client), (
        "with a client, the marker in a `files` page must be found too")


def test_verify_predecessor_checks_files_pages_for_prior_8ks(cfg, conn):
    """The predecessor's only 8-K before the window can likewise sit in a
    `files` page rather than `recent`."""
    subs = submissions(["4"])  # nothing in `recent` counts as prior evidence
    subs["filings"]["files"] = [{"name": "old.json"}]
    pages = {"old.json": {
        "form": ["8-K"], "acceptanceDateTime": ["2020-01-02T20:00:00Z"],
        "accessionNumber": ["old-1"],
    }}
    client = FakeClient({"0000099999": subs}, pages=pages)
    ok, why = verify_predecessor(cfg, conn, client, "0000099999", "2911",
                                 WINDOW_START)
    assert ok and "1 8-K" in why


# -- link_predecessors: a total EDGAR outage must not look like "nothing to link"

def _no_history_filing(cik, ticker, acceptance_utc):
    return {"accession_no": f"seed-{cik}", "cik": cik, "ticker": ticker,
            "form": "8-K", "items": "", "acceptance_utc": acceptance_utc,
            "filing_date_utc": None, "report_date_utc": None,
            "primary_doc": None, "fetched_utc": 0}


def test_every_candidate_fetch_failing_raises(cfg, conn):
    """A bare `continue` on every EdgarRequestError made a total outage during
    `--link-predecessors` look identical to a clean 'nothing to link' run."""
    db.upsert_companies(conn, [{"cik": "0002115436", "ticker": "XOM",
                               "name": "ExxonMobil Holdings", "exchange": "NYSE"}])
    db.upsert_filings(conn, [
        _no_history_filing("0002115436", "XOM", WINDOW_START + 3600)])
    client = FakeClient({"0002115436": EdgarRequestError("HTTP 503 from EDGAR")})
    with pytest.raises(SystemExit, match="EVERY one of 1"):
        link_predecessors(cfg, conn, client=client)


def test_more_candidates_than_the_cap_is_reported_not_truncated(cfg, conn,
                                                               monkeypatch):
    """A prefix of a long candidate list is worse than no answer.

    Verifying only the first N can leave exactly one survivor, which reads as a
    confident link, while the real predecessor sat at N+1 and was never looked
    at. A wrong link stores another company's CIK under this ticker, so its
    8-Ks are collected as this company's — and nothing in the output shows it.
    "Too many to check" has to be an answer of its own.
    """
    db.upsert_companies(conn, [{"cik": "0002115436", "ticker": "XOM",
                                "name": "Common Holdings", "exchange": "NYSE"}])
    db.upsert_filings(conn, [
        _no_history_filing("0002115436", "XOM", WINDOW_START + 3600)])
    client = FakeClient({"0002115436": submissions(
        ["8-K12B"], name="Common Holdings Corp")})

    crowded = [(f"COMMON HOLDINGS {i} CORP", f"{i:010d}")
               for i in range(MAX_PREDECESSOR_CANDIDATES + 1)]
    monkeypatch.setattr(edgar, "load_cik_lookup", lambda cfg_, client_: {})
    monkeypatch.setattr(edgar, "propose_predecessors",
                        lambda name, lookup, cik: crowded)

    result = link_predecessors(cfg, conn, client=client, dry_run=True)

    assert result["linked"] == []
    assert len(result["unresolved"]) == 1
    assert "more than the" in result["unresolved"][0]["reasons"][0]
    # ...and nothing was written under the successor's ticker.
    assert conn.execute(
        "SELECT COUNT(*) FROM companies WHERE successor_cik IS NOT NULL"
    ).fetchone()[0] == 0


def test_partial_candidate_fetch_failures_are_counted_not_dropped(cfg, conn):
    db.upsert_companies(conn, [
        {"cik": "0000000020", "ticker": "AAA", "name": "AAA Corp", "exchange": "NYSE"},
        {"cik": "0000000021", "ticker": "BBB", "name": "BBB Corp", "exchange": "NYSE"},
    ])
    db.upsert_filings(conn, [
        _no_history_filing("0000000020", "AAA", WINDOW_START + 3600),
        _no_history_filing("0000000021", "BBB", WINDOW_START + 3600),
    ])
    client = FakeClient({"0000000020": EdgarRequestError("HTTP 503"),
                         "0000000021": submissions(["8-K", "10-Q"])})
    result = link_predecessors(cfg, conn, client=client)
    assert result["candidates"] == 2
    assert result["fetch_failed"] == 1
    assert result["successors"] == 0, "the reachable candidate filed no marker"
