"""SEC EDGAR collector tests (plan §6.4 confirmation source). No network."""

import json

import pytest
import requests

from src.collectors import edgar
from src.collectors.edgar import (
    EdgarUnavailable, describe, filing_url, filings_to_rows, load_cik_map,
)


def _cfg(tmp_path=None):
    return {
        "news": {"edgar_base": "https://efts.example/search",
                 "edgar_tickers_url": "https://sec.example/tickers.json",
                 "edgar_forms": ["8-K", "425"],
                 "edgar_filing_hour_utc": 21},
        "paths": {"symbol_dir": str(tmp_path) if tmp_path else "."},
    }


def _hit(items=("2.01",), file_date="2025-03-04", form="8-K",
         _id="0000320193-25-000007:doc.htm", cik="0000320193"):
    return {"_id": _id,
            "_source": {"root_form": form, "items": list(items),
                        "file_date": file_date, "ciks": [cik],
                        "display_names": ["Apple Inc.  (AAPL)  (CIK 0000320193)"]}}


# --- rendering a filing as a readable claim ---------------------------------

def test_item_codes_become_readable_claims():
    """'2.01' means the acquisition actually closed — the labeler needs that."""
    text = describe(_hit(items=["2.01"]))
    assert "Completion of Acquisition or Disposition of Assets" in text
    assert "Apple Inc." in text
    assert "8-K" in text


def test_several_items_are_all_described():
    text = describe(_hit(items=["2.02", "9.01"]))
    assert "Results of Operations" in text and "Financial Statements" in text


def test_an_unknown_item_code_is_dropped_not_guessed():
    text = describe(_hit(items=["2.01", "99.99"]))
    assert "99.99" not in text and "Completion of Acquisition" in text


def test_a_filing_with_no_items_still_names_its_form():
    """S-4s and 425s carry no item codes but are still evidence."""
    text = describe(_hit(items=[], form="425"))
    assert "425" in text and "Apple Inc." in text


# --- rows -------------------------------------------------------------------

def test_rows_are_attributed_to_sec_so_they_count_as_credible():
    (row,) = filings_to_rows([_hit()], "AAPL", _cfg())
    assert row[3] == "sec.gov" and row[5] == "edgar" and row[1] == "AAPL"


def test_the_filing_timestamp_errs_late_within_the_day():
    """Erring early would invent foresight; late only forfeits the bonus."""
    (row,) = filings_to_rows([_hit(file_date="2025-03-04")], "AAPL", _cfg())
    from src.utils.timeutils import date_str_to_ts
    assert row[4] == date_str_to_ts("2025-03-04") + 21 * 3600


def test_a_filing_without_a_date_is_skipped():
    hit = _hit()
    del hit["_source"]["file_date"]
    assert filings_to_rows([hit], "AAPL", _cfg()) == []


def test_the_url_is_the_canonical_edgar_archive_path():
    url = filing_url(_hit(_id="0000320193-25-000007:a8k.htm"))
    assert url == ("https://www.sec.gov/Archives/edgar/data/320193/"
                   "000032019325000007/a8k.htm")


def test_two_filings_produce_distinct_keys():
    """url is the news table's primary key; a collision would drop evidence."""
    rows = filings_to_rows([_hit(_id="0000320193-25-000007:a.htm"),
                            _hit(_id="0000320193-25-000008:b.htm")],
                           "AAPL", _cfg())
    assert len({r[0] for r in rows}) == 2


# --- transport --------------------------------------------------------------

class _Resp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload if payload is not None else {"hits": {"hits": []}}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def test_a_transient_500_is_retried(monkeypatch):
    """EDGAR 500s intermittently on queries that succeed on retry."""
    monkeypatch.setattr(edgar.Backoff, "sleep", lambda self, why=None: None)
    replies = [_Resp(500), _Resp(200, {"hits": {"hits": [_hit()]}})]

    class Session:
        def get(self, *a, **k):
            return replies.pop(0)

    assert len(edgar.fetch_filings(_cfg(), Session(), "0000320193", 0, 1)) == 1


def test_giving_up_raises_rather_than_reporting_no_filings(monkeypatch):
    """Silence is read as FALSE, so a failed query must never look empty."""
    monkeypatch.setattr(edgar.Backoff, "sleep", lambda self, why=None: None)

    class Session:
        def get(self, *a, **k):
            return _Resp(500)

    with pytest.raises(EdgarUnavailable):
        edgar.fetch_filings(_cfg(), Session(), "0000320193", 0, 1)


def test_the_cik_map_is_cached_after_one_download(tmp_path):
    cfg = _cfg(tmp_path)
    calls = []

    class Session:
        def get(self, url, **k):
            calls.append(url)
            return _Resp(200, {"0": {"cik_str": 320193, "ticker": "aapl"}})

    session = Session()
    assert load_cik_map(cfg, session)["AAPL"] == "0000320193"
    assert load_cik_map(cfg, session)["AAPL"] == "0000320193"
    assert len(calls) == 1


def test_cik_is_zero_padded_as_edgar_expects(tmp_path):
    cfg = _cfg(tmp_path)
    (tmp_path / "sec_company_tickers.json").write_text(
        json.dumps({"0": {"cik_str": 1234, "ticker": "tiny"}}))
    assert load_cik_map(cfg, None)["TINY"] == "0000001234"
